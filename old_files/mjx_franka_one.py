"""
MJX Diffsim Franka — 1-DOF (Joint 0 Only)

Newton diffsim_franka_one.py'ın MJX portudur.
Sadece base joint (joint 0) hedefini optimize eder.

Pipeline:
  1. Forward: mjx.step ile SIM_SUBSTEPS adım fizik çalıştır
  2. Loss:    ||EE_pos - target_pos||²
  3. Backward: jax.grad ile ∂loss/∂ctrl[0] hesapla
  4. Adam:    ctrl[0] güncelle

SOLVER PATCH:
  MJX'in solver'ı iterations>1'de while_loop kullanır → jax.grad uyumsuz.
  Bu script solver'ı monkey-patch'leyerek while_loop'u fori_loop ile
  değiştirir. Böylece iterations=4 ile hem gradient hem stabil contact alırız.
  Cube stacking'de de aynı yaklaşım çalışacak.

Usage:
    source ~/mjx_diffrl_env/bin/activate
    python diffsim_franka_one_mjx.py
    python diffsim_franka_one_mjx.py --iters 300 --lr 0.02 --solver-iters 4
"""

import argparse
import math
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
import mujoco
from mujoco import mjx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# SOLVER MONKEY-PATCH: while_loop → fori_loop
# ---------------------------------------------------------------------------
# MJX solver.py'daki solve() fonksiyonu iterations>1'de jax.lax.while_loop
# kullanıyor. while_loop dynamic termination'a sahip olduğu için jax.grad
# ile uyumsuz. Biz bunu static-bound fori_loop ile değiştiriyoruz.
#
# Bu yaklaşım:
# - Solver'ı her zaman tam N iterasyon çalıştırır (erken çıkış yok)
# - jax.grad ile tam uyumlu
# - Contact çözümü doğru (iterations=4 ile küp kaymaz)
# - MJX update'lerinden etkilenmez (sadece while_loop çağrısını yakalar)
# ---------------------------------------------------------------------------
import mujoco.mjx._src.solver as _mjx_solver

_original_solve = _mjx_solver.solve

def _patched_solve(m, d):
    """Solver wrapper: while_loop'u fori_loop'a çevirmek için
    jax.lax.while_loop'u geçici olarak override eder."""
    
    # Orijinal while_loop'u sakla
    original_while_loop = jax.lax.while_loop
    
    def fixed_iter_while_loop(cond_fun, body_fun, init_val):
        """while_loop yerine sabit N iterasyonlu fori_loop kullan.
        
        body_fun'ı m.opt.iterations kez çalıştırır.
        Erken çıkış yapmaz ama jax.grad ile uyumlu olur.
        """
        def fori_body(_, carry):
            return body_fun(carry)
        return jax.lax.fori_loop(0, m.opt.iterations, fori_body, init_val)
    
    # while_loop'u geçici olarak override et
    jax.lax.while_loop = fixed_iter_while_loop
    try:
        result = _original_solve(m, d)
    finally:
        # Geri al
        jax.lax.while_loop = original_while_loop
    
    return result

# Patch'i uygula
_mjx_solver.solve = _patched_solve
print("✓ MJX solver patched: while_loop → fori_loop (differentiable iterations>1)")

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="MJX 1-DOF Diffsim Franka")
parser.add_argument("--iters", type=int, default=200, help="Training iterations")
parser.add_argument("--lr", type=float, default=3e-2, help="Learning rate")
parser.add_argument("--substeps", type=int, default=100, help="Physics substeps")
parser.add_argument("--solver-iters", type=int, default=4, help="MJX solver iterations")
parser.add_argument("--grad-clip", type=float, default=5.0, help="Max gradient magnitude")
parser.add_argument("--no-viewer", action="store_true", help="Live viewer açma")
parser.add_argument("--xml", type=str,
                    default="/home/emre/mujoco_playground/mujoco_playground/"
                            "external_deps/mujoco_menagerie/franka_emika_panda/"
                            "mjx_single_cube.xml")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SIM_SUBSTEPS   = args.substeps
TRAIN_ITERS    = args.iters
TRAIN_LR       = args.lr
GRAD_CLIP      = args.grad_clip

TARGET_POS = jnp.array([0.55, 0.35, 0.45])
EE_BODY_IDX = 9  # "hand"

# ---------------------------------------------------------------------------
# Load MuJoCo model
# ---------------------------------------------------------------------------
print("\nLoading MuJoCo model...")
mj_model = mujoco.MjModel.from_xml_path(args.xml)
mj_data = mujoco.MjData(mj_model)

key_id = mj_model.keyframe("home").id
mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
mujoco.mj_forward(mj_model, mj_data)

sim_dt = mj_model.opt.timestep
print(f"  nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu}, dt={sim_dt}")
print(f"  EE body: {EE_BODY_IDX} ({mj_model.body(EE_BODY_IDX).name})")
print(f"  EE home pos: {mj_data.xpos[EE_BODY_IDX]}")

home_ctrl = jnp.array(mj_data.ctrl)
print(f"  Home ctrl: {home_ctrl}")

# ---------------------------------------------------------------------------
# MJX model — tek model, solver patch sayesinde iterations>1 differentiable
# ---------------------------------------------------------------------------
print(f"\nSetting up MJX (iterations={args.solver_iters})...")
mj_model.opt.iterations = args.solver_iters
mj_model.opt.ls_iterations = 4

mjx_model = mjx.put_model(mj_model)
mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
mjx_data_init = mjx.put_data(mj_model, mj_data)

# ---------------------------------------------------------------------------
# Learnable parameter
# ---------------------------------------------------------------------------
init_q0 = float(mj_data.ctrl[0])
param = jnp.array(init_q0)

optimizer = optax.adam(TRAIN_LR)
opt_state = optimizer.init(param)

print(f"\n{'='*60}")
print(f"1-DOF DIFFSIM (MJX): Optimizing ctrl[0] (base rotation)")
print(f"  Initial ctrl[0]: {init_q0:.4f} rad ({math.degrees(init_q0):.1f}°)")
print(f"  Target EE pos: [{TARGET_POS[0]:.3f}, {TARGET_POS[1]:.3f}, {TARGET_POS[2]:.3f}]")
print(f"  Substeps: {SIM_SUBSTEPS}")
print(f"  Solver iterations: {args.solver_iters}")
print(f"  LR: {TRAIN_LR}")
print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------
def loss_fn(ctrl_0_val):
    """Forward rollout + EE loss."""
    ctrl = home_ctrl.at[0].set(ctrl_0_val)
    
    def step_fn(state, _):
        state = state.replace(ctrl=ctrl)
        state = mjx.step(mjx_model, state)
        return state, None
    
    final_state, _ = jax.lax.scan(step_fn, mjx_data_init, None, length=SIM_SUBSTEPS)
    
    ee_pos = final_state.xpos[EE_BODY_IDX]
    delta = ee_pos - TARGET_POS
    return jnp.dot(delta, delta)

# ---------------------------------------------------------------------------
# Forward rollout (for viewer — same model, no separate sim needed)
# ---------------------------------------------------------------------------
def forward_rollout(ctrl_0_val):
    """Forward rollout — returns final state."""
    ctrl = home_ctrl.at[0].set(ctrl_0_val)
    
    def step_fn(state, _):
        state = state.replace(ctrl=ctrl)
        state = mjx.step(mjx_model, state)
        return state, state
    
    final_state, all_states = jax.lax.scan(step_fn, mjx_data_init, None, length=SIM_SUBSTEPS)
    return final_state, all_states

# ---------------------------------------------------------------------------
# JIT compile
# ---------------------------------------------------------------------------
print("JIT compiling loss + grad...")
t0 = time.time()
value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

loss_val, grad_val = value_and_grad_fn(param)
loss_val.block_until_ready()
print(f"  JIT compile: {time.time()-t0:.1f}s")
print(f"  Initial loss: {float(loss_val):.6f}")
print(f"  Initial grad: {float(grad_val):.6f}")

print("\nJIT compiling forward rollout...")
t0 = time.time()
jit_rollout = jax.jit(forward_rollout)
final, _ = jit_rollout(param)
final.xpos.block_until_ready()
print(f"  JIT compile: {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# Live viewer setup
# ---------------------------------------------------------------------------
USE_VIEWER = not args.no_viewer

if USE_VIEWER:
    from mujoco import viewer as mj_viewer
    
    mj_model_render = mujoco.MjModel.from_xml_path(args.xml)
    mj_data_render = mujoco.MjData(mj_model_render)
    
    mj_data_render.qpos[:] = np.array(mjx_data_init.qpos)
    mj_data_render.qvel[:] = np.array(mjx_data_init.qvel)
    mujoco.mj_forward(mj_model_render, mj_data_render)
    
    mocap_id = mj_model_render.body("mocap_target").mocapid[0]
    mj_data_render.mocap_pos[mocap_id] = np.array(TARGET_POS)
    
    viewer_handle = mj_viewer.launch_passive(mj_model_render, mj_data_render)
    
    print("\n" + "=" * 60)
    print("CANLI VIEWER AÇILDI")
    print("  Kırmızı kutu = hedef pozisyon")
    print("  Robot eğitim boyunca hedefe yaklaşacak")
    print("  ESC = kapat")
    print("=" * 60 + "\n")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
loss_history = []
q0_history = []
ee_history = []

print("Training başlıyor...\n")
train_start = time.time()

for iteration in range(TRAIN_ITERS):
    if USE_VIEWER and not viewer_handle.is_running():
        print("\nViewer kapatıldı, eğitim durduruluyor.")
        break
    
    loss_val, grad_val = value_and_grad_fn(param)
    
    loss_f = float(loss_val)
    grad_f = float(grad_val)
    q0_f = float(param)
    
    loss_history.append(loss_f)
    q0_history.append(q0_f)
    
    grad_clipped = jnp.clip(grad_val, -GRAD_CLIP, GRAD_CLIP)
    updates, opt_state = optimizer.update(grad_clipped, opt_state, param)
    param = optax.apply_updates(param, updates)
    
    final_state, _ = jit_rollout(param)
    ee_pos = np.array(final_state.xpos[EE_BODY_IDX])
    ee_history.append(ee_pos.copy())
    
    print(f"Iter {iteration:4d}:  "
          f"q0={q0_f:+.4f} rad ({math.degrees(q0_f):+.1f}°)  "
          f"grad={grad_f:+.6f}  "
          f"loss={loss_f:.6f}  "
          f"EE=[{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    
    if USE_VIEWER and viewer_handle.is_running():
        mj_data_render.qpos[:] = np.array(final_state.qpos)
        mj_data_render.qvel[:] = np.array(final_state.qvel)
        mujoco.mj_forward(mj_model_render, mj_data_render)
        mj_data_render.mocap_pos[mocap_id] = np.array(TARGET_POS)
        viewer_handle.sync()

train_time = time.time() - train_start
print(f"\nTraining done: {train_time:.1f}s ({TRAIN_ITERS} iters, "
      f"{train_time/max(1,len(loss_history)):.3f}s/iter)")

# ---------------------------------------------------------------------------
# Final results
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"SONUÇLAR")
print(f"{'='*60}")
print(f"  Initial loss: {loss_history[0]:.6f}")
print(f"  Final loss:   {loss_history[-1]:.6f}")
print(f"  Reduction:    {loss_history[-1]/loss_history[0]*100:.1f}%")
print(f"  Initial q0:   {q0_history[0]:+.4f} rad ({math.degrees(q0_history[0]):+.1f}°)")
print(f"  Final q0:     {q0_history[-1]:+.4f} rad ({math.degrees(q0_history[-1]):+.1f}°)")
print(f"  Final EE:     [{ee_history[-1][0]:.4f}, {ee_history[-1][1]:.4f}, {ee_history[-1][2]:.4f}]")
print(f"  Target:       [{float(TARGET_POS[0]):.4f}, {float(TARGET_POS[1]):.4f}, {float(TARGET_POS[2]):.4f}]")
print(f"  Final dist:   {np.linalg.norm(ee_history[-1] - np.array(TARGET_POS)):.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

axes[0].plot(loss_history, linewidth=1.5)
axes[0].set_ylabel("Loss (||EE - target||²)")
axes[0].set_title("Loss Curve")
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale("log")

q0_deg = [math.degrees(q) for q in q0_history]
axes[1].plot(q0_deg, linewidth=1.5)
axes[1].set_ylabel("ctrl[0] (degrees)")
axes[1].set_title("Joint 0 Target")
axes[1].grid(True, alpha=0.3)

ee_arr = np.array(ee_history)
target_np = np.array(TARGET_POS)
axes[2].plot(ee_arr[:, 0], label="EE_x", linewidth=1.5)
axes[2].plot(ee_arr[:, 1], label="EE_y", linewidth=1.5)
axes[2].plot(ee_arr[:, 2], label="EE_z", linewidth=1.5)
axes[2].axhline(target_np[0], color="C0", linestyle="--", alpha=0.5, label=f"target_x={target_np[0]:.2f}")
axes[2].axhline(target_np[1], color="C1", linestyle="--", alpha=0.5, label=f"target_y={target_np[1]:.2f}")
axes[2].axhline(target_np[2], color="C2", linestyle="--", alpha=0.5, label=f"target_z={target_np[2]:.2f}")
axes[2].set_ylabel("Position (m)")
axes[2].set_xlabel("Iteration")
axes[2].set_title("EE Position vs Target")
axes[2].legend(loc="center right", fontsize=8)
axes[2].grid(True, alpha=0.3)

fig.suptitle(f"MJX 1-DOF Diffsim — {TRAIN_ITERS} iters, LR={TRAIN_LR}, solver_iters={args.solver_iters}")
fig.tight_layout()
plot_path = "diffsim_1dof_results.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {plot_path}")

# ---------------------------------------------------------------------------
# Keep viewer open
# ---------------------------------------------------------------------------
if USE_VIEWER and viewer_handle.is_running():
    print("\nEğitim bitti. Viewer açık — ESC ile kapat.")
    try:
        while viewer_handle.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nCtrl+C — çıkılıyor.")