"""
MJX Diffsim Franka — EE-Velocity (v2, Simplified)

Newton diffsim_franka_ee_vel.py'ın MJX portudur.

Learnable parameter: v_target ∈ ℝ³ — EE hedef hızı (world frame)

ÖNCEKİ VERSİYONDAKİ SORUNLAR VE ÇÖZÜMLER:
  1. Jacobian CPU'da hesaplanıp her 10 iter JIT re-trace → YAVAŞ
     → Çözüm: Jacobian'ı bir kez hesapla, JIT'e sabit olarak ver
     
  2. ctrl = q + q_dot*dt (position mode integration) → HATALI
     → Çözüm: Jacobian'ı kullanarak q_dot hesapla, sonra 
       ctrl[0:7] = q_home + J^† v * frame_dt (sabit referanstan offset)
       
  3. current_q güncelleniyor ama mjx_data_init hep home → TUTARSIZ
     → Çözüm: Her iter aynı home'dan başla, v_target tek shot
       (Newton versiyonunda da TRAJ_STEPS=1 idi)

YENİ YAKLAŞIM:
  - Jacobian home pozisyonunda bir kez hesaplanır (script başında)
  - v_target (3,) optimize edilir  
  - Her iterasyon: home'dan başla → ctrl = home_q + J^† v * dt → physics
  - jax.grad tüm pipeline'dan geçer (J^† differentiable)
  - JIT sadece bir kez compile eder

Usage:
    source ~/mjx_diffrl_env/bin/activate
    python diffsim_franka_ee_vel_mjx.py
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
# SOLVER MONKEY-PATCH
# ---------------------------------------------------------------------------
import mujoco.mjx._src.solver as _mjx_solver

_original_solve = _mjx_solver.solve

def _patched_solve(m, d):
    original_while_loop = jax.lax.while_loop
    def fixed_iter_while_loop(cond_fun, body_fun, init_val):
        def fori_body(_, carry):
            return body_fun(carry)
        return jax.lax.fori_loop(0, m.opt.iterations, fori_body, init_val)
    jax.lax.while_loop = fixed_iter_while_loop
    try:
        result = _original_solve(m, d)
    finally:
        jax.lax.while_loop = original_while_loop
    return result

_mjx_solver.solve = _patched_solve
print("✓ MJX solver patched")

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="MJX EE-Velocity Diffsim")
parser.add_argument("--iters", type=int, default=200)
parser.add_argument("--lr", type=float, default=3e-2)
parser.add_argument("--substeps", type=int, default=100)
parser.add_argument("--solver-iters", type=int, default=4)
parser.add_argument("--grad-clip", type=float, default=50.0)
parser.add_argument("--damping", type=float, default=0.001)
parser.add_argument("--no-viewer", action="store_true")
parser.add_argument("--xml", type=str,
                    default="/home/emre/mujoco_playground/mujoco_playground/"
                            "external_deps/mujoco_menagerie/franka_emika_panda/"
                            "mjx_single_cube.xml")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SIM_SUBSTEPS = args.substeps
TRAIN_ITERS  = args.iters
TRAIN_LR     = args.lr
GRAD_CLIP    = args.grad_clip
DAMPING      = args.damping
V_MAX        = 3.0

TARGET_POS = jnp.array([0.4, 0.15, 0.55])
EE_BODY_IDX = 9  # "hand"
N_JOINTS = 7

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
frame_dt = SIM_SUBSTEPS * sim_dt

print(f"  nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu}")
print(f"  sim_dt={sim_dt}, frame_dt={frame_dt:.4f}s")
print(f"  EE home pos: {mj_data.xpos[EE_BODY_IDX]}")

home_ctrl = jnp.array(mj_data.ctrl)
home_q_arm = jnp.array(mj_data.qpos[:N_JOINTS])

# ---------------------------------------------------------------------------
# Jacobian — bir kez hesapla, home pozisyonunda
# ---------------------------------------------------------------------------
print("\nComputing Jacobian at home position...")
jacp = np.zeros((3, mj_model.nv))
jacr = np.zeros((3, mj_model.nv))
mujoco.mj_jac(mj_model, mj_data, jacp, jacr,
              mj_data.site_xpos[0], EE_BODY_IDX)

# Sadece arm joint sütunları (ilk 7)
J_p = jnp.array(jacp[:, :N_JOINTS])
print(f"  J_p shape: {J_p.shape}")
print(f"  J_p:\n{np.array(J_p)}")

# Jacobian'ın rank'ı kontrol et
s = np.linalg.svd(np.array(J_p), compute_uv=False)
print(f"  Singular values: {s}")
print(f"  Rank: {np.sum(s > 1e-6)}")

# ---------------------------------------------------------------------------
# MJX model
# ---------------------------------------------------------------------------
print(f"\nSetting up MJX (solver_iters={args.solver_iters})...")
mj_model.opt.iterations = args.solver_iters
mj_model.opt.ls_iterations = 4
mjx_model = mjx.put_model(mj_model)

mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
mjx_data_init = mjx.put_data(mj_model, mj_data)

# ---------------------------------------------------------------------------
# Learnable parameter
# ---------------------------------------------------------------------------
param = jnp.zeros(3)
optimizer = optax.adam(TRAIN_LR)
opt_state = optimizer.init(param)

print(f"\n{'='*60}")
print(f"EE-VELOCITY DIFFSIM (MJX v2)")
print(f"  Learnable: v_target ∈ ℝ³")
print(f"  Target: [{TARGET_POS[0]:.3f}, {TARGET_POS[1]:.3f}, {TARGET_POS[2]:.3f}]")
print(f"  EE home: [{float(mj_data.xpos[EE_BODY_IDX][0]):.3f}, "
      f"{float(mj_data.xpos[EE_BODY_IDX][1]):.3f}, "
      f"{float(mj_data.xpos[EE_BODY_IDX][2]):.3f}]")
print(f"  Substeps: {SIM_SUBSTEPS}, frame_dt: {frame_dt:.4f}s")
print(f"  Damping: {DAMPING}, LR: {TRAIN_LR}")
print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# Loss function — tamamen JIT-friendly, re-trace yok
# ---------------------------------------------------------------------------
def loss_fn(v_target):
    """
    v_target (3,) → loss
    
    Pipeline:
      1. q_dot = J^T (JJ^T + λI)^{-1} v_target   (damped pinv)
      2. ctrl[0:7] = home_q + q_dot * frame_dt      (position target)
      3. mjx.step × N_SUBSTEPS                      (physics)
      4. loss = ||ee_pos - target||²
      
    Gradient akar: v_target → q_dot → ctrl → physics → ee_pos → loss
    J_p sabit (closure'dan) — gradient geçmez ama sorun değil,
    v_target üzerinden tüm yönleri keşfedebilir.
    """
    # 1. Damped pseudoinverse: v_target → q_dot
    A = J_p @ J_p.T + DAMPING * jnp.eye(3)
    y = jnp.linalg.solve(A, v_target)
    q_dot = J_p.T @ y
    
    # 2. Position target = home + q_dot * dt
    q_target_arm = home_q_arm + q_dot * frame_dt
    ctrl = home_ctrl.at[:N_JOINTS].set(q_target_arm)
    
    # 3. Forward physics
    def step_fn(state, _):
        state = state.replace(ctrl=ctrl)
        state = mjx.step(mjx_model, state)
        return state, None
    
    final_state, _ = jax.lax.scan(
        step_fn, mjx_data_init, None, length=SIM_SUBSTEPS
    )
    
    # 4. Loss
    ee_pos = final_state.site_xpos[0]
    delta = ee_pos - TARGET_POS
    return jnp.dot(delta, delta)


def forward_rollout(v_target):
    """Aynı pipeline ama final state döndürür (viewer için)."""
    A = J_p @ J_p.T + DAMPING * jnp.eye(3)
    y = jnp.linalg.solve(A, v_target)
    q_dot = J_p.T @ y
    
    q_target_arm = home_q_arm + q_dot * frame_dt
    ctrl = home_ctrl.at[:N_JOINTS].set(q_target_arm)
    
    def step_fn(state, _):
        state = state.replace(ctrl=ctrl)
        state = mjx.step(mjx_model, state)
        return state, None
    
    final_state, _ = jax.lax.scan(
        step_fn, mjx_data_init, None, length=SIM_SUBSTEPS
    )
    return final_state


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
print(f"  Initial grad: [{float(grad_val[0]):.4f}, {float(grad_val[1]):.4f}, {float(grad_val[2]):.4f}]")

# Gradient yönü mantıklı mı kontrol et
ee_home = np.array(mj_data.site_xpos[0])
desired_direction = np.array(TARGET_POS) - ee_home
print(f"\n  Sanity check:")
print(f"    EE→Target direction: [{desired_direction[0]:+.3f}, {desired_direction[1]:+.3f}, {desired_direction[2]:+.3f}]")
print(f"    Gradient direction:  [{float(grad_val[0]):+.3f}, {float(grad_val[1]):+.3f}, {float(grad_val[2]):+.3f}]")
print(f"    (Gradient negatif olmalı — v_target'ı artırınca loss azalmalı)")

print("\nJIT compiling forward rollout...")
t0 = time.time()
jit_rollout = jax.jit(forward_rollout)
final = jit_rollout(param)
final.xpos.block_until_ready()
print(f"  JIT compile: {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# Live viewer
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
    print("CANLI VIEWER AÇILDI — ESC ile kapat")
    print("=" * 60 + "\n")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
loss_history = []
v_history = []
ee_history = []

print("Training başlıyor...\n")
train_start = time.time()

for iteration in range(TRAIN_ITERS):
    if USE_VIEWER and not viewer_handle.is_running():
        print("\nViewer kapatıldı.")
        break
    
    # Forward + backward
    loss_val, grad_val = value_and_grad_fn(param)
    
    loss_f = float(loss_val)
    grad_np = np.array(grad_val)
    v_np = np.array(param)
    
    loss_history.append(loss_f)
    v_history.append(v_np.copy())
    
    # Gradient clipping
    grad_norm = np.linalg.norm(grad_np)
    if grad_norm > GRAD_CLIP:
        grad_val = grad_val * (GRAD_CLIP / grad_norm)
    
    # Adam update
    updates, opt_state = optimizer.update(grad_val, opt_state, param)
    param = optax.apply_updates(param, updates)
    
    # Velocity clamping
    param = jnp.clip(param, -V_MAX, V_MAX)
    
    # Viewer update
    final_state = jit_rollout(param)
    ee_pos = np.array(final_state.site_xpos[0])
    ee_history.append(ee_pos.copy())
    
    print(f"Iter {iteration:4d}:  "
          f"v=[{v_np[0]:+.3f}, {v_np[1]:+.3f}, {v_np[2]:+.3f}]  "
          f"grad=[{grad_np[0]:+.4f}, {grad_np[1]:+.4f}, {grad_np[2]:+.4f}]  "
          f"loss={loss_f:.6f}  "
          f"EE=[{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    
    if USE_VIEWER and viewer_handle.is_running():
        mj_data_render.qpos[:] = np.array(final_state.qpos)
        mj_data_render.qvel[:] = np.array(final_state.qvel)
        mujoco.mj_forward(mj_model_render, mj_data_render)
        mj_data_render.mocap_pos[mocap_id] = np.array(TARGET_POS)
        viewer_handle.sync()

train_time = time.time() - train_start
print(f"\nTraining done: {train_time:.1f}s ({len(loss_history)} iters, "
      f"{train_time/max(1,len(loss_history)):.3f}s/iter)")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"SONUÇLAR")
print(f"{'='*60}")
print(f"  Initial loss: {loss_history[0]:.6f}")
print(f"  Final loss:   {loss_history[-1]:.6f}")
print(f"  Reduction:    {loss_history[-1]/loss_history[0]*100:.1f}%")
print(f"  Final v:      [{v_history[-1][0]:+.4f}, {v_history[-1][1]:+.4f}, {v_history[-1][2]:+.4f}]")
print(f"  Final EE:     [{ee_history[-1][0]:.4f}, {ee_history[-1][1]:.4f}, {ee_history[-1][2]:.4f}]")
print(f"  Target:       [{float(TARGET_POS[0]):.4f}, {float(TARGET_POS[1]):.4f}, {float(TARGET_POS[2]):.4f}]")
print(f"  Final dist:   {np.linalg.norm(ee_history[-1] - np.array(TARGET_POS)):.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

axes[0].plot(loss_history, linewidth=1.5)
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss Curve")
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale("log")

v_arr = np.array(v_history)
axes[1].plot(v_arr[:, 0], label="vx", linewidth=1.5)
axes[1].plot(v_arr[:, 1], label="vy", linewidth=1.5)
axes[1].plot(v_arr[:, 2], label="vz", linewidth=1.5)
axes[1].set_ylabel("v_target (m/s)")
axes[1].set_title("Learned EE Velocity")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

ee_arr = np.array(ee_history)
target_np = np.array(TARGET_POS)
axes[2].plot(ee_arr[:, 0], label="EE_x", linewidth=1.5)
axes[2].plot(ee_arr[:, 1], label="EE_y", linewidth=1.5)
axes[2].plot(ee_arr[:, 2], label="EE_z", linewidth=1.5)
axes[2].axhline(target_np[0], color="C0", linestyle="--", alpha=0.5)
axes[2].axhline(target_np[1], color="C1", linestyle="--", alpha=0.5)
axes[2].axhline(target_np[2], color="C2", linestyle="--", alpha=0.5)
axes[2].set_ylabel("Position (m)")
axes[2].set_xlabel("Iteration")
axes[2].set_title("EE Position vs Target")
axes[2].legend(loc="center right", fontsize=8)
axes[2].grid(True, alpha=0.3)

fig.suptitle(f"MJX EE-Velocity Diffsim — {len(loss_history)} iters")
fig.tight_layout()
plot_path = "diffsim_ee_vel_results.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {plot_path}")

if USE_VIEWER and viewer_handle.is_running():
    print("\nViewer açık — ESC ile kapat.")
    try:
        while viewer_handle.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nÇıkılıyor.")