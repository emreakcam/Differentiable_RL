"""
MJX BPTT — Closed-Loop EE-Velocity (Fast Compile)

EE-vel'in hızlı compile + iyi convergence'ı
+ BPTT'nin closed-loop yapısı
+ Nested lax.scan ile hızlı JIT

MLP her frame step'te:
  obs → v_target(3) + finger(1)
  q_dot = J† v_target (Jacobian bir kez hesaplanmış)
  ctrl = current_q + q_dot * frame_dt

Tüm step'ler lax.scan içinde — JIT tek step trace eder.

Usage:
    source ~/mjx_diffrl_env/bin/activate
    python diffsim_bptt_mjx.py
    python diffsim_bptt_mjx.py --steps 32 --iters 500
"""

import argparse
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
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--substeps", type=int, default=16)
parser.add_argument("--iters", type=int, default=300)
parser.add_argument("--lr", type=float, default=3e-3)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--solver-iters", type=int, default=4)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--damping", type=float, default=0.01)
parser.add_argument("--no-viewer", action="store_true")
parser.add_argument("--xml", type=str,
                    default="/home/emre/mujoco_playground/mujoco_playground/"
                            "external_deps/mujoco_menagerie/franka_emika_panda/"
                            "mjx_single_cube.xml")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_STEPS     = args.steps
N_SUBSTEPS  = args.substeps
TRAIN_ITERS = args.iters
TRAIN_LR    = args.lr
GAMMA       = args.gamma
GRAD_CLIP   = args.grad_clip
DAMPING     = args.damping
V_MAX       = 1.0
N_JOINTS    = 7

EE_SITE_IDX   = 0
CUBE_BODY_IDX = 12
FINGER_OPEN   = 0.04

Q_LO = jnp.array([-2.897, -1.763, -2.897, -3.072, -2.897, -0.018, -2.897])
Q_HI = jnp.array([ 2.897,  1.763,  2.897, -0.070,  2.897,  3.752,  2.897])

TARGET_POS = jnp.array([0.4, 0.15, 0.55])

OBS_DIM = 10  # ee(3) + ee_to_target(3) + ee_vel(3) + finger(1)
ACT_DIM = 4   # v_target(3) + finger(1)
HIDDEN  = 64

# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------
def init_mlp(key):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        'w1': jax.random.normal(k1, (OBS_DIM, HIDDEN)) * jnp.sqrt(2.0 / OBS_DIM),
        'b1': jnp.zeros(HIDDEN),
        'w2': jax.random.normal(k2, (HIDDEN, HIDDEN)) * jnp.sqrt(2.0 / HIDDEN),
        'b2': jnp.zeros(HIDDEN),
        'w3': jax.random.normal(k3, (HIDDEN, ACT_DIM)) * 0.01,
        'b3': jnp.zeros(ACT_DIM),
    }


def mlp_forward(params, obs):
    x = jnp.tanh(obs @ params['w1'] + params['b1'])
    x = jnp.tanh(x @ params['w2'] + params['b2'])
    x = x @ params['w3'] + params['b3']
    v_target = jnp.tanh(x[:3]) * V_MAX
    finger = jax.nn.sigmoid(x[3]) * FINGER_OPEN
    return v_target, finger

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("\nLoading MuJoCo model...")
mj_model = mujoco.MjModel.from_xml_path(args.xml)
mj_data = mujoco.MjData(mj_model)

key_id = mj_model.keyframe("home").id
mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
mujoco.mj_forward(mj_model, mj_data)

sim_dt = mj_model.opt.timestep
frame_dt = N_SUBSTEPS * sim_dt
total_time = N_STEPS * frame_dt
total_physics = N_STEPS * N_SUBSTEPS

print(f"  sim_dt={sim_dt}, frame_dt={frame_dt:.4f}s")
print(f"  {N_STEPS} steps × {N_SUBSTEPS} substeps = {total_physics} physics steps")
print(f"  Total trajectory: {total_time:.3f}s")
print(f"  EE home (site): {mj_data.site_xpos[EE_SITE_IDX]}")
print(f"  Target: {TARGET_POS}")

home_ctrl = jnp.array(mj_data.ctrl)

# ---------------------------------------------------------------------------
# Jacobian — bir kez, home'da
# ---------------------------------------------------------------------------
print("\nComputing Jacobian at home position...")
jacp = np.zeros((3, mj_model.nv))
jacr = np.zeros((3, mj_model.nv))
mujoco.mj_jac(mj_model, mj_data, jacp, jacr,
              mj_data.site_xpos[EE_SITE_IDX], 9)  # body 9 = hand
J_p = jnp.array(jacp[:, :N_JOINTS])

# Precompute damped pinv components (sabit — JIT'e closure olarak girer)
A_inv = jnp.linalg.inv(J_p @ J_p.T + DAMPING * jnp.eye(3))
J_pinv = J_p.T @ A_inv  # (7, 3) — q_dot = J_pinv @ v_target

s = np.linalg.svd(np.array(J_p), compute_uv=False)
print(f"  J_p shape: {J_p.shape}, rank: {np.sum(s > 1e-6)}")
print(f"  J_pinv precomputed: {J_pinv.shape}")

# ---------------------------------------------------------------------------
# MJX
# ---------------------------------------------------------------------------
print(f"\nSetting up MJX (solver_iters={args.solver_iters})...")
mj_model.opt.iterations = args.solver_iters
mj_model.opt.ls_iterations = 4
mjx_model = mjx.put_model(mj_model)

mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
mjx_data_init = mjx.put_data(mj_model, mj_data)

# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------
rng = jax.random.PRNGKey(42)
mlp_params = init_mlp(rng)
n_params = sum(p.size for p in jax.tree.leaves(mlp_params))

optimizer = optax.adam(TRAIN_LR)
opt_state = optimizer.init(mlp_params)

print(f"\n{'='*60}")
print(f"BPTT CLOSED-LOOP — EE Velocity + Nested lax.scan")
print(f"  MLP: {OBS_DIM}→{HIDDEN}→{HIDDEN}→{ACT_DIM} ({n_params} params)")
print(f"  Action: v_target(3) → J†→ q_dot → ctrl = q + q_dot*dt")
print(f"  {N_STEPS}×{N_SUBSTEPS} = {total_physics} physics steps ({total_time:.3f}s)")
print(f"  Gamma: {GAMMA}, LR: {TRAIN_LR}, Damping: {DAMPING}")
print(f"  Target: {TARGET_POS}")
print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# BPTT Loss — nested lax.scan (fast compile)
# ---------------------------------------------------------------------------
def bptt_loss_fn(mlp_params):
    """
    Nested lax.scan:
      Outer scan: N_STEPS frame steps (MLP called each)
      Inner scan: N_SUBSTEPS physics substeps (ctrl constant)
    
    JIT traces 1 frame step + 1 substep → compiles fast.
    """
    def frame_step_fn(carry, _):
        state, prev_ee_pos, total_reward, gamma_acc = carry

        # 1. Observation
        ee_pos = state.site_xpos[EE_SITE_IDX]
        ee_to_target = TARGET_POS - ee_pos
        ee_vel = (ee_pos - prev_ee_pos) / frame_dt
        finger_width = state.qpos[7] + state.qpos[8]
        obs = jnp.concatenate([ee_pos, ee_to_target, ee_vel,
                                jnp.array([finger_width])])

        # 2. MLP → v_target + finger
        v_target, finger_target = mlp_forward(mlp_params, obs)

        # 3. EE vel → joint targets (precomputed J_pinv)
        q_dot = J_pinv @ v_target
        current_q = state.qpos[:N_JOINTS]
        q_target = jnp.clip(current_q + q_dot * frame_dt, Q_LO, Q_HI)

        ctrl = home_ctrl.at[:N_JOINTS].set(q_target)
        ctrl = ctrl.at[7].set(finger_target)

        # 4. Physics substeps (inner scan)
        def substep_fn(s, _):
            s = s.replace(ctrl=ctrl)
            s = mjx.step(mjx_model, s)
            return s, None

        state, _ = jax.lax.scan(substep_fn, state, None, length=N_SUBSTEPS)

        # 5. Reward
        ee_new = state.site_xpos[EE_SITE_IDX]
        delta = ee_new - TARGET_POS
        dist = jnp.sqrt(jnp.dot(delta, delta) + 1e-6)
        rew = -dist

        total_reward = total_reward + gamma_acc * rew
        gamma_acc = gamma_acc * GAMMA

        return (state, ee_pos, total_reward, gamma_acc), None

    init_carry = (
        mjx_data_init,
        mjx_data_init.site_xpos[EE_SITE_IDX],  # prev_ee_pos
        jnp.float32(0.0),                        # total_reward
        jnp.float32(1.0),                        # gamma_acc
    )

    (final_state, _, total_reward, _), _ = jax.lax.scan(
        frame_step_fn, init_carry, None, length=N_STEPS)

    return -total_reward


def bptt_forward_fn(mlp_params):
    """Forward rollout — per-step info + qpos for rendering."""
    def frame_step_fn(carry, _):
        state, prev_ee_pos = carry

        ee_pos = state.site_xpos[EE_SITE_IDX]
        ee_to_target = TARGET_POS - ee_pos
        ee_vel = (ee_pos - prev_ee_pos) / frame_dt
        finger_width = state.qpos[7] + state.qpos[8]
        obs = jnp.concatenate([ee_pos, ee_to_target, ee_vel,
                                jnp.array([finger_width])])

        v_target, finger_target = mlp_forward(mlp_params, obs)

        q_dot = J_pinv @ v_target
        current_q = state.qpos[:N_JOINTS]
        q_target = jnp.clip(current_q + q_dot * frame_dt, Q_LO, Q_HI)

        ctrl = home_ctrl.at[:N_JOINTS].set(q_target)
        ctrl = ctrl.at[7].set(finger_target)

        def substep_fn(s, _):
            s = s.replace(ctrl=ctrl)
            s = mjx.step(mjx_model, s)
            return s, s.qpos

        state, step_qpos = jax.lax.scan(substep_fn, state, None, length=N_SUBSTEPS)

        ee_new = state.site_xpos[EE_SITE_IDX]
        delta = ee_new - TARGET_POS
        dist = jnp.sqrt(jnp.dot(delta, delta) + 1e-6)

        step_info = jnp.concatenate([ee_new, v_target, jnp.array([finger_target, -dist])])
        # step_info: ee(3) + v(3) + finger(1) + rew(1) = 8

        return (state, ee_pos), (step_info, step_qpos)

    init_carry = (mjx_data_init, mjx_data_init.site_xpos[EE_SITE_IDX])

    (final_state, _), (all_step_info, all_qpos) = jax.lax.scan(
        frame_step_fn, init_carry, None, length=N_STEPS)

    return final_state, all_step_info, all_qpos

# ---------------------------------------------------------------------------
# JIT
# ---------------------------------------------------------------------------
print("JIT compiling BPTT loss + grad...")
t0 = time.time()
value_and_grad_fn = jax.jit(jax.value_and_grad(bptt_loss_fn))
loss_val, grad_val = value_and_grad_fn(mlp_params)
loss_val.block_until_ready()
print(f"  JIT compile: {time.time()-t0:.1f}s")
print(f"  Initial loss: {float(loss_val):.6f}")
print(f"  Initial grad norm: {float(optax.global_norm(grad_val)):.4f}")

print("\nJIT compiling forward rollout...")
t0 = time.time()
jit_forward = jax.jit(bptt_forward_fn)
_, info_test, qpos_test = jit_forward(mlp_params)
info_test.block_until_ready()
print(f"  JIT compile: {time.time()-t0:.1f}s")
print(f"  all_step_info shape: {info_test.shape}")   # (N_STEPS, 8)
print(f"  all_qpos shape: {qpos_test.shape}")         # (N_STEPS, N_SUBSTEPS, nq)

# ---------------------------------------------------------------------------
# Viewer
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
    print("✓ Viewer açıldı\n")


def render_trajectory(all_qpos, skip=1):
    """all_qpos shape: (N_STEPS, N_SUBSTEPS, nq)"""
    if not USE_VIEWER or not viewer_handle.is_running():
        return
    qpos_np = np.array(all_qpos)
    for step in range(qpos_np.shape[0]):
        for sub in range(0, qpos_np.shape[1], skip):
            mj_data_render.qpos[:] = qpos_np[step, sub]
            mujoco.mj_forward(mj_model_render, mj_data_render)
            mj_data_render.mocap_pos[mocap_id] = np.array(TARGET_POS)
            viewer_handle.sync()
            time.sleep(sim_dt * skip)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
loss_history = []
dist_history = []

print("Training başlıyor...\n")
train_start = time.time()

for iteration in range(TRAIN_ITERS):
    if USE_VIEWER and not viewer_handle.is_running():
        print("\nViewer kapatıldı.")
        break

    loss_val, grad_val = value_and_grad_fn(mlp_params)
    loss_f = float(loss_val)

    if np.isnan(loss_f):
        print(f"\n⚠ NaN at iter {iteration}!")
        break

    loss_history.append(loss_f)

    grad_norm = float(optax.global_norm(grad_val))
    if grad_norm > GRAD_CLIP:
        grad_val = jax.tree.map(lambda g: g * (GRAD_CLIP / grad_norm), grad_val)

    updates, opt_state = optimizer.update(grad_val, opt_state, mlp_params)
    mlp_params = optax.apply_updates(mlp_params, updates)

    # Log + render every 5 iters
    if iteration % 5 == 0 or iteration < 3:
        _, all_info, all_qpos = jit_forward(mlp_params)
        info_np = np.array(all_info)  # (N_STEPS, 8)

        ee_final = info_np[-1, :3]
        dist = np.linalg.norm(ee_final - np.array(TARGET_POS))
        dist_history.append(dist)

        print(f"\nIter {iteration:4d}:  loss={loss_f:.4f}  dist={dist:.4f}  "
              f"grad_norm={grad_norm:.4f}")
        for s in range(N_STEPS):
            ee = info_np[s, :3]
            v = info_np[s, 3:6]
            fng = info_np[s, 6]
            rew = info_np[s, 7]
            print(f"  step {s}: EE=[{ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}]  "
                  f"v=[{v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f}]  "
                  f"fng={fng:.4f}  rew={rew:.4f}")

        render_trajectory(all_qpos, skip=1)
    else:
        dist_history.append(dist_history[-1] if dist_history else 0)

train_time = time.time() - train_start
print(f"\nTraining done: {train_time:.1f}s ({len(loss_history)} iters, "
      f"{train_time/max(1,len(loss_history)):.3f}s/iter)")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"SONUÇLAR")
print(f"{'='*60}")
print(f"  Initial loss: {loss_history[0]:.4f}")
print(f"  Final loss:   {loss_history[-1]:.4f}")
print(f"  Final dist:   {dist_history[-1]:.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(loss_history, linewidth=1.5)
axes[0].set_ylabel("Loss (-reward)")
axes[0].set_title("Loss")
axes[0].grid(True, alpha=0.3)

axes[1].plot(dist_history, linewidth=1.5)
axes[1].set_ylabel("Distance (m)")
axes[1].set_xlabel("Iteration")
axes[1].set_title("EE → Target Distance")
axes[1].grid(True, alpha=0.3)

fig.suptitle(f"BPTT EE-Vel — {N_STEPS}×{N_SUBSTEPS}, {len(loss_history)} iters")
fig.tight_layout()
plot_path = "bptt_results.png"
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