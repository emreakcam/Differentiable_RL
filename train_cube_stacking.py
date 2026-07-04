"""
MJX BPTT — Cube Stacking Training Script

Usage:
    python train.py
    python train.py --batch 8 --iters 900 --lr 5e-4
    python train.py --xml /path/to/scene.xml --no-viewer
"""
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")

import argparse
import time
import pickle
import numpy as np
import jax.numpy as jnp
import optax
import mujoco
from mujoco import mjx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from helpers.mjx_utils import patch_solver, safe_action
from helpers.obs import ObsRMS
from models.policy import init_all_mlps, policy_forward
from rewards.cube_stack import select_reward
from envs.cube_stack_env import (
    N_JOINTS, FINGER_OPEN, Q_LO, Q_HI, VEL_LIMITS,
    OBS_DIM, ACT_DIM, CUBE2_HEIGHT, STACK_OFFSET,
    discover_indices, make_task_cfg, build_obs, create_batch,
)

# ---------------------------------------------------------------------------
# Solver patch
# ---------------------------------------------------------------------------
patch_solver()

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--seg0", type=int, default=16, help="Pre-grasp steps")
parser.add_argument("--seg1", type=int, default=12, help="Descend steps")
parser.add_argument("--seg2", type=int, default=6, help="Close steps")
parser.add_argument("--seg3", type=int, default=16, help="Move steps")
parser.add_argument("--seg4", type=int, default=8, help="Align+Release steps")
parser.add_argument("--substeps", type=int, default=4)
parser.add_argument("--iters", type=int, default=900)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--gamma", type=float, default=0.999)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--solver-iters", type=int, default=6)
parser.add_argument("--grad-clip", type=float, default=300.0)
parser.add_argument("--hidden-trunk", type=int, default=64)
parser.add_argument("--hidden-head", type=int, default=8)
parser.add_argument("--no-viewer", action="store_true")
parser.add_argument("--xml", type=str,
                    default="assets/common/franka_emika_panda/mjx_single_cube.xml")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Segment config
# ---------------------------------------------------------------------------
N_SEG = [args.seg0, args.seg1, args.seg2, args.seg3, args.seg4]
N_TOTAL = sum(N_SEG)
SEG_BOUNDARIES = []
acc = 0
for s in N_SEG[:-1]:
    acc += s
    SEG_BOUNDARIES.append(acc)
# SEG_BOUNDARIES = [SEG1_START, SEG2_START, SEG3_START, SEG4_START]

N_SUBSTEPS  = args.substeps
BATCH_SIZE  = args.batch
TRAIN_ITERS = args.iters
TRAIN_LR    = args.lr
GAMMA       = args.gamma
GRAD_CLIP   = args.grad_clip

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("\nLoading MuJoCo model...")
mj_model = mujoco.MjModel.from_xml_path(args.xml)
mj_data = mujoco.MjData(mj_model)

indices = discover_indices(mj_model)
task_cfg = make_task_cfg(indices)

key_id = mj_model.keyframe("home").id
mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
mujoco.mj_forward(mj_model, mj_data)

sim_dt = mj_model.opt.timestep
frame_dt = N_SUBSTEPS * sim_dt
total_time = N_TOTAL * frame_dt
total_physics = N_TOTAL * N_SUBSTEPS

print(f"  sim_dt={sim_dt}, frame_dt={frame_dt:.4f}s")
for i, n in enumerate(N_SEG):
    seg_names = ["pre-grasp", "descend", "close", "move", "release"]
    print(f"  Seg{i} ({seg_names[i]}): {n} steps")
print(f"  Total: {N_TOTAL} steps × {N_SUBSTEPS} substeps = "
      f"{total_physics} physics ({total_time:.3f}s)")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  EE home (site): {mj_data.site_xpos[indices['ee_site_idx']]}")

home_ctrl = jnp.array(mj_data.ctrl)
init_cube_pos = jnp.array(mj_data.xpos[indices['cube1_body_idx']])
init_cube2_pos = jnp.array(mj_data.xpos[indices['cube2_body_idx']])
print(f"  Cube1 pos: {init_cube_pos}")
print(f"  Cube2 pos: {init_cube2_pos}")

# ---------------------------------------------------------------------------
# MJX
# ---------------------------------------------------------------------------
print(f"\nSetting up MJX (solver_iters={args.solver_iters})...")
mj_model.opt.iterations = args.solver_iters
mj_model.opt.ls_iterations = 8
mjx_model = mjx.put_model(mj_model)

mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
mjx_data_init = mjx.put_data(mj_model, mj_data)
mjx_data_init = mjx.forward(mjx_model, mjx_data_init)

# ---------------------------------------------------------------------------
# Batch data
# ---------------------------------------------------------------------------
print(f"\nCreating batch of {BATCH_SIZE} envs...")
mjx_data_batch, init_cube_pos_batch = create_batch(
    BATCH_SIZE, mj_model, mj_data, mjx_model, key_id)

for i in range(BATCH_SIZE):
    print(f"  Env {i}: cube1 pos = [{float(init_cube_pos_batch[i,0]):.4f}, "
          f"{float(init_cube_pos_batch[i,1]):.4f}, "
          f"{float(init_cube_pos_batch[i,2]):.4f}]")

# Logging uses batch[0]
mjx_data_single0 = jax.tree.map(lambda x: x[0], mjx_data_batch)
init_cube_pos_single0 = init_cube_pos_batch[0]

# ---------------------------------------------------------------------------
# Initialize policy + optimizer
# ---------------------------------------------------------------------------
rng = jax.random.PRNGKey(42)
finger_biases = [2.0, 2.0, 0.0, 0.0, 0.0]  # seg0,1 open; seg2,3,4 neutral
mlp_params = init_all_mlps(
    rng, OBS_DIM, args.hidden_trunk, args.hidden_head, ACT_DIM,
    n_segments=5, finger_biases=finger_biases)
n_params = sum(p.size for p in jax.tree.leaves(mlp_params))

optimizer = optax.chain(
    optax.clip_by_global_norm(GRAD_CLIP),
    optax.adam(TRAIN_LR),
)
opt_state = optimizer.init(mlp_params)
obs_rms = ObsRMS(OBS_DIM)

print(f"\n{'='*60}")
print(f"BPTT 5-SEGMENT + ObsNorm + Batch={BATCH_SIZE}")
print(f"  MLP: {OBS_DIM}→{args.hidden_trunk}→{args.hidden_trunk}"
      f"→{args.hidden_head}→{ACT_DIM} × 5 heads ({n_params} params)")
print(f"  Segments: {'+'.join(str(s) for s in N_SEG)} = {N_TOTAL} steps")
print(f"  Physics per env: {total_physics} substeps ({total_time:.3f}s)")
print(f"  Total physics per iter: {total_physics * BATCH_SIZE}")
print(f"  LR: {TRAIN_LR}, Gamma: {GAMMA}, Grad clip: {GRAD_CLIP}")
print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# Single env loss (vmapped)
# ---------------------------------------------------------------------------
def single_env_loss_fn(mlp_params, obs_mean, obs_std,
                       mjx_data_single, init_cube_single):
    def frame_step_fn(carry, step_idx):
        state, prev_ee_pos, total_reward, gamma_acc = carry

        is_boundary = jnp.zeros((), dtype=bool)
        for b in SEG_BOUNDARIES:
            is_boundary = is_boundary | (step_idx == b)

        state = jax.lax.cond(
            is_boundary,
            lambda s: jax.lax.stop_gradient(s),
            lambda s: s, state)
        prev_ee_pos = jax.lax.cond(
            is_boundary,
            lambda p: jax.lax.stop_gradient(p),
            lambda p: p, prev_ee_pos)
        gamma_acc = jnp.where(is_boundary, jnp.float32(1.0), gamma_acc)

        obs, ee_pos, current_q = build_obs(
            state, prev_ee_pos, frame_dt, step_idx, task_cfg)
        obs_norm = (obs.astype(jnp.float32) - obs_mean) / obs_std

        q_dot_target, finger_target = policy_forward(
            mlp_params, obs_norm, step_idx, SEG_BOUNDARIES,
            N_JOINTS, VEL_LIMITS, FINGER_OPEN)
        safe_q_dot = safe_action(q_dot_target, current_q, Q_LO, Q_HI)

        ctrl = home_ctrl.at[:N_JOINTS].set(safe_q_dot)
        ctrl = ctrl.at[7].set(finger_target)

        def substep_fn(s, _):
            s = s.replace(ctrl=ctrl)
            s = mjx.step(mjx_model, s)
            return s, None

        state, _ = jax.lax.scan(substep_fn, state, None, length=N_SUBSTEPS)

        rew = select_reward(state, step_idx, task_cfg, SEG_BOUNDARIES)
        total_reward = total_reward + gamma_acc * rew
        gamma_acc = gamma_acc * GAMMA

        return (state, ee_pos, total_reward, gamma_acc), obs

    init_carry = (
        mjx_data_single,
        mjx_data_single.site_xpos[indices['ee_site_idx']],
        jnp.float64(0.0),
        jnp.float64(1.0),
    )
    (_, _, total_reward, _), all_obs = jax.lax.scan(
        frame_step_fn, init_carry, jnp.arange(N_TOTAL))
    return -total_reward, all_obs


# ---------------------------------------------------------------------------
# Batch loss
# ---------------------------------------------------------------------------
def batch_loss_fn(mlp_params, obs_mean, obs_std):
    losses, all_obs_batch = jax.vmap(
        single_env_loss_fn,
        in_axes=(None, None, None, 0, 0)
    )(mlp_params, obs_mean, obs_std, mjx_data_batch, init_cube_pos_batch)
    return jnp.mean(losses), (all_obs_batch, losses)


# ---------------------------------------------------------------------------
# Forward rollout for logging (single env, batch[0])
# ---------------------------------------------------------------------------
def single_forward_fn(mlp_params, obs_mean, obs_std):
    def frame_step_fn(carry, step_idx):
        state, prev_ee_pos = carry

        obs, ee_pos, current_q = build_obs(
            state, prev_ee_pos, frame_dt, step_idx, task_cfg)
        obs_norm = (obs.astype(jnp.float32) - obs_mean) / obs_std

        q_dot_target, finger_target = policy_forward(
            mlp_params, obs_norm, step_idx, SEG_BOUNDARIES,
            N_JOINTS, VEL_LIMITS, FINGER_OPEN)
        safe_q_dot = safe_action(q_dot_target, current_q, Q_LO, Q_HI)

        ctrl = home_ctrl.at[:N_JOINTS].set(safe_q_dot)
        ctrl = ctrl.at[7].set(finger_target)

        def substep_fn(s, _):
            s = s.replace(ctrl=ctrl)
            s = mjx.step(mjx_model, s)
            return s, s.qpos

        state, step_qpos = jax.lax.scan(
            substep_fn, state, None, length=N_SUBSTEPS)

        ee_new = state.site_xpos[indices['ee_site_idx']]
        ee_vel_new = (ee_new - ee_pos) / frame_dt
        actual_qvel = state.qvel[:N_JOINTS]
        cube_pos = state.xpos[indices['cube1_body_idx']]
        actual_rew = select_reward(state, step_idx, task_cfg, SEG_BOUNDARIES)
        cube2_pos = state.xpos[indices['cube2_body_idx']]

        step_info = jnp.concatenate([
            ee_new,                                     # 0:3
            ee_vel_new,                                 # 3:6
            jnp.array([finger_target, actual_rew]),     # 6:8
            safe_q_dot,                                 # 8:15
            actual_qvel,                                # 15:22
            cube_pos,                                   # 22:25
            cube2_pos,                                  # 25:28
        ])
        return (state, ee_pos), (step_info, step_qpos)

    init_carry = (mjx_data_single0,
                  mjx_data_single0.site_xpos[indices['ee_site_idx']])
    (_, _), (all_step_info, all_qpos) = jax.lax.scan(
        frame_step_fn, init_carry, jnp.arange(N_TOTAL))
    return all_step_info, all_qpos


# ---------------------------------------------------------------------------
# JIT compile
# ---------------------------------------------------------------------------
print("JIT compiling batch loss + grad...")
t0 = time.time()
value_and_grad_fn = jax.jit(jax.value_and_grad(batch_loss_fn, has_aux=True))
obs_mean, obs_std = obs_rms.get_jnp()
(loss_val, (all_obs_init, _)), grad_val = value_and_grad_fn(
    mlp_params, obs_mean, obs_std)
loss_val.block_until_ready()
print(f"  JIT compile: {time.time()-t0:.1f}s")
print(f"  Initial loss: {float(loss_val):.6f}")
print(f"  Initial grad norm: {float(optax.global_norm(grad_val)):.4f}")

obs_rms.update(np.array(all_obs_init).reshape(-1, OBS_DIM))
obs_mean, obs_std = obs_rms.get_jnp()
print(f"  Obs RMS seeded: mean range "
      f"[{float(obs_mean.min()):.3f}, {float(obs_mean.max()):.3f}]")

print("\nJIT compiling forward rollout...")
t0 = time.time()
jit_forward = jax.jit(single_forward_fn)
info_test, qpos_test = jit_forward(mlp_params, obs_mean, obs_std)
info_test.block_until_ready()
print(f"  JIT compile: {time.time()-t0:.1f}s")

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

    viewer_handle = mj_viewer.launch_passive(mj_model_render, mj_data_render)
    print("✓ Viewer açıldı\n")


def render_trajectory(all_qpos, skip=1):
    if not USE_VIEWER or not viewer_handle.is_running():
        return
    qpos_np = np.array(all_qpos)
    for step in range(qpos_np.shape[0]):
        for sub in range(0, qpos_np.shape[1], skip):
            mj_data_render.qpos[:] = qpos_np[step, sub]
            mujoco.mj_forward(mj_model_render, mj_data_render)
            viewer_handle.sync()
            time.sleep(sim_dt * skip)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
loss_history = []
seg_dist_history = [[] for _ in range(5)]
grad_norm_history = []
per_env_history = []

print("Training başlıyor...\n")
train_start = time.time()

for iteration in range(TRAIN_ITERS):
    if USE_VIEWER and not viewer_handle.is_running():
        print("\nViewer kapatıldı.")
        break

    obs_mean, obs_std = obs_rms.get_jnp()
    (loss_val, (all_obs_batch, per_env_losses)), grad_val = \
        value_and_grad_fn(mlp_params, obs_mean, obs_std)
    loss_f = float(loss_val)

    obs_rms.update(np.array(all_obs_batch).reshape(-1, OBS_DIM))
    per_env_history.append(np.array(-per_env_losses))

    if np.isnan(loss_f):
        print(f"\n⚠ NaN at iter {iteration}!")
        break

    loss_history.append(loss_f)
    grad_norm = float(optax.global_norm(grad_val))
    grad_norm_history.append(grad_norm)

    # Log every 5 iters
    if iteration % 5 == 0 or iteration < 3:
        all_info, all_qpos = jit_forward(mlp_params, obs_mean, obs_std)
        info_np = np.array(all_info)

        # Segment end distances
        seg_ends = SEG_BOUNDARIES + [N_TOTAL]
        for si in range(4):
            end_idx = seg_ends[si] - 1
            ee = info_np[end_idx, :3]
            cube = info_np[end_idx, 22:25]
            seg_dist_history[si].append(np.linalg.norm(ee - cube))

        # Seg4: stack alignment
        cube1_f = info_np[-1, 22:25]
        cube2_f = info_np[-1, 25:28]
        xy_err = np.linalg.norm(cube1_f[:2] - cube2_f[:2])
        z_err = abs(cube1_f[2] - (cube2_f[2] + CUBE2_HEIGHT + STACK_OFFSET))
        seg_dist_history[4].append(xy_err + z_err)

        print(f"\nIter {iteration:4d}:  loss={loss_f:.4f}  "
              f"grad_norm={grad_norm:.4f}")
        for si, name in enumerate(["Pre-grasp", "Descend", "Close",
                                    "Move", "Release"]):
            print(f"  {name}: {seg_dist_history[si][-1]:.4f}")

        render_trajectory(all_qpos, skip=1)
    else:
        for si in range(5):
            seg_dist_history[si].append(
                seg_dist_history[si][-1] if seg_dist_history[si] else 0)

    updates, opt_state = optimizer.update(grad_val, opt_state, mlp_params)
    mlp_params = optax.apply_updates(mlp_params, updates)

train_time = time.time() - train_start
print(f"\nTraining done: {train_time:.1f}s ({len(loss_history)} iters, "
      f"{train_time/max(1,len(loss_history)):.3f}s/iter)")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
with open("mlp_params_cube_stack_batch.pkl", "wb") as f:
    pickle.dump(jax.tree.map(lambda x: np.array(x), mlp_params), f)
print("Model saved: mlp_params_cube_stack_batch.pkl")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"SONUÇLAR")
print(f"{'='*60}")
print(f"  Batch size:      {BATCH_SIZE}")
print(f"  Initial loss:    {loss_history[0]:.4f}")
print(f"  Final loss:      {loss_history[-1]:.4f}")
for si, name in enumerate(["Pre-grasp", "Descend", "Close",
                            "Move", "Release"]):
    print(f"  {name} final:  {seg_dist_history[si][-1]:.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(7, 1, figsize=(10, 21))

per_env_arr = np.array(per_env_history)
for env_i in range(BATCH_SIZE):
    axes[0].plot(per_env_arr[:, env_i], alpha=0.15, linewidth=0.5, color='gray')
axes[0].plot([-l for l in loss_history], linewidth=2, color='blue',
             label='mean reward')
axes[0].set_ylabel("Reward")
axes[0].set_title("Per-env Reward (gray) + Mean (blue)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(grad_norm_history, linewidth=1.5, color='red')
axes[1].set_ylabel("Grad Norm")
axes[1].set_title("Gradient Norm (before clip)")
axes[1].grid(True, alpha=0.3)

seg_names = ["Pre-grasp: EE→Cube", "Descend: EE→Cube", "Close: EE→Cube",
             "Move: Cube→Target", "Release: Align Error"]
for si in range(5):
    axes[si + 2].plot(seg_dist_history[si], linewidth=1.5)
    axes[si + 2].set_ylabel("Distance (m)")
    axes[si + 2].set_title(f"Seg{si} ({seg_names[si]})")
    axes[si + 2].grid(True, alpha=0.3)

axes[-1].set_xlabel("Iteration")
fig.suptitle(f"BPTT 5-Seg + ObsNorm + Batch={BATCH_SIZE} — "
             f"{'+'.join(str(s) for s in N_SEG)}, {len(loss_history)} iters")
fig.tight_layout()
plot_path = "bptt_5seg_batch_results.png"
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