"""
MJX BPTT — 15-Segment Container Sorting (Progressive)

Place the 3 objects into the box: Cube → Ball → Prism
For each object: pre-grasp → descend → grasp → move → release
Usage:
    python train_container.py
    python train_container.py --iters 3000 --no-viewer
"""
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")

import argparse
import time
import pickle
import gc
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
from rewards.container import select_reward
from envs.container_env import (
    N_JOINTS, FINGER_OPEN, Q_LO, Q_HI, VEL_LIMITS,
    OBS_DIM, ACT_DIM, N_SEGMENTS, PHASES_PER_OBJECT,
    OBJECT_ORDER, PHASE_NAMES, PRE_GRASP_Z_OFFSET,
    discover_indices, make_task_cfg, build_obs, create_batch,
    evaluate_phase_metric, metric_satisfied, PATIENCE,
)

# ---------------------------------------------------------------------------
patch_solver()

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# Per-object steps (aynı 5 phase × 3 nesne)
parser.add_argument("--pregrasp-steps", type=int, default=20)
parser.add_argument("--descend-steps", type=int, default=12)
parser.add_argument("--grasp-steps", type=int, default=8)
parser.add_argument("--move-steps", type=int, default=20)
parser.add_argument("--release-steps", type=int, default=8)
parser.add_argument("--substeps", type=int, default=4)
parser.add_argument("--iters", type=int, default=3000)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--gamma", type=float, default=0.999)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--solver-iters", type=int, default=6)
parser.add_argument("--grad-clip", type=float, default=400.0)
parser.add_argument("--hidden-trunk", type=int, default=64)
parser.add_argument("--hidden-head", type=int, default=16)
parser.add_argument("--no-viewer", action="store_true")
parser.add_argument("--xml", type=str, default="assets/common/franka_emika_panda/mjx_container.xml")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Segment config — 5 phase × 3 objects = 15
# ---------------------------------------------------------------------------
STEPS_PER_OBJ = [args.pregrasp_steps, args.descend_steps,
                 args.grasp_steps, args.move_steps, args.release_steps]
N_SEG = STEPS_PER_OBJ * len(OBJECT_ORDER)  # [16,12,6,16,8] × 3
N_TOTAL = sum(N_SEG)

SEG_BOUNDARIES = []
acc = 0
for s in N_SEG[:-1]:
    acc += s
    SEG_BOUNDARIES.append(acc)
# 14 boundaries

PHASE_BOUNDARIES = SEG_BOUNDARIES + [N_TOTAL]  # 15 elements

# Segment isimleri
SEG_NAMES = []
starts = [0] + SEG_BOUNDARIES
ends = SEG_BOUNDARIES + [N_TOTAL]
SEGMENTS = []
for obj_i, obj_name in enumerate(OBJECT_ORDER):
    for phase_i, phase_name in enumerate(PHASE_NAMES):
        seg_i = obj_i * PHASES_PER_OBJECT + phase_i
        name = f"{obj_name}_{phase_name}"
        SEG_NAMES.append(name)
        SEGMENTS.append((name, starts[seg_i], ends[seg_i]))

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

print(f"  Indices: {indices}")
print(f"  sim_dt={sim_dt}, frame_dt={frame_dt:.4f}s")
for i, (name, s, e) in enumerate(SEGMENTS):
    print(f"  Seg{i:2d} ({name:20s}): steps {s}-{e-1} ({N_SEG[i]} steps)")
print(f"  Total: {N_TOTAL} × {N_SUBSTEPS} substeps = "
      f"{total_physics} physics ({total_time:.3f}s)")
print(f"  Batch: {BATCH_SIZE}")

home_ctrl = jnp.array(mj_data.ctrl)

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
mjx_data_batch = create_batch(
    BATCH_SIZE, mj_model, mj_data, mjx_model, key_id, indices)
mjx_data_single0 = jax.tree.map(lambda x: x[0], mjx_data_batch)

# ---------------------------------------------------------------------------
# Initialize policy + optimizer
# ---------------------------------------------------------------------------
rng = jax.random.PRNGKey(42)
# finger biases: [open, open, close, close, open] × 3 objects
finger_biases = [2.0, 2.0, 0.0, 0.0, 0.0] * len(OBJECT_ORDER)
mlp_params = init_all_mlps(
    rng, OBS_DIM, 32, ACT_DIM,
    n_segments=N_SEGMENTS, finger_biases=finger_biases)
n_params = sum(p.size for p in jax.tree.leaves(mlp_params))

optimizer = optax.chain(
    optax.clip_by_global_norm(GRAD_CLIP),
    optax.adam(TRAIN_LR),
)
opt_state = optimizer.init(mlp_params)
obs_rms = ObsRMS(OBS_DIM)

print(f"\n{'='*60}")
print(f"BPTT 15-SEG CONTAINER SORT — Batch={BATCH_SIZE}")
print(f"  MLP: {OBS_DIM}→{args.hidden_trunk}→{args.hidden_trunk}"
      f"→{args.hidden_head}→{ACT_DIM} × {N_SEGMENTS} heads ({n_params} params)")
print(f"  Per-object: {'+'.join(str(s) for s in STEPS_PER_OBJ)} = "
      f"{sum(STEPS_PER_OBJ)} steps")
print(f"  Total: {N_TOTAL} steps, LR: {TRAIN_LR}, Gamma: {GAMMA}")
print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# Single env loss
# ---------------------------------------------------------------------------
def single_env_loss_fn(mlp_params, obs_mean, obs_std,
                       mjx_data_single, active_steps):
    @jax.checkpoint
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
        gamma_acc = jnp.where(is_boundary, jnp.float64(1.0), gamma_acc)

        obs, ee_pos, current_q = build_obs(
            state, prev_ee_pos, frame_dt, step_idx, task_cfg)
        obs_norm = (obs.astype(jnp.float32) - obs_mean) / obs_std

        q_dot_target, finger_target = policy_forward(
            mlp_params, obs_norm, step_idx, SEG_BOUNDARIES,
            N_JOINTS, VEL_LIMITS, FINGER_OPEN)
        safe_q_dot = safe_action(q_dot_target, current_q, Q_LO, Q_HI)

        ctrl = home_ctrl.at[:N_JOINTS].set(safe_q_dot)
        ctrl = ctrl.at[7].set(finger_target)

        def do_substeps(s):
            @jax.checkpoint
            def substep_fn(s, _):
                s = s.replace(ctrl=ctrl)
                s = mjx.step(mjx_model, s)
                return s, None
            s, _ = jax.lax.scan(substep_fn, s, None, length=N_SUBSTEPS)
            return s

        state = jax.lax.cond(
            step_idx < active_steps, do_substeps, lambda s: s, state)

        rew = select_reward(state, step_idx, task_cfg, SEG_BOUNDARIES)
        rew = jnp.where(step_idx < active_steps, rew, 0.0)
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
def batch_loss_fn(mlp_params, obs_mean, obs_std, active_steps):
    losses, all_obs_batch = jax.vmap(
        single_env_loss_fn,
        in_axes=(None, None, None, 0, None)
    )(mlp_params, obs_mean, obs_std, mjx_data_batch, active_steps)
    return jnp.mean(losses), (all_obs_batch, losses)


# ---------------------------------------------------------------------------
# Forward rollout for logging
# ---------------------------------------------------------------------------
def single_forward_fn(mlp_params, obs_mean, obs_std, active_steps):
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

        def do_substeps_log(s):
            def substep_fn(s, _):
                s = s.replace(ctrl=ctrl)
                s = mjx.step(mjx_model, s)
                return s, s.qpos
            return jax.lax.scan(substep_fn, s, None, length=N_SUBSTEPS)

        def skip_substeps_log(s):
            dummy_qpos = jnp.broadcast_to(s.qpos, (N_SUBSTEPS,) + s.qpos.shape)
            return s, dummy_qpos

        state, step_qpos = jax.lax.cond(
            step_idx < active_steps,
            do_substeps_log, skip_substeps_log, state)

        ee_new = state.site_xpos[indices['ee_site_idx']]
        actual_rew = select_reward(state, step_idx, task_cfg, SEG_BOUNDARIES)
        cube_pos = state.xpos[indices['cube_body_idx']]
        ball_pos = state.xpos[indices['ball_body_idx']]
        prism_pos = state.xpos[indices['prism_body_idx']]
        container_pos = state.site_xpos[indices['container_inside_site_idx']]

        # step_info layout:
        #  0:3 ee, 3:6 ee_vel, 6 finger, 7 rew,
        #  8:11 cube, 11:14 ball, 14:17 prism, 17:20 container
        ee_vel = (ee_new - ee_pos) / frame_dt
        step_info = jnp.concatenate([
            ee_new,                                     # 0:3
            ee_vel,                                     # 3:6
            jnp.array([finger_target, actual_rew]),     # 6:8
            cube_pos,                                   # 8:11
            ball_pos,                                   # 11:14
            prism_pos,                                  # 14:17
            container_pos,                              # 17:20
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
active_steps = jnp.int32(N_TOTAL)
(loss_val, (all_obs_init, _)), grad_val = value_and_grad_fn(
    mlp_params, obs_mean, obs_std, active_steps)
loss_val.block_until_ready()
print(f"  JIT compile: {time.time()-t0:.1f}s, initial loss: {float(loss_val):.6f}")

obs_rms.update(np.array(all_obs_init).reshape(-1, OBS_DIM))
obs_mean, obs_std = obs_rms.get_jnp()

print("JIT compiling forward rollout...")
t0 = time.time()
jit_forward = jax.jit(single_forward_fn)
info_test, _ = jit_forward(mlp_params, obs_mean, obs_std, active_steps)
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
    print("✓ Viewer\n")


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
# Progressive training state
# ---------------------------------------------------------------------------
current_phase = 0
active_steps = jnp.int32(PHASE_BOUNDARIES[min(1, len(PHASE_BOUNDARIES) - 1)])
patience_count = 0

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
loss_history = []
grad_norm_history = []
per_env_history = []
seg_histories = {i: [] for i in range(N_SEGMENTS)}

print("Training is about to begin...\n")
train_start = time.time()

for iteration in range(TRAIN_ITERS):
    if USE_VIEWER and not viewer_handle.is_running():
        print("\nThe viewer has been closed.")
        break

    obs_mean, obs_std = obs_rms.get_jnp()
    (loss_val, (all_obs_batch, per_env_losses)), grad_val = \
        value_and_grad_fn(mlp_params, obs_mean, obs_std, active_steps)
    loss_f = float(loss_val)

    obs_rms.update(np.array(all_obs_batch).reshape(-1, OBS_DIM))
    per_env_history.append(np.array(-per_env_losses))

    if np.isnan(loss_f):
        print(f"\n⚠ NaN at iter {iteration}!")
        break

    loss_history.append(loss_f)
    grad_norm = float(optax.global_norm(grad_val))
    grad_norm_history.append(grad_norm)

    if iteration % 5 == 0 or iteration < 3:
        all_info, all_qpos = jit_forward(
            mlp_params, obs_mean, obs_std, active_steps)
        info_np = np.array(all_info)

        # Segment metrics
        metrics = {}
        for si in range(N_SEGMENTS):
            seg_end = (PHASE_BOUNDARIES[si] - 1
                       if si < len(PHASE_BOUNDARIES) else N_TOTAL - 1)
            obj_group = si // PHASES_PER_OBJECT
            phase_in_obj = si % PHASES_PER_OBJECT
            obj_slice = slice(8 + obj_group * 3, 8 + obj_group * 3 + 3)

            if phase_in_obj == 0:  # pre-grasp
                ee = info_np[seg_end, :3]
                obj_pos = info_np[seg_end, obj_slice]
                above = obj_pos + np.array([0, 0, PRE_GRASP_Z_OFFSET])
                metrics[si] = np.linalg.norm(ee - above)
            elif phase_in_obj in (1, 2):  # descend, grasp
                ee = info_np[seg_end, :3]
                obj_pos = info_np[seg_end, obj_slice]
                metrics[si] = np.linalg.norm(ee - obj_pos)
            elif phase_in_obj == 3:  # move
                obj_pos = info_np[seg_end, obj_slice]
                container = info_np[seg_end, 17:20]
                metrics[si] = np.linalg.norm(obj_pos[:2] - container[:2])
            else:  # release
                obj_pos = info_np[seg_end, obj_slice]
                container = info_np[seg_end, 17:20]
                metrics[si] = np.linalg.norm(obj_pos - container)

        for i in range(N_SEGMENTS):
            seg_histories[i].append(metrics[i])

        print(f"\nIter {iteration:4d}:  loss={loss_f:.4f}  "
              f"grad={grad_norm:.1f}  phase={current_phase}  "
              f"active={int(active_steps)}  "
              f"patience={patience_count}/{PATIENCE}")
        for i, (name, _, _) in enumerate(SEGMENTS):
            marker = "→" if i == current_phase else " "
            print(f"  {marker} Seg{i:2d} {name:20s}: {metrics[i]:.4f}")

        # Phase progression
        metric = evaluate_phase_metric(
            info_np, current_phase, PHASE_BOUNDARIES, task_cfg)
        if metric_satisfied(metric, current_phase):
            patience_count += 1
        else:
            patience_count = 0

        if patience_count >= PATIENCE:
            current_phase += 1
            if current_phase >= len(PHASE_BOUNDARIES):
                print(f"\n✓ Tüm phase'ler tamamlandı! iter={iteration}")
                break
            active_steps = jnp.int32(
                PHASE_BOUNDARIES[min(current_phase + 1,
                                     len(PHASE_BOUNDARIES) - 1)])
            patience_count = 0
            seg_name = (SEGMENTS[current_phase][0]
                       if current_phase < len(SEGMENTS) else 'ALL')
            print(f"\n→ Phase {current_phase}: "
                  f"active={int(active_steps)}, seg={seg_name}")

        render_trajectory(all_qpos, skip=1)
    else:
        for i in range(N_SEGMENTS):
            seg_histories[i].append(
                seg_histories[i][-1] if seg_histories[i] else 0)

    updates, opt_state = optimizer.update(grad_val, opt_state, mlp_params)
    mlp_params = optax.apply_updates(mlp_params, updates)

    if iteration % 50 == 0:
        plt.close('all')
        gc.collect()

train_time = time.time() - train_start
print(f"\nTraining done: {train_time:.1f}s ({len(loss_history)} iters, "
      f"{train_time/max(1,len(loss_history)):.3f}s/iter)")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
with open("container_sort_params.pkl", "wb") as f:
    pickle.dump(jax.tree.map(lambda x: np.array(x), mlp_params), f)
print("Model saved: container_sort_params.pkl")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"RESULTS — 15-SEG CONTAINER SORT")
print(f"{'='*60}")
print(f"  Final loss: {loss_history[-1]:.4f}")
for i, (name, _, _) in enumerate(SEGMENTS):
    print(f"  Seg{i:2d} ({name:20s}): {seg_histories[i][-1]:.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
n_plot_rows = 2 + (N_SEGMENTS + 1) // 2  # reward + grad + segment pairs
fig, axes = plt.subplots(n_plot_rows, 2, figsize=(14, 3 * n_plot_rows))
axes = axes.flatten()

# Reward
per_env_arr = np.array(per_env_history)
for env_i in range(BATCH_SIZE):
    axes[0].plot(per_env_arr[:, env_i], alpha=0.15, linewidth=0.5, color='gray')
axes[0].plot([-l for l in loss_history], linewidth=2, color='blue',
             label='mean reward')
axes[0].set_title("Reward")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Grad norm
axes[1].plot(grad_norm_history, linewidth=1.5, color='red')
axes[1].set_title("Gradient Norm")
axes[1].grid(True, alpha=0.3)

# Segment metrics
colors = {'cube': 'green', 'ball': 'blue', 'prism': 'orange'}
for i in range(N_SEGMENTS):
    ax = axes[i + 2]
    obj_name = OBJECT_ORDER[i // PHASES_PER_OBJECT]
    phase_name = PHASE_NAMES[i % PHASES_PER_OBJECT]
    ax.plot(seg_histories[i], linewidth=1.5, color=colors[obj_name])
    ax.set_title(f"Seg{i} {obj_name} {phase_name}")
    ax.grid(True, alpha=0.3)

# Hide unused axes
for i in range(N_SEGMENTS + 2, len(axes)):
    axes[i].set_visible(False)

fig.suptitle(f"BPTT 15-Seg Container Sort — Batch={BATCH_SIZE}, "
             f"{len(loss_history)} iters", fontsize=14)
fig.tight_layout()
plot_path = "bptt_container_sort_results.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {plot_path}")

if USE_VIEWER and viewer_handle.is_running():
    print("\nViewer is open — Press ESC to close.")
    try:
        while viewer_handle.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting.")