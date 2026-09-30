"""
MJX BPTT — Cube Stacking Training Script (5 segments, progressive curriculum)

Segments: Pre-grasp → Descend → Close → Move → Release

Curriculum metrics are averaged over the whole batch (not env 0), so phase
advancement reflects the policy rather than one lucky initial condition.

Usage:
    python train_cube_stacking.py --no-viewer
    python train_cube_stacking.py --batch 32 --iters 900 --lr 1e-3
    python train_cube_stacking.py --no-curriculum      # ablation
    python train_cube_stacking.py --no-rebatch         # fixed batch ablation
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
from rewards.cube_stack import select_reward
from envs.cube_stack_env import (
    N_JOINTS, FINGER_OPEN, Q_LO, Q_HI, VEL_LIMITS,
    OBS_DIM, ACT_DIM, PRE_GRASP_Z_OFFSET, CUBE2_HEIGHT, STACK_OFFSET,
    discover_indices, make_task_cfg, build_obs,
)

# ---------------------------------------------------------------------------
# Solver patch
# ---------------------------------------------------------------------------
patch_solver()

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--seg0", type=int, default=24, help="Pre-grasp steps")
parser.add_argument("--seg1", type=int, default=20, help="Descend steps")
parser.add_argument("--seg2", type=int, default=8, help="Close steps")
parser.add_argument("--seg3", type=int, default=24, help="Move steps")
parser.add_argument("--seg4", type=int, default=8, help="Align+Release steps")
parser.add_argument("--substeps", type=int, default=4)
parser.add_argument("--iters", type=int, default=900)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--gamma", type=float, default=0.999)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--solver-iters", type=int, default=6)
parser.add_argument("--grad-clip", type=float, default=300.0)
parser.add_argument("--patience", type=int, default=4,
                    help="Logging intervals a phase criterion must hold")
parser.add_argument("--log-every", type=int, default=5)
parser.add_argument("--seed", type=int, default=42, help="Policy init seed")
parser.add_argument("--no-curriculum", action="store_true",
                    help="Ablation: all segments active from iteration 0")
parser.add_argument("--no-rebatch", action="store_true",
                    help="Ablation: fixed batch instead of fresh ICs per iter")
parser.add_argument("--no-viewer", action="store_true")
parser.add_argument("--xml", type=str,
                    default="assets/common/franka_emika_panda/mjx_single_cube.xml")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Segment config
# ---------------------------------------------------------------------------
N_SEG = [args.seg0, args.seg1, args.seg2, args.seg3, args.seg4]
N_TOTAL = sum(N_SEG)
N_SEGMENTS = len(N_SEG)

SEG_BOUNDARIES = []
acc = 0
for s in N_SEG[:-1]:
    acc += s
    SEG_BOUNDARIES.append(acc)
# SEG_BOUNDARIES = [SEG1_START, SEG2_START, SEG3_START, SEG4_START]

PHASE_BOUNDS = SEG_BOUNDARIES + [N_TOTAL]      # cumulative end of each segment
SEG_ENDS = [b - 1 for b in PHASE_BOUNDS]        # last step index of each segment
SEG_NAMES = ["Pre-grasp", "Descend", "Close", "Move", "Release"]

# Phase advancement criteria — (type, threshold), evaluated on the batch mean
CRITERIA = [
    ("dist",   0.04),   # 0 Pre-grasp: EE → above cube
    ("dist",   0.04),   # 1 Descend:   EE → cube
    ("dist",   0.04),   # 2 Close:     EE → cube
    ("dist",   0.10),   # 3 Move:      cube1 → stack target
    ("always", 0.0),    # 4 Release
]

N_SUBSTEPS  = args.substeps
BATCH_SIZE  = args.batch
TRAIN_ITERS = args.iters
TRAIN_LR    = args.lr
GAMMA       = args.gamma
GRAD_CLIP   = args.grad_clip
PATIENCE    = args.patience

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("\nLoading MuJoCo model...")
mj_model = mujoco.MjModel.from_xml_path(args.xml)
mj_model.opt.iterations = args.solver_iters
mj_model.opt.ls_iterations = 8
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
    print(f"  Seg{i} ({SEG_NAMES[i]}): {n} steps")
print(f"  Total: {N_TOTAL} steps x {N_SUBSTEPS} substeps = "
      f"{total_physics} physics ({total_time:.3f}s)")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  EE home (site): {mj_data.site_xpos[indices['ee_site_idx']]}")

home_ctrl = jnp.array(mj_data.ctrl)

# ---------------------------------------------------------------------------
# MJX
# ---------------------------------------------------------------------------
print(f"\nSetting up MJX (solver_iters={args.solver_iters})...")
mjx_model = mjx.put_model(mj_model)

mjx_data_init = mjx.put_data(mj_model, mj_data)
mjx_data_init = mjx.forward(mjx_model, mjx_data_init)

# Home state used as the template for JAX-side randomization
home_data = mjx.put_data(mj_model, mj_data)
home_data = mjx.forward(mjx_model, home_data)


# ---------------------------------------------------------------------------
# Batch creation — pure JAX, so fresh ICs cost nothing per iteration
# qpos[9], qpos[10] are the cube1 freejoint x, y
# ---------------------------------------------------------------------------
def rebatch_jax(rng_key, batch_size):
    keys = jax.random.split(rng_key, batch_size)

    def randomize_one(k):
        noise = jax.random.uniform(k, (2,), minval=-0.05, maxval=0.05)
        new_qpos = home_data.qpos.at[9].add(noise[0])
        new_qpos = new_qpos.at[10].add(noise[1])
        d = home_data.replace(qpos=new_qpos)
        return mjx.forward(mjx_model, d)

    return jax.vmap(randomize_one)(keys)


rebatch_jit = jax.jit(rebatch_jax, static_argnums=(1,))

print(f"\nCreating batch of {BATCH_SIZE} envs...")
batch = rebatch_jit(jax.random.PRNGKey(0), BATCH_SIZE)
data0 = jax.tree.map(lambda x: x[0], batch)

cube_xy = np.array(batch.xpos[:, indices['cube1_body_idx'], :2])
print(f"  cube1 x range: [{cube_xy[:, 0].min():.4f}, {cube_xy[:, 0].max():.4f}]")
print(f"  cube1 y range: [{cube_xy[:, 1].min():.4f}, {cube_xy[:, 1].max():.4f}]")
print(f"  Rebatch per iteration: {not args.no_rebatch}")

# ---------------------------------------------------------------------------
# Initialize policy + optimizer
# ---------------------------------------------------------------------------
rng = jax.random.PRNGKey(args.seed)
finger_biases = [2.0, 2.0, 0.0, 0.0, 2.0]  # seg0,1 open; seg2,3,4 neutral
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

print(f"\n{'='*64}")
print(f"BPTT 5-SEGMENT + ObsNorm + Curriculum + Batch={BATCH_SIZE}")
print(f"  MLP: {OBS_DIM}->32->32->{ACT_DIM} x {N_SEGMENTS} segs ({n_params} params)")
print(f"  Segments: {'+'.join(str(s) for s in N_SEG)} = {N_TOTAL} steps")
print(f"  Physics per env: {total_physics} substeps ({total_time:.3f}s)")
print(f"  Total physics per iter: {total_physics * BATCH_SIZE}")
print(f"  LR: {TRAIN_LR}, Gamma: {GAMMA}, Grad clip: {GRAD_CLIP}")
print(f"  Curriculum: {'OFF (ablation)' if args.no_curriculum else f'ON (patience={PATIENCE})'}")
print(f"  Seed: {args.seed}")
print(f"{'='*64}\n")


# ---------------------------------------------------------------------------
# Single env loss (vmapped)
# ---------------------------------------------------------------------------
def single_env_loss_fn(mlp_params, obs_mean, obs_std, data, active_steps):
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

        @jax.checkpoint
        def substep_fn(s, _):
            s = s.replace(ctrl=ctrl)
            s = mjx.step(mjx_model, s)
            return s, None

        def do_substeps(s):
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
        data,
        data.site_xpos[indices['ee_site_idx']],
        jnp.float64(0.0),
        jnp.float64(1.0),
    )
    (_, _, total_reward, _), all_obs = jax.lax.scan(
        frame_step_fn, init_carry, jnp.arange(N_TOTAL))
    return -total_reward, all_obs


# ---------------------------------------------------------------------------
# Batch loss
# ---------------------------------------------------------------------------
def batch_loss_fn(mlp_params, obs_mean, obs_std, active_steps, batch_data):
    losses, all_obs_batch = jax.vmap(
        single_env_loss_fn,
        in_axes=(None, None, None, 0, None)
    )(mlp_params, obs_mean, obs_std, batch_data, active_steps)
    return jnp.mean(losses), (all_obs_batch, losses)


# ---------------------------------------------------------------------------
# Forward rollout for logging — vmapped over the whole batch
#
# step_info layout (11 floats):
#   0:3  ee_pos      3:6  cube1_pos    6:9  cube2_pos
#   9    finger      10   reward
# ---------------------------------------------------------------------------
def single_forward_fn(mlp_params, obs_mean, obs_std, active_steps, data):
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

        def do_substeps(s):
            def substep_fn(s, _):
                s = s.replace(ctrl=ctrl)
                s = mjx.step(mjx_model, s)
                return s, s.qpos
            return jax.lax.scan(substep_fn, s, None, length=N_SUBSTEPS)

        def skip_substeps(s):
            return s, jnp.broadcast_to(s.qpos, (N_SUBSTEPS,) + s.qpos.shape)

        state, step_qpos = jax.lax.cond(
            step_idx < active_steps, do_substeps, skip_substeps, state)

        actual_rew = select_reward(state, step_idx, task_cfg, SEG_BOUNDARIES)
        step_info = jnp.concatenate([
            state.site_xpos[indices['ee_site_idx']],       # 0:3
            state.xpos[indices['cube1_body_idx']],         # 3:6
            state.xpos[indices['cube2_body_idx']],         # 6:9
            jnp.array([finger_target, actual_rew]),        # 9:11
        ])
        return (state, ee_pos), (step_info, step_qpos)

    init_carry = (data, data.site_xpos[indices['ee_site_idx']])
    (_, _), (all_step_info, all_qpos) = jax.lax.scan(
        frame_step_fn, init_carry, jnp.arange(N_TOTAL))
    return all_step_info, all_qpos


batch_forward_fn = jax.vmap(single_forward_fn,
                            in_axes=(None, None, None, None, 0))


# ---------------------------------------------------------------------------
# Per-segment metrics — averaged across the batch
# ---------------------------------------------------------------------------
def compute_metrics(info_np):
    """info_np: (B, N_TOTAL, 11) → list of (mean, std) per segment."""
    ee = info_np[:, :, 0:3]
    c1 = info_np[:, :, 3:6]
    c2 = info_np[:, :, 6:9]

    out = []
    for si, se in enumerate(SEG_ENDS):
        if si == 0:      # Pre-grasp: EE → above cube1
            target = c1[:, se] + np.array([0.0, 0.0, PRE_GRASP_Z_OFFSET])
            d = np.linalg.norm(ee[:, se] - target, axis=-1)
        elif si == 1:    # Descend: EE → cube1 holding pose
            target = c1[:, se] + np.array([0.0, 0.0, -0.01])
            d = np.linalg.norm(ee[:, se] - target, axis=-1)
        elif si == 2:    # Close: EE → cube1
            d = np.linalg.norm(ee[:, se] - c1[:, se], axis=-1)
        elif si == 3:    # Move: cube1 → stack target
            target = c2[:, se] + np.array([0.0, 0.0,
                                           CUBE2_HEIGHT + STACK_OFFSET])
            d = np.linalg.norm(c1[:, se] - target, axis=-1)
        else:            # Release: xy misalignment + z error
            xy = np.linalg.norm(c1[:, se, :2] - c2[:, se, :2], axis=-1)
            z = np.abs(c1[:, se, 2] -
                       (c2[:, se, 2] + CUBE2_HEIGHT + STACK_OFFSET))
            d = xy + z
        out.append((float(d.mean()), float(d.std())))
    return out


def criterion_met(metric_mean, phase):
    crit_type, thresh = CRITERIA[phase]
    if crit_type == "always":
        return True
    if crit_type == "lift":
        return metric_mean > thresh
    return metric_mean < thresh


# ---------------------------------------------------------------------------
# JIT compile
# ---------------------------------------------------------------------------
print("JIT compiling batch loss + grad...")
t0 = time.time()
value_and_grad_fn = jax.jit(jax.value_and_grad(batch_loss_fn, has_aux=True))
obs_mean, obs_std = obs_rms.get_jnp()

if args.no_curriculum:
    current_phase = N_SEGMENTS - 1
    active_steps = jnp.int32(N_TOTAL)
else:
    current_phase = 0
    active_steps = jnp.int32(PHASE_BOUNDS[min(1, len(PHASE_BOUNDS) - 1)])
patience_count = 0

(loss_val, (all_obs_init, _)), grad_val = value_and_grad_fn(
    mlp_params, obs_mean, obs_std, active_steps, batch)
loss_val.block_until_ready()
print(f"  JIT compile: {time.time()-t0:.1f}s")
print(f"  Initial loss: {float(loss_val):.6f}")
print(f"  Initial grad norm: {float(optax.global_norm(grad_val)):.4f}")

obs_rms.update(np.array(all_obs_init).reshape(-1, OBS_DIM))
obs_mean, obs_std = obs_rms.get_jnp()
print(f"  Obs RMS seeded: mean range "
      f"[{float(obs_mean.min()):.3f}, {float(obs_mean.max()):.3f}]")

print("\nJIT compiling batched forward rollout...")
t0 = time.time()
jit_forward = jax.jit(batch_forward_fn)
info_test, qpos_test = jit_forward(mlp_params, obs_mean, obs_std,
                                   active_steps, batch)
info_test.block_until_ready()
print(f"  JIT compile: {time.time()-t0:.1f}s, info shape: {info_test.shape}")

print(f"  Curriculum: phase={current_phase}, active_steps={int(active_steps)}")

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
    print("Viewer opened\n")


def render_trajectory(all_qpos, skip=1):
    """all_qpos: (N_TOTAL, N_SUBSTEPS, nq) for a single env."""
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
grad_norm_history = []
per_env_history = []
seg_mean_history = [[] for _ in range(N_SEGMENTS)]
seg_std_history = [[] for _ in range(N_SEGMENTS)]
phase_history = []

print("Training is about to begin...\n")
train_start = time.time()

for iteration in range(TRAIN_ITERS):
    if USE_VIEWER and not viewer_handle.is_running():
        print("\nThe viewer has been closed.")
        break

    if not args.no_rebatch:
        batch = rebatch_jit(jax.random.PRNGKey(1000 + iteration), BATCH_SIZE)

    obs_mean, obs_std = obs_rms.get_jnp()
    (loss_val, (all_obs_batch, per_env_losses)), grad_val = \
        value_and_grad_fn(mlp_params, obs_mean, obs_std, active_steps, batch)
    loss_f = float(loss_val)

    obs_rms.update(np.array(all_obs_batch).reshape(-1, OBS_DIM))
    per_env_history.append(np.array(-per_env_losses))

    if np.isnan(loss_f):
        print(f"\nNaN at iter {iteration}!")
        break

    loss_history.append(loss_f)
    grad_norm = float(optax.global_norm(grad_val))
    grad_norm_history.append(grad_norm)
    phase_history.append(current_phase)

    if iteration % args.log_every == 0 or iteration < 3:
        all_info, all_qpos = jit_forward(
            mlp_params, obs_mean, obs_std, active_steps, batch)
        info_np = np.array(all_info)

        metrics = compute_metrics(info_np)
        for si in range(N_SEGMENTS):
            seg_mean_history[si].append(metrics[si][0])
            seg_std_history[si].append(metrics[si][1])

        fng = info_np[:, :, 9]
        print(f"\nIter {iteration:4d}:  loss={loss_f:.4f}  "
              f"grad={grad_norm:.1f}  phase={current_phase}  "
              f"active={int(active_steps)}  "
              f"patience={patience_count}/{PATIENCE}")
        for si in range(N_SEGMENTS):
            marker = "->" if si == current_phase else "  "
            m, s = metrics[si]
            print(f"  {marker} {SEG_NAMES[si]:10s}: {m:.4f} +/- {s:.4f}   "
                  f"fng={fng[:, SEG_ENDS[si]].mean():.4f}")

        render_trajectory(all_qpos[0], skip=1)

        # ---- Progressive advancement, on the batch mean ----
        if not args.no_curriculum and current_phase < N_SEGMENTS - 1:
            if criterion_met(metrics[current_phase][0], current_phase):
                patience_count += 1
            else:
                patience_count = 0

            if patience_count >= PATIENCE:
                current_phase += 1
                patience_count = 0
                active_steps = jnp.int32(
                    PHASE_BOUNDS[min(current_phase + 1, N_SEGMENTS - 1)])
                print(f"  --> Phase {current_phase} ({SEG_NAMES[current_phase]}), "
                      f"active={int(active_steps)}")
    else:
        for si in range(N_SEGMENTS):
            seg_mean_history[si].append(
                seg_mean_history[si][-1] if seg_mean_history[si] else 0.0)
            seg_std_history[si].append(
                seg_std_history[si][-1] if seg_std_history[si] else 0.0)

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
ckpt_path = f"mlp_params_cube_stack_seed{args.seed}.pkl"
with open(ckpt_path, "wb") as f:
    pickle.dump(jax.tree.map(lambda x: np.array(x), mlp_params), f)
print(f"Model saved: {ckpt_path}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print(f"\n{'='*64}")
print("RESULTS")
print(f"{'='*64}")
print(f"  Batch size:      {BATCH_SIZE}")
print(f"  Initial loss:    {loss_history[0]:.4f}")
print(f"  Final loss:      {loss_history[-1]:.4f}")
print(f"  Final phase:     {current_phase} / {N_SEGMENTS - 1}")
for si in range(N_SEGMENTS):
    print(f"  {SEG_NAMES[si]:10s} final: {seg_mean_history[si][-1]:.4f} "
          f"+/- {seg_std_history[si][-1]:.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(N_SEGMENTS + 3, 1, figsize=(10, 3 * (N_SEGMENTS + 3)))

per_env_arr = np.array(per_env_history)
for env_i in range(per_env_arr.shape[1]):
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

axes[2].step(range(len(phase_history)), phase_history, linewidth=1.5,
             color='purple', where='post')
axes[2].set_ylabel("Phase")
axes[2].set_yticks(range(N_SEGMENTS))
axes[2].set_title("Curriculum Phase")
axes[2].grid(True, alpha=0.3)

seg_titles = [
    "Pre-grasp: EE→above cube", "Descend: EE→cube", "Close: EE→cube",
    "Move: cube1→stack target", "Release: xy + z error",
]
for si in range(N_SEGMENTS):
    ax = axes[si + 3]
    m = np.array(seg_mean_history[si])
    s = np.array(seg_std_history[si])
    x = np.arange(len(m))
    ax.plot(x, m, linewidth=1.5, color='tab:blue')
    ax.fill_between(x, m - s, m + s, alpha=0.2, color='tab:blue')
    crit_type, thresh = CRITERIA[si]
    if crit_type != "always":
        ax.axhline(y=thresh, color='red', linestyle='--', alpha=0.5,
                   label=f'threshold {thresh}')
        ax.legend(fontsize=8)
    ax.set_ylabel("Distance (m)")
    ax.set_title(f"Seg{si} ({seg_titles[si]}) — batch mean ± std")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Iteration")
fig.suptitle(f"BPTT 5-Seg + Curriculum + Batch={BATCH_SIZE} — "
             f"{'+'.join(str(s) for s in N_SEG)}, {len(loss_history)} iters")
fig.tight_layout()
plot_path = f"bptt_cube_stack_seed{args.seed}.png"
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
