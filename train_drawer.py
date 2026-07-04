"""
MJX BPTT — 10-Segment Drawer + Cube + Close (Progressive)

Usage:
    python train_drawer.py
    python train_drawer.py --iters 3000 --no-viewer
    python train_drawer.py --xml assets/drawer/mjx_drawer.xml
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
from rewards.drawer import select_reward
from envs.drawer_env import (
    N_JOINTS, FINGER_OPEN, Q_LO, Q_HI, VEL_LIMITS,
    OBS_DIM, ACT_DIM, LIFT_Z_TARGET, ABOVE_CUBE_Z,
    discover_indices, make_task_cfg, build_obs, create_batch,
    evaluate_phase_metric, metric_satisfied, PATIENCE,
)

# ---------------------------------------------------------------------------
# Solver patch
# ---------------------------------------------------------------------------
patch_solver()

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--seg0", type=int, default=20, help="Approach handle")
parser.add_argument("--seg1", type=int, default=6,  help="Grasp handle")
parser.add_argument("--seg2", type=int, default=20, help="Pull drawer")
parser.add_argument("--seg3", type=int, default=12, help="Release + lift")
parser.add_argument("--seg4", type=int, default=24, help="Above cube")
parser.add_argument("--seg5", type=int, default=12, help="Descend to cube")
parser.add_argument("--seg6", type=int, default=6,  help="Grasp cube")
parser.add_argument("--seg7", type=int, default=16, help="Place cube")
parser.add_argument("--seg8", type=int, default=8,  help="Release + re-handle")
parser.add_argument("--seg9", type=int, default=12, help="Push closed")
parser.add_argument("--substeps", type=int, default=4)
parser.add_argument("--iters", type=int, default=3000)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--gamma", type=float, default=0.999)
parser.add_argument("--batch", type=int, default=13)
parser.add_argument("--solver-iters", type=int, default=6)
parser.add_argument("--grad-clip", type=float, default=400.0)
parser.add_argument("--hidden-trunk", type=int, default=64)
parser.add_argument("--hidden-head", type=int, default=24)
parser.add_argument("--no-viewer", action="store_true")
parser.add_argument("--xml", type=str, default="assets/common/franka_emika_panda/mjx_drawer.xml")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Segment config
# ---------------------------------------------------------------------------
N_SEG = [args.seg0, args.seg1, args.seg2, args.seg3, args.seg4,
         args.seg5, args.seg6, args.seg7, args.seg8, args.seg9]
N_TOTAL = sum(N_SEG)
N_SEGMENTS = len(N_SEG)

SEG_BOUNDARIES = []
acc = 0
for s in N_SEG[:-1]:
    acc += s
    SEG_BOUNDARIES.append(acc)
# SEG_BOUNDARIES = [SEG1..SEG9_START], 9 elements

PHASE_BOUNDARIES = SEG_BOUNDARIES + [N_TOTAL]  # 10 elements

SEG_NAMES = [
    "APPROACH", "GRASP HDL", "PULL", "LIFT", "ABOVE CUBE",
    "TO CUBE", "GRAB CUBE", "PLACE", "RE-HANDLE", "PUSH",
]

SEGMENTS = []
starts = [0] + SEG_BOUNDARIES
ends = SEG_BOUNDARIES + [N_TOTAL]
for i, name in enumerate(SEG_NAMES):
    SEGMENTS.append((name, starts[i], ends[i]))

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
    print(f"  Seg{i} ({name:12s}): steps {s}-{e-1} ({N_SEG[i]} steps)")
print(f"  Total: {N_TOTAL} × {N_SUBSTEPS} substeps = "
      f"{total_physics} physics ({total_time:.3f}s)")
print(f"  Batch: {BATCH_SIZE}")
print(f"  EE home:  {mj_data.site_xpos[indices['ee_site_idx']]}")
print(f"  Handle:   {mj_data.site_xpos[indices['handle_site_idx']]}")
print(f"  Cube:     {mj_data.xpos[indices['cube_body_idx']]}")

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
    BATCH_SIZE, mj_model, mj_data, mjx_model, key_id)
mjx_data_single0 = jax.tree.map(lambda x: x[0], mjx_data_batch)

# ---------------------------------------------------------------------------
# Initialize policy + optimizer
# ---------------------------------------------------------------------------
rng = jax.random.PRNGKey(42)
finger_biases = [2.0, 0.0, -2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, -2.0]
mlp_params = init_all_mlps(
    rng, OBS_DIM, args.hidden_trunk, args.hidden_head, ACT_DIM,
    n_segments=N_SEGMENTS, finger_biases=finger_biases)
n_params = sum(p.size for p in jax.tree.leaves(mlp_params))

optimizer = optax.chain(
    optax.clip_by_global_norm(GRAD_CLIP),
    optax.adam(TRAIN_LR),
)
opt_state = optimizer.init(mlp_params)
obs_rms = ObsRMS(OBS_DIM)

print(f"\n{'='*60}")
print(f"BPTT 10-SEG DRAWER+CUBE — Batch={BATCH_SIZE}")
print(f"  MLP: {OBS_DIM}→{args.hidden_trunk}→{args.hidden_trunk}"
      f"→{args.hidden_head}→{ACT_DIM} × {N_SEGMENTS} heads ({n_params} params)")
print(f"  Segments: {'+'.join(str(s) for s in N_SEG)} = {N_TOTAL} steps")
print(f"  LR: {TRAIN_LR}, Gamma: {GAMMA}, Grad clip: {GRAD_CLIP}")
print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# Single env loss (vmapped)
# ---------------------------------------------------------------------------
def single_env_loss_fn(mlp_params, obs_mean, obs_std,
                       mjx_data_single, active_steps):
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
# Forward rollout for logging (env 0)
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
        handle_pos = state.site_xpos[indices['handle_site_idx']]
        drawer_pos = state.qpos[indices['drawer_joint_qpos_idx']]
        actual_rew = select_reward(state, step_idx, task_cfg, SEG_BOUNDARIES)
        cube_pos = state.xpos[indices['cube_body_idx']]
        inside_pos = state.site_xpos[indices['drawer_inside_site_idx']]

        step_info = jnp.concatenate([
            ee_new,                                     # 0:3
            handle_pos,                                 # 3:6
            jnp.array([finger_target, actual_rew]),     # 6:8
            safe_q_dot,                                 # 8:15
            state.qvel[:N_JOINTS],                      # 15:22
            jnp.array([drawer_pos]),                    # 22
            cube_pos,                                   # 23:26
            inside_pos,                                 # 26:29
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
print(f"  JIT compile: {time.time()-t0:.1f}s, step_info shape: {info_test.shape}")

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

print("Training başlıyor...\n")
train_start = time.time()

for iteration in range(TRAIN_ITERS):
    if USE_VIEWER and not viewer_handle.is_running():
        print("\nViewer kapatıldı.")
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

    # Log every 5 iters
    if iteration % 5 == 0 or iteration < 3:
        all_info, all_qpos = jit_forward(
            mlp_params, obs_mean, obs_std, active_steps)
        info_np = np.array(all_info)

        # Segment metrics
        metrics = {}
        seg_ends = SEG_BOUNDARIES + [N_TOTAL - 1]
        # seg0,1,8: EE-handle
        for si in (0, 1):
            se = seg_ends[si] - 1
            metrics[si] = np.linalg.norm(info_np[se, :3] - info_np[se, 3:6])
        # seg2: drawer position
        metrics[2] = info_np[seg_ends[2] - 1, 22]
        # seg3: EE-lift_target
        se3 = seg_ends[3] - 1
        h3 = info_np[se3, 3:6]
        metrics[3] = np.linalg.norm(
            info_np[se3, :3] - (h3 + np.array([0, 0, LIFT_Z_TARGET])))
        # seg4: EE-above_cube
        se4 = seg_ends[4] - 1
        c4 = info_np[se4, 23:26]
        metrics[4] = np.linalg.norm(
            info_np[se4, :3] - (c4 + np.array([0, 0, ABOVE_CUBE_Z])))
        # seg5,6: EE-cube
        for si in (5, 6):
            se = seg_ends[si] - 1
            metrics[si] = np.linalg.norm(
                info_np[se, :3] - info_np[se, 23:26])
        # seg7,8: cube-inside
        for si in (7, 8):
            se = seg_ends[si] - 1
            metrics[si] = np.linalg.norm(
                info_np[se, 23:26] - info_np[se, 26:29])
        # seg9: drawer close
        metrics[9] = info_np[-1, 22]

        for i in range(N_SEGMENTS):
            seg_histories[i].append(metrics[i])

        print(f"\nIter {iteration:4d}:  loss={loss_f:.4f}  "
              f"grad={grad_norm:.1f}  phase={current_phase}  "
              f"active={int(active_steps)}  "
              f"patience={patience_count}/{PATIENCE}")
        for i, (name, _, _) in enumerate(SEGMENTS):
            print(f"  Seg{i} {name:12s}: {metrics[i]:.4f}")

        # Per-step detail
        for seg_name, s_start, s_end in SEGMENTS:
            print(f"  --- {seg_name} (step {s_start}-{s_end-1}) ---")
            for s in range(s_start, s_end):
                ee = info_np[s, :3]
                fng = info_np[s, 6]
                rew = info_np[s, 7]
                drw = info_np[s, 22]
                cube = info_np[s, 23:26]
                print(f"    step {s:3d}: "
                      f"EE=[{ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}]  "
                      f"cube=[{cube[0]:.3f},{cube[1]:.3f},{cube[2]:.3f}]  "
                      f"fng={fng:.4f}  drw={drw:.4f}  rew={rew:.4f}")

        # Phase progression check
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

    # Update params
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
with open("drawer_mlp_params.pkl", "wb") as f:
    pickle.dump(jax.tree.map(lambda x: np.array(x), mlp_params), f)
print("Model saved: drawer_mlp_params.pkl")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"SONUÇLAR — 10-SEG DRAWER+CUBE")
print(f"{'='*60}")
print(f"  Final loss: {loss_history[-1]:.4f}")
for i, (name, _, _) in enumerate(SEGMENTS):
    print(f"  Seg{i} ({name:12s}): {seg_histories[i][-1]:.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(6, 2, figsize=(14, 20))
axes = axes.flatten()

per_env_arr = np.array(per_env_history)
for env_i in range(BATCH_SIZE):
    axes[0].plot(per_env_arr[:, env_i], alpha=0.15, linewidth=0.5, color='gray')
axes[0].plot([-l for l in loss_history], linewidth=2, color='blue',
             label='mean reward')
axes[0].set_title("Reward")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(grad_norm_history, linewidth=1.5, color='red')
axes[1].set_title("Gradient Norm")
axes[1].grid(True, alpha=0.3)

seg_titles = [
    "Seg0 EE→Handle", "Seg1 Grasp Handle", "Seg2 Drawer Open",
    "Seg3 Lift", "Seg4 Above Cube", "Seg5 EE→Cube",
    "Seg6 Grasp Cube", "Seg7 Place", "Seg8 Re-Handle",
    "Seg9 Drawer Close",
]
for i in range(N_SEGMENTS):
    ax = axes[i + 2]
    ax.plot(seg_histories[i], linewidth=1.5)
    ax.set_title(seg_titles[i])
    ax.grid(True, alpha=0.3)
    if i == 2:
        ax.axhline(y=-0.12, color='red', linestyle='--', alpha=0.5)
    if i == 9:
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

fig.suptitle(f"BPTT 10-Seg Drawer+Cube — Batch={BATCH_SIZE}, "
             f"{len(loss_history)} iters", fontsize=14)
fig.tight_layout()
plot_path = "bptt_drawer_10seg_results.png"
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