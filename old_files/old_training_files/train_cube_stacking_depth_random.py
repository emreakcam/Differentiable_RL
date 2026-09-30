"""
MJX BPTT — Cube Stacking + Differentiable Depth CNN + Progressive Curriculum

2 overhead + 1 wrist kamera → diff depth render → CNN → latent
+ proprioception → segment action head → MJX physics → BPTT

Progressive curriculum: phase başına 1 segment açılır (bir sonraki segment
de açık). Metric threshold geçilince patience sonrası phase ilerler.

Usage:
    python train_cube_stacking_depth.py --no-viewer
    python train_cube_stacking_depth.py --batch 16 --iters 1500 --lr 3e-4
"""
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")

import argparse, time, pickle, gc
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
from helpers.diff_render import (
    setup_cameras, setup_wrist_camera, render_all_cameras,
)
from models.depth_policy import (
    init_depth_policy, depth_policy_forward, cnn_forward,
)
from rewards.cube_stack_depth import select_reward
from envs.cube_stack_depth_env import (
    N_JOINTS, FINGER_OPEN, Q_LO, Q_HI, VEL_LIMITS, TARGET_QUAT,
    ACT_DIM, PROPRIO_DIM, LATENT_DIM, N_CHANNELS,
    IMG_RES, OVERHEAD_MAX_DEPTH, OVERHEAD_FOVY_RAD,
    WRIST_MAX_DEPTH,
    PRE_GRASP_Z, CUBE2_HEIGHT, STACK_OFFSET,
    SEG_STEPS, N_TOTAL, SEG_BOUNDARIES, FINGER_BIASES,
    CAM_NAMES,
    discover_indices, make_task_cfg, build_proprio, create_batch,
)

patch_solver()

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--substeps", type=int, default=4)
parser.add_argument("--iters", type=int, default=2000)
parser.add_argument("--lr", type=float, default=8e-4)
parser.add_argument("--gamma", type=float, default=0.999)
parser.add_argument("--grad-clip", type=float, default=500.0)
parser.add_argument("--batch", type=int, default=48)
parser.add_argument("--hidden", type=int, default=256)
parser.add_argument("--solver-iters", type=int, default=6)
parser.add_argument("--no-viewer", action="store_true")
parser.add_argument("--xml", type=str,
                    default="assets/common/franka_emika_panda/"
                            "mjx_single_cube_depth_cam.xml")
parser.add_argument("--cameras", type=str, default="overhead1,overhead2,wrist",
                    help="Comma-separated camera selection: overhead1,overhead2,wrist")
args = parser.parse_args()

N_SUBSTEPS = args.substeps
GAMMA = args.gamma
BATCH_SIZE = args.batch
OBS_DIM = LATENT_DIM + PROPRIO_DIM

# ── Camera selection ──────────────────────────────────────────────────────
# Değiştirmek için: --cameras overhead1,wrist  veya  --cameras wrist  vb.
cam_selection = [c.strip() for c in args.cameras.split(",")]
use_overhead1 = "overhead1" in cam_selection
use_overhead2 = "overhead2" in cam_selection
use_wrist     = "wrist" in cam_selection
ACTIVE_N_CHANNELS = len(cam_selection)

ACTIVE_CAM_NAMES = []
if use_overhead1: ACTIVE_CAM_NAMES.append(CAM_NAMES[0])  # 'overhead_cam1'
if use_overhead2: ACTIVE_CAM_NAMES.append(CAM_NAMES[1])  # 'overhead_cam2'

MAX_DEPTHS = []
if use_overhead1: MAX_DEPTHS.append(OVERHEAD_MAX_DEPTH)
if use_overhead2: MAX_DEPTHS.append(OVERHEAD_MAX_DEPTH)
if use_wrist:     MAX_DEPTHS.append(WRIST_MAX_DEPTH)

print(f"  Cameras: {cam_selection} ({ACTIVE_N_CHANNELS} channels)")

# ═══════════════════════════════════════════════════════════════════════════
# MODEL + CAMERAS
# ═══════════════════════════════════════════════════════════════════════════
print("\n[DEPTH] Loading model...")
mj_model = mujoco.MjModel.from_xml_path(args.xml)
mj_model.opt.iterations = args.solver_iters
mj_model.opt.ls_iterations = 8
mj_data = mujoco.MjData(mj_model)

idx = discover_indices(mj_model)
cfg = make_task_cfg(idx)

# Home keyframe
key_id = mj_model.keyframe("home").id
mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
mujoco.mj_forward(mj_model, mj_data)

# Kamera setup — seçime göre
overhead_cams = setup_cameras(mj_model, mj_data, ACTIVE_CAM_NAMES,
                               OVERHEAD_FOVY_RAD, IMG_RES, IMG_RES)

if use_wrist:
    wrist_cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    WRIST_FOVY_RAD = jnp.deg2rad(float(mj_model.cam_fovy[wrist_cam_id]))
    wrist_info = setup_wrist_camera(mj_model, WRIST_FOVY_RAD, IMG_RES, IMG_RES)
else:
    wrist_info = None

print(f"  Overhead cameras: {len(overhead_cams)}")
for i, cam in enumerate(overhead_cams):
    print(f"    cam{i} pos: {cam['pos']}")
if wrist_info is not None:
    print(f"  Wrist camera: cam_id={wrist_info['cam_id']}")
else:
    print(f"  Wrist camera: disabled")

home_ctrl = jnp.array(mj_data.ctrl)
frame_dt = N_SUBSTEPS * mj_model.opt.timestep

# MJX
print(f"  Setting up MJX (solver_iters={args.solver_iters})...")
mjx_model = mjx.put_model(mj_model)

# Batch
batch = create_batch(mj_model, mj_data, mjx_model, BATCH_SIZE)
data0 = jax.tree.map(lambda x: x[0], batch)
print(f"  Batch: {BATCH_SIZE} envs, {N_TOTAL} steps, segs={SEG_STEPS}")

# Debug: depth render test
mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
mujoco.mj_forward(mj_model, mj_data)
d_test = mjx.put_data(mj_model, mj_data)
d_test = mjx.forward(mjx_model, d_test)
home_data = mjx.put_data(mj_model, mj_data)
home_data = mjx.forward(mjx_model, home_data)

depth_all = render_all_cameras(d_test, overhead_cams, wrist_info,
                                cfg['box_body_indices'], cfg['box_halfs'],
                                cfg['finger_body_indices'], cfg['finger_half'],
                                cfg['hand_body_idx'], cfg['hand_half'],
                                OVERHEAD_MAX_DEPTH, WRIST_MAX_DEPTH)
depth_np = np.array(depth_all)
print(f"  depth_all.shape: {depth_all.shape}")

cam_labels = []
cam_vmaxs = []
if use_overhead1: cam_labels.append('Overhead 1'); cam_vmaxs.append(OVERHEAD_MAX_DEPTH)
if use_overhead2: cam_labels.append('Overhead 2'); cam_vmaxs.append(OVERHEAD_MAX_DEPTH)
if use_wrist:     cam_labels.append('Wrist');      cam_vmaxs.append(WRIST_MAX_DEPTH)

fig, axes = plt.subplots(1, ACTIVE_N_CHANNELS,
                          figsize=(5 * ACTIVE_N_CHANNELS, 5))
if ACTIVE_N_CHANNELS == 1:
    axes = [axes]
for i in range(ACTIVE_N_CHANNELS):
    im = axes[i].imshow(depth_np[:, :, i], cmap='viridis',
                         vmin=0, vmax=float(cam_vmaxs[i]))
    axes[i].set_title(cam_labels[i])
    plt.colorbar(im, ax=axes[i], label='depth (m)')
fig.suptitle(f"Depth Render — {IMG_RES}×{IMG_RES} ({args.cameras})")
fig.tight_layout()
fig.savefig("debug_depth_cameras.png", dpi=150)
plt.close(fig)
print("✓ Debug depth saved: debug_depth_cameras.png")

# ═══════════════════════════════════════════════════════════════════════════
# POLICY + OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════
rng = jax.random.PRNGKey(42)
all_params = init_depth_policy(
    rng, ACTIVE_N_CHANNELS, LATENT_DIM, PROPRIO_DIM,
    args.hidden, ACT_DIM, IMG_RES, IMG_RES,
    n_segments=5, finger_biases=FINGER_BIASES)

n_cnn_params = sum(p.size for p in jax.tree.leaves(all_params['cnn']))
n_head_params = sum(
    p.size for k, v in all_params.items() if k != 'cnn'
    for p in jax.tree.leaves(v))
n_total_params = n_cnn_params + n_head_params

optimizer = optax.chain(
    optax.clip_by_global_norm(args.grad_clip),
    optax.adam(args.lr),
)
opt_state = optimizer.init(all_params)
obs_rms = ObsRMS(PROPRIO_DIM)

print(f"\n{'='*65}")
print(f"BPTT DEPTH CNN — Cube Stacking + Progressive Curriculum")
print(f"  CNN: ({IMG_RES},{IMG_RES},{ACTIVE_N_CHANNELS})→8→16→32→FC→{LATENT_DIM} "
      f"({n_cnn_params} params)")
print(f"  Trunk: {LATENT_DIM+PROPRIO_DIM}→128→128 (shared)")
print(f"  Heads: 128→32→{ACT_DIM} × 5")
print(f"  Total: {n_total_params} params (CNN: {n_cnn_params})")
print(f"  Total: {n_total_params} params")
print(f"  Obs: {LATENT_DIM} (CNN, {ACTIVE_N_CHANNELS}ch) + {PROPRIO_DIM} (proprio) = {OBS_DIM}")
print(f"  Segments: {'+'.join(map(str, SEG_STEPS))} = {N_TOTAL} steps")
print(f"  LR: {args.lr}, Gamma: {GAMMA}, Clip: {args.grad_clip}")
print(f"{'='*65}\n")


# ═══════════════════════════════════════════════════════════════════════════
# PROGRESSIVE CURRICULUM
# ═══════════════════════════════════════════════════════════════════════════
PATIENCE = 4

PHASE_BOUNDS = []
acc = 0
for s in SEG_STEPS:
    acc += s
    PHASE_BOUNDS.append(acc)

CRITERIA = [
    ("dist", 0.04),   # 0 Reach: EE→cube < 4cm
    ("dist", 0.04),   # 1 Descend: EE→cube < 4cm
    ("dist", 0.04),   # 2 Grasp: EE→cube < 4cm
    ("dist", 0.10),   # 3 Move: cube→stack < 10cm
    ("always", 0),    # 4 Release
]

phase = 0
patience_cnt = 0
# Başta seg0 + seg1 aktif (bir sonraki segment de açık)
active_steps = jnp.int32(PHASE_BOUNDS[min(1, len(PHASE_BOUNDS) - 1)])

seg_labels = ['Reach', 'Descend', 'Grasp', 'Move', 'Release']
seg_ends = []
acc = 0
for s in SEG_STEPS:
    acc += s
    seg_ends.append(acc - 1)

print(f"  Curriculum: phase=0, active_steps={int(active_steps)}")
print(f"  Criteria: {CRITERIA}")


# ═══════════════════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════
def single_env_loss(all_params, obs_mean, obs_std, data, active_steps):
    @jax.checkpoint
    def frame_step(carry, step_idx):
        state, prev_ee, total_rew, gamma_acc = carry

        # stop_gradient at segment boundaries
        is_boundary = jnp.zeros((), dtype=bool)
        for b in SEG_BOUNDARIES:
            is_boundary = is_boundary | (step_idx == b)
        state = jax.lax.cond(is_boundary,
            lambda s: jax.lax.stop_gradient(s), lambda s: s, state)
        prev_ee = jax.lax.cond(is_boundary,
            lambda p: jax.lax.stop_gradient(p), lambda p: p, prev_ee)
        gamma_acc = jnp.where(is_boundary, jnp.float64(1.0), gamma_acc)

        # Depth render → CNN → latent
        depth_img = render_all_cameras(
            state, overhead_cams, wrist_info,
            cfg['box_body_indices'], cfg['box_halfs'],
            cfg['finger_body_indices'], cfg['finger_half'],
            cfg['hand_body_idx'], cfg['hand_half'],
            OVERHEAD_MAX_DEPTH, WRIST_MAX_DEPTH)
        depth_img = jax.lax.stop_gradient(depth_img)
        cnn_latent = cnn_forward(all_params['cnn'], depth_img, MAX_DEPTHS)

        # Proprioception
        proprio, ee_pos, current_q = build_proprio(
            state, prev_ee, frame_dt, idx)
        proprio_norm = (proprio.astype(jnp.float32) - obs_mean) / obs_std

        # Policy forward
        qd, f = depth_policy_forward(
            all_params, cnn_latent, proprio_norm,
            step_idx, SEG_BOUNDARIES,
            N_JOINTS, VEL_LIMITS, FINGER_OPEN)
        safe_qd = safe_action(qd, current_q, Q_LO, Q_HI)

        ctrl = home_ctrl.at[:N_JOINTS].set(safe_qd)
        ctrl = ctrl.at[7].set(f)

        # Physics substeps — sadece aktif step'lerde
        @jax.checkpoint
        def substep(s, _):
            s = s.replace(ctrl=ctrl)
            s = mjx.step(mjx_model, s)
            return s, None
        def do_substeps(s):
            s, _ = jax.lax.scan(substep, s, None, length=N_SUBSTEPS)
            return s
        state = jax.lax.cond(step_idx < active_steps,
                             do_substeps, lambda s: s, state)

        # Reward — sadece aktif step'lerde
        rew = select_reward(state, step_idx, cfg, SEG_BOUNDARIES)
        rew = jnp.where(step_idx < active_steps, rew, 0.0)

        total_rew = total_rew + gamma_acc * rew
        gamma_acc = gamma_acc * GAMMA

        return (state, ee_pos, total_rew, gamma_acc), proprio

    init = (data, data.site_xpos[idx['ee_site']],
            jnp.float64(0.0), jnp.float64(1.0))
    (_, _, total_rew, _), all_proprio = jax.lax.scan(
        frame_step, init, jnp.arange(N_TOTAL))
    return -total_rew, all_proprio


def batch_loss(all_params, obs_mean, obs_std, active_steps, batch_data):
    losses, all_obs = jax.vmap(
        single_env_loss, in_axes=(None, None, None, 0, None)
    )(all_params, obs_mean, obs_std, batch_data, active_steps)
    return jnp.mean(losses), (all_obs, losses)

def rebatch_jax(rng_key, batch_size):
    keys = jax.random.split(rng_key, batch_size)
    def randomize_one(k):
        noise = jax.random.uniform(k, (2,), minval=-0.05, maxval=0.05)
        new_qpos = home_data.qpos.at[9].add(noise[0])
        new_qpos = new_qpos.at[10].add(noise[1])
        d = home_data.replace(qpos=new_qpos)
        d = mjx.forward(mjx_model, d)
        return d
    return jax.vmap(randomize_one)(keys)

rebatch_jit = jax.jit(rebatch_jax, static_argnums=(1,))


# ═══════════════════════════════════════════════════════════════════════════
# JIT COMPILE
# ═══════════════════════════════════════════════════════════════════════════
print("JIT compiling batch loss + grad...")
t0 = time.time()
value_and_grad_fn = jax.jit(jax.value_and_grad(batch_loss, has_aux=True))

_ = rebatch_jit(jax.random.PRNGKey(9999), BATCH_SIZE)
print("✓ rebatch_jit compiled")

obs_mean, obs_std = obs_rms.get_jnp()
(loss_val, (all_obs_init, _)), grad_val = value_and_grad_fn(
    all_params, obs_mean, obs_std, active_steps, batch)
loss_val.block_until_ready()
print(f"  JIT: {time.time()-t0:.1f}s, loss: {float(loss_val):.4f}")
print(f"  Grad norm: {float(optax.global_norm(grad_val)):.4f}")

# Seed obs_rms
obs_rms.update(np.array(all_obs_init).reshape(-1, PROPRIO_DIM))


# ═══════════════════════════════════════════════════════════════════════════
# FORWARD ROLLOUT (logging + viewer)
# ═══════════════════════════════════════════════════════════════════════════
def single_forward(all_params, obs_mean, obs_std, active_steps, fwd_data):
    def frame_step(carry, step_idx):
        state, prev_ee = carry

        depth_img = render_all_cameras(
            state, overhead_cams, wrist_info,
            cfg['box_body_indices'], cfg['box_halfs'],
            cfg['finger_body_indices'], cfg['finger_half'],
            cfg['hand_body_idx'], cfg['hand_half'],
            OVERHEAD_MAX_DEPTH, WRIST_MAX_DEPTH)
        cnn_latent = cnn_forward(all_params['cnn'], depth_img, MAX_DEPTHS)

        proprio, ee_pos, current_q = build_proprio(
            state, prev_ee, frame_dt, idx)
        proprio_norm = (proprio.astype(jnp.float32) - obs_mean) / obs_std

        qd, f = depth_policy_forward(
            all_params, cnn_latent, proprio_norm,
            step_idx, SEG_BOUNDARIES,
            N_JOINTS, VEL_LIMITS, FINGER_OPEN)
        safe_qd = safe_action(qd, current_q, Q_LO, Q_HI)

        ctrl = home_ctrl.at[:N_JOINTS].set(safe_qd)
        ctrl = ctrl.at[7].set(f)

        def do_sub(s):
            def substep(s, _):
                s = s.replace(ctrl=ctrl)
                s = mjx.step(mjx_model, s)
                return s, s.qpos
            return jax.lax.scan(substep, s, None, length=N_SUBSTEPS)
        def skip_sub(s):
            return s, jnp.broadcast_to(s.qpos, (N_SUBSTEPS,) + s.qpos.shape)

        state, step_qpos = jax.lax.cond(
            step_idx < active_steps, do_sub, skip_sub, state)

        return (state, ee_pos), (state, step_qpos, f)

    init = (fwd_data, fwd_data.site_xpos[idx['ee_site']])
    (final, _), (all_states, all_qpos, all_fingers) = jax.lax.scan(
        frame_step, init, jnp.arange(N_TOTAL))
    return all_states, all_qpos, all_fingers

print("\nJIT compiling forward rollout...")
t0 = time.time()
jit_forward = jax.jit(single_forward)
_ = jit_forward(all_params, obs_mean, obs_std, active_steps, data0)
print(f"  Forward JIT: {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════
# VIEWER
# ═══════════════════════════════════════════════════════════════════════════
USE_VIEWER = not args.no_viewer
if USE_VIEWER:
    from mujoco import viewer as mj_viewer
    mj_model_render = mujoco.MjModel.from_xml_path(args.xml)
    mj_data_render = mujoco.MjData(mj_model_render)
    mujoco.mj_resetDataKeyframe(mj_model_render, mj_data_render, key_id)
    mujoco.mj_forward(mj_model_render, mj_data_render)
    viewer_handle = mj_viewer.launch_passive(mj_model_render, mj_data_render)
    print("✓ Viewer açıldı\n")


def render_trajectory(all_qpos, skip=1):
    if not USE_VIEWER or not viewer_handle.is_running():
        return
    qpos_np = np.array(all_qpos)
    sim_dt = mj_model.opt.timestep
    for step in range(qpos_np.shape[0]):
        for sub in range(0, qpos_np.shape[1], skip):
            mj_data_render.qpos[:] = qpos_np[step, sub]
            mujoco.mj_forward(mj_model_render, mj_data_render)
            viewer_handle.sync()
            time.sleep(sim_dt * skip)


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════
loss_history = []
grad_norm_history = []
seg_dist_history = [[] for _ in range(5)]

print("\nTraining başlıyor...\n")
train_start = time.time()

for iteration in range(args.iters):
    if USE_VIEWER and not viewer_handle.is_running():
        print("\nViewer kapatıldı.")
        break
    
    batch = rebatch_jit(jax.random.PRNGKey(iteration), BATCH_SIZE)
    data0 = jax.tree.map(lambda x: x[0], batch)

    obs_mean, obs_std = obs_rms.get_jnp()

    (loss_val, (all_obs, per_env_losses)), grad_val = value_and_grad_fn(
        all_params, obs_mean, obs_std, active_steps, batch)

    loss_f = float(loss_val)
    obs_rms.update(np.array(all_obs).reshape(-1, PROPRIO_DIM))

    if np.isnan(loss_f):
        print(f"\n⚠ NaN at iter {iteration}!")
        break

    loss_history.append(loss_f)
    grad_norm = float(optax.global_norm(grad_val))
    grad_norm_history.append(grad_norm)

    if iteration % 5 == 0 or iteration < 3:
        all_states, all_qpos, all_fng = jit_forward(
            all_params, obs_mean, obs_std, active_steps, data0)
        s_ee = np.array(all_states.site_xpos[:, idx['ee_site']])
        s_c1 = np.array(all_states.xpos[:, idx['cube1']])
        s_c2 = np.array(all_states.xpos[:, idx['cube2']])
        s_fng = np.array(all_fng)

        metrics = []
        for si in range(5):
            se = seg_ends[si]
            if si == 0:  # Reach → pre_grasp target
                target = s_c1[se] + np.array([0, 0, PRE_GRASP_Z])
                metrics.append(np.linalg.norm(s_ee[se] - target))
            elif si == 1:  # Descend → cube
                target = s_c1[se] + np.array([0, 0, -0.01])
                metrics.append(np.linalg.norm(s_ee[se] - target))
            elif si == 2:  # Grasp → cube
                metrics.append(np.linalg.norm(s_ee[se] - s_c1[se]))
            elif si == 3:  # Move → stack target
                stack_t = s_c2[se] + np.array([0, 0,
                                                CUBE2_HEIGHT + STACK_OFFSET])
                metrics.append(np.linalg.norm(s_c1[se] - stack_t))
            else:  # Release
                stack_t = s_c2[se] + np.array([0, 0,
                                                CUBE2_HEIGHT + STACK_OFFSET])
                c1_final = s_c1[se]
                xy_err = np.linalg.norm(c1_final[:2] - s_c2[se][:2])
                z_err = abs(c1_final[2] - stack_t[2])
                metrics.append(xy_err + z_err)

        for si in range(5):
            seg_dist_history[si].append(metrics[si])

        # Print
        print(f"\nIter {iteration:4d}:  loss={loss_f:.4f}  grad={grad_norm:.1f}"
              f"  phase={phase}  active={int(active_steps)}")
        for si in range(5):
            marker = "→" if si == phase else " "
            print(f"  {marker} {seg_labels[si]:8s}: {metrics[si]:.4f}  "
                  f"fng={s_fng[seg_ends[si]]:.4f}")

        render_trajectory(all_qpos, skip=2)

        # ── Progressive advancement ──
        if phase < len(PHASE_BOUNDS) - 1:
            crit_type, thresh = CRITERIA[phase]
            met = metrics[phase]
            if crit_type == "always":
                satisfied = True
            elif crit_type == "lift":
                satisfied = met > thresh
            else:
                satisfied = met < thresh

            if satisfied:
                patience_cnt += 1
            else:
                patience_cnt = 0

            if patience_cnt >= PATIENCE:
                phase += 1
                patience_cnt = 0
                active_steps = jnp.int32(
                    PHASE_BOUNDS[min(phase + 1, len(PHASE_BOUNDS) - 1)])
                print(f"  ──→ Phase {phase}, active={int(active_steps)}")
    else:
        for si in range(5):
            seg_dist_history[si].append(
                seg_dist_history[si][-1] if seg_dist_history[si] else 0)

    updates, opt_state = optimizer.update(grad_val, opt_state, all_params)
    all_params = optax.apply_updates(all_params, updates)

    if iteration % 50 == 0:
        plt.close('all')
        gc.collect()

train_time = time.time() - train_start
print(f"\nTraining done: {train_time:.1f}s ({len(loss_history)} iters)")

# ═══════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════
with open("depth_cnn_params.pkl", "wb") as f:
    pickle.dump(jax.tree.map(lambda x: np.array(x), all_params), f)
print("Model saved: depth_cnn_params.pkl")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════════════════
n_plots = 2 + len(seg_labels)
fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots))

axes[0].plot([-l for l in loss_history], linewidth=2, color='blue')
axes[0].set_ylabel("Reward")
axes[0].set_title("Mean Reward")
axes[0].grid(True, alpha=0.3)

axes[1].plot(grad_norm_history, linewidth=1.5, color='red')
axes[1].set_ylabel("Grad Norm")
axes[1].set_title("Gradient Norm")
axes[1].grid(True, alpha=0.3)

for si in range(5):
    axes[si + 2].plot(seg_dist_history[si], linewidth=1.5)
    axes[si + 2].set_ylabel("Distance (m)")
    axes[si + 2].set_title(f"{seg_labels[si]}")
    axes[si + 2].grid(True, alpha=0.3)

axes[-1].set_xlabel("Iteration")
fig.suptitle(f"BPTT Depth CNN — Batch={BATCH_SIZE}, {IMG_RES}×{IMG_RES}, "
             f"cams={args.cameras}, latent={LATENT_DIM}, {len(loss_history)} iters")
fig.tight_layout()
fig.savefig("bptt_depth_cnn_results.png", dpi=150)
plt.close(fig)
print(f"Plot saved: bptt_depth_cnn_results.png")

if USE_VIEWER and viewer_handle.is_running():
    print("\nViewer açık — ESC ile kapat.")
    try:
        while viewer_handle.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nÇıkılıyor.")