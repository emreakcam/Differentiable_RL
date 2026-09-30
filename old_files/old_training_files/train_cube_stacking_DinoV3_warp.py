"""
Vision DINOv3 + DiffRL — Cube Stacking (Chunk-Based BPTT)
==========================================================
OPTIMIZED: MJWarp GPU render, depth channel removed
  - Batched MJWarp render (all envs per chunk, GPU)
  - Batched DINOv3 inference (all envs × cameras in one batch)
  - vmapped forward_chunk (all envs in one JIT call)
  - No depth channel — spatial softmax on proj_dim only

Usage:
    python train_cube_stacking_DinoV3_warp.py --no-viewer
    python train_cube_stacking_DinoV3_warp.py --cameras wrist
    python train_cube_stacking_DinoV3_warp.py --cameras overhead1,wrist
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
from mujoco.mjx import get_rgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from helpers.mjx_utils import patch_solver, safe_action
from helpers.obs import ObsRMS
from models.vision_backbone import DINOv3Backbone
from models.vision_policy import (
    init_vision_policy, vision_policy_forward,
    N_JOINTS, FINGER_OPEN, N_SEGMENTS,
)
from rewards.cube_stack_depth import select_reward
from envs.cube_stack_depth_env import (
    Q_LO, Q_HI, VEL_LIMITS, TARGET_QUAT,
    PROPRIO_DIM, PRE_GRASP_Z, CUBE2_HEIGHT, STACK_OFFSET,
    SEG_STEPS, N_TOTAL, SEG_BOUNDARIES, CAM_NAMES,
    discover_indices, make_task_cfg, build_proprio, create_batch,
)

patch_solver()

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--img-size", type=int, default=64)
parser.add_argument("--substeps", type=int, default=4)
parser.add_argument("--iters", type=int, default=1500)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--gamma", type=float, default=0.999)
parser.add_argument("--grad-clip", type=float, default=500.0)
parser.add_argument("--chunk-size", type=int, default=5)
parser.add_argument("--proj-dim", type=int, default=64)
parser.add_argument("--batch", type=int, default=20)
parser.add_argument("--no-viewer", action="store_true")
parser.add_argument("--xml", type=str,
                    default="assets/common/franka_emika_panda/"
                            "mjx_single_cube_depth_cam.xml")
parser.add_argument("--cameras", type=str, default="overhead1,overhead2,wrist",
                    help="Comma-separated camera selection: overhead1,overhead2,wrist")
args = parser.parse_args()

N_SUBSTEPS = args.substeps
CHUNK_SIZE = args.chunk_size
GAMMA = args.gamma
BATCH_SIZE = args.batch

# ── Camera selection ──────────────────────────────────────────────────────
cam_selection = [c.strip() for c in args.cameras.split(",")]
use_overhead1 = "overhead1" in cam_selection
use_overhead2 = "overhead2" in cam_selection
use_wrist     = "wrist" in cam_selection
ACTIVE_N_CAMERAS = len(cam_selection)

ACTIVE_CAM_NAMES = []
if use_overhead1: ACTIVE_CAM_NAMES.append(CAM_NAMES[0])
if use_overhead2: ACTIVE_CAM_NAMES.append(CAM_NAMES[1])
if use_wrist:     ACTIVE_CAM_NAMES.append('wrist_cam')

print(f"  Cameras: {cam_selection} ({ACTIVE_N_CAMERAS} channels)")

N_CHUNKS = N_TOTAL // CHUNK_SIZE
N_FEAT_SLOTS = N_CHUNKS + 1

# ═══════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════════════════
print("\n[1] MuJoCo model yükleniyor...")
mj_model = mujoco.MjModel.from_xml_path(args.xml)
mj_model.opt.iterations = 6
mj_model.opt.ls_iterations = 8
mj_data = mujoco.MjData(mj_model)

idx = discover_indices(mj_model)
cfg = make_task_cfg(idx)

key_id = mj_model.keyframe("home").id
mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
mujoco.mj_forward(mj_model, mj_data)

home_ctrl = jnp.array(mj_data.ctrl)
frame_dt = N_SUBSTEPS * mj_model.opt.timestep
mjx_model = mjx.put_model(mj_model)

home_data = mjx.put_data(mj_model, mj_data)
home_data = mjx.forward(mjx_model, home_data)

print(f"  Steps: {N_TOTAL}, Chunks: {N_CHUNKS}, Chunk size: {CHUNK_SIZE}")
print(f"  Feature slots: {N_FEAT_SLOTS}")

print("\n[2] DINOv3 backbone yükleniyor...")
backbone = DINOv3Backbone(img_size=args.img_size, device="cuda")
GRID_H = backbone.grid_h
GRID_W = backbone.grid_w

print("\n[3] MJWarp GPU renderer oluşturuluyor...")
for i in range(mj_model.ncam):
    mj_model.cam_resolution[i] = [args.img_size, args.img_size]

def _to_f32(tree):
    return jax.tree.map(
        lambda x: x.astype(jnp.float32)
        if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating)
        else x, tree)

mx_warp = _to_f32(mjx.put_model(mj_model, impl='warp'))
d_warp_template = _to_f32(mjx.put_data(mj_model, mj_data, impl='warp'))
rc = mjx.create_render_context(mj_model, nworld=BATCH_SIZE)
rc_pytree = rc.pytree()

@jax.jit
@jax.vmap
def warp_batched_forward(qpos):
    d = d_warp_template.replace(qpos=qpos)
    return mjx.forward(mx_warp, d)

@jax.jit
def warp_render(mx_w, d_w, rc_pt):
    d_w = mjx.refit_bvh(mx_w, d_w, rc_pt)
    pixels, _ = mjx.render(mx_w, d_w, rc_pt)
    return pixels

WARP_CAM_IDS = []
for name in ACTIVE_CAM_NAMES:
    cid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    WARP_CAM_IDS.append(cid)

print(f"  Active cameras: {ACTIVE_CAM_NAMES} → cam_ids: {WARP_CAM_IDS}")
print(f"  Render context: nworld={BATCH_SIZE}, {args.img_size}×{args.img_size}")

print("\n[4] Vision policy oluşturuluyor...")
rng = jax.random.PRNGKey(42)
policy_params = init_vision_policy(
    rng, n_cameras=ACTIVE_N_CAMERAS, proj_dim=args.proj_dim,
    proprio_dim=PROPRIO_DIM)

optimizer = optax.chain(
    optax.clip_by_global_norm(args.grad_clip),
    optax.adam(args.lr),
)
opt_state = optimizer.init(policy_params)
obs_rms = ObsRMS(PROPRIO_DIM)


# ═══════════════════════════════════════════════════════════════════════════
# DEBUG: Camera render + DINOv3 feature test
# ═══════════════════════════════════════════════════════════════════════════
print("\n[5] Debug camera render (MJWarp)...")
mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
mujoco.mj_forward(mj_model, mj_data)

d_debug = _to_f32(mjx.put_data(mj_model, mj_data, impl='warp'))
rc_debug = mjx.create_render_context(mj_model, nworld=1)
rc_debug_pt = rc_debug.pytree()

@jax.jit
def debug_render(mx_w, d_w, rc_pt):
    d_w = mjx.refit_bvh(mx_w, d_w, rc_pt)
    pixels, _ = mjx.render(mx_w, d_w, rc_pt)
    return pixels

debug_pixels = debug_render(mx_warp, d_debug, rc_debug_pt)

fig, axes = plt.subplots(1, ACTIVE_N_CAMERAS,
                          figsize=(6 * ACTIVE_N_CAMERAS, 5))
if ACTIVE_N_CAMERAS == 1:
    axes = [axes]
for i, cam_id in enumerate(WARP_CAM_IDS):
    rgb = np.clip(np.array(get_rgb(rc_debug_pt, cam_id, debug_pixels)[0]), 0, 1)
    axes[i].imshow(rgb)
    axes[i].set_title(f"RGB — {ACTIVE_CAM_NAMES[i]}")
    axes[i].axis("off")

fig.suptitle(f"MJWarp Debug — {args.img_size}×{args.img_size} ({args.cameras})")
fig.tight_layout()
fig.savefig("debug_vision_cameras.png", dpi=150)
plt.close(fig)
print("✓ debug_vision_cameras.png saved")

debug_rgb = [(np.clip(np.array(get_rgb(rc_debug_pt, cid, debug_pixels)[0]), 0, 1) * 255
              ).astype(np.uint8) for cid in WARP_CAM_IDS]
feats_test = backbone.extract_batch(np.stack(debug_rgb))
print(f"  DINOv3 features: {feats_test.shape}")
print(f"  Grid: {GRID_H}×{GRID_W}, dim={backbone.hidden_dim}")


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════
_RENDER_FIELDS = [
    'xpos', 'xquat', 'xmat',
    'geom_xpos', 'geom_xmat',
    'cam_xpos', 'cam_xmat',
    'light_xpos', 'light_xdir',
    'site_xpos', 'site_xmat',
]

def warp_render_to_rgb(states_batched):
    """JAX state → field kopyala (f32) → warp render. Forward yok."""
    d_render = d_warp_batched_template
    for field in _RENDER_FIELDS:
        if hasattr(states_batched, field):
            val = getattr(states_batched, field).astype(jnp.float32)
            d_render = d_render.replace(**{field: val})

    pixels = warp_render(mx_warp, d_render, rc_pytree)
    jax.block_until_ready(pixels)

    rgb_per_cam = []
    for cam_id in WARP_CAM_IDS:
        rgb = np.array(get_rgb(rc_pytree, cam_id, pixels))
        rgb_per_cam.append(rgb)
    return rgb_per_cam


def make_initial_state(seed=0):
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    rng_np = np.random.RandomState(seed)
    noise = rng_np.uniform(-0.05, 0.05, size=2)
    mj_data.qpos[9] += noise[0]
    mj_data.qpos[10] += noise[1]
    mujoco.mj_forward(mj_model, mj_data)
    d = mjx.put_data(mj_model, mj_data)
    d = mjx.forward(mjx_model, d)
    return d


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
# FORWARD CHUNK — single env (JIT)
# ═══════════════════════════════════════════════════════════════════════════
def forward_chunk_single(policy_params, vis_list,
                         state, prev_ee, obs_mean, obs_std,
                         chunk_start, active_steps):
    def step_fn(carry, step_offset):
        state, prev_ee_pos = carry
        step_idx = chunk_start + step_offset

        proprio, ee_pos, current_q = build_proprio(
            state, prev_ee_pos, frame_dt, idx)
        proprio_norm = (proprio.astype(jnp.float32) - obs_mean) / obs_std

        qd, f = vision_policy_forward(
            policy_params, vis_list,
            proprio_norm, step_idx, SEG_BOUNDARIES)
        safe_qd = safe_action(qd, current_q, Q_LO, Q_HI)

        ctrl = home_ctrl.at[:N_JOINTS].set(safe_qd)
        ctrl = ctrl.at[7].set(f)

        def do_sub(s):
            def sub(s, _):
                s = s.replace(ctrl=ctrl)
                s = mjx.step(mjx_model, s)
                return s, s.qpos
            return jax.lax.scan(sub, s, None, length=N_SUBSTEPS)
        def skip_sub(s):
            return s, jnp.broadcast_to(s.qpos, (N_SUBSTEPS,) + s.qpos.shape)

        state, sub_qpos = jax.lax.cond(
            step_idx < active_steps, do_sub, skip_sub, state)

        return (state, ee_pos), sub_qpos

    (state, ee), all_qpos = jax.lax.scan(
        step_fn, (state, prev_ee), jnp.arange(CHUNK_SIZE))
    return state, ee, all_qpos


# ═══════════════════════════════════════════════════════════════════════════
# BATCHED FORWARD CHUNK — all envs in one call (vmap)
# ═══════════════════════════════════════════════════════════════════════════
def forward_chunk_batched(policy_params, vis_batch,
                          states, prev_ees, obs_mean, obs_std,
                          chunk_start, active_steps):
    def single(vis_arr, state, prev_ee):
        vis_list = [vis_arr[c] for c in range(ACTIVE_N_CAMERAS)]
        return forward_chunk_single(
            policy_params, vis_list,
            state, prev_ee, obs_mean, obs_std,
            chunk_start, active_steps)

    return jax.vmap(single)(vis_batch, states, prev_ees)

forward_chunk_batched_jit = jax.jit(forward_chunk_batched,
                                     static_argnums=())


# ═══════════════════════════════════════════════════════════════════════════
# BATCHED COLLECT FEATURES — chunk-by-chunk across all envs
# ═══════════════════════════════════════════════════════════════════════════
def collect_features_batched(policy_params, batched_states,
                              obs_mean, obs_std, active_steps,
                              collect_qpos=False):
    """Tüm env'leri chunk-by-chunk paralel işle — MJWarp GPU render."""
    import torch

    n_active_chunks = int(np.ceil(float(active_steps) / CHUNK_SIZE)) + 1
    n_active_chunks = min(n_active_chunks, N_CHUNKS + 1)

    all_vision = np.zeros((BATCH_SIZE, N_FEAT_SLOTS, ACTIVE_N_CAMERAS,
                            GRID_H, GRID_W, 768), dtype=np.float32)
    all_qpos_env0 = []

    states = batched_states
    prev_ees = batched_states.site_xpos[:, idx['ee_site']]

    t_sync = 0; t_render = 0; t_dino = 0; t_forward = 0

    for chunk_idx in range(N_CHUNKS + 1):
        if chunk_idx >= n_active_chunks:
            break

        # ── 1. JAX sync ──
        t0 = time.time()
        jax.block_until_ready(states.qpos)
        t_sync += time.time() - t0

        # ── 2. MJWarp GPU render ──
        t0 = time.time()
        rgb_per_cam = warp_render_to_rgb(states)
        t_render += time.time() - t0

        # ── 3. Batch DINOv3 ──
        t0 = time.time()
        flat_rgb = []
        for env_i in range(BATCH_SIZE):
            for cam_i in range(ACTIVE_N_CAMERAS):
                img = rgb_per_cam[cam_i][env_i]
                if img.dtype in [np.float32, np.float64]:
                    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                flat_rgb.append(img)
        flat_rgb = np.stack(flat_rgb, axis=0)

        feats_flat = backbone.extract_batch(flat_rgb)
        torch.cuda.synchronize()

        feats_all = feats_flat.reshape(BATCH_SIZE, ACTIVE_N_CAMERAS,
                                        GRID_H, GRID_W, 768)
        t_dino += time.time() - t0

        # Store
        all_vision[:, chunk_idx] = feats_all

        # ── 4. Batched forward chunk (vmap, MJX-JAX physics) ──
        if chunk_idx < N_CHUNKS and chunk_idx < n_active_chunks - 1:
            t0 = time.time()
            vis_batch = jnp.array(feats_all)
            chunk_start = jnp.int32(chunk_idx * CHUNK_SIZE)

            states, prev_ees, chunk_qpos = forward_chunk_batched_jit(
                policy_params, vis_batch,
                states, prev_ees, obs_mean, obs_std,
                chunk_start, active_steps)

            if collect_qpos:
                all_qpos_env0.append(np.array(chunk_qpos[0]))
            t_forward += time.time() - t0

    return (jnp.array(all_vision),
            all_qpos_env0, t_sync, t_render, t_dino, t_forward)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: EPISODE LOSS
# ═══════════════════════════════════════════════════════════════════════════
def episode_loss(policy_params, vision_array,
                 initial_state, obs_mean, obs_std, active_steps):
    @jax.checkpoint
    def frame_step(carry, step_idx):
        state, prev_ee, total_rew, gamma_acc = carry

        is_boundary = jnp.zeros((), dtype=bool)
        for b in SEG_BOUNDARIES:
            is_boundary = is_boundary | (step_idx == b)
        state = jax.lax.cond(is_boundary,
            lambda s: jax.lax.stop_gradient(s), lambda s: s, state)
        prev_ee = jax.lax.cond(is_boundary,
            lambda p: jax.lax.stop_gradient(p), lambda p: p, prev_ee)
        gamma_acc = jnp.where(is_boundary, jnp.float64(1.0), gamma_acc)

        chunk_idx = step_idx // CHUNK_SIZE
        vis_list = [vision_array[chunk_idx, c] for c in range(ACTIVE_N_CAMERAS)]

        proprio, ee_pos, current_q = build_proprio(
            state, prev_ee, frame_dt, idx)
        proprio_norm = (proprio.astype(jnp.float32) - obs_mean) / obs_std

        qd, f = vision_policy_forward(
            policy_params, vis_list,
            proprio_norm, step_idx, SEG_BOUNDARIES)
        safe_qd = safe_action(qd, current_q, Q_LO, Q_HI)

        ctrl = home_ctrl.at[:N_JOINTS].set(safe_qd)
        ctrl = ctrl.at[7].set(f)

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

        rew = select_reward(state, step_idx, cfg, SEG_BOUNDARIES)
        rew = jnp.where(step_idx < active_steps, rew, 0.0)

        total_rew = total_rew + gamma_acc * rew
        gamma_acc = gamma_acc * GAMMA

        return (state, ee_pos, total_rew, gamma_acc), proprio

    init = (initial_state, initial_state.site_xpos[idx['ee_site']],
            jnp.float64(0.0), jnp.float64(1.0))
    (final_state, _, total_rew, _), all_proprio = jax.lax.scan(
        frame_step, init, jnp.arange(N_TOTAL))

    return -total_rew, (all_proprio, final_state)


def batch_episode_loss(policy_params, all_vision,
                       all_states, obs_mean, obs_std, active_steps):
    losses, aux = jax.vmap(
        episode_loss, in_axes=(None, 0, 0, None, None, None)
    )(policy_params, all_vision, all_states,
      obs_mean, obs_std, active_steps)
    return jnp.mean(losses), aux


# ═══════════════════════════════════════════════════════════════════════════
# JIT COMPILE
# ═══════════════════════════════════════════════════════════════════════════
print("\n[6] JIT compiling...")
t0 = time.time()
value_and_grad_fn = jax.jit(jax.value_and_grad(batch_episode_loss, has_aux=True))

dummy_state = make_initial_state(seed=9999)
dummy_states = jax.tree.map(
    lambda x: jnp.stack([x] * BATCH_SIZE), dummy_state)
dummy_vision = jnp.zeros((BATCH_SIZE, N_FEAT_SLOTS, ACTIVE_N_CAMERAS,
                           GRID_H, GRID_W, 768), dtype=jnp.float32)
obs_mean, obs_std = obs_rms.get_jnp()

PATIENCE = 4
PHASE_BOUNDS = []
acc = 0
for s in SEG_STEPS:
    acc += s
    PHASE_BOUNDS.append(acc)

phase = 0
patience_cnt = 0
active_steps = jnp.int32(PHASE_BOUNDS[min(1, len(PHASE_BOUNDS) - 1)])

_ = rebatch_jit(jax.random.PRNGKey(9999), BATCH_SIZE)
print("✓ rebatch_jit compiled")

# forward_chunk_batched warmup
dummy_batch = rebatch_jit(jax.random.PRNGKey(8888), BATCH_SIZE)
dummy_prev_ees = dummy_batch.site_xpos[:, idx['ee_site']]
dummy_vis_b = jnp.zeros((BATCH_SIZE, ACTIVE_N_CAMERAS, GRID_H, GRID_W, 768),
                         dtype=jnp.float32)
_ = forward_chunk_batched_jit(
    policy_params, dummy_vis_b,
    dummy_batch, dummy_prev_ees,
    obs_mean, obs_std, jnp.int32(0), active_steps)
print("✓ forward_chunk_batched_jit compiled")

# Warp render warmup — template oluştur (bir kez)
print("  Warming up warp render...")
dummy_qpos_f32 = dummy_batch.qpos.astype(jnp.float32)
d_warp_batched_template = warp_batched_forward(dummy_qpos_f32)
jax.block_until_ready(d_warp_batched_template.qpos)
warmup_pixels = warp_render(mx_warp, d_warp_batched_template, rc_pytree)
jax.block_until_ready(warmup_pixels)
print("✓ warp _impl template cached + render compiled")

CRITERIA = [
    ("dist", 0.05),   # 0 Reach
    ("dist", 0.05),   # 1 Descend
    ("dist", 0.04),   # 2 Grasp
    ("dist", 0.10),   # 3 Move
    ("always", 0),    # 4 Release
]

seg_labels = ['Reach', 'Descend', 'Grasp', 'Move', 'Release']
seg_ends = []
acc = 0
for s in SEG_STEPS:
    acc += s
    seg_ends.append(acc - 1)

(loss_val, _), grad_val = value_and_grad_fn(
    policy_params, dummy_vision, dummy_states,
    obs_mean, obs_std, active_steps)
loss_val.block_until_ready()
print(f"  JIT: {time.time()-t0:.1f}s, loss: {float(loss_val):.4f}")
print(f"  Grad norm: {float(optax.global_norm(grad_val)):.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# FORWARD ROLLOUT (metrics)
# ═══════════════════════════════════════════════════════════════════════════
def forward_for_metrics(policy_params, vision_array,
                        initial_state, obs_mean, obs_std, active_steps):
    def frame_step(carry, step_idx):
        state, prev_ee = carry

        chunk_idx = step_idx // CHUNK_SIZE
        vis_list = [vision_array[chunk_idx, c] for c in range(ACTIVE_N_CAMERAS)]

        proprio, ee_pos, current_q = build_proprio(
            state, prev_ee, frame_dt, idx)
        proprio_norm = (proprio.astype(jnp.float32) - obs_mean) / obs_std

        qd, f = vision_policy_forward(
            policy_params, vis_list,
            proprio_norm, step_idx, SEG_BOUNDARIES)
        safe_qd = safe_action(qd, current_q, Q_LO, Q_HI)

        ctrl = home_ctrl.at[:N_JOINTS].set(safe_qd)
        ctrl = ctrl.at[7].set(f)

        def do_sub(s):
            def sub(s, _):
                s = s.replace(ctrl=ctrl)
                s = mjx.step(mjx_model, s)
                return s, s.qpos
            return jax.lax.scan(sub, s, None, length=N_SUBSTEPS)
        def skip_sub(s):
            return s, jnp.broadcast_to(s.qpos, (N_SUBSTEPS,) + s.qpos.shape)

        state, step_qpos = jax.lax.cond(
            step_idx < active_steps, do_sub, skip_sub, state)

        return (state, ee_pos), (state, step_qpos, f)

    init = (initial_state, initial_state.site_xpos[idx['ee_site']])
    (_, _), (all_states, all_qpos, all_fingers) = jax.lax.scan(
        frame_step, init, jnp.arange(N_TOTAL))
    return all_states, all_qpos, all_fingers

jit_forward_metrics = jax.jit(forward_for_metrics)
print("  Forward metrics JIT compiled")


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


def render_trajectory(all_qpos_chunks, skip=2):
    if not USE_VIEWER or not viewer_handle.is_running():
        return
    sim_dt = mj_model.opt.timestep
    for chunk_qpos in all_qpos_chunks:
        for step in range(chunk_qpos.shape[0]):
            for sub in range(0, chunk_qpos.shape[1], skip):
                mj_data_render.qpos[:] = chunk_qpos[step, sub]
                mujoco.mj_forward(mj_model_render, mj_data_render)
                viewer_handle.sync()
                time.sleep(sim_dt * skip)


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("VISION DINOv3 + DiffRL — Cube Stacking (MJWarp, no depth)")
print(f"  Cameras: {args.cameras} ({ACTIVE_N_CAMERAS} cameras)")
print(f"  Image: {args.img_size}×{args.img_size}")
print(f"  Grid: {GRID_H}×{GRID_W} patches")
print(f"  Batch: {BATCH_SIZE} envs")
print(f"  Chunks: {N_CHUNKS}+1, size={CHUNK_SIZE}")
print(f"  Segments: {'+'.join(map(str, SEG_STEPS))} = {N_TOTAL} steps")
print(f"  LR: {args.lr}, Gamma: {GAMMA}, Clip: {args.grad_clip}")
print("=" * 65 + "\n")

loss_history = []
grad_norm_history = []
seg_dist_history = [[] for _ in range(5)]

train_start = time.time()

for iteration in range(args.iters):
    if USE_VIEWER and not viewer_handle.is_running():
        print("\nViewer kapatıldı.")
        break

    obs_mean, obs_std = obs_rms.get_jnp()

    # ── PHASE 1: Batched collect ──
    t_phase1 = time.time()

    batched_states = rebatch_jit(jax.random.PRNGKey(iteration), BATCH_SIZE)
    jax.block_until_ready(batched_states.qpos)

    (batched_vision,
     all_qpos_env0, t_s, t_r, t_d, t_f) = collect_features_batched(
        policy_params, batched_states, obs_mean, obs_std,
        active_steps, collect_qpos=(iteration % 5 == 0 or iteration < 3))

    t_phase1 = time.time() - t_phase1

    # ── PHASE 2: BPTT ──
    t_bptt = time.time()
    (loss_val, aux), grads = value_and_grad_fn(
        policy_params, batched_vision,
        batched_states, obs_mean, obs_std, active_steps)
    t_bptt = time.time() - t_bptt

    loss_f = float(loss_val)
    all_proprio_batch = aux[0]
    obs_rms.update(np.array(all_proprio_batch).reshape(-1, PROPRIO_DIM))

    if np.isnan(loss_f):
        print(f"\n⚠ NaN at iter {iteration}!")
        break

    updates, opt_state = optimizer.update(grads, opt_state, policy_params)
    policy_params = optax.apply_updates(policy_params, updates)

    loss_history.append(-loss_f)
    grad_norm = float(optax.global_norm(grads))
    grad_norm_history.append(grad_norm)

    # ── Metrics + Print ──
    if iteration % 5 == 0 or iteration < 3:
        all_states, _, all_fng = jit_forward_metrics(
            policy_params, batched_vision[0],
            jax.tree.map(lambda x: x[0], batched_states),
            obs_mean, obs_std, active_steps)

        s_ee = np.array(all_states.site_xpos[:, idx['ee_site']])
        s_c1 = np.array(all_states.xpos[:, idx['cube1']])
        s_c2 = np.array(all_states.xpos[:, idx['cube2']])
        s_fng = np.array(all_fng)

        metrics = []
        for si in range(5):
            se = seg_ends[si]
            if si == 0:
                target = s_c1[se] + np.array([0, 0, PRE_GRASP_Z])
                metrics.append(np.linalg.norm(s_ee[se] - target))
            elif si == 1:
                target = s_c1[se] + np.array([0, 0, -0.01])
                metrics.append(np.linalg.norm(s_ee[se] - target))
            elif si == 2:
                metrics.append(np.linalg.norm(s_ee[se] - s_c1[se]))
            elif si == 3:
                stack_t = s_c2[se] + np.array([0, 0,
                                                CUBE2_HEIGHT + STACK_OFFSET])
                metrics.append(np.linalg.norm(s_c1[se] - stack_t))
            else:
                stack_t = s_c2[se] + np.array([0, 0,
                                                CUBE2_HEIGHT + STACK_OFFSET])
                xy_err = np.linalg.norm(s_c1[se][:2] - s_c2[se][:2])
                z_err = abs(s_c1[se][2] - stack_t[2])
                metrics.append(xy_err + z_err)

        for si in range(5):
            seg_dist_history[si].append(metrics[si])

        elapsed = time.time() - train_start
        print(f"\nIter {iteration:4d}:  rew={-loss_f:.3f}  "
              f"grad={grad_norm:.1f}  phase={phase}  "
              f"active={int(active_steps)}  "
              f"[phase1={t_phase1:.1f}s bptt={t_bptt:.1f}s "
              f"sync={t_s:.1f} rend={t_r:.1f} dino={t_d:.1f} fwd={t_f:.1f}]")

        for si in range(5):
            marker = "→" if si == phase else " "
            print(f"  {marker} {seg_labels[si]:8s}: {metrics[si]:.4f}  "
                  f"fng={s_fng[seg_ends[si]]:.4f}")

        if all_qpos_env0:
            render_trajectory(all_qpos_env0, skip=2)

        # Progressive advancement
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

    if iteration % 50 == 0:
        plt.close('all')
        gc.collect()

train_time = time.time() - train_start
print(f"\nTraining done: {train_time:.1f}s ({len(loss_history)} iters)")

# ═══════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════
with open("vision_policy_params.pkl", "wb") as f:
    pickle.dump(jax.tree.map(lambda x: np.array(x), policy_params), f)
print("Saved: vision_policy_params.pkl")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════════════════
n_plots = 2 + len(seg_labels)
fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots))

axes[0].plot(loss_history, linewidth=2, color='green')
axes[0].set_ylabel("Reward")
axes[0].set_title("Episode Reward")
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
fig.suptitle(f"Vision DINOv3 (no depth) — {args.img_size}×{args.img_size}, "
             f"cams={args.cameras}, {len(loss_history)} iters")
fig.tight_layout()
fig.savefig("vision_training_results.png", dpi=150)
plt.close(fig)
print(f"Plot saved: vision_training_results.png")

if USE_VIEWER and viewer_handle.is_running():
    print("\nViewer açık — ESC ile kapat.")
    try:
        while viewer_handle.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nÇıkılıyor.")