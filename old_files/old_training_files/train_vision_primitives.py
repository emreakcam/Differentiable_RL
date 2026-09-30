"""
Vision-based Primitive Library — MJWarp render + DINOv3 (wrist cam) + Shared Trunk
+ Per-Primitive Heads, trained jointly across 3 tasks in one combined backward pass.

Combines:
  - train_cube_stacking_DinoV3_warp.py: MJWarp batched GPU rendering, batched DINOv3
    inference, chunk-based BPTT (rendering is non-differentiable, cached per chunk
    and held fixed while CHUNK_SIZE differentiable physics/policy steps run under it).
  - train_primitives.py: one MLP per primitive shared across all tasks, per-primitive
    optimizers + grad clipping, multi-task combined_loss (single backward pass across
    cube/peg/container), per-task progressive curriculum.

Policy input is vision (wrist-cam DINOv3 features, projected + spatial-softmaxed) +
proprio only — no privileged target_pos / obj_rel. Reward shaping still uses
privileged sim state (as in train_primitives.py) — only the policy is vision-only.

Physics stepping and MJWarp rendering cannot be batched across tasks (different
MjModel structure per task → different qpos/qvel shapes and render-context geometry),
so each task keeps its own model / render context / chunked rollout. DINOv3 inference
*is* task-agnostic (same wrist image shape for all 3 tasks), so per chunk step the
rendered wrist images of every still-active task are concatenated into one batched
DINOv3 call and split back out afterward.

Usage:
    python train_vision_primitives.py --no-viewer
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
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from helpers.mjx_utils import patch_solver
from helpers.obs import ObsRMS
from models.vision_backbone import DINOv3Backbone
from models.vision_primitives import init_vision_primitives, vision_primitive_forward
from models.primitives import (
    PRIM_NAMES, REACH, DESCEND, GRASP, MOVE, INSERT_ALIGN, INSERT_PUSH, RELEASE,
    PROPRIO_DIM, DOWN_QUAT, N_JOINTS, FINGER_OPEN,
    build_proprio, safe_action,
    reward_reach, reward_descend, reward_grasp,
    reward_move_generic, reward_release_generic,
    reward_insert_align, reward_insert_push,
)

patch_solver()

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--img-size", type=int, default=64)
parser.add_argument("--substeps", type=int, default=4)
parser.add_argument("--iters", type=int, default=2000)
parser.add_argument("--lr", type=float, default=2e-4)  # shared projection+trunk
parser.add_argument("--lr-reach", type=float, default=2e-4)
parser.add_argument("--lr-descend", type=float, default=2e-4)
parser.add_argument("--lr-grasp", type=float, default=2e-4)
parser.add_argument("--lr-move", type=float, default=2e-4)
parser.add_argument("--lr-insert-align", type=float, default=2e-5)
parser.add_argument("--lr-insert-push", type=float, default=2e-5)
parser.add_argument("--lr-release", type=float, default=2e-4)
parser.add_argument("--gamma", type=float, default=0.999)
parser.add_argument("--grad-clip", type=float, default=1000.0)
parser.add_argument("--chunk-size", type=int, default=5)
parser.add_argument("--proj-dim", type=int, default=64)
parser.add_argument("--batch", type=int, default=20)
parser.add_argument("--solver-iters", type=int, default=6)
parser.add_argument("--no-viewer", action="store_true")
parser.add_argument("--cube-xml", type=str,
                    default="assets/common/franka_emika_panda/mjx_single_cube_depth_cam.xml")
parser.add_argument("--peg-xml", type=str,
                    default="assets/common/franka_emika_panda/mjx_peg_slot.xml")
parser.add_argument("--cont-xml", type=str,
                    default="assets/common/franka_emika_panda/mjx_container.xml")
args = parser.parse_args()

N_SUBSTEPS = args.substeps
CHUNK_SIZE = args.chunk_size
GAMMA = args.gamma
BATCH_SIZE = args.batch
TASK_NAMES = ['cube', 'peg', 'cont']

PRIM_LRS = {name: getattr(args, f'lr_{name}') for name in PRIM_NAMES}

# Task-agnostic geometry constants (shared across all 3 tasks, same as train_primitives.py)
PRE_GRASP_Z  = 0.075
CUBE2_HEIGHT = 0.05
STACK_OFFSET = 0.01
LIFT_Z_TARGET = 0.18
PLACE_Z_OFFSET = 0.08


def _to_f32(tree):
    return jax.tree.map(
        lambda x: x.astype(jnp.float32)
        if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating)
        else x, tree)


def _select_by_step(values, step_idx, boundaries):
    """Nested where — pick values[i] for step_idx < boundaries[i]."""
    result = values[-1]
    for i in range(len(boundaries) - 1, -1, -1):
        result = jax.tree.map(
            lambda r, v: jnp.where(step_idx < boundaries[i], v, r),
            result, values[i])
    return result


_RENDER_FIELDS = [
    'xpos', 'xquat', 'xmat',
    'geom_xpos', 'geom_xmat',
    'cam_xpos', 'cam_xmat',
    'light_xpos', 'light_xdir',
    'site_xpos', 'site_xmat',
]


# ═══════════════════════════════════════════════════════════════════════════
# INDEX DISCOVERY (per task)
# ═══════════════════════════════════════════════════════════════════════════
def cube_idx_fn(mj):
    return {
        'ee_site':   0,
        'hand_body': mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "hand"),
        'cube1':     mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "box"),
        'cube2':     mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "box2"),
    }


def peg_idx_fn(mj):
    return {
        'ee_site':     mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "gripper"),
        'hand_body':   mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "hand"),
        'peg_body':    mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "peg"),
        'peg_bottom':  mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "peg_bottom"),
        'peg_grip':    mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "peg_grip"),
        'slot_entry':  mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "slot_entry"),
        'slot_target': mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "slot_target"),
    }


def cont_idx_fn(mj):
    idx = {
        'ee_site':    mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "gripper"),
        'hand_body':  mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "hand"),
        'ball_body':  mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "ball"),
        'cube_body':  mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "cube"),
        'prism_body': mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "prism"),
        'container':  mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "container_inside"),
    }
    idx['obj_idxs'] = [idx['ball_body'], idx['cube_body'], idx['prism_body']]
    return idx


CUBE_SEG_STEPS = [20, 12, 8, 16, 8]           # Reach, Descend, Grasp, Move, Release
CUBE_PRIM_SEQ  = [REACH, DESCEND, GRASP, MOVE, RELEASE]

PEG_SEG_STEPS = [20, 12, 8, 12, 16, 12, 8]
PEG_PRIM_SEQ  = [REACH, DESCEND, GRASP, MOVE, INSERT_ALIGN, INSERT_PUSH, RELEASE]

CONT_STEPS_PER_OBJ = [20, 12, 8, 16, 8]
CONT_SEG_STEPS = CONT_STEPS_PER_OBJ * 3
CONT_PRIM_SEQ  = [REACH, DESCEND, GRASP, MOVE, RELEASE] * 3

TASK_SPECS = {
    'cube': dict(xml=args.cube_xml, idx_fn=cube_idx_fn,
                 seg_steps=CUBE_SEG_STEPS, prim_seq=CUBE_PRIM_SEQ, rand_offsets=[9]),
    'peg':  dict(xml=args.peg_xml, idx_fn=peg_idx_fn,
                 seg_steps=PEG_SEG_STEPS, prim_seq=PEG_PRIM_SEQ, rand_offsets=[9]),
    'cont': dict(xml=args.cont_xml, idx_fn=cont_idx_fn,
                 seg_steps=CONT_SEG_STEPS, prim_seq=CONT_PRIM_SEQ, rand_offsets=[9, 16, 23]),
}


# ═══════════════════════════════════════════════════════════════════════════
# DINOv3 BACKBONE (shared across tasks)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[1] DINOv3 backbone yükleniyor...")
backbone = DINOv3Backbone(img_size=args.img_size, device="cuda")
GRID_H = backbone.grid_h
GRID_W = backbone.grid_w


# ═══════════════════════════════════════════════════════════════════════════
# PER-TASK SETUP — model, indices, MJWarp render context, rebatch fn
# ═══════════════════════════════════════════════════════════════════════════
def setup_task(name, spec):
    xml_path = spec['xml']
    print(f"\n[TASK:{name}] Loading {xml_path} ...")
    mj = mujoco.MjModel.from_xml_path(xml_path)
    mj.opt.iterations = args.solver_iters
    mj.opt.ls_iterations = 8
    md = mujoco.MjData(mj)

    idx = spec['idx_fn'](mj)

    key_id = mj.keyframe("home").id
    mujoco.mj_resetDataKeyframe(mj, md, key_id)
    mujoco.mj_forward(mj, md)
    home_ctrl = jnp.array(md.ctrl)
    frame_dt = N_SUBSTEPS * mj.opt.timestep
    mjx_model = mjx.put_model(mj)

    home_data = mjx.put_data(mj, md)
    home_data = mjx.forward(mjx_model, home_data)

    seg_steps = spec['seg_steps']
    boundaries = []
    acc = 0
    for s in seg_steps[:-1]:
        acc += s
        boundaries.append(acc)
    n_total = sum(seg_steps)
    phase_bounds = boundaries + [n_total]

    wrist_id = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    mj.cam_resolution[wrist_id] = [args.img_size, args.img_size]

    mx_warp = _to_f32(mjx.put_model(mj, impl='warp'))
    d_warp_template = _to_f32(mjx.put_data(mj, md, impl='warp'))
    rc = mjx.create_render_context(mj, nworld=BATCH_SIZE)
    rc_pt = rc.pytree()
    # `rc` must be kept alive for as long as `rc_pt` is used: RenderContext.__del__
    # deregisters its GPU buffers from mjx's global registry when GC'd, and rc_pt
    # only holds an int key into that registry — so it must be stashed on `task`
    # below, not left as a local that gets collected when setup_task() returns.

    @jax.jit
    @jax.vmap
    def warp_batched_forward(qpos):
        d = d_warp_template.replace(qpos=qpos)
        return mjx.forward(mx_warp, d)

    @jax.jit
    def warp_render(mx_w, d_w, rc_pt_):
        d_w = mjx.refit_bvh(mx_w, d_w, rc_pt_)
        pixels, _ = mjx.render(mx_w, d_w, rc_pt_)
        return pixels

    rand_offsets = spec['rand_offsets']

    def rebatch_jax(rng_key, batch_size):
        keys = jax.random.split(rng_key, batch_size)

        def randomize_one(k):
            new_qpos = home_data.qpos
            sub_keys = jax.random.split(k, len(rand_offsets))
            for off, sk in zip(rand_offsets, sub_keys):
                n = jax.random.uniform(sk, (2,), minval=-0.05, maxval=0.05)
                new_qpos = new_qpos.at[off].add(n[0])
                new_qpos = new_qpos.at[off + 1].add(n[1])
            d = home_data.replace(qpos=new_qpos)
            d = mjx.forward(mjx_model, d)
            return d
        return jax.vmap(randomize_one)(keys)
    rebatch_jit = jax.jit(rebatch_jax, static_argnums=(1,))

    n_chunks = n_total // CHUNK_SIZE

    task = dict(
        name=name, mj=mj, md=md, idx=idx, home_ctrl=home_ctrl, frame_dt=frame_dt,
        mjx_model=mjx_model, home_data=home_data, seg_steps=seg_steps,
        boundaries=boundaries, n_total=n_total, phase_bounds=phase_bounds,
        prim_seq=spec['prim_seq'], wrist_id=wrist_id,
        mx_warp=mx_warp, rc=rc, rc_pt=rc_pt,
        warp_batched_forward=warp_batched_forward, warp_render=warp_render,
        rebatch_jit=rebatch_jit, n_chunks=n_chunks, n_feat_slots=n_chunks + 1,
    )
    print(f"  {name}: {n_total} steps, {n_chunks}+1 chunks, segs={seg_steps}")
    return task


tasks = {name: setup_task(name, TASK_SPECS[name]) for name in TASK_NAMES}


# ═══════════════════════════════════════════════════════════════════════════
# POLICY + OPTIMIZERS
# ═══════════════════════════════════════════════════════════════════════════
print("\n[2] Vision-primitive policy oluşturuluyor...")
rng = jax.random.PRNGKey(42)
policy_params = init_vision_primitives(rng, proj_dim=args.proj_dim, proprio_dim=PROPRIO_DIM)

shared_optimizer = optax.chain(
    optax.clip_by_global_norm(args.grad_clip),
    optax.adam(args.lr),
)
shared_opt_state = shared_optimizer.init(
    {'projection': policy_params['projection'], 'trunk': policy_params['trunk']})

optimizers = {}
opt_states = {}
for name in PRIM_NAMES:
    optimizers[name] = optax.adam(PRIM_LRS[name])
    opt_states[name] = optimizers[name].init(policy_params[name])

obs_rms = ObsRMS(PROPRIO_DIM)


# ═══════════════════════════════════════════════════════════════════════════
# WARP RENDER → RGB (wrist cam only)
# ═══════════════════════════════════════════════════════════════════════════
def warp_render_to_rgb(task, states_batched):
    d_render = task['d_warp_batched_template']
    for field in _RENDER_FIELDS:
        if hasattr(states_batched, field):
            val = getattr(states_batched, field).astype(jnp.float32)
            d_render = d_render.replace(**{field: val})
    pixels = task['warp_render'](task['mx_warp'], d_render, task['rc_pt'])
    jax.block_until_ready(pixels)
    return np.array(get_rgb(task['rc_pt'], task['wrist_id'], pixels))  # (B, H, W, 3)


# ═══════════════════════════════════════════════════════════════════════════
# FORWARD CHUNK (physics advance, used during phase-1 feature collection)
# ═══════════════════════════════════════════════════════════════════════════
def make_forward_chunk_fns(task):
    mjx_model = task['mjx_model']
    home_ctrl = task['home_ctrl']
    frame_dt = task['frame_dt']
    ee_i, hand_i = task['idx']['ee_site'], task['idx']['hand_body']
    boundaries = task['boundaries']
    prim_seq_arr = jnp.array(task['prim_seq'])
    n_segs = len(task['prim_seq'])

    def forward_chunk_single(policy_params, feat_map, state, prev_ee,
                             obs_mean, obs_std, chunk_start, active_steps):
        def step_fn(carry, step_offset):
            state, prev_ee_pos = carry
            step_idx = chunk_start + step_offset

            proprio, ee_pos, current_q = build_proprio(
                state, prev_ee_pos, frame_dt, DOWN_QUAT, ee_i, hand_i)
            proprio_norm = (proprio.astype(jnp.float32) - obs_mean) / obs_std

            prim_idx = _select_by_step(
                [prim_seq_arr[i] for i in range(n_segs)], step_idx, boundaries)

            qd, f = vision_primitive_forward(policy_params, feat_map, proprio_norm, prim_idx)
            safe_qd = safe_action(qd, current_q)

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

    def forward_chunk_batched(policy_params, feat_batch, states, prev_ees,
                              obs_mean, obs_std, chunk_start, active_steps):
        def single(feat_map, state, prev_ee):
            return forward_chunk_single(
                policy_params, feat_map, state, prev_ee,
                obs_mean, obs_std, chunk_start, active_steps)
        return jax.vmap(single)(feat_batch, states, prev_ees)

    return jax.jit(forward_chunk_batched)


for name in TASK_NAMES:
    tasks[name]['forward_chunk_batched_jit'] = make_forward_chunk_fns(tasks[name])


# ═══════════════════════════════════════════════════════════════════════════
# WARMUP — rebatch + warp render template per task
# ═══════════════════════════════════════════════════════════════════════════
print("\n[3] Warp render templates ısıtılıyor...")
for name in TASK_NAMES:
    t = tasks[name]
    _ = t['rebatch_jit'](jax.random.PRNGKey(9999), BATCH_SIZE)
    dummy_batch = t['rebatch_jit'](jax.random.PRNGKey(8888), BATCH_SIZE)
    dummy_qpos_f32 = dummy_batch.qpos.astype(jnp.float32)
    d_warp_batched_template = t['warp_batched_forward'](dummy_qpos_f32)
    jax.block_until_ready(d_warp_batched_template.qpos)
    t['d_warp_batched_template'] = d_warp_batched_template
    warmup_pixels = t['warp_render'](t['mx_warp'], d_warp_batched_template, t['rc_pt'])
    jax.block_until_ready(warmup_pixels)
    print(f"  ✓ {name} warp template + render ready")


# ═══════════════════════════════════════════════════════════════════════════
# DEBUG: wrist render + DINOv3 feature sanity check
# ═══════════════════════════════════════════════════════════════════════════
print("\n[4] Debug wrist render (MJWarp, all 3 tasks)...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, name in enumerate(TASK_NAMES):
    t = tasks[name]
    debug_batch = t['rebatch_jit'](jax.random.PRNGKey(0), BATCH_SIZE)
    rgb = warp_render_to_rgb(t, debug_batch)
    axes[i].imshow(np.clip(rgb[0], 0, 1))
    axes[i].set_title(f"{name} — wrist_cam")
    axes[i].axis("off")
fig.suptitle(f"MJWarp Debug — {args.img_size}×{args.img_size} (wrist only)")
fig.tight_layout()
fig.savefig("debug_vision_primitives_cameras.png", dpi=150)
plt.close(fig)
print("✓ debug_vision_primitives_cameras.png saved")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 (fused across tasks): render → concat → single DINOv3 call → split
# ═══════════════════════════════════════════════════════════════════════════
def collect_all_features(policy_params, states_dict, obs_mean, obs_std,
                         active_dict, collect_qpos_tasks=()):
    n_active_chunks = {}
    for name in TASK_NAMES:
        t = tasks[name]
        nac = int(np.ceil(float(int(active_dict[name])) / CHUNK_SIZE)) + 1
        n_active_chunks[name] = min(nac, t['n_chunks'] + 1)
    max_chunks = max(n_active_chunks.values())

    vision_arrays = {
        name: np.zeros((BATCH_SIZE, tasks[name]['n_feat_slots'], GRID_H, GRID_W, 768),
                        dtype=np.float32)
        for name in TASK_NAMES
    }
    all_qpos = {name: [] for name in TASK_NAMES}

    cur_states = dict(states_dict)
    prev_ees = {name: states_dict[name].site_xpos[:, tasks[name]['idx']['ee_site']]
                for name in TASK_NAMES}

    t_render = t_dino = t_forward = 0.0

    for chunk_idx in range(max_chunks):
        active = [name for name in TASK_NAMES if chunk_idx < n_active_chunks[name]]
        if not active:
            break

        t0 = time.time()
        jax.block_until_ready(cur_states[active[0]].qpos)
        rgb_per_task = {name: warp_render_to_rgb(tasks[name], cur_states[name])
                         for name in active}
        t_render += time.time() - t0

        t0 = time.time()
        flat = []
        for name in active:
            img = rgb_per_task[name]
            if img.dtype in (np.float32, np.float64):
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            flat.append(img)
        flat_rgb = np.concatenate(flat, axis=0)
        feats_flat = backbone.extract_batch(flat_rgb)
        torch.cuda.synchronize()
        t_dino += time.time() - t0

        t0 = time.time()
        offset = 0
        for name in active:
            feats = feats_flat[offset:offset + BATCH_SIZE].reshape(
                BATCH_SIZE, GRID_H, GRID_W, 768)
            offset += BATCH_SIZE
            vision_arrays[name][:, chunk_idx] = feats

            t = tasks[name]
            if chunk_idx < t['n_chunks'] and chunk_idx < n_active_chunks[name] - 1:
                vis_b = jnp.array(feats)
                chunk_start = jnp.int32(chunk_idx * CHUNK_SIZE)
                new_state, new_prev_ee, chunk_qpos = t['forward_chunk_batched_jit'](
                    policy_params, vis_b, cur_states[name], prev_ees[name],
                    obs_mean, obs_std, chunk_start, jnp.int32(active_dict[name]))
                cur_states[name] = new_state
                prev_ees[name] = new_prev_ee
                if name in collect_qpos_tasks:
                    all_qpos[name].append(np.array(chunk_qpos[0]))
        t_forward += time.time() - t0

    return ({name: jnp.array(vision_arrays[name]) for name in TASK_NAMES},
            all_qpos, t_render, t_dino, t_forward)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: PER-TASK EPISODE LOSS (differentiable BPTT under cached vision)
# ═══════════════════════════════════════════════════════════════════════════
def make_cube_loss_fn(task):
    idx = task['idx']; mjx_model = task['mjx_model']; home_ctrl = task['home_ctrl']
    frame_dt = task['frame_dt']; boundaries = task['boundaries']; n_total = task['n_total']
    ee_i, hand_i = idx['ee_site'], idx['hand_body']
    prim_seq_arr = jnp.array(task['prim_seq'])

    def cube_loss(policy_params, vision_array, obs_mean, obs_std, data, active_steps):
        @jax.checkpoint
        def frame_step(carry, step_idx):
            state, prev_ee, total_rew, gamma_acc = carry

            is_boundary = jnp.zeros((), dtype=bool)
            for b in boundaries:
                is_boundary = is_boundary | (step_idx == b)
            state = jax.lax.cond(is_boundary,
                lambda s: jax.lax.stop_gradient(s), lambda s: s, state)
            prev_ee = jax.lax.cond(is_boundary,
                lambda p: jax.lax.stop_gradient(p), lambda p: p, prev_ee)
            gamma_acc = jnp.where(is_boundary, jnp.float64(1.0), gamma_acc)

            cube_pos = state.xpos[idx['cube1']]
            cube2_pos = state.xpos[idx['cube2']]
            stack_target = cube2_pos + jnp.array([0., 0., CUBE2_HEIGHT + STACK_OFFSET])

            feat_map = vision_array[step_idx // CHUNK_SIZE]

            proprio, ee_pos, current_q = build_proprio(
                state, prev_ee, frame_dt, DOWN_QUAT, ee_i, hand_i)
            proprio_norm = (proprio.astype(jnp.float32) - obs_mean) / obs_std

            prim_idx = _select_by_step(
                [prim_seq_arr[i] for i in range(5)], step_idx, boundaries)

            qd, f = vision_primitive_forward(policy_params, feat_map, proprio_norm, prim_idx)
            safe_qd = safe_action(qd, current_q)

            ctrl = home_ctrl.at[:N_JOINTS].set(safe_qd)
            ctrl = ctrl.at[7].set(f)

            def do_substeps(s):
                @jax.checkpoint
                def sub(s, _):
                    s = s.replace(ctrl=ctrl)
                    s = mjx.step(mjx_model, s)
                    return s, None
                s, _ = jax.lax.scan(sub, s, None, length=N_SUBSTEPS)
                return s
            state = jax.lax.cond(step_idx < active_steps, do_substeps, lambda s: s, state)

            rews = [
                reward_reach(state, cube_pos + jnp.array([0., 0., PRE_GRASP_Z]),
                             DOWN_QUAT, ee_i, hand_i),
                reward_descend(state, cube_pos + jnp.array([0., 0., -0.01]),
                               DOWN_QUAT, ee_i, hand_i),
                reward_grasp(state, cube_pos, DOWN_QUAT, ee_i, hand_i),
                reward_move_generic(state, cube_pos, stack_target, ee_i, hand_i, DOWN_QUAT),
                reward_release_generic(state, stack_target, ee_i, hand_i, DOWN_QUAT),
            ]
            rew = _select_by_step(rews, step_idx, boundaries)
            rew = jnp.where(step_idx < active_steps, rew, 0.0)
            total_rew = total_rew + gamma_acc * rew
            gamma_acc = gamma_acc * GAMMA

            return (state, ee_pos, total_rew, gamma_acc), proprio

        init = (data, data.site_xpos[ee_i], jnp.float64(0.0), jnp.float64(1.0))
        (_, _, total_rew, _), all_proprio = jax.lax.scan(
            frame_step, init, jnp.arange(n_total))
        return -total_rew, all_proprio
    return cube_loss


def make_peg_loss_fn(task):
    idx = task['idx']; mjx_model = task['mjx_model']; home_ctrl = task['home_ctrl']
    frame_dt = task['frame_dt']; boundaries = task['boundaries']; n_total = task['n_total']
    ee_i, hand_i = idx['ee_site'], idx['hand_body']
    prim_seq_arr = jnp.array(task['prim_seq'])

    def peg_loss(policy_params, vision_array, obs_mean, obs_std, data, active_steps):
        @jax.checkpoint
        def frame_step(carry, step_idx):
            state, prev_ee, total_rew, gamma_acc = carry

            is_boundary = jnp.zeros((), dtype=bool)
            for b in boundaries:
                is_boundary = is_boundary | (step_idx == b)
            state = jax.lax.cond(is_boundary,
                lambda s: jax.lax.stop_gradient(s), lambda s: s, state)
            prev_ee = jax.lax.cond(is_boundary,
                lambda p: jax.lax.stop_gradient(p), lambda p: p, prev_ee)
            gamma_acc = jnp.where(is_boundary, jnp.float64(1.0), gamma_acc)

            peg_grip = state.site_xpos[idx['peg_grip']]
            peg_bottom = state.site_xpos[idx['peg_bottom']]
            slot_entry = state.site_xpos[idx['slot_entry']]
            slot_target = state.site_xpos[idx['slot_target']]
            lift_target = jnp.array([slot_entry[0], slot_entry[1], LIFT_Z_TARGET])

            feat_map = vision_array[step_idx // CHUNK_SIZE]

            proprio, ee_pos, current_q = build_proprio(
                state, prev_ee, frame_dt, DOWN_QUAT, ee_i, hand_i)
            proprio_norm = (proprio.astype(jnp.float32) - obs_mean) / obs_std

            prim_idx = _select_by_step(
                [prim_seq_arr[i] for i in range(7)], step_idx, boundaries)

            qd, f = vision_primitive_forward(policy_params, feat_map, proprio_norm, prim_idx)
            safe_qd = safe_action(qd, current_q)

            ctrl = home_ctrl.at[:N_JOINTS].set(safe_qd)
            ctrl = ctrl.at[7].set(f)

            def do_substeps(s):
                @jax.checkpoint
                def sub(s, _):
                    s = s.replace(ctrl=ctrl)
                    s = mjx.step(mjx_model, s)
                    return s, None
                s, _ = jax.lax.scan(sub, s, None, length=N_SUBSTEPS)
                return s
            state = jax.lax.cond(step_idx < active_steps, do_substeps, lambda s: s, state)

            peg_cfg = {
                'ee_site_idx': ee_i, 'hand_body_idx': hand_i,
                'peg_body_idx': idx['peg_body'],
                'peg_bottom_site_idx': idx['peg_bottom'],
                'peg_grip_site_idx': idx['peg_grip'],
                'slot_entry_site_idx': idx['slot_entry'],
                'slot_target_site_idx': idx['slot_target'],
                'target_quat': DOWN_QUAT,
            }

            rews = [
                reward_reach(state, peg_grip + jnp.array([0., 0., PRE_GRASP_Z]),
                             DOWN_QUAT, ee_i, hand_i),
                reward_descend(state, peg_grip, DOWN_QUAT, ee_i, hand_i),
                reward_grasp(state, peg_grip, DOWN_QUAT, ee_i, hand_i),
                reward_move_generic(state, peg_bottom, lift_target, ee_i, hand_i, DOWN_QUAT),
                reward_insert_align(state, peg_cfg),
                reward_insert_push(state, peg_cfg),
                reward_release_generic(state, slot_target, ee_i, hand_i, DOWN_QUAT),
            ]
            rew = _select_by_step(rews, step_idx, boundaries)
            rew = jnp.where(step_idx < active_steps, rew, 0.0)
            total_rew = total_rew + gamma_acc * rew
            gamma_acc = gamma_acc * GAMMA

            return (state, ee_pos, total_rew, gamma_acc), proprio

        init = (data, data.site_xpos[ee_i], jnp.float64(0.0), jnp.float64(1.0))
        (_, _, total_rew, _), all_proprio = jax.lax.scan(
            frame_step, init, jnp.arange(n_total))
        return -total_rew, all_proprio
    return peg_loss


def make_cont_loss_fn(task):
    idx = task['idx']; mjx_model = task['mjx_model']; home_ctrl = task['home_ctrl']
    frame_dt = task['frame_dt']; boundaries = task['boundaries']; n_total = task['n_total']
    ee_i, hand_i = idx['ee_site'], idx['hand_body']
    obj_idxs = idx['obj_idxs']
    prim_seq_arr = jnp.array(task['prim_seq'])

    def cont_loss(policy_params, vision_array, obs_mean, obs_std, data, active_steps):
        @jax.checkpoint
        def frame_step(carry, step_idx):
            state, prev_ee, total_rew, gamma_acc = carry

            is_boundary = jnp.zeros((), dtype=bool)
            for b in boundaries:
                is_boundary = is_boundary | (step_idx == b)
            state = jax.lax.cond(is_boundary,
                lambda s: jax.lax.stop_gradient(s), lambda s: s, state)
            prev_ee = jax.lax.cond(is_boundary,
                lambda p: jax.lax.stop_gradient(p), lambda p: p, prev_ee)
            gamma_acc = jnp.where(is_boundary, jnp.float64(1.0), gamma_acc)

            obj_positions = jnp.stack([state.xpos[i] for i in obj_idxs])
            container_pos = state.site_xpos[idx['container']]
            place_target = container_pos + jnp.array([0., 0., PLACE_Z_OFFSET])

            feat_map = vision_array[step_idx // CHUNK_SIZE]

            proprio, ee_pos, current_q = build_proprio(
                state, prev_ee, frame_dt, DOWN_QUAT, ee_i, hand_i)
            proprio_norm = (proprio.astype(jnp.float32) - obs_mean) / obs_std

            prim_idx = _select_by_step(
                [prim_seq_arr[i] for i in range(15)], step_idx, boundaries)

            qd, f = vision_primitive_forward(policy_params, feat_map, proprio_norm, prim_idx)
            safe_qd = safe_action(qd, current_q)

            ctrl = home_ctrl.at[:N_JOINTS].set(safe_qd)
            ctrl = ctrl.at[7].set(f)

            def do_substeps(s):
                @jax.checkpoint
                def sub(s, _):
                    s = s.replace(ctrl=ctrl)
                    s = mjx.step(mjx_model, s)
                    return s, None
                s, _ = jax.lax.scan(sub, s, None, length=N_SUBSTEPS)
                return s
            state = jax.lax.cond(step_idx < active_steps, do_substeps, lambda s: s, state)

            rews = []
            for oi in range(3):
                op = obj_positions[oi]
                obj_p = state.xpos[obj_idxs[oi]]
                rews.extend([
                    reward_reach(state, op + jnp.array([0., 0., PRE_GRASP_Z]),
                                DOWN_QUAT, ee_i, hand_i),
                    reward_descend(state, op + jnp.array([0., 0., -0.01]),
                                  DOWN_QUAT, ee_i, hand_i),
                    reward_grasp(state, op, DOWN_QUAT, ee_i, hand_i),
                    reward_move_generic(state, obj_p, place_target, ee_i, hand_i, DOWN_QUAT),
                    reward_release_generic(state, place_target, ee_i, hand_i, DOWN_QUAT),
                ])

            rew = _select_by_step(rews, step_idx, boundaries)
            rew = jnp.where(step_idx < active_steps, rew, 0.0)
            total_rew = total_rew + gamma_acc * rew
            gamma_acc = gamma_acc * GAMMA

            return (state, ee_pos, total_rew, gamma_acc), proprio

        init = (data, data.site_xpos[ee_i], jnp.float64(0.0), jnp.float64(1.0))
        (_, _, total_rew, _), all_proprio = jax.lax.scan(
            frame_step, init, jnp.arange(n_total))
        return -total_rew, all_proprio
    return cont_loss


cube_loss_fn = make_cube_loss_fn(tasks['cube'])
peg_loss_fn = make_peg_loss_fn(tasks['peg'])
cont_loss_fn = make_cont_loss_fn(tasks['cont'])


def combined_loss(policy_params, obs_mean, obs_std,
                  cube_vision, peg_vision, cont_vision,
                  cube_batch, peg_batch, cont_batch,
                  cube_active, peg_active, cont_active):
    cube_losses, cube_obs = jax.vmap(
        cube_loss_fn, in_axes=(None, 0, None, None, 0, None)
    )(policy_params, cube_vision, obs_mean, obs_std, cube_batch, cube_active)

    peg_losses, peg_obs = jax.vmap(
        peg_loss_fn, in_axes=(None, 0, None, None, 0, None)
    )(policy_params, peg_vision, obs_mean, obs_std, peg_batch, peg_active)

    cont_losses, cont_obs = jax.vmap(
        cont_loss_fn, in_axes=(None, 0, None, None, 0, None)
    )(policy_params, cont_vision, obs_mean, obs_std, cont_batch, cont_active)

    total = jnp.mean(cube_losses) + jnp.mean(peg_losses) + jnp.mean(cont_losses)
    return total, (cube_obs, peg_obs, cont_obs, cube_losses, peg_losses, cont_losses)


# ═══════════════════════════════════════════════════════════════════════════
# FORWARD ROLLOUT (single env, metrics + progressive curriculum)
# ═══════════════════════════════════════════════════════════════════════════
def make_forward_metrics_fn(task):
    idx = task['idx']; mjx_model = task['mjx_model']; home_ctrl = task['home_ctrl']
    frame_dt = task['frame_dt']; boundaries = task['boundaries']; n_total = task['n_total']
    ee_i, hand_i = idx['ee_site'], idx['hand_body']
    prim_seq_arr = jnp.array(task['prim_seq'])
    n_segs = len(task['prim_seq'])

    def forward_fn(policy_params, vision_array, obs_mean, obs_std, data, active_steps):
        def frame_step(carry, step_idx):
            state, prev_ee = carry
            feat_map = vision_array[step_idx // CHUNK_SIZE]

            proprio, ee_pos, current_q = build_proprio(
                state, prev_ee, frame_dt, DOWN_QUAT, ee_i, hand_i)
            proprio_norm = (proprio.astype(jnp.float32) - obs_mean) / obs_std

            prim_idx = _select_by_step(
                [prim_seq_arr[i] for i in range(n_segs)], step_idx, boundaries)

            qd, f = vision_primitive_forward(policy_params, feat_map, proprio_norm, prim_idx)
            safe_qd = safe_action(qd, current_q)
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

        init = (data, data.site_xpos[ee_i])
        (_, _), (all_states, all_qpos, all_fingers) = jax.lax.scan(
            frame_step, init, jnp.arange(n_total))
        return all_states, all_qpos, all_fingers
    return jax.jit(forward_fn)


for name in TASK_NAMES:
    tasks[name]['forward_metrics_jit'] = make_forward_metrics_fn(tasks[name])


# ═══════════════════════════════════════════════════════════════════════════
# JIT COMPILE COMBINED LOSS
# ═══════════════════════════════════════════════════════════════════════════
print("\n[5] JIT compiling combined loss + grad...")
value_and_grad_fn = jax.jit(jax.value_and_grad(combined_loss, has_aux=True))

PATIENCE = 4
active_steps = {}
for name in TASK_NAMES:
    pb = tasks[name]['phase_bounds']
    active_steps[name] = jnp.int32(pb[min(1, len(pb) - 1)])

obs_mean, obs_std = obs_rms.get_jnp()

t0 = time.time()
dummy_states = {name: tasks[name]['rebatch_jit'](jax.random.PRNGKey(1234), BATCH_SIZE)
                for name in TASK_NAMES}
dummy_visions, _, _, _, _ = collect_all_features(
    policy_params, dummy_states, obs_mean, obs_std, active_steps)

(loss_val, aux), grads = value_and_grad_fn(
    policy_params, obs_mean, obs_std,
    dummy_visions['cube'], dummy_visions['peg'], dummy_visions['cont'],
    dummy_states['cube'], dummy_states['peg'], dummy_states['cont'],
    active_steps['cube'], active_steps['peg'], active_steps['cont'])
loss_val.block_until_ready()
print(f"  Combined JIT: {time.time()-t0:.1f}s, loss={float(loss_val):.4f}")
print(f"  Grad norm: {float(optax.global_norm(grads)):.4f}")

# Seed obs_rms from warmup rollout
cube_obs, peg_obs, cont_obs, _, _, _ = aux
all_proprio_seed = np.concatenate([
    np.array(cube_obs).reshape(-1, PROPRIO_DIM),
    np.array(peg_obs).reshape(-1, PROPRIO_DIM),
    np.array(cont_obs).reshape(-1, PROPRIO_DIM),
], axis=0)
obs_rms.update(all_proprio_seed)
obs_mean, obs_std = obs_rms.get_jnp()
print(f"  obs_rms seeded: mean_range=[{float(obs_mean.min()):.3f}, {float(obs_mean.max()):.3f}], "
      f"std_range=[{float(obs_std.min()):.3f}, {float(obs_std.max()):.3f}]")


# ═══════════════════════════════════════════════════════════════════════════
# VIEWERS (cube + container, optional)
# ═══════════════════════════════════════════════════════════════════════════
USE_VIEWER = not args.no_viewer
viewers = {}
if USE_VIEWER:
    from mujoco import viewer as mj_viewer
    for vname, vxml in [('cont', args.cont_xml)]:
        v_mj = mujoco.MjModel.from_xml_path(vxml)
        v_md = mujoco.MjData(v_mj)
        mujoco.mj_resetDataKeyframe(v_mj, v_md, v_mj.keyframe("home").id)
        mujoco.mj_forward(v_mj, v_md)
        handle = mj_viewer.launch_passive(v_mj, v_md)
        viewers[vname] = {'model': v_mj, 'data': v_md, 'handle': handle}
        print(f"✓ Viewer açıldı ({vname})")
    print()


def render_trajectory(task_name, all_qpos_chunks, skip=2):
    if not USE_VIEWER or task_name not in viewers:
        return
    v = viewers[task_name]
    if not v['handle'].is_running():
        return
    sim_dt = tasks[task_name]['mj'].opt.timestep
    for chunk_qpos in all_qpos_chunks:
        for step in range(chunk_qpos.shape[0]):
            for sub in range(0, chunk_qpos.shape[1], skip):
                v['data'].qpos[:] = chunk_qpos[step, sub]
                mujoco.mj_forward(v['model'], v['data'])
                v['handle'].sync()
                time.sleep(sim_dt * skip)


# ═══════════════════════════════════════════════════════════════════════════
# PROGRESSIVE CURRICULUM CRITERIA
# ═══════════════════════════════════════════════════════════════════════════
CUBE_CRITERIA = [
    ("dist", 0.04), ("dist", 0.04), ("dist", 0.04), ("dist", 0.04), ("always", 0),
]
PEG_CRITERIA = [
    ("dist", 0.04), ("dist", 0.04), ("dist", 0.04), ("lift", 0.15),
    ("dist", 0.03), ("dist", 0.03), ("always", 0),
]
CONT_CRITERIA = ([
    ("dist", 0.04), ("dist", 0.04), ("dist", 0.04), ("dist", 0.125), ("always", 0),
] * 3)
CRITERIA = {'cube': CUBE_CRITERIA, 'peg': PEG_CRITERIA, 'cont': CONT_CRITERIA}

phase = {name: 0 for name in TASK_NAMES}
patience_cnt = {name: 0 for name in TASK_NAMES}


def _check_advance(metrics, phase, criteria, patience_cnt, phase_bounds, label):
    if phase >= len(phase_bounds) - 1:
        return phase, patience_cnt, None
    crit_type, thresh = criteria[phase]
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
    new_active = None
    if patience_cnt >= PATIENCE:
        phase += 1
        patience_cnt = 0
        new_active = jnp.int32(phase_bounds[min(phase + 1, len(phase_bounds) - 1)])
        print(f"  → {label} phase {phase}, active={int(new_active)}")
    return phase, patience_cnt, new_active


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("VISION-BASED PRIMITIVES — 7 Primitives × 3 Tasks (wrist cam only)")
print(f"  Image: {args.img_size}×{args.img_size}, Grid: {GRID_H}×{GRID_W}")
print(f"  Batch: {BATCH_SIZE} per task, Chunk: {CHUNK_SIZE}")
print(f"  LR: shared={args.lr}, Gamma: {GAMMA}, Clip: {args.grad_clip}")
print("=" * 65 + "\n")

combined_loss_hist = []
task_loss_hist = {name: [] for name in TASK_NAMES}
grad_norm_hist = []
prim_grad_hists = {name: [] for name in PRIM_NAMES}

train_start = time.time()

for iteration in range(args.iters):
    if USE_VIEWER and not any(v['handle'].is_running() for v in viewers.values()):
        print("\nTüm viewer'lar kapatıldı.")
        break

    obs_mean, obs_std = obs_rms.get_jnp()

    # ── rebatch every task, every iteration ──
    states = {name: tasks[name]['rebatch_jit'](
                  jax.random.PRNGKey(iteration * 3 + i), BATCH_SIZE)
              for i, name in enumerate(TASK_NAMES)}
    for name in TASK_NAMES:
        jax.block_until_ready(states[name].qpos)

    log_now = (iteration % 5 == 0 or iteration < 3)
    collect_qpos_tasks = ('cont',) if log_now else ()

    # ── PHASE 1: fused render + DINOv3 across tasks, per-task physics advance ──
    t_phase1 = time.time()
    vision_arrays, all_qpos, t_r, t_d, t_f = collect_all_features(
        policy_params, states, obs_mean, obs_std, active_steps,
        collect_qpos_tasks=collect_qpos_tasks)
    t_phase1 = time.time() - t_phase1

    # ── PHASE 2: combined BPTT backward pass ──
    t_bptt = time.time()
    (loss_val, aux), grads = value_and_grad_fn(
        policy_params, obs_mean, obs_std,
        vision_arrays['cube'], vision_arrays['peg'], vision_arrays['cont'],
        states['cube'], states['peg'], states['cont'],
        active_steps['cube'], active_steps['peg'], active_steps['cont'])
    t_bptt = time.time() - t_bptt

    loss_f = float(loss_val)
    if np.isnan(loss_f):
        print(f"\n⚠ NaN at iter {iteration}!")
        break

    cube_obs, peg_obs, cont_obs, cube_l, peg_l, cont_l = aux
    all_proprio = np.concatenate([
        np.array(cube_obs).reshape(-1, PROPRIO_DIM),
        np.array(peg_obs).reshape(-1, PROPRIO_DIM),
        np.array(cont_obs).reshape(-1, PROPRIO_DIM),
    ], axis=0)
    obs_rms.update(all_proprio)

    # ── shared update (projection + trunk) ──
    shared_grad = {'projection': grads['projection'], 'trunk': grads['trunk']}
    shared_params = {'projection': policy_params['projection'], 'trunk': policy_params['trunk']}
    shared_updates, shared_opt_state = shared_optimizer.update(
        shared_grad, shared_opt_state, shared_params)
    new_shared = optax.apply_updates(shared_params, shared_updates)
    policy_params['projection'] = new_shared['projection']
    policy_params['trunk'] = new_shared['trunk']

    # ── per-primitive clip + update ──
    prim_grad_norms = {}
    for name in PRIM_NAMES:
        g = grads[name]
        norm = optax.global_norm(g)
        prim_grad_norms[name] = float(norm)
        scale = jnp.minimum(1.0, args.grad_clip / (norm + 1e-6))
        g = jax.tree.map(lambda x: x * scale, g)
        updates, opt_states[name] = optimizers[name].update(
            g, opt_states[name], policy_params[name])
        policy_params[name] = optax.apply_updates(policy_params[name], updates)

    grad_norm = float(optax.global_norm(grads))
    grad_norm_hist.append(grad_norm)
    combined_loss_hist.append(-loss_f)
    task_loss_hist['cube'].append(float(-jnp.mean(cube_l)))
    task_loss_hist['peg'].append(float(-jnp.mean(peg_l)))
    task_loss_hist['cont'].append(float(-jnp.mean(cont_l)))
    for name in PRIM_NAMES:
        prim_grad_hists[name].append(prim_grad_norms[name])

    if log_now:
        elapsed = time.time() - train_start
        print(f"\nIter {iteration:4d}:  rew={-loss_f:.3f}  grad={grad_norm:.1f}  "
              f"[phase1={t_phase1:.1f}s bptt={t_bptt:.1f}s "
              f"rend={t_r:.1f} dino={t_d:.1f} fwd={t_f:.1f}]  elapsed={elapsed:.0f}s")

        for name in TASK_NAMES:
            t = tasks[name]
            obs0 = jax.tree.map(lambda x: x[0], states[name])
            vis0 = vision_arrays[name][0]
            f_states, f_qpos, f_fng = t['forward_metrics_jit'](
                policy_params, vis0, obs_mean, obs_std, obs0, active_steps[name])

            seg_ends = [b - 1 for b in t['phase_bounds']]
            idx = t['idx']
            metrics = []

            if name == 'cube':
                ee = np.array(f_states.site_xpos[:, idx['ee_site']])
                c1 = np.array(f_states.xpos[:, idx['cube1']])
                c2 = np.array(f_states.xpos[:, idx['cube2']])
                for si, se in enumerate(seg_ends):
                    if si == 0:
                        metrics.append(np.linalg.norm(ee[se] - (c1[se] + [0, 0, PRE_GRASP_Z])))
                    elif si == 1:
                        metrics.append(np.linalg.norm(ee[se] - (c1[se] + [0, 0, -0.01])))
                    elif si == 2:
                        metrics.append(np.linalg.norm(ee[se] - c1[se]))
                    else:
                        stack_t = c2[se] + np.array([0, 0, CUBE2_HEIGHT + STACK_OFFSET])
                        metrics.append(np.linalg.norm(c1[se] - stack_t))
            elif name == 'peg':
                ee = np.array(f_states.site_xpos[:, idx['ee_site']])
                peg_b = np.array(f_states.xpos[:, idx['peg_body']])
                pb = np.array(f_states.site_xpos[:, idx['peg_bottom']])
                pg = np.array(f_states.site_xpos[:, idx['peg_grip']])
                se_ = np.array(f_states.site_xpos[:, idx['slot_entry']])
                st_ = np.array(f_states.site_xpos[:, idx['slot_target']])
                for si, se in enumerate(seg_ends):
                    if si == 0:
                        metrics.append(np.linalg.norm(ee[se] - (pg[se] + [0, 0, PRE_GRASP_Z])))
                    elif si in (1, 2):
                        metrics.append(np.linalg.norm(ee[se] - pg[se]))
                    elif si == 3:
                        metrics.append(peg_b[se][2])
                    elif si == 4:
                        metrics.append(np.linalg.norm(pb[se] - se_[se]))
                    elif si == 5:
                        metrics.append(np.linalg.norm(pb[se] - st_[se]))
                    else:
                        metrics.append(float(np.array(f_fng)[se]))
            else:  # cont
                ee = np.array(f_states.site_xpos[:, idx['ee_site']])
                objs = [np.array(f_states.xpos[:, i]) for i in idx['obj_idxs']]
                cont_p = np.array(f_states.site_xpos[:, idx['container']])
                for si, se in enumerate(seg_ends):
                    oi, ph = si // 5, si % 5
                    if ph == 0:
                        metrics.append(np.linalg.norm(
                            ee[se] - (objs[oi][se] + [0, 0, PRE_GRASP_Z])))
                    elif ph == 1:
                        metrics.append(np.linalg.norm(
                            ee[se] - (objs[oi][se] + [0, 0, -0.01])))
                    elif ph == 2:
                        metrics.append(np.linalg.norm(ee[se] - objs[oi][se]))
                    elif ph == 3:
                        metrics.append(np.linalg.norm(
                            objs[oi][se][:2] - cont_p[se][:2]))
                    else:
                        metrics.append(np.linalg.norm(objs[oi][se] - cont_p[se]))

            print(f"  {name:5s} rew={task_loss_hist[name][-1]:.3f}  "
                  f"phase={phase[name]}  active={int(active_steps[name])}  "
                  f"metrics={['%.4f' % m for m in metrics]}")

            new_phase, new_patience, new_active = _check_advance(
                metrics, phase[name], CRITERIA[name], patience_cnt[name],
                t['phase_bounds'], name)
            phase[name] = new_phase
            patience_cnt[name] = new_patience
            if new_active is not None:
                active_steps[name] = new_active

            if all_qpos.get('cont'):
                render_trajectory('cont', all_qpos['cont'], skip=2)

    if iteration % 50 == 0:
        plt.close('all')
        gc.collect()

train_time = time.time() - train_start
print(f"\nTraining done: {train_time:.1f}s ({len(combined_loss_hist)} iters)")

# ═══════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════
with open("vision_primitives_params.pkl", "wb") as f:
    pickle.dump(jax.tree.map(lambda x: np.array(x), policy_params), f)
print("Saved: vision_primitives_params.pkl")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()

axes[0].plot(combined_loss_hist, linewidth=2, color='black')
axes[0].set_title("Combined Reward (all 3 tasks)")
axes[0].grid(True, alpha=0.3)

for name in PRIM_NAMES:
    axes[1].plot(prim_grad_hists[name], linewidth=1, label=name, alpha=0.8)
axes[1].set_title("Per-Primitive Gradient Norm (raw)")
axes[1].set_yscale('log')
axes[1].legend(fontsize=7)
axes[1].grid(True, alpha=0.3)

axes[2].plot(task_loss_hist['cube'], linewidth=1.5, color='green')
axes[2].set_title("Cube Stacking Reward")
axes[2].grid(True, alpha=0.3)

axes[3].plot(task_loss_hist['peg'], linewidth=1.5, color='orange')
axes[3].set_title("Peg Insertion Reward")
axes[3].grid(True, alpha=0.3)

axes[4].plot(task_loss_hist['cont'], linewidth=1.5, color='blue')
axes[4].set_title("Container Sorting Reward")
axes[4].grid(True, alpha=0.3)

axes[5].plot(grad_norm_hist, linewidth=1.5, color='red', alpha=0.6)
axes[5].axhline(y=args.grad_clip, color='black', linestyle='--', alpha=0.5,
                label=f'clip={args.grad_clip}')
axes[5].set_title("Total Gradient Norm")
axes[5].legend(fontsize=8)
axes[5].grid(True, alpha=0.3)

fig.suptitle(f"Vision Primitives — 7 prims × 3 tasks, wrist-only, "
             f"{len(combined_loss_hist)} iters", fontsize=14)
fig.tight_layout()
fig.savefig("vision_primitives_training_results.png", dpi=150)
plt.close(fig)
print("Plot saved: vision_primitives_training_results.png")

if USE_VIEWER and any(v['handle'].is_running() for v in viewers.values()):
    print("\nViewer açık — ESC ile kapat.")
    try:
        while any(v['handle'].is_running() for v in viewers.values()):
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nÇıkılıyor.")