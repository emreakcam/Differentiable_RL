"""
MJX BPTT — 9 Primitives × 5 Tasks Parallel Training

Primitives: Reach, Descend, Grasp, Move, InsertAlign, InsertPush, Release, Pull, Swing
Tasks:
  1. Cube Stacking  (5 seg):  Reach→Descend→Grasp→Move→Release
  2. Peg Insertion   (7 seg):  Reach→Descend→Grasp→Move→InsertAlign→InsertPush→Release
  3. Container Sort (15 seg):  3×(Reach→Descend→Grasp→Move→Release)
  4. Drawer Opening  (4 seg):  Reach→Grasp→Pull→Release
  5. Cabinet Opening (4 seg):  Reach→Grasp→Swing→Release
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

from helpers.mjx_utils import patch_solver
from helpers.obs import ObsRMS
from models.primitives_all import (
        PRIM_NAMES, REACH, DESCEND, GRASP, MOVE, INSERT_ALIGN, INSERT_PUSH,
    RELEASE, PULL, SWING,
    PROPRIO_DIM, DOWN_QUAT, N_JOINTS, FINGER_OPEN,
    init_primitives, primitive_forward, build_proprio, safe_action,
    reward_reach, reward_descend, reward_grasp,
    reward_move_generic, reward_release_generic,
    reward_insert_align, reward_insert_push, reward_pull, reward_swing,
)

patch_solver()

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--substeps", type=int, default=4)
parser.add_argument("--iters", type=int, default=3000)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr-reach", type=float, default=1e-3)
parser.add_argument("--lr-descend", type=float, default=1e-3)
parser.add_argument("--lr-grasp", type=float, default=1e-3)
parser.add_argument("--lr-move", type=float, default=1e-3)
parser.add_argument("--lr-insert-align", type=float, default=1e-4)
parser.add_argument("--lr-insert-push", type=float, default=1e-4)
parser.add_argument("--lr-release", type=float, default=1e-3)
parser.add_argument("--lr-pull", type=float, default=1e-3)
parser.add_argument("--lr-swing", type=float, default=1e-3)
parser.add_argument("--gamma", type=float, default=0.999)
parser.add_argument("--grad-clip", type=float, default=600.0)
parser.add_argument("--cube-batch", type=int, default=32)
parser.add_argument("--peg-batch", type=int, default=32)
parser.add_argument("--cont-batch", type=int, default=32)
parser.add_argument("--drawer-batch", type=int, default=32)
parser.add_argument("--cabinet-batch", type=int, default=32)
parser.add_argument("--solver-iters", type=int, default=6)
parser.add_argument("--no-viewer", action="store_true")
parser.add_argument("--cube-xml", type=str,
                    default="assets/common/franka_emika_panda/mjx_single_cube.xml")
parser.add_argument("--peg-xml", type=str,
                    default="assets/common/franka_emika_panda/mjx_peg_slot.xml")
parser.add_argument("--cont-xml", type=str,
                    default="assets/common/franka_emika_panda/mjx_container.xml")
parser.add_argument("--drawer-xml", type=str,
                    default="assets/common/franka_emika_panda/mjx_drawer.xml")
parser.add_argument("--cabinet-xml", type=str,
                    default="assets/common/franka_emika_panda/mjx_revolute_door.xml")
args = parser.parse_args()

N_SUBSTEPS = args.substeps
GAMMA = args.gamma

# ═══════════════════════════════════════════════════════════════════════════
# TASK 1: CUBE STACKING
# ═══════════════════════════════════════════════════════════════════════════
print("\n[CUBE] Loading model...")
cube_mj = mujoco.MjModel.from_xml_path(args.cube_xml)
cube_mj.opt.iterations = args.solver_iters
cube_mj.opt.ls_iterations = 8
cube_md = mujoco.MjData(cube_mj)

CUBE_IDX = {
    'ee_site':   0,  # gripper site
    'hand_body': mujoco.mj_name2id(cube_mj, mujoco.mjtObj.mjOBJ_BODY, "hand"),
    'cube1':     mujoco.mj_name2id(cube_mj, mujoco.mjtObj.mjOBJ_BODY, "box"),
    'cube2':     mujoco.mj_name2id(cube_mj, mujoco.mjtObj.mjOBJ_BODY, "box2"),
}

CUBE_SEG_STEPS = [16, 12, 6, 16, 8]  # Reach, Descend, Grasp, Move, Release
CUBE_N_TOTAL = sum(CUBE_SEG_STEPS)
CUBE_BOUNDARIES = []
acc = 0
for s in CUBE_SEG_STEPS[:-1]:
    acc += s; CUBE_BOUNDARIES.append(acc)
CUBE_PHASE_BOUNDS = CUBE_BOUNDARIES + [CUBE_N_TOTAL]
CUBE_PRIM_SEQ = [REACH, DESCEND, GRASP, MOVE, RELEASE]

PRE_GRASP_Z = 0.075
CUBE2_HEIGHT = 0.05
STACK_OFFSET = 0.01

cube_key_id = cube_mj.keyframe("home").id
mujoco.mj_resetDataKeyframe(cube_mj, cube_md, cube_key_id)
mujoco.mj_forward(cube_mj, cube_md)
cube_home_ctrl = jnp.array(cube_md.ctrl)
cube_frame_dt = N_SUBSTEPS * cube_mj.opt.timestep
cube_mjx_model = mjx.put_model(cube_mj)

def cube_make_batch(batch_size, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), batch_size)
    dlist = []
    for k in keys:
        mujoco.mj_resetDataKeyframe(cube_mj, cube_md, cube_key_id)
        n = jax.random.uniform(k, (2,), minval=-0.05, maxval=0.05)
        cube_md.qpos[9] += float(n[0])
        cube_md.qpos[10] += float(n[1])
        mujoco.mj_forward(cube_mj, cube_md)
        d = mjx.put_data(cube_mj, cube_md)
        d = mjx.forward(cube_mjx_model, d)
        dlist.append(d)
    return jax.tree.map(lambda *xs: jnp.stack(xs, 0), *dlist)

cube_batch = cube_make_batch(args.cube_batch)
print(f"  Cube: {CUBE_N_TOTAL} steps, batch={args.cube_batch}, "
      f"segs={CUBE_SEG_STEPS}")


# ═══════════════════════════════════════════════════════════════════════════
# TASK 2: PEG INSERTION
# ═══════════════════════════════════════════════════════════════════════════
print("\n[PEG] Loading model...")
peg_mj = mujoco.MjModel.from_xml_path(args.peg_xml)
peg_mj.opt.iterations = args.solver_iters
peg_mj.opt.ls_iterations = 8
peg_md = mujoco.MjData(peg_mj)

PEG_IDX = {
    'ee_site':     mujoco.mj_name2id(peg_mj, mujoco.mjtObj.mjOBJ_SITE, "gripper"),
    'hand_body':   mujoco.mj_name2id(peg_mj, mujoco.mjtObj.mjOBJ_BODY, "hand"),
    'peg_body':    mujoco.mj_name2id(peg_mj, mujoco.mjtObj.mjOBJ_BODY, "peg"),
    'peg_bottom':  mujoco.mj_name2id(peg_mj, mujoco.mjtObj.mjOBJ_SITE, "peg_bottom"),
    'peg_grip':    mujoco.mj_name2id(peg_mj, mujoco.mjtObj.mjOBJ_SITE, "peg_grip"),
    'slot_entry':  mujoco.mj_name2id(peg_mj, mujoco.mjtObj.mjOBJ_SITE, "slot_entry"),
    'slot_target': mujoco.mj_name2id(peg_mj, mujoco.mjtObj.mjOBJ_SITE, "slot_target"),
}

PEG_SEG_STEPS = [16, 12, 6, 12, 20, 20, 6]
PEG_N_TOTAL = sum(PEG_SEG_STEPS)
PEG_BOUNDARIES = []
acc = 0
for s in PEG_SEG_STEPS[:-1]:
    acc += s; PEG_BOUNDARIES.append(acc)
PEG_PHASE_BOUNDS = PEG_BOUNDARIES + [PEG_N_TOTAL]
PEG_PRIM_SEQ = [REACH, DESCEND, GRASP, MOVE, INSERT_ALIGN, INSERT_PUSH, RELEASE]

LIFT_Z_TARGET = 0.18

peg_key_id = peg_mj.keyframe("home").id
mujoco.mj_resetDataKeyframe(peg_mj, peg_md, peg_key_id)
mujoco.mj_forward(peg_mj, peg_md)
peg_home_ctrl = jnp.array(peg_md.ctrl)
peg_frame_dt = N_SUBSTEPS * peg_mj.opt.timestep
peg_mjx_model = mjx.put_model(peg_mj)

def peg_make_batch(batch_size, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), batch_size)
    dlist = []
    for k in keys:
        mujoco.mj_resetDataKeyframe(peg_mj, peg_md, peg_key_id)
        n = jax.random.uniform(k, (2,), minval=-0.05, maxval=0.05)
        peg_md.qpos[9] += float(n[0])
        peg_md.qpos[10] += float(n[1])
        mujoco.mj_forward(peg_mj, peg_md)
        d = mjx.put_data(peg_mj, peg_md)
        d = mjx.forward(peg_mjx_model, d)
        dlist.append(d)
    return jax.tree.map(lambda *xs: jnp.stack(xs, 0), *dlist)

peg_batch = peg_make_batch(args.peg_batch)
print(f"  Peg: {PEG_N_TOTAL} steps, batch={args.peg_batch}, "
      f"segs={PEG_SEG_STEPS}")


# ═══════════════════════════════════════════════════════════════════════════
# TASK 3: CONTAINER SORTING
# ═══════════════════════════════════════════════════════════════════════════
print("\n[CONT] Loading model...")
cont_mj = mujoco.MjModel.from_xml_path(args.cont_xml)
cont_mj.opt.iterations = args.solver_iters
cont_mj.opt.ls_iterations = 8
cont_md = mujoco.MjData(cont_mj)

CONT_IDX = {
    'ee_site':    mujoco.mj_name2id(cont_mj, mujoco.mjtObj.mjOBJ_SITE, "gripper"),
    'hand_body':  mujoco.mj_name2id(cont_mj, mujoco.mjtObj.mjOBJ_BODY, "hand"),
    'ball_body':  mujoco.mj_name2id(cont_mj, mujoco.mjtObj.mjOBJ_BODY, "ball"),
    'cube_body':  mujoco.mj_name2id(cont_mj, mujoco.mjtObj.mjOBJ_BODY, "cube"),
    'prism_body': mujoco.mj_name2id(cont_mj, mujoco.mjtObj.mjOBJ_BODY, "prism"),
    'container':  mujoco.mj_name2id(cont_mj, mujoco.mjtObj.mjOBJ_SITE,
                                     "container_inside"),
}
# Nesne sırası: ball, cube, prism
CONT_OBJ_KEYS = ['ball_body', 'cube_body', 'prism_body']
CONT_OBJ_IDXS = [CONT_IDX[k] for k in CONT_OBJ_KEYS]

CONT_STEPS_PER_OBJ = [16, 12, 8, 16, 6]  # reach, descend, grasp, move, release
CONT_SEG_STEPS = CONT_STEPS_PER_OBJ * 3
CONT_N_TOTAL = sum(CONT_SEG_STEPS)
CONT_BOUNDARIES = []
acc = 0
for s in CONT_SEG_STEPS[:-1]:
    acc += s; CONT_BOUNDARIES.append(acc)
CONT_PHASE_BOUNDS = CONT_BOUNDARIES + [CONT_N_TOTAL]
CONT_PRIM_SEQ = [REACH, DESCEND, GRASP, MOVE, RELEASE] * 3

PLACE_Z_OFFSET = 0.08

cont_key_id = cont_mj.keyframe("home").id
mujoco.mj_resetDataKeyframe(cont_mj, cont_md, cont_key_id)
mujoco.mj_forward(cont_mj, cont_md)
cont_home_ctrl = jnp.array(cont_md.ctrl)
cont_frame_dt = N_SUBSTEPS * cont_mj.opt.timestep
cont_mjx_model = mjx.put_model(cont_mj)

def cont_make_batch(batch_size, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), batch_size)
    dlist = []
    for k in keys:
        mujoco.mj_resetDataKeyframe(cont_mj, cont_md, cont_key_id)
        k1, k2, k3 = jax.random.split(k, 3)
        for i, offset in enumerate([9, 16, 23]):
            n = jax.random.uniform([k1, k2, k3][i], (2,),
                                   minval=-0.05, maxval=0.05)
            cont_md.qpos[offset] += float(n[0])
            cont_md.qpos[offset + 1] += float(n[1])
        mujoco.mj_forward(cont_mj, cont_md)
        d = mjx.put_data(cont_mj, cont_md)
        d = mjx.forward(cont_mjx_model, d)
        dlist.append(d)
    return jax.tree.map(lambda *xs: jnp.stack(xs, 0), *dlist)

cont_batch = cont_make_batch(args.cont_batch)
print(f"  Cont: {CONT_N_TOTAL} steps, batch={args.cont_batch}, "
      f"segs={CONT_SEG_STEPS}")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 4: DRAWER
# ═══════════════════════════════════════════════════════════════════════════
DRAWER_TARGET_QUAT = jnp.array([0.3430, 0.6184, 0.6183, 0.3431])
DRAWER_OPEN_TARGET = -0.12

print("\n[DRAWER] Loading model...")
drawer_mj = mujoco.MjModel.from_xml_path(args.drawer_xml)
drawer_mj.opt.iterations = args.solver_iters
drawer_mj.opt.ls_iterations = 8
drawer_md = mujoco.MjData(drawer_mj)

DRAWER_IDX = {
    'ee_site':     mujoco.mj_name2id(drawer_mj, mujoco.mjtObj.mjOBJ_SITE, "gripper"),
    'hand_body':   mujoco.mj_name2id(drawer_mj, mujoco.mjtObj.mjOBJ_BODY, "hand"),
    'handle_site': mujoco.mj_name2id(drawer_mj, mujoco.mjtObj.mjOBJ_SITE, "drawer_handle"),
    'joint_qpos_idx': 11,
}

DRAWER_SEG_STEPS = [16, 8, 20, 8]
DRAWER_N_TOTAL = sum(DRAWER_SEG_STEPS)
DRAWER_BOUNDARIES = []
acc = 0
for s in DRAWER_SEG_STEPS[:-1]:
    acc += s; DRAWER_BOUNDARIES.append(acc)
DRAWER_PHASE_BOUNDS = DRAWER_BOUNDARIES + [DRAWER_N_TOTAL]
DRAWER_PRIM_SEQ = [REACH, GRASP, PULL, RELEASE]

drawer_key_id = drawer_mj.keyframe("home").id
mujoco.mj_resetDataKeyframe(drawer_mj, drawer_md, drawer_key_id)
mujoco.mj_forward(drawer_mj, drawer_md)
drawer_home_ctrl = jnp.array(drawer_md.ctrl)
drawer_frame_dt = N_SUBSTEPS * drawer_mj.opt.timestep
drawer_mjx_model = mjx.put_model(drawer_mj)

def drawer_make_batch(batch_size, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), batch_size)
    dlist = []
    for k in keys:
        mujoco.mj_resetDataKeyframe(drawer_mj, drawer_md, drawer_key_id)
        n = jax.random.uniform(k, (N_JOINTS,), minval=-0.02, maxval=0.02)
        for i in range(N_JOINTS):
            drawer_md.qpos[i] += float(n[i])
        mujoco.mj_forward(drawer_mj, drawer_md)
        d = mjx.put_data(drawer_mj, drawer_md)
        d = mjx.forward(drawer_mjx_model, d)
        dlist.append(d)
    return jax.tree.map(lambda *xs: jnp.stack(xs, 0), *dlist)

drawer_batch = drawer_make_batch(args.drawer_batch)
print(f"  Drawer: {DRAWER_N_TOTAL} steps, batch={args.drawer_batch}")


# ═══════════════════════════════════════════════════════════════════════════
# TASK 5: CABINET
# ═══════════════════════════════════════════════════════════════════════════
CABINET_TARGET_QUAT = jnp.array([0.2150, 0.2427, 0.9444, 0.0552])
HINGE_OPEN_TARGET = -1.2
LEFT_HINGE_QPOS_IDX = 9

print("\n[CABINET] Loading model...")
cabinet_mj = mujoco.MjModel.from_xml_path(args.cabinet_xml)
cabinet_mj.opt.iterations = args.solver_iters
cabinet_mj.opt.ls_iterations = 8
cabinet_md = mujoco.MjData(cabinet_mj)

CABINET_IDX = {
    'ee_site':     mujoco.mj_name2id(cabinet_mj, mujoco.mjtObj.mjOBJ_SITE, "gripper"),
    'hand_body':   mujoco.mj_name2id(cabinet_mj, mujoco.mjtObj.mjOBJ_BODY, "hand"),
    'handle_site': mujoco.mj_name2id(cabinet_mj, mujoco.mjtObj.mjOBJ_SITE, "leftdoor_handle"),
    'hinge_qpos_idx': LEFT_HINGE_QPOS_IDX,
}

CABINET_SEG_STEPS = [16, 8, 20, 8]
CABINET_N_TOTAL = sum(CABINET_SEG_STEPS)
CABINET_BOUNDARIES = []
acc = 0
for s in CABINET_SEG_STEPS[:-1]:
    acc += s; CABINET_BOUNDARIES.append(acc)
CABINET_PHASE_BOUNDS = CABINET_BOUNDARIES + [CABINET_N_TOTAL]
CABINET_PRIM_SEQ = [REACH, GRASP, SWING, RELEASE]

cabinet_key_id = cabinet_mj.keyframe("home").id
mujoco.mj_resetDataKeyframe(cabinet_mj, cabinet_md, cabinet_key_id)
mujoco.mj_forward(cabinet_mj, cabinet_md)
cabinet_home_ctrl = jnp.array(cabinet_md.ctrl)
cabinet_frame_dt = N_SUBSTEPS * cabinet_mj.opt.timestep
cabinet_mjx_model = mjx.put_model(cabinet_mj)

def cabinet_make_batch(batch_size, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), batch_size)
    dlist = []
    for k in keys:
        mujoco.mj_resetDataKeyframe(cabinet_mj, cabinet_md, cabinet_key_id)
        n = jax.random.uniform(k, (N_JOINTS,), minval=-0.02, maxval=0.02)
        for i in range(N_JOINTS):
            cabinet_md.qpos[i] += float(n[i])
        mujoco.mj_forward(cabinet_mj, cabinet_md)
        d = mjx.put_data(cabinet_mj, cabinet_md)
        d = mjx.forward(cabinet_mjx_model, d)
        dlist.append(d)
    return jax.tree.map(lambda *xs: jnp.stack(xs, 0), *dlist)

cabinet_batch = cabinet_make_batch(args.cabinet_batch)
print(f"  Cabinet: {CABINET_N_TOTAL} steps, batch={args.cabinet_batch}")

# ═══════════════════════════════════════════════════════════════════════════
# PRIMITIVES + OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════
rng = jax.random.PRNGKey(42)
prim_params = init_primitives(rng)
n_params = sum(p.size for p in jax.tree.leaves(prim_params))

GRAD_CLIP = args.grad_clip

# Per-primitive LRs (fall back to global --lr)
PRIM_LRS = {}
for name in PRIM_NAMES:
    attr = f'lr_{name}'.replace('-', '_')
    override = getattr(args, attr, None)
    PRIM_LRS[name] = override if override is not None else args.lr

# Per-primitive optimizers
optimizers = {}
opt_states = {}
for name in PRIM_NAMES:
    optimizers[name] = optax.adam(PRIM_LRS[name])
    opt_states[name] = optimizers[name].init(prim_params[name])

obs_rms = ObsRMS(PROPRIO_DIM) 

print(f"\n{'='*65}")
print(f"PRIMITIVE LIBRARY — 9 Primitives × 5 Tasks")
print(f"  Params: {n_params} (9 MLPs × 36→256→256→128→8)")
print(f"  Cube: {CUBE_N_TOTAL} steps, Peg: {PEG_N_TOTAL} steps, "
      f"Cont: {CONT_N_TOTAL} steps")
print(f"  Batch: cube={args.cube_batch} peg={args.peg_batch} "
      f"cont={args.cont_batch}")
print(f"  LR: {args.lr}, Gamma: {GAMMA}, Clip: {args.grad_clip}")
print(f"{'='*65}\n")


# ═══════════════════════════════════════════════════════════════════════════
# PER-TASK LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

# ---- Segment config helpers ----
def _select_by_step(values, step_idx, boundaries):
    """Nested where ile step_idx'e göre değer seç."""
    result = values[-1]
    for i in range(len(boundaries) - 1, -1, -1):
        result = jax.tree.map(
            lambda r, v: jnp.where(step_idx < boundaries[i], v, r),
            result, values[i])
    return result


# ---- CUBE STACKING LOSS ----
def cube_single_loss(prim_params, obs_mean, obs_std, data, active_steps):
    @jax.checkpoint
    def frame_step(carry, step_idx):
        state, prev_ee, total_rew, gamma_acc = carry

        is_boundary = jnp.zeros((), dtype=bool)
        for b in CUBE_BOUNDARIES:
            is_boundary = is_boundary | (step_idx == b)
        state = jax.lax.cond(is_boundary,
            lambda s: jax.lax.stop_gradient(s), lambda s: s, state)
        prev_ee = jax.lax.cond(is_boundary,
            lambda p: jax.lax.stop_gradient(p), lambda p: p, prev_ee)
        gamma_acc = jnp.where(is_boundary, jnp.float64(1.0), gamma_acc)

        # Live positions
        cube_pos = state.xpos[CUBE_IDX['cube1']]
        cube2_pos = state.xpos[CUBE_IDX['cube2']]
        stack_target = cube2_pos + jnp.array([0.0, 0.0,
                                               CUBE2_HEIGHT + STACK_OFFSET])

        # Per-segment config
        seg_configs = [
            (REACH,   cube_pos + jnp.array([0., 0., PRE_GRASP_Z])),
            (DESCEND, cube_pos + jnp.array([0., 0., -0.01])),
            (GRASP,   cube_pos),
            (MOVE,    stack_target),
            (RELEASE, stack_target),
        ]
        prim_indices = jnp.array([c[0] for c in seg_configs])
        target_positions = jnp.stack([c[1] for c in seg_configs])

        prim_idx = _select_by_step(
            [prim_indices[i] for i in range(5)],
            step_idx, CUBE_BOUNDARIES)
        target_pos = _select_by_step(
            [target_positions[i] for i in range(5)],
            step_idx, CUBE_BOUNDARIES)

        # Proprio + forward
        proprio, ee_pos, current_q = build_proprio(
            state, prev_ee, cube_frame_dt, DOWN_QUAT,
            CUBE_IDX['ee_site'], CUBE_IDX['hand_body'],
            joint_state=0.0)
        proprio_norm = (proprio.astype(jnp.float32) - obs_mean) / obs_std

        # obj_rel: Reach/Descend → ee-target, Grasp/Move/Release → cube-target
        obj_rel = jnp.where(prim_idx <= DESCEND,
                            ee_pos - target_pos,
                            cube_pos - target_pos)

        qd, f = primitive_forward(prim_params, prim_idx,
                                  proprio_norm, target_pos, obj_rel)
        safe_qd = safe_action(qd, current_q)

        ctrl = cube_home_ctrl.at[:N_JOINTS].set(safe_qd)
        ctrl = ctrl.at[7].set(f)

        def do_substeps(s):
            @jax.checkpoint
            def sub(s, _):
                s = s.replace(ctrl=ctrl)
                s = mjx.step(cube_mjx_model, s)
                return s, None
            s, _ = jax.lax.scan(sub, s, None, length=N_SUBSTEPS)
            return s
        state = jax.lax.cond(step_idx < active_steps,
                             do_substeps, lambda s: s, state)

        # Reward dispatch
        ei, hi = CUBE_IDX['ee_site'], CUBE_IDX['hand_body']

        rews = [
            reward_reach(state, target_positions[0], DOWN_QUAT, ei, hi),
            reward_descend(state, target_positions[1], DOWN_QUAT, ei, hi),
            reward_grasp(state, target_positions[2], DOWN_QUAT, ei, hi),
            reward_move_generic(state, cube_pos, stack_target, ei, hi, DOWN_QUAT),
            reward_release_generic(state, stack_target, ei, hi, DOWN_QUAT),
        ]
        rew = _select_by_step(rews, step_idx, CUBE_BOUNDARIES)
        rew = jnp.where(step_idx < active_steps, rew, 0.0)
        total_rew = total_rew + gamma_acc * rew
        gamma_acc = gamma_acc * GAMMA

        return (state, ee_pos, total_rew, gamma_acc), proprio

    init = (data, data.site_xpos[CUBE_IDX['ee_site']],
            jnp.float64(0.0), jnp.float64(1.0))
    (_, _, total_rew, _), all_proprio = jax.lax.scan(
        frame_step, init, jnp.arange(CUBE_N_TOTAL))
    return -total_rew, all_proprio


# ---- PEG INSERTION LOSS ----
def peg_single_loss(prim_params, obs_mean, obs_std, data, active_steps):
    @jax.checkpoint
    def frame_step(carry, step_idx):
        state, prev_ee, total_rew, gamma_acc = carry

        is_boundary = jnp.zeros((), dtype=bool)
        for b in PEG_BOUNDARIES:
            is_boundary = is_boundary | (step_idx == b)
        state = jax.lax.cond(is_boundary,
            lambda s: jax.lax.stop_gradient(s), lambda s: s, state)
        prev_ee = jax.lax.cond(is_boundary,
            lambda p: jax.lax.stop_gradient(p), lambda p: p, prev_ee)
        gamma_acc = jnp.where(is_boundary, jnp.float64(1.0), gamma_acc)

        peg_grip = state.site_xpos[PEG_IDX['peg_grip']]
        peg_bottom = state.site_xpos[PEG_IDX['peg_bottom']]
        peg_pos = state.xpos[PEG_IDX['peg_body']]
        slot_entry = state.site_xpos[PEG_IDX['slot_entry']]
        slot_target = state.site_xpos[PEG_IDX['slot_target']]
        lift_target = jnp.array([slot_entry[0], slot_entry[1],
                                  LIFT_Z_TARGET])

        # 7 segments
        target_positions = jnp.stack([
            peg_grip + jnp.array([0., 0., PRE_GRASP_Z]),  # 0 Reach
            peg_grip,                                       # 1 Descend
            peg_grip,                                       # 2 Grasp
            lift_target,                                    # 3 Move (lift)
            slot_entry,                                     # 4 Insert (align)
            slot_target,                                    # 5 Insert (push)
            slot_target,                                    # 6 Release
        ])
        prim_indices = jnp.array(PEG_PRIM_SEQ)

        prim_idx = _select_by_step(
            [prim_indices[i] for i in range(7)],
            step_idx, PEG_BOUNDARIES)
        target_pos = _select_by_step(
            [target_positions[i] for i in range(7)],
            step_idx, PEG_BOUNDARIES)

        proprio, ee_pos, current_q = build_proprio(
            state, prev_ee, peg_frame_dt, DOWN_QUAT,
            PEG_IDX['ee_site'], PEG_IDX['hand_body'],
            joint_state=0.0)
        proprio_norm = (proprio.astype(jnp.float32) - obs_mean) / obs_std

        # obj_rel: Reach/Descend → ee-target, Grasp+ → peg-target
        obj_rel = jnp.where(prim_idx <= DESCEND,
                            ee_pos - target_pos,
                            peg_pos - target_pos)

        qd, f = primitive_forward(prim_params, prim_idx,
                                  proprio_norm, target_pos, obj_rel)
        safe_qd = safe_action(qd, current_q)

        ctrl = peg_home_ctrl.at[:N_JOINTS].set(safe_qd)
        ctrl = ctrl.at[7].set(f)

        def do_substeps(s):
            @jax.checkpoint
            def sub(s, _):
                s = s.replace(ctrl=ctrl)
                s = mjx.step(peg_mjx_model, s)
                return s, None
            s, _ = jax.lax.scan(sub, s, None, length=N_SUBSTEPS)
            return s
        state = jax.lax.cond(step_idx < active_steps,
                             do_substeps, lambda s: s, state)

        ei, hi = PEG_IDX['ee_site'], PEG_IDX['hand_body']

        peg_cfg = {
            'ee_site_idx': ei, 'hand_body_idx': hi,
            'peg_body_idx': PEG_IDX['peg_body'],
            'peg_bottom_site_idx': PEG_IDX['peg_bottom'],
            'peg_grip_site_idx': PEG_IDX['peg_grip'],
            'slot_entry_site_idx': PEG_IDX['slot_entry'],
            'slot_target_site_idx': PEG_IDX['slot_target'],
            'target_quat': DOWN_QUAT,
        }

        rews = [
            reward_reach(state, target_positions[0], DOWN_QUAT, ei, hi),
            reward_descend(state, target_positions[1], DOWN_QUAT, ei, hi),
            reward_grasp(state, target_positions[2], DOWN_QUAT, ei, hi),
            reward_move_generic(state, peg_bottom, lift_target, ei, hi, DOWN_QUAT),
            reward_insert_align(state, peg_cfg),
            reward_insert_push(state, peg_cfg),
            reward_release_generic(state, slot_target, ei, hi, DOWN_QUAT),
        ]
        rew = _select_by_step(rews, step_idx, PEG_BOUNDARIES)
        rew = jnp.where(step_idx < active_steps, rew, 0.0)
        total_rew = total_rew + gamma_acc * rew
        gamma_acc = gamma_acc * GAMMA

        return (state, ee_pos, total_rew, gamma_acc), proprio

    init = (data, data.site_xpos[PEG_IDX['ee_site']],
            jnp.float64(0.0), jnp.float64(1.0))
    (_, _, total_rew, _), all_proprio = jax.lax.scan(
        frame_step, init, jnp.arange(PEG_N_TOTAL))
    return -total_rew, all_proprio


# ---- CONTAINER SORTING LOSS ----
def cont_single_loss(prim_params, obs_mean, obs_std, data, active_steps):
    @jax.checkpoint
    def frame_step(carry, step_idx):
        state, prev_ee, total_rew, gamma_acc = carry

        is_boundary = jnp.zeros((), dtype=bool)
        for b in CONT_BOUNDARIES:
            is_boundary = is_boundary | (step_idx == b)
        state = jax.lax.cond(is_boundary,
            lambda s: jax.lax.stop_gradient(s), lambda s: s, state)
        prev_ee = jax.lax.cond(is_boundary,
            lambda p: jax.lax.stop_gradient(p), lambda p: p, prev_ee)
        gamma_acc = jnp.where(is_boundary, jnp.float64(1.0), gamma_acc)

        # Live positions — 3 objects
        obj_positions = jnp.stack([state.xpos[idx] for idx in CONT_OBJ_IDXS])
        container_pos = state.site_xpos[CONT_IDX['container']]
        place_target = container_pos + jnp.array([0., 0., PLACE_Z_OFFSET])

        # 15 segments = 3 obj × 5 phases
        all_targets = []
        all_prims = []
        all_obj_idxs = []
        for oi in range(3):
            op = obj_positions[oi]
            obj_idx = CONT_OBJ_IDXS[oi]
            all_targets.extend([
                op + jnp.array([0., 0., PRE_GRASP_Z]),  # Reach
                op + jnp.array([0., 0., -0.01]),         # Descend
                op,                                       # Grasp
                place_target,                             # Move
                place_target,                             # Release
            ])
            all_prims.extend([REACH, DESCEND, GRASP, MOVE, RELEASE])
            all_obj_idxs.extend([obj_idx] * 5)

        target_positions = jnp.stack(all_targets)
        prim_indices = jnp.array(all_prims)
        obj_body_indices = jnp.array(all_obj_idxs)

        prim_idx = _select_by_step(
            [prim_indices[i] for i in range(15)],
            step_idx, CONT_BOUNDARIES)
        target_pos = _select_by_step(
            [target_positions[i] for i in range(15)],
            step_idx, CONT_BOUNDARIES)
        obj_idx = _select_by_step(
            [obj_body_indices[i] for i in range(15)],
            step_idx, CONT_BOUNDARIES)

        proprio, ee_pos, current_q = build_proprio(
            state, prev_ee, cont_frame_dt, DOWN_QUAT,
            CONT_IDX['ee_site'], CONT_IDX['hand_body'],
            joint_state=0.0)
        proprio_norm = (proprio.astype(jnp.float32) - obs_mean) / obs_std

        # obj_rel: Reach/Descend → ee-target, Grasp+ → obj-target
        current_obj_pos = state.xpos[obj_idx]
        obj_rel = jnp.where(prim_idx <= DESCEND,
                            ee_pos - target_pos,
                            current_obj_pos - target_pos)

        qd, f = primitive_forward(prim_params, prim_idx,
                                  proprio_norm, target_pos, obj_rel)
        safe_qd = safe_action(qd, current_q)

        ctrl = cont_home_ctrl.at[:N_JOINTS].set(safe_qd)
        ctrl = ctrl.at[7].set(f)

        def do_substeps(s):
            @jax.checkpoint
            def sub(s, _):
                s = s.replace(ctrl=ctrl)
                s = mjx.step(cont_mjx_model, s)
                return s, None
            s, _ = jax.lax.scan(sub, s, None, length=N_SUBSTEPS)
            return s
        state = jax.lax.cond(step_idx < active_steps,
                             do_substeps, lambda s: s, state)

        ei, hi = CONT_IDX['ee_site'], CONT_IDX['hand_body']

        rews = []
        for oi_loop in range(3):
            op = obj_positions[oi_loop]
            oi_body = CONT_OBJ_IDXS[oi_loop]
            obj_p = state.xpos[oi_body]
            rews.extend([
                reward_reach(state, op + jnp.array([0.,0.,PRE_GRASP_Z]),
                             DOWN_QUAT, ei, hi),
                reward_descend(state, op + jnp.array([0.,0.,-0.01]),
                               DOWN_QUAT, ei, hi),
                reward_grasp(state, op, DOWN_QUAT, ei, hi),
                reward_move_generic(state, obj_p, place_target, ei, hi, DOWN_QUAT),
                reward_release_generic(state, place_target, ei, hi, DOWN_QUAT),
            ])

        rew = _select_by_step(rews, step_idx, CONT_BOUNDARIES)
        rew = jnp.where(step_idx < active_steps, rew, 0.0)
        total_rew = total_rew + gamma_acc * rew
        gamma_acc = gamma_acc * GAMMA

        return (state, ee_pos, total_rew, gamma_acc), proprio

    init = (data, data.site_xpos[CONT_IDX['ee_site']],
            jnp.float64(0.0), jnp.float64(1.0))
    (_, _, total_rew, _), all_proprio = jax.lax.scan(
        frame_step, init, jnp.arange(CONT_N_TOTAL))
    return -total_rew, all_proprio


# ---- DRAWER LOSS ----
def drawer_single_loss(prim_params, obs_mean, obs_std, data, active_steps):
    @jax.checkpoint
    def frame_step(carry, step_idx):
        state, prev_ee, total_rew, gamma_acc = carry

        is_boundary = jnp.zeros((), dtype=bool)
        for b in DRAWER_BOUNDARIES:
            is_boundary = is_boundary | (step_idx == b)
        state = jax.lax.cond(is_boundary,
            lambda s: jax.lax.stop_gradient(s), lambda s: s, state)
        prev_ee = jax.lax.cond(is_boundary,
            lambda p: jax.lax.stop_gradient(p), lambda p: p, prev_ee)
        gamma_acc = jnp.where(is_boundary, jnp.float64(1.0), gamma_acc)

        handle_pos = state.site_xpos[DRAWER_IDX['handle_site']]
        drawer_joint = state.qpos[DRAWER_IDX['joint_qpos_idx']]

        target_positions = jnp.stack([
            handle_pos,  # 0 Reach
            handle_pos,  # 1 Grasp
            handle_pos,  # 2 Pull (dynamic)
            handle_pos,  # 3 Release
        ])
        prim_indices = jnp.array(DRAWER_PRIM_SEQ)

        prim_idx = _select_by_step(
            [prim_indices[i] for i in range(4)],
            step_idx, DRAWER_BOUNDARIES)
        target_pos = _select_by_step(
            [target_positions[i] for i in range(4)],
            step_idx, DRAWER_BOUNDARIES)

        proprio, ee_pos, current_q = build_proprio(
            state, prev_ee, drawer_frame_dt, DRAWER_TARGET_QUAT,
            DRAWER_IDX['ee_site'], DRAWER_IDX['hand_body'],
            joint_state=drawer_joint)
        proprio_norm = (proprio.astype(jnp.float32) - obs_mean) / obs_std

        obj_rel = ee_pos - handle_pos

        qd, f = primitive_forward(prim_params, prim_idx,
                                  proprio_norm, target_pos, obj_rel)
        safe_qd = safe_action(qd, current_q)

        ctrl = drawer_home_ctrl.at[:N_JOINTS].set(safe_qd)
        ctrl = ctrl.at[7].set(f)

        def do_substeps(s):
            @jax.checkpoint
            def sub(s, _):
                s = s.replace(ctrl=ctrl)
                s = mjx.step(drawer_mjx_model, s)
                return s, None
            s, _ = jax.lax.scan(sub, s, None, length=N_SUBSTEPS)
            return s
        state = jax.lax.cond(step_idx < active_steps,
                             do_substeps, lambda s: s, state)

        ei = DRAWER_IDX['ee_site']
        hi = DRAWER_IDX['hand_body']

        rews = [
            reward_reach(state, handle_pos, DRAWER_TARGET_QUAT, ei, hi),
            reward_grasp(state, handle_pos, DRAWER_TARGET_QUAT, ei, hi),
            reward_pull(state, DRAWER_IDX['handle_site'],
                       DRAWER_IDX['joint_qpos_idx'], DRAWER_OPEN_TARGET,
                       ei, hi, DRAWER_TARGET_QUAT),
            reward_release_generic(state, handle_pos, ei, hi, DRAWER_TARGET_QUAT),
        ]
        rew = _select_by_step(rews, step_idx, DRAWER_BOUNDARIES)
        rew = jnp.where(step_idx < active_steps, rew, 0.0)
        total_rew = total_rew + gamma_acc * rew
        gamma_acc = gamma_acc * GAMMA

        return (state, ee_pos, total_rew, gamma_acc), proprio

    init = (data, data.site_xpos[DRAWER_IDX['ee_site']],
            jnp.float64(0.0), jnp.float64(1.0))
    (_, _, total_rew, _), all_proprio = jax.lax.scan(
        frame_step, init, jnp.arange(DRAWER_N_TOTAL))
    return -total_rew, all_proprio


# ---- CABINET LOSS ----
def cabinet_single_loss(prim_params, obs_mean, obs_std, data, active_steps):
    @jax.checkpoint
    def frame_step(carry, step_idx):
        state, prev_ee, total_rew, gamma_acc = carry

        is_boundary = jnp.zeros((), dtype=bool)
        for b in CABINET_BOUNDARIES:
            is_boundary = is_boundary | (step_idx == b)
        state = jax.lax.cond(is_boundary,
            lambda s: jax.lax.stop_gradient(s), lambda s: s, state)
        prev_ee = jax.lax.cond(is_boundary,
            lambda p: jax.lax.stop_gradient(p), lambda p: p, prev_ee)
        gamma_acc = jnp.where(is_boundary, jnp.float64(1.0), gamma_acc)

        handle_pos = state.site_xpos[CABINET_IDX['handle_site']]
        hinge_angle = state.qpos[CABINET_IDX['hinge_qpos_idx']]

        target_positions = jnp.stack([
            handle_pos,  # 0 Reach
            handle_pos,  # 1 Grasp
            handle_pos,  # 2 Swing (dynamic)
            handle_pos,  # 3 Release
        ])
        prim_indices = jnp.array(CABINET_PRIM_SEQ)

        prim_idx = _select_by_step(
            [prim_indices[i] for i in range(4)],
            step_idx, CABINET_BOUNDARIES)
        target_pos = _select_by_step(
            [target_positions[i] for i in range(4)],
            step_idx, CABINET_BOUNDARIES)

        proprio, ee_pos, current_q = build_proprio(
            state, prev_ee, cabinet_frame_dt, CABINET_TARGET_QUAT,
            CABINET_IDX['ee_site'], CABINET_IDX['hand_body'],
            joint_state=hinge_angle)
        proprio_norm = (proprio.astype(jnp.float32) - obs_mean) / obs_std

        obj_rel = ee_pos - handle_pos

        qd, f = primitive_forward(prim_params, prim_idx,
                                  proprio_norm, target_pos, obj_rel)
        safe_qd = safe_action(qd, current_q)

        ctrl = cabinet_home_ctrl.at[:N_JOINTS].set(safe_qd)
        ctrl = ctrl.at[7].set(f)

        def do_substeps(s):
            @jax.checkpoint
            def sub(s, _):
                s = s.replace(ctrl=ctrl)
                s = mjx.step(cabinet_mjx_model, s)
                return s, None
            s, _ = jax.lax.scan(sub, s, None, length=N_SUBSTEPS)
            return s
        state = jax.lax.cond(step_idx < active_steps,
                             do_substeps, lambda s: s, state)

        ei = CABINET_IDX['ee_site']
        hi = CABINET_IDX['hand_body']

        rews = [
            reward_reach(state, handle_pos, CABINET_TARGET_QUAT, ei, hi),
            reward_grasp(state, handle_pos, CABINET_TARGET_QUAT, ei, hi),
            reward_swing(state, CABINET_IDX['handle_site'],
                        CABINET_IDX['hinge_qpos_idx'], HINGE_OPEN_TARGET,
                        ei, hi, CABINET_TARGET_QUAT),
            reward_release_generic(state, handle_pos, ei, hi, CABINET_TARGET_QUAT),
        ]
        rew = _select_by_step(rews, step_idx, CABINET_BOUNDARIES)
        rew = jnp.where(step_idx < active_steps, rew, 0.0)
        total_rew = total_rew + gamma_acc * rew
        gamma_acc = gamma_acc * GAMMA

        return (state, ee_pos, total_rew, gamma_acc), proprio

    init = (data, data.site_xpos[CABINET_IDX['ee_site']],
            jnp.float64(0.0), jnp.float64(1.0))
    (_, _, total_rew, _), all_proprio = jax.lax.scan(
        frame_step, init, jnp.arange(CABINET_N_TOTAL))
    return -total_rew, all_proprio

# ═══════════════════════════════════════════════════════════════════════════
# COMBINED LOSS (single backward pass)
# ═══════════════════════════════════════════════════════════════════════════
def combined_loss(prim_params, obs_mean, obs_std,
                  cube_active, peg_active, cont_active,
                  drawer_active, cabinet_active):
    cube_losses, cube_obs = jax.vmap(
        cube_single_loss, in_axes=(None, None, None, 0, None)
    )(prim_params, obs_mean, obs_std, cube_batch, cube_active)

    peg_losses, peg_obs = jax.vmap(
        peg_single_loss, in_axes=(None, None, None, 0, None)
    )(prim_params, obs_mean, obs_std, peg_batch, peg_active)

    cont_losses, cont_obs = jax.vmap(
        cont_single_loss, in_axes=(None, None, None, 0, None)
    )(prim_params, obs_mean, obs_std, cont_batch, cont_active)

    drawer_losses, drawer_obs = jax.vmap(
        drawer_single_loss, in_axes=(None, None, None, 0, None)
    )(prim_params, obs_mean, obs_std, drawer_batch, drawer_active)

    cabinet_losses, cabinet_obs = jax.vmap(
        cabinet_single_loss, in_axes=(None, None, None, 0, None)
    )(prim_params, obs_mean, obs_std, cabinet_batch, cabinet_active)

    total = (jnp.mean(cube_losses) + jnp.mean(peg_losses) +
             jnp.mean(cont_losses) + jnp.mean(drawer_losses) +
             jnp.mean(cabinet_losses))
    return total, (cube_obs, peg_obs, cont_obs, drawer_obs, cabinet_obs,
                   cube_losses, peg_losses, cont_losses,
                   drawer_losses, cabinet_losses)


# ═══════════════════════════════════════════════════════════════════════════
# JIT COMPILE
# ═══════════════════════════════════════════════════════════════════════════
print("JIT compiling combined loss + grad...")
t0 = time.time()
value_and_grad_fn = jax.jit(jax.value_and_grad(combined_loss, has_aux=True))

obs_mean, obs_std = obs_rms.get_jnp()
cube_active = jnp.int32(CUBE_N_TOTAL)
peg_active = jnp.int32(PEG_N_TOTAL)
cont_active = jnp.int32(CONT_N_TOTAL)
drawer_active = jnp.int32(DRAWER_N_TOTAL)
cabinet_active = jnp.int32(CABINET_N_TOTAL)

(loss_val, aux), grad_val = value_and_grad_fn(
    prim_params, obs_mean, obs_std,
    cube_active, peg_active, cont_active, drawer_active, cabinet_active)
loss_val.block_until_ready()
print(f"  JIT: {time.time()-t0:.1f}s, loss: {float(loss_val):.4f}")

# Seed obs_rms
all_proprio = np.concatenate([
    np.array(aux[0]).reshape(-1, PROPRIO_DIM),
    np.array(aux[1]).reshape(-1, PROPRIO_DIM),
    np.array(aux[2]).reshape(-1, PROPRIO_DIM),
    np.array(aux[3]).reshape(-1, PROPRIO_DIM),
    np.array(aux[4]).reshape(-1, PROPRIO_DIM),
], axis=0)
obs_rms.update(all_proprio)

# ═══════════════════════════════════════════════════════════════════════════
# FORWARD ROLLOUT FUNCTIONS (logging + viewer)
# ═══════════════════════════════════════════════════════════════════════════

def _make_forward_fn(mjx_model, home_ctrl, frame_dt, n_total,
                     boundaries, ee_site, hand_body,
                     get_step_info_fn, get_joint_state_fn=None):
    def forward_fn(prim_params, obs_mean, obs_std, data, active_steps):
        def frame_step(carry, step_idx):
            state, prev_ee = carry

            prim_idx, target_pos, target_quat, obj_pos = get_step_info_fn(
                state, step_idx, boundaries)

            js = get_joint_state_fn(state) if get_joint_state_fn else 0.0

            proprio, ee_pos, current_q = build_proprio(
                state, prev_ee, frame_dt, target_quat, ee_site, hand_body,
                joint_state=js)
            proprio_norm = (proprio.astype(jnp.float32) - obs_mean) / obs_std

            # obj_rel: Reach/Descend → ee-target, Grasp+ → obj-target
            obj_rel = jnp.where(prim_idx <= DESCEND,
                                ee_pos - target_pos,
                                obj_pos - target_pos)

            qd, f = primitive_forward(prim_params, prim_idx,
                                      proprio_norm, target_pos, obj_rel)
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
                return s, jnp.broadcast_to(s.qpos, (N_SUBSTEPS,)+s.qpos.shape)

            state, step_qpos = jax.lax.cond(
                step_idx < active_steps, do_sub, skip_sub, state)

            ee_new = state.site_xpos[ee_site]
            return (state, ee_pos), (state, step_qpos, f)

        init = (data, data.site_xpos[ee_site])
        (final, _), (all_states, all_qpos, all_fingers) = jax.lax.scan(
            frame_step, init, jnp.arange(n_total))
        return all_states, all_qpos, all_fingers
    return forward_fn


# Cube step info helper
def cube_step_info(state, step_idx, boundaries):
    cube_pos = state.xpos[CUBE_IDX['cube1']]
    cube2_pos = state.xpos[CUBE_IDX['cube2']]
    stack_t = cube2_pos + jnp.array([0., 0., CUBE2_HEIGHT + STACK_OFFSET])
    targets = jnp.stack([
        cube_pos + jnp.array([0., 0., PRE_GRASP_Z]),
        cube_pos + jnp.array([0., 0., -0.01]),
        cube_pos, stack_t, stack_t])
    prim_ids = jnp.array(CUBE_PRIM_SEQ)
    idx = _select_by_step([prim_ids[i] for i in range(5)],
                          step_idx, boundaries)
    pos = _select_by_step([targets[i] for i in range(5)],
                          step_idx, boundaries)
    return idx, pos, DOWN_QUAT, cube_pos

# Peg step info helper
def peg_step_info(state, step_idx, boundaries):
    pg = state.site_xpos[PEG_IDX['peg_grip']]
    peg_pos = state.xpos[PEG_IDX['peg_body']]
    se = state.site_xpos[PEG_IDX['slot_entry']]
    st = state.site_xpos[PEG_IDX['slot_target']]
    lt = jnp.array([se[0], se[1], LIFT_Z_TARGET])
    targets = jnp.stack([
        pg + jnp.array([0., 0., PRE_GRASP_Z]), pg, pg, lt, se, st, st])
    prim_ids = jnp.array(PEG_PRIM_SEQ)
    idx = _select_by_step([prim_ids[i] for i in range(7)],
                          step_idx, boundaries)
    pos = _select_by_step([targets[i] for i in range(7)],
                          step_idx, boundaries)
    return idx, pos, DOWN_QUAT, peg_pos

# Container step info helper
def cont_step_info(state, step_idx, boundaries):
    obj_positions = jnp.stack([state.xpos[idx] for idx in CONT_OBJ_IDXS])
    container_pos = state.site_xpos[CONT_IDX['container']]
    place_t = container_pos + jnp.array([0., 0., PLACE_Z_OFFSET])
    all_targets = []
    all_obj_pos = []
    for oi in range(3):
        op = obj_positions[oi]
        all_targets.extend([
            op + jnp.array([0., 0., PRE_GRASP_Z]),
            op + jnp.array([0., 0., -0.01]),
            op, place_t, place_t])
        all_obj_pos.extend([op] * 5)
    targets = jnp.stack(all_targets)
    obj_pos_stack = jnp.stack(all_obj_pos)
    prim_ids = jnp.array(CONT_PRIM_SEQ)
    idx = _select_by_step([prim_ids[i] for i in range(15)],
                          step_idx, boundaries)
    pos = _select_by_step([targets[i] for i in range(15)],
                          step_idx, boundaries)
    obj_pos = _select_by_step([obj_pos_stack[i] for i in range(15)],
                              step_idx, boundaries)
    return idx, pos, DOWN_QUAT, obj_pos

def drawer_step_info(state, step_idx, boundaries):
    handle_pos = state.site_xpos[DRAWER_IDX['handle_site']]
    targets = jnp.stack([handle_pos, handle_pos, handle_pos, handle_pos])
    prim_ids = jnp.array(DRAWER_PRIM_SEQ)
    idx = _select_by_step([prim_ids[i] for i in range(4)],
                          step_idx, boundaries)
    pos = _select_by_step([targets[i] for i in range(4)],
                          step_idx, boundaries)
    return idx, pos, DRAWER_TARGET_QUAT, handle_pos

def cabinet_step_info(state, step_idx, boundaries):
    handle_pos = state.site_xpos[CABINET_IDX['handle_site']]
    targets = jnp.stack([handle_pos, handle_pos, handle_pos, handle_pos])
    prim_ids = jnp.array(CABINET_PRIM_SEQ)
    idx = _select_by_step([prim_ids[i] for i in range(4)],
                          step_idx, boundaries)
    pos = _select_by_step([targets[i] for i in range(4)],
                          step_idx, boundaries)
    return idx, pos, CABINET_TARGET_QUAT, handle_pos

cube_fwd = jax.jit(_make_forward_fn(
    cube_mjx_model, cube_home_ctrl, cube_frame_dt, CUBE_N_TOTAL,
    CUBE_BOUNDARIES, CUBE_IDX['ee_site'], CUBE_IDX['hand_body'],
    cube_step_info))

peg_fwd = jax.jit(_make_forward_fn(
    peg_mjx_model, peg_home_ctrl, peg_frame_dt, PEG_N_TOTAL,
    PEG_BOUNDARIES, PEG_IDX['ee_site'], PEG_IDX['hand_body'],
    peg_step_info))

cont_fwd = jax.jit(_make_forward_fn(
    cont_mjx_model, cont_home_ctrl, cont_frame_dt, CONT_N_TOTAL,
    CONT_BOUNDARIES, CONT_IDX['ee_site'], CONT_IDX['hand_body'],
    cont_step_info))

drawer_fwd = jax.jit(_make_forward_fn(
    drawer_mjx_model, drawer_home_ctrl, drawer_frame_dt, DRAWER_N_TOTAL,
    DRAWER_BOUNDARIES, DRAWER_IDX['ee_site'], DRAWER_IDX['hand_body'],
    drawer_step_info,
    get_joint_state_fn=lambda s: s.qpos[DRAWER_IDX['joint_qpos_idx']]))

cabinet_fwd = jax.jit(_make_forward_fn(
    cabinet_mjx_model, cabinet_home_ctrl, cabinet_frame_dt, CABINET_N_TOTAL,
    CABINET_BOUNDARIES, CABINET_IDX['ee_site'], CABINET_IDX['hand_body'],
    cabinet_step_info,
    get_joint_state_fn=lambda s: s.qpos[CABINET_IDX['hinge_qpos_idx']]))

# Container env 0 for forward calls
cont_data0 = jax.tree.map(lambda x: x[0], cont_batch)
cube_data0 = jax.tree.map(lambda x: x[0], cube_batch)
peg_data0 = jax.tree.map(lambda x: x[0], peg_batch)
drawer_data0 = jax.tree.map(lambda x: x[0], drawer_batch)
cabinet_data0 = jax.tree.map(lambda x: x[0], cabinet_batch)

print("\nJIT compiling forward rollouts...")
t0 = time.time()
_ = cube_fwd(prim_params, obs_mean, obs_std, cube_data0, cube_active)
_ = peg_fwd(prim_params, obs_mean, obs_std, peg_data0, peg_active)
_ = cont_fwd(prim_params, obs_mean, obs_std, cont_data0, cont_active)
_ = drawer_fwd(prim_params, obs_mean, obs_std, drawer_data0, drawer_active)
_ = cabinet_fwd(prim_params, obs_mean, obs_std, cabinet_data0, cabinet_active)
print(f"  Forward JIT: {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════
# VIEWERS (all tasks)
# ═══════════════════════════════════════════════════════════════════════════
USE_VIEWER = not args.no_viewer
viewers = {}
if USE_VIEWER:
    from mujoco import viewer as mj_viewer
    mj_m = mujoco.MjModel.from_xml_path(args.cabinet_xml)
    mj_d = mujoco.MjData(mj_m)
    mujoco.mj_resetDataKeyframe(mj_m, mj_d, mj_m.keyframe("home").id)
    mujoco.mj_forward(mj_m, mj_d)
    handle = mj_viewer.launch_passive(mj_m, mj_d)
    viewers['cabinet'] = {'model': mj_m, 'data': mj_d, 'handle': handle}
    print("✓ cabinet viewer\n")

def render_trajectory(task_name, all_qpos, sim_dt, skip=2):
    if not USE_VIEWER or task_name not in viewers:
        return
    v = viewers[task_name]
    if not v['handle'].is_running():
        return
    qpos_np = np.array(all_qpos)
    for step in range(qpos_np.shape[0]):
        for sub in range(0, qpos_np.shape[1], skip):
            v['data'].qpos[:] = qpos_np[step, sub]
            mujoco.mj_forward(v['model'], v['data'])
            v['handle'].sync()
            time.sleep(sim_dt * skip)


# ═══════════════════════════════════════════════════════════════════════════
# PROGRESSIVE TRAINING STATE
# ═══════════════════════════════════════════════════════════════════════════
PATIENCE = 4

# Per-task phase tracking
cube_phase = 0;  cube_patience = 0
peg_phase = 0;   peg_patience = 0
cont_phase = 0;  cont_patience = 0
drawer_phase = 0;  drawer_patience = 0
cabinet_phase = 0; cabinet_patience = 0

cube_active = jnp.int32(CUBE_PHASE_BOUNDS[min(1, len(CUBE_PHASE_BOUNDS)-1)])
peg_active = jnp.int32(PEG_PHASE_BOUNDS[min(1, len(PEG_PHASE_BOUNDS)-1)])
cont_active = jnp.int32(CONT_PHASE_BOUNDS[min(1, len(CONT_PHASE_BOUNDS)-1)])
drawer_active = jnp.int32(DRAWER_PHASE_BOUNDS[min(1, len(DRAWER_PHASE_BOUNDS)-1)])
cabinet_active = jnp.int32(CABINET_PHASE_BOUNDS[min(1, len(CABINET_PHASE_BOUNDS)-1)])

# Phase criteria: (metric_type, threshold)
CUBE_CRITERIA = [
    ("dist", 0.04),   # 0 Reach: EE→cube
    ("dist", 0.04),   # 1 Descend: EE→cube
    ("dist", 0.04),   # 2 Grasp: EE→cube
    ("dist", 0.04),   # 3 Move: cube→stack
    ("always", 0),    # 4 Release
]
PEG_CRITERIA = [
    ("dist", 0.04),   # 0 Reach: EE→peg
    ("dist", 0.04),   # 1 Descend: EE→peg
    ("dist", 0.04),   # 2 Grasp: EE→peg
    ("lift", 0.15),   # 3 Move: peg_z > 0.15
    ("dist", 0.03),   # 4 Insert: pb→slot
    ("dist", 0.03),   # 5 Insert: pb→slot_target
    ("always", 0),    # 6 Release
]
CONT_CRITERIA = ([
    ("dist", 0.04), ("dist", 0.04), ("dist", 0.04),
    ("dist", 0.125), ("always", 0),
] * 3)

DRAWER_CRITERIA = [
    ("dist", 0.04),      # 0 Reach: EE→handle
    ("dist", 0.04),      # 1 Grasp: EE→handle
    ("drawer", -0.10),   # 2 Pull: drawer_pos < -0.10
    ("always", 0),       # 3 Release
]
CABINET_CRITERIA = [
    ("dist", 0.04),      # 0 Reach: EE→handle
    ("dist", 0.04),      # 1 Grasp: EE→handle
    ("hinge", -0.8),     # 2 Swing: hinge < -0.8
    ("always", 0),       # 3 Release
]

# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════
loss_history = []
cube_loss_hist = []
peg_loss_hist = []
cont_loss_hist = []
grad_norm_hist = []
drawer_loss_hist = []
cabinet_loss_hist = []

print("\nTraining is about to begin...\n")
train_start = time.time()

for iteration in range(args.iters):
    if USE_VIEWER and not any(v['handle'].is_running() for v in viewers.values()):
        print("\nAll viewers closed.")
        break

    obs_mean, obs_std = obs_rms.get_jnp()

    # ── Single backward pass ──
    (loss_val, aux), grad_val = value_and_grad_fn(
        prim_params, obs_mean, obs_std,
        cube_active, peg_active, cont_active,
        drawer_active, cabinet_active)

    # ── Per-primitive clip + per-primitive optimizer update ──
    new_params = {}
    for name in PRIM_NAMES:
        prim_grad = grad_val[name]
        prim_norm = optax.global_norm(prim_grad)
        scale = jnp.minimum(1.0, GRAD_CLIP / (prim_norm + 1e-6))
        prim_grad = jax.tree.map(lambda g: g * scale, prim_grad)

        updates, opt_states[name] = optimizers[name].update(
            prim_grad, opt_states[name], prim_params[name])
        new_params[name] = optax.apply_updates(prim_params[name], updates)
    prim_params = new_params

    # ── Bookkeeping ──
    cube_obs, peg_obs, cont_obs = aux[0], aux[1], aux[2]
    drawer_obs, cabinet_obs = aux[3], aux[4]
    cube_l, peg_l, cont_l = aux[5], aux[6], aux[7]
    drawer_l, cabinet_l = aux[8], aux[9]

    loss_f = float(loss_val)
    raw_grad_norm = float(optax.global_norm(grad_val))

    # Per-primitive gradient norms (raw before clipping)
    prim_grad_norms = {}
    for name in PRIM_NAMES:
        prim_grad_norms[name] = float(optax.global_norm(grad_val[name]))

    all_proprio = np.concatenate([
        np.array(cube_obs).reshape(-1, PROPRIO_DIM),
        np.array(peg_obs).reshape(-1, PROPRIO_DIM),
        np.array(cont_obs).reshape(-1, PROPRIO_DIM),
        np.array(drawer_obs).reshape(-1, PROPRIO_DIM),
        np.array(cabinet_obs).reshape(-1, PROPRIO_DIM),
    ], axis=0)
    obs_rms.update(all_proprio)

    if np.isnan(loss_f):
        print(f"\n⚠ NaN at iter {iteration}!")
        break

    loss_history.append(loss_f)
    cube_loss_hist.append(float(-jnp.mean(cube_l)))
    peg_loss_hist.append(float(-jnp.mean(peg_l)))
    cont_loss_hist.append(float(-jnp.mean(cont_l)))
    drawer_loss_hist.append(float(-jnp.mean(drawer_l)))
    cabinet_loss_hist.append(float(-jnp.mean(cabinet_l)))
    grad_norm_hist.append(raw_grad_norm)
    if not hasattr(args, '_prim_grad_hists'):
        args._prim_grad_hists = {name: [] for name in PRIM_NAMES}
    for name in PRIM_NAMES:
        args._prim_grad_hists[name].append(prim_grad_norms[name])

    if iteration % 5 == 0 or iteration < 3:
        obs_mean_l, obs_std_l = obs_rms.get_jnp()

        # ── Cube forward ──
        cube_states, cube_qpos, cube_fng = cube_fwd(
            prim_params, obs_mean_l, obs_std_l, cube_data0, cube_active)
        c_ee = np.array(cube_states.site_xpos[:, CUBE_IDX['ee_site']])
        c_c1 = np.array(cube_states.xpos[:, CUBE_IDX['cube1']])
        c_c2 = np.array(cube_states.xpos[:, CUBE_IDX['cube2']])
        c_fng = np.array(cube_fng)

        c_seg_end = [b-1 for b in CUBE_PHASE_BOUNDS]
        c_metrics = []
        for si, se in enumerate(c_seg_end):
            if si == 0:  # Reach → ee vs pre-grasp
                c_metrics.append(np.linalg.norm(
                    c_ee[se] - (c_c1[se] + np.array([0,0,PRE_GRASP_Z]))))
            elif si == 1:  # Descend → ee vs descend target
                c_metrics.append(np.linalg.norm(
                    c_ee[se] - (c_c1[se] + np.array([0,0,-0.01]))))
            elif si == 2:  # Grasp → ee vs cube
                c_metrics.append(np.linalg.norm(c_ee[se] - c_c1[se]))
            else:  # Move, Release → cube vs stack
                stack_t = c_c2[se] + np.array([0,0,CUBE2_HEIGHT+STACK_OFFSET])
                c_metrics.append(np.linalg.norm(c_c1[se] - stack_t))

        # ── Peg forward ──
        peg_states, peg_qpos, peg_fng = peg_fwd(
            prim_params, obs_mean_l, obs_std_l, peg_data0, peg_active)
        p_ee = np.array(peg_states.site_xpos[:, PEG_IDX['ee_site']])
        p_peg = np.array(peg_states.xpos[:, PEG_IDX['peg_body']])
        p_pb = np.array(peg_states.site_xpos[:, PEG_IDX['peg_bottom']])
        p_se = np.array(peg_states.site_xpos[:, PEG_IDX['slot_entry']])
        p_st = np.array(peg_states.site_xpos[:, PEG_IDX['slot_target']])
        p_fng = np.array(peg_fng)

        p_seg_end = [b-1 for b in PEG_PHASE_BOUNDS]
        p_metrics = []
        p_pg = np.array(peg_states.site_xpos[:, PEG_IDX['peg_grip']])
        for si, se in enumerate(p_seg_end):
            if si == 0:  # Reach → ee vs peg_grip + pre_grasp
                c_target = p_pg[se] + np.array([0,0,PRE_GRASP_Z])
                p_metrics.append(np.linalg.norm(p_ee[se] - c_target))
            elif si == 1:  # Descend → ee vs peg_grip
                p_metrics.append(np.linalg.norm(p_ee[se] - p_pg[se]))
            elif si == 2:  # Grasp → ee vs peg_grip
                p_metrics.append(np.linalg.norm(p_ee[se] - p_pg[se]))
            elif si == 3:  # Move → peg z height
                p_metrics.append(p_peg[se][2])
            elif si == 4:  # Insert align → peg_bottom vs slot_entry
                p_metrics.append(np.linalg.norm(p_pb[se] - p_se[se]))
            elif si == 5:  # Insert push → peg_bottom vs slot_target
                p_metrics.append(np.linalg.norm(p_pb[se] - p_st[se]))
            else:  # Release → finger
                p_metrics.append(p_fng[se])

        # ── Container forward ──
        cont_states, cont_qpos_all, cont_fng = cont_fwd(
            prim_params, obs_mean_l, obs_std_l, cont_data0, cont_active)
        ct_ee = np.array(cont_states.site_xpos[:, CONT_IDX['ee_site']])
        ct_objs = [np.array(cont_states.xpos[:, idx])
                   for idx in CONT_OBJ_IDXS]
        ct_cont = np.array(
            cont_states.site_xpos[:, CONT_IDX['container']])
        ct_fng = np.array(cont_fng)

        ct_seg_end = [b-1 for b in CONT_PHASE_BOUNDS]
        ct_metrics = []
        for si, se in enumerate(ct_seg_end):
            oi = si // 5
            phase = si % 5
            if phase == 0:  # Reach → ee vs obj + pre_grasp
                ct_metrics.append(np.linalg.norm(
                    ct_ee[se] - (ct_objs[oi][se] + np.array([0,0,PRE_GRASP_Z]))))
            elif phase == 1:  # Descend → ee vs obj + (-0.01)
                ct_metrics.append(np.linalg.norm(
                    ct_ee[se] - (ct_objs[oi][se] + np.array([0,0,-0.01]))))
            elif phase == 2:  # Grasp → ee vs obj
                ct_metrics.append(np.linalg.norm(ct_ee[se] - ct_objs[oi][se]))
            elif phase == 3:  # Move → obj xy vs container xy
                ct_metrics.append(np.linalg.norm(
                    ct_objs[oi][se][:2] - ct_cont[se][:2]))
            else:  # Release → obj vs container
                ct_metrics.append(np.linalg.norm(
                    ct_objs[oi][se] - ct_cont[se]))
                
        # ── Drawer forward ──
        drawer_states, drawer_qpos, drawer_fng = drawer_fwd(
            prim_params, obs_mean_l, obs_std_l, drawer_data0, drawer_active)
        d_ee = np.array(drawer_states.site_xpos[:, DRAWER_IDX['ee_site']])
        d_handle = np.array(drawer_states.site_xpos[:, DRAWER_IDX['handle_site']])
        d_fng = np.array(drawer_fng)
        d_joint = np.array(drawer_states.qpos[:, DRAWER_IDX['joint_qpos_idx']])

        d_seg_end = [b-1 for b in DRAWER_PHASE_BOUNDS]
        d_metrics = []
        for si, se in enumerate(d_seg_end):
            if si in (0, 1):
                d_metrics.append(np.linalg.norm(d_ee[se] - d_handle[se]))
            elif si == 2:
                d_metrics.append(float(d_joint[se]))
            else:
                d_metrics.append(float(d_fng[se]))

        # ── Cabinet forward ──
        cabinet_states, cabinet_qpos_all, cabinet_fng = cabinet_fwd(
            prim_params, obs_mean_l, obs_std_l, cabinet_data0, cabinet_active)
        cb_ee = np.array(cabinet_states.site_xpos[:, CABINET_IDX['ee_site']])
        cb_handle = np.array(cabinet_states.site_xpos[:, CABINET_IDX['handle_site']])
        cb_fng = np.array(cabinet_fng)
        cb_hinge = np.array(cabinet_states.qpos[:, CABINET_IDX['hinge_qpos_idx']])

        cb_seg_end = [b-1 for b in CABINET_PHASE_BOUNDS]
        cb_metrics = []
        for si, se in enumerate(cb_seg_end):
            if si in (0, 1):
                cb_metrics.append(np.linalg.norm(cb_ee[se] - cb_handle[se]))
            elif si == 2:
                cb_metrics.append(float(cb_hinge[se]))
            else:
                cb_metrics.append(float(cb_fng[se]))

        # ── Print ──
        print(f"\nIter {iteration:4d}:  loss={loss_f:.4f}  "
              f"grad={raw_grad_norm:.1f}")
        print(f"  ┌─ GRADIENTS (raw → clipped to {GRAD_CLIP})")
        for name in PRIM_NAMES:
            raw = prim_grad_norms[name]
            clipped = min(raw, GRAD_CLIP)
            print(f"  │  {name:8s}: raw={raw:10.1f}  "
                  f"clip={clipped:8.1f}")

        # Cube
        prim_labels = ['Reach','Desc','Grasp','Move','Rel']
        print(f"  ┌─ CUBE  rew={cube_loss_hist[-1]:.3f}  "
              f"phase={cube_phase}  active={int(cube_active)}")
        for si in range(5):
            m = "→" if si == cube_phase else " "
            print(f"  │ {m} {prim_labels[si]:5s}: {c_metrics[si]:.4f}  "
                  f"fng={c_fng[c_seg_end[si]]:.4f}")

        # Peg
        prim_labels_p = ['Reach','Desc','Grasp','Move','Ins1','Ins2','Rel']
        print(f"  ├─ PEG   rew={peg_loss_hist[-1]:.3f}  "
              f"phase={peg_phase}  active={int(peg_active)}")
        for si in range(7):
            m = "→" if si == peg_phase else " "
            unit = "z" if si == 3 else ("fng" if si == 6 else "dist")
            print(f"  │ {m} {prim_labels_p[si]:5s}: "
                  f"{unit}={p_metrics[si]:.4f}")

        # Container
        obj_names = ['ball', 'cube', 'prism']
        print(f"  └─ CONT  rew={cont_loss_hist[-1]:.3f}  "
              f"phase={cont_phase}  active={int(cont_active)}")
        for oi in range(3):
            for pi in range(5):
                si = oi * 5 + pi
                m = "→" if si == cont_phase else " "
                print(f"    {m} {obj_names[oi]:5s}/{prim_labels[pi]:5s}: "
                      f"{ct_metrics[si]:.4f}")
            
        # Drawer
        drawer_labels = ['Reach','Grasp','Pull','Rel']
        print(f"  ├─ DRAWER rew={drawer_loss_hist[-1]:.3f}  "
              f"phase={drawer_phase}  active={int(drawer_active)}")
        for si in range(4):
            m = "→" if si == drawer_phase else " "
            unit = "joint" if si == 2 else ("fng" if si == 3 else "dist")
            print(f"  │ {m} {drawer_labels[si]:5s}: "
                  f"{unit}={d_metrics[si]:.4f}")

        # Cabinet
        cabinet_labels = ['Reach','Grasp','Swing','Rel']
        print(f"  └─ CABINET rew={cabinet_loss_hist[-1]:.3f}  "
              f"phase={cabinet_phase}  active={int(cabinet_active)}")
        for si in range(4):
            m = "→" if si == cabinet_phase else " "
            unit = "hinge" if si == 2 else ("fng" if si == 3 else "dist")
            print(f"    {m} {cabinet_labels[si]:5s}: "
                  f"{unit}={cb_metrics[si]:.4f}")

        render_trajectory('cabinet', cabinet_qpos_all, cabinet_mj.opt.timestep, skip=2)
        
        # ── Progressive advancement (metric-based) ──
        def _check_advance(metrics, phase, criteria, patience_cnt,
                           phase_bounds, label):
            if phase >= len(phase_bounds) - 1:
                return phase, patience_cnt, None
            crit_type, thresh = criteria[phase]
            met = metrics[phase]
            if crit_type == "always":
                satisfied = True
            elif crit_type == "lift":
                satisfied = met > thresh
            elif crit_type in ("drawer", "hinge"):
                satisfied = met <= thresh
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
                new_active = jnp.int32(
                    phase_bounds[min(phase + 1, len(phase_bounds) - 1)])
                print(f"  → {label} phase {phase}, active={int(new_active)}")
            return phase, patience_cnt, new_active

        cube_phase, cube_patience, ca = _check_advance(
            c_metrics, cube_phase, CUBE_CRITERIA, cube_patience,
            CUBE_PHASE_BOUNDS, "Cube")
        if ca is not None: cube_active = ca

        peg_phase, peg_patience, pa = _check_advance(
            p_metrics, peg_phase, PEG_CRITERIA, peg_patience,
            PEG_PHASE_BOUNDS, "Peg")
        if pa is not None: peg_active = pa

        cont_phase, cont_patience, cta = _check_advance(
            ct_metrics, cont_phase, CONT_CRITERIA, cont_patience,
            CONT_PHASE_BOUNDS, "Cont")
        if cta is not None: cont_active = cta

        drawer_phase, drawer_patience, da = _check_advance(
            d_metrics, drawer_phase, DRAWER_CRITERIA, drawer_patience,
            DRAWER_PHASE_BOUNDS, "Drawer")
        if da is not None: drawer_active = da

        cabinet_phase, cabinet_patience, cba = _check_advance(
            cb_metrics, cabinet_phase, CABINET_CRITERIA, cabinet_patience,
            CABINET_PHASE_BOUNDS, "Cabinet")
        if cba is not None: cabinet_active = cba

    if iteration % 50 == 0:
        plt.close('all')
        gc.collect()

train_time = time.time() - train_start
print(f"\nTraining done: {train_time:.1f}s ({len(loss_history)} iters)")

# ═══════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════
with open("primitive_params.pkl", "wb") as f:
    pickle.dump(jax.tree.map(lambda x: np.array(x), prim_params), f)
print("Model saved: primitive_params.pkl")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(5, 2, figsize=(14, 20))
axes = axes.flatten()

axes[0].plot([-l for l in loss_history], linewidth=2, color='black')
axes[0].set_title("Combined Reward")
axes[0].grid(True, alpha=0.3)

prim_colors = {'reach': 'tab:blue', 'descend': 'tab:cyan',
               'grasp': 'tab:green', 'move': 'tab:red',
               'insert_align': 'tab:orange', 'insert_push': 'gold',
               'release': 'tab:purple', 'pull': 'tab:brown',
               'swing': 'tab:pink'}
for name in PRIM_NAMES:
    axes[1].plot(args._prim_grad_hists[name], linewidth=1,
                 color=prim_colors[name], label=name, alpha=0.8)
axes[1].set_title("Per-Primitive Gradient Norm (raw)")
axes[1].legend(fontsize=7)
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3)

axes[2].plot(cube_loss_hist, linewidth=1.5, color='green')
axes[2].set_title("Cube Stacking Reward")
axes[2].grid(True, alpha=0.3)

axes[3].plot(peg_loss_hist, linewidth=1.5, color='orange')
axes[3].set_title("Peg Insertion Reward")
axes[3].grid(True, alpha=0.3)

axes[4].plot(cont_loss_hist, linewidth=1.5, color='blue')
axes[4].set_title("Container Sorting Reward")
axes[4].grid(True, alpha=0.3)

axes[5].plot(drawer_loss_hist, linewidth=1.5, color='brown')
axes[5].set_title("Drawer Reward")
axes[5].grid(True, alpha=0.3)

axes[6].plot(cabinet_loss_hist, linewidth=1.5, color='deeppink')
axes[6].set_title("Cabinet Reward")
axes[6].grid(True, alpha=0.3)

# All tasks overlaid
axes[7].plot(cube_loss_hist, linewidth=1.5, color='green', label='cube')
axes[7].plot(peg_loss_hist, linewidth=1.5, color='orange', label='peg')
axes[7].plot(cont_loss_hist, linewidth=1.5, color='blue', label='cont')
axes[7].plot(drawer_loss_hist, linewidth=1.5, color='brown', label='drawer')
axes[7].plot(cabinet_loss_hist, linewidth=1.5, color='deeppink', label='cabinet')
axes[7].set_title("All Task Rewards")
axes[7].legend(fontsize=7)
axes[7].grid(True, alpha=0.3)

# Total gradient norm
axes[8].plot(grad_norm_hist, linewidth=1.5, color='red', alpha=0.3,
             label='raw total')
axes[8].axhline(y=GRAD_CLIP, color='black', linestyle='--', alpha=0.5,
                label=f'clip={GRAD_CLIP}')
axes[8].set_title("Total Gradient Norm")
axes[8].legend(fontsize=8)
axes[8].grid(True, alpha=0.3)

axes[9].set_visible(False)

fig.suptitle(f"Primitive Library — 9 Prims × 5 Tasks, "
             f"{len(loss_history)} iters, clip={GRAD_CLIP}", fontsize=14)
fig.tight_layout()
fig.savefig("primitive_training_results.png", dpi=150)
plt.close(fig)
print(f"Plot saved: primitive_training_results.png")