"""
MJX BPTT — 3-Segment Grasp (Single MLP + One-Hot Phase)

Tek MLP, 3 segment, one-hot phase input:
  Segment 0 (step 0..23):   Pre-grasp — küpün üstüne git, finger açık
  Segment 1 (step 24..35):  Descend — küpe in, finger açık
  Segment 2 (step 36..41):  Close — finger kapat, küpü tut

stop_gradient her segment sınırında — gradient segmentler arası akmaz.

Usage:
    python mjx_franka_bptt_3seg.py
    python mjx_franka_bptt_3seg.py --seg0 24 --seg1 12 --seg2 6 --iters 400
"""
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")
import argparse
import time
import numpy as np
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
parser.add_argument("--seg0", type=int, default=20, help="Pre-grasp steps")
parser.add_argument("--seg1", type=int, default=12, help="Descend steps")
parser.add_argument("--seg2", type=int, default=6, help="Close steps")
parser.add_argument("--seg3", type=int, default=24, help="Move steps")
parser.add_argument("--seg4", type=int, default=8, help="Align+Release steps")
parser.add_argument("--substeps", type=int, default=4)
parser.add_argument("--iters", type=int, default=900)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--gamma", type=float, default=0.999)
parser.add_argument("--solver-iters", type=int, default=8)
parser.add_argument("--grad-clip", type=float, default=10.0)
parser.add_argument("--no-viewer", action="store_true")
parser.add_argument("--xml", type=str,
                    default="/home/emre/mjx_diffsim/franka_emika_panda2/"
                            "mjx_single_cube.xml")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_SEG0      = args.seg0       # pre-grasp
N_SEG1      = args.seg1       # descend
N_SEG2      = args.seg2       # close
N_SEG3      = args.seg3
N_SEG4      = args.seg4
N_TOTAL     = N_SEG0 + N_SEG1 + N_SEG2 + N_SEG3 + N_SEG4
SEG1_START  = N_SEG0
SEG2_START  = N_SEG0 + N_SEG1
SEG3_START  = N_SEG0 + N_SEG1 + N_SEG2
SEG4_START  = N_SEG0 + N_SEG1 + N_SEG2 + N_SEG3
N_SUBSTEPS  = args.substeps
TRAIN_ITERS = args.iters
TRAIN_LR    = args.lr
GAMMA       = args.gamma
GRAD_CLIP   = args.grad_clip
N_JOINTS    = 7


FINGER_OPEN   = 0.04
GRASP_Z_OFFSET     = 0.0 
PRE_GRASP_Z_OFFSET = 0.075   # seg1: küpün kendisi
CUBE2_HEIGHT = 0.05  # küp yarı boyutu
STACK_OFFSET = 0.01  # cube1 yarı boyutu

Q_LO = jnp.array([-2.897, -1.763, -2.897, -3.072, -2.897, -0.018, -2.897])
Q_HI = jnp.array([ 2.897,  1.763,  2.897, -0.070,  2.897,  3.752,  2.897])
VEL_LIMITS = jnp.array([2.17, 2.17, 2.17, 2.17, 2.61, 2.61, 2.61])

TARGET_POS  = jnp.array([0.4, 0.15, 0.55])
TARGET_QUAT = jnp.array([0.0, 0.7071068, 0.7071068, 0.0])

OBS_DIM = 54  # ee(3) + cube_pos(3) + ee_to_cube(3) + ee_vel(3) + finger(1) 
                #+ q(7) + qdot(7) + ee_quat(4) + quat_err(4) + cube_quat(4) + one_hot(5)
                #+ cube2_pos(3) + cube2_quat(4) + cube1_tocube2 (3)
ACT_DIM = N_JOINTS + 1
HIDDEN  = 128

# ---------------------------------------------------------------------------
# Shaped distance function
# ---------------------------------------------------------------------------
def shaped_distance(a, b, s, t):
    dist = jnp.sqrt(jnp.dot(a - b, a - b) + 1e-6)
    scale = 1.8318 / jnp.maximum(s, 1e-6)
    shaped = (1.0 - jnp.tanh(dist * scale)) ** 2
    return jnp.where(dist < t, 1.0, shaped)

# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------
def init_mlp(key):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        'w1': jax.random.normal(k1, (OBS_DIM, HIDDEN)) * jnp.sqrt(2.0 / OBS_DIM).astype(jnp.float32),
        'b1': jnp.zeros(HIDDEN, dtype=jnp.float32),
        'w2': jax.random.normal(k2, (HIDDEN, HIDDEN)) * jnp.sqrt(2.0 / HIDDEN).astype(jnp.float32),
        'b2': jnp.zeros(HIDDEN, dtype=jnp.float32),
        'w3': jax.random.normal(k3, (HIDDEN, ACT_DIM)).astype(jnp.float32) * 0.01,
        'b3': jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0], dtype=jnp.float32),  # finger açık başla
    }


def mlp_forward(params, obs):
    obs32 = obs.astype(jnp.float32)
    x = jnp.tanh(obs32 @ params['w1'] + params['b1'])
    x = jnp.tanh(x @ params['w2'] + params['b2'])
    x = x @ params['w3'] + params['b3']
    q_dot_target = jnp.tanh(x[:N_JOINTS]) * VEL_LIMITS
    finger = jax.nn.sigmoid(x[N_JOINTS]) * FINGER_OPEN
    # physics'e geri dönerken float64'e çevrilecek otomatik
    return q_dot_target, finger

# ---------------------------------------------------------------------------
# One-hot encoding
# ---------------------------------------------------------------------------
def get_one_hot(step_idx):
    seg0 = (step_idx < SEG1_START).astype(jnp.float32)
    seg1 = ((step_idx >= SEG1_START) & (step_idx < SEG2_START)).astype(jnp.float32)
    seg2 = ((step_idx >= SEG2_START) & (step_idx < SEG3_START)).astype(jnp.float32)
    seg3 = ((step_idx >= SEG3_START) & (step_idx < SEG4_START)).astype(jnp.float32)
    seg4 = (step_idx >= SEG4_START).astype(jnp.float32)
    return jnp.array([seg0, seg1, seg2, seg3, seg4])

# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------
def build_obs(state, prev_ee_pos, frame_dt, step_idx):
    ee_pos = state.site_xpos[EE_SITE_IDX]
    ee_vel = (ee_pos - prev_ee_pos) / frame_dt
    cube_pos = state.xpos[CUBE1_BODY_IDX]
    cube_quat = state.xquat[CUBE1_BODY_IDX]
    ee_to_cube = cube_pos - ee_pos
    finger_width = state.qpos[7] + state.qpos[8]
    current_q = state.qpos[:N_JOINTS]
    current_q_dot = state.qvel[:N_JOINTS]
    ee_quat_obs = state.xquat[9]
    quat_err = TARGET_QUAT - ee_quat_obs
    cube2_pos = state.xpos[CUBE2_BODY_IDX]
    cube2_quat = state.xquat[CUBE2_BODY_IDX]
    cube1_to_cube2 = cube2_pos - cube_pos  # cube1'den cube2'ye
    one_hot = get_one_hot(step_idx)
    obs = jnp.concatenate([ee_pos, cube_pos, ee_to_cube, ee_vel,
                            jnp.array([finger_width]), current_q, current_q_dot,
                            ee_quat_obs, quat_err, cube_quat, one_hot,
                            cube2_pos, cube2_quat, cube1_to_cube2])
    return obs, ee_pos, current_q

# ---------------------------------------------------------------------------
# Action safety
# ---------------------------------------------------------------------------
def safe_action(q_dot_target, current_q):
    at_min = (current_q <= Q_LO + 0.01)
    at_max = (current_q >= Q_HI - 0.01)
    safe = jnp.where(at_min, jnp.maximum(0, q_dot_target), q_dot_target)
    safe = jnp.where(at_max, jnp.minimum(0, safe), safe)
    return safe

# ---------------------------------------------------------------------------
# Reward functions — 5 segments
# ---------------------------------------------------------------------------
def reward_seg0_pregrasp(state, init_cube_pos):
    ee_pos = state.site_xpos[EE_SITE_IDX]
    cube_pos = state.xpos[CUBE1_BODY_IDX]
    ee_pos_z = ee_pos[2]

    pre_grip_pos = cube_pos + jnp.array([0.0, 0.0, PRE_GRASP_Z_OFFSET])
    delta = ee_pos - pre_grip_pos
    r_approach = -jnp.sqrt(jnp.dot(delta, delta) + 1e-6) * 1.0 + shaped_distance(ee_pos, pre_grip_pos, 0.3, 0.001) * 1.0

    ee_quat = state.xquat[9]
    quat_dot = jnp.dot(ee_quat, TARGET_QUAT)
    ori_loss = (1.0 - quat_dot * quat_dot) * r_approach

    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_open = finger_width * 5.0  # açık = iyi

    ground_penalty = jnp.where(ee_pos_z < 0.1, 5.0 * (0.1 - ee_pos_z) / 0.1, 0.0)

    return r_approach * 2.0 - ori_loss * 5.0 - ground_penalty * 1.0 + r_finger_open


def reward_seg1_descend(state, init_cube_pos):
    ee_pos = state.site_xpos[EE_SITE_IDX]
    cube_pos = state.xpos[CUBE1_BODY_IDX]
    cube_holding_pos = cube_pos + jnp.array([0.0, 0.0, -0.01])

    # EE-cube mesafesi (yakın kalmalı)
    r_approach = shaped_distance(ee_pos, cube_holding_pos, 0.1, 0.001) * 1

    # Orientation koruma
    ee_quat = state.xquat[9]
    quat_dot = jnp.dot(ee_quat, TARGET_QUAT)
    ori_loss = (1.0 - quat_dot * quat_dot) * r_approach

    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_open = finger_width * 1.0  # açık = iyi

    # Ground penalty
    ee_pos_z = ee_pos[2]
    ground_penalty = jnp.where(ee_pos_z < 0.04, 4.0 * (0.04 - ee_pos_z) / 0.04, 0.0)

    return r_approach * 2.0 - ori_loss * 5.0 - ground_penalty + r_finger_open

def reward_seg2_close(state, init_cube_pos):
    """Segment 2: gripper kapat, küpü tut, yerinde kal."""
    ee_pos = state.site_xpos[EE_SITE_IDX]
    cube_pos = state.xpos[CUBE1_BODY_IDX]
    cube_holding_pos = cube_pos + jnp.array([0.0, 0.0, -0.01])

    # EE-cube yakın kalmalı
    r_proximity = shaped_distance(ee_pos, cube_holding_pos, 0.1, 0.001)

    # Gripper kapatma
    finger_width = state.qpos[7] + state.qpos[8]
    r_grip_close = -finger_width * 2.0  # kapalı = iyi

    # # Küp z yerinde kalsın (düşmesin)
    # cube_z_drift = jnp.abs(cube_pos[2] - init_cube_pos[2])
    # r_cube_z = -cube_z_drift * 10.0

    # Orientation koru
    ee_quat = state.xquat[9]
    quat_dot = jnp.dot(ee_quat, TARGET_QUAT)
    ori_loss = (1.0 - quat_dot * quat_dot) * r_proximity

    # Ground penalty
    ee_pos_z = ee_pos[2]
    ground_penalty = jnp.where(ee_pos_z < 0.04, 4.0 * (0.04 - ee_pos_z) / 0.04, 0.0)

    return r_proximity * 2.0 + r_grip_close - ori_loss * 5.0 -ground_penalty

def reward_seg3_move(state, init_cube_pos):
    """Segment 3: küpü hedefe götür, finger kapalı, küp elde kalsın."""
    ee_pos = state.site_xpos[EE_SITE_IDX]
    cube1_pos = state.xpos[CUBE1_BODY_IDX]
    cube2_pos = state.xpos[CUBE2_BODY_IDX]
    cube1_z = state.xpos[CUBE1_BODY_IDX][2]
    cube_holding_pos = cube1_pos + jnp.array([0.0, 0.0, 0.015])

    stack_target = cube2_pos + jnp.array([0.0, 0.0, CUBE2_HEIGHT + STACK_OFFSET])
    min_height = 0.05    # Has it risen 1 cm off the ground?
    max_height = 0.075 + 0.01   # Reward up to 10 cm

    lift_weight = jnp.clip((cube1_z - min_height) / (max_height - min_height), 0.0, 1.0)

    # Küp hedefe yaklaşsın
    delta = cube1_pos - stack_target
    r_cube_to_target = (shaped_distance(cube1_pos, stack_target, 0.5, 0.001) + shaped_distance(cube1_pos, stack_target, 0.1, 0.001)) / 2
    r_cube_to_target = jnp.clip(r_cube_to_target, 0.0, 1.0) * lift_weight

    # Gripper kapalı kalsın
    finger_width = state.qpos[7] + state.qpos[8]
    r_grip_close = -finger_width * 2.0

    # Orientation
    ee_quat = state.xquat[9]
    quat_dot = jnp.dot(ee_quat, TARGET_QUAT)
    ori_loss = 1.0 - quat_dot * quat_dot

    # Ground penalty
    ee_pos_z = ee_pos[2]
    ground_penalty = jnp.where(ee_pos_z < 0.04, 4.0 * (0.04 - ee_pos_z) / 0.04, 0.0)

    return r_cube_to_target * 1.0 + lift_weight + r_grip_close - ori_loss * 3.0 - ground_penalty

def reward_seg4_release(state, init_cube_pos):
    """Segment 4: küpü hizala ve bırak."""
    ee_pos = state.site_xpos[EE_SITE_IDX]
    cube1_pos = state.xpos[CUBE1_BODY_IDX]
    cube2_pos = state.xpos[CUBE2_BODY_IDX]

    stack_target = cube2_pos + jnp.array([0.0, 0.0, CUBE2_HEIGHT + 0.005])

    r_align = shaped_distance(cube1_pos, stack_target, 0.1, 0.001)

    # Gripper açılmalı (bırak)
    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_open = finger_width * r_align

    # Orientation
    ee_quat = state.xquat[9]
    quat_dot = jnp.dot(ee_quat, TARGET_QUAT)
    ori_loss = 1.0 - quat_dot * quat_dot

    return r_align * 2.0 + r_finger_open - ori_loss * 3.0


def select_reward(state, step_idx, init_cube_pos):
    rew0 = reward_seg0_pregrasp(state, init_cube_pos)
    rew1 = reward_seg1_descend(state, init_cube_pos)
    rew2 = reward_seg2_close(state, init_cube_pos)
    rew3 = reward_seg3_move(state, init_cube_pos)
    rew4 = reward_seg4_release(state, init_cube_pos)

    rew = jnp.where(step_idx < SEG1_START, rew0,
          jnp.where(step_idx < SEG2_START, rew1,
          jnp.where(step_idx < SEG3_START, rew2,
          jnp.where(step_idx < SEG4_START, rew3, rew4))))
    return rew

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("\nLoading MuJoCo model...")
mj_model = mujoco.MjModel.from_xml_path(args.xml)
mj_data = mujoco.MjData(mj_model)

CUBE1_BODY_IDX = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "box")
CUBE2_BODY_IDX = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "box2")
EE_SITE_IDX   = 0

key_id = mj_model.keyframe("home").id
mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
mujoco.mj_forward(mj_model, mj_data)

sim_dt = mj_model.opt.timestep
frame_dt = N_SUBSTEPS * sim_dt
total_time = N_TOTAL * frame_dt
total_physics = N_TOTAL * N_SUBSTEPS

print(f"  sim_dt={sim_dt}, frame_dt={frame_dt:.4f}s")
print(f"  Seg0 (pre-grasp): {N_SEG0} steps")
print(f"  Seg1 (descend):   {N_SEG1} steps")
print(f"  Seg2 (close):     {N_SEG2} steps")
print(f"  Total: {N_TOTAL} steps × {N_SUBSTEPS} substeps = {total_physics} physics ({total_time:.3f}s)")
print(f"  EE home (site): {mj_data.site_xpos[EE_SITE_IDX]}")

home_ctrl = jnp.array(mj_data.ctrl)

init_cube_pos = jnp.array(mj_data.xpos[CUBE1_BODY_IDX])
init_cube2_pos = jnp.array(mj_data.xpos[CUBE2_BODY_IDX])
print(f"  Cube pos: {init_cube_pos}")
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
# Initialize
# ---------------------------------------------------------------------------
rng = jax.random.PRNGKey(42)
mlp_params = init_mlp(rng)
n_params = sum(p.size for p in jax.tree.leaves(mlp_params))

optimizer = optax.chain(
    optax.clip_by_global_norm(GRAD_CLIP),
    optax.adam(TRAIN_LR),
)
opt_state = optimizer.init(mlp_params)

ema_params = jax.tree.map(lambda p: p.copy(), mlp_params)
EMA_TAU = 0.6

print(f"\n{'='*60}")
print(f"BPTT 3-SEGMENT — Single MLP + One-Hot Phase")
print(f"  MLP: {OBS_DIM}→{HIDDEN}→{HIDDEN}→{ACT_DIM} ({n_params} params)")
print(f"  Segments: {N_SEG0} + {N_SEG1} + {N_SEG2} = {N_TOTAL} steps")
print(f"  Physics: {total_physics} substeps ({total_time:.3f}s)")
print(f"  LR: {TRAIN_LR}, Gamma: {GAMMA}, Grad clip: {GRAD_CLIP}")
print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# Combined Loss — stop_gradient at segment boundaries
# ---------------------------------------------------------------------------
def combined_loss_fn(mlp_params):
    def frame_step_fn(carry, step_idx):
        state, prev_ee_pos, total_reward, gamma_acc = carry

        # Stop gradient at segment boundaries
        is_boundary = ((step_idx == SEG1_START) | (step_idx == SEG2_START) | 
               (step_idx == SEG3_START) | (step_idx == SEG4_START))
        state = jax.lax.cond(
            is_boundary,
            lambda s: jax.lax.stop_gradient(s),
            lambda s: s,
            state
        )
        prev_ee_pos = jax.lax.cond(
            is_boundary,
            lambda p: jax.lax.stop_gradient(p),
            lambda p: p,
            prev_ee_pos
        )

        gamma_acc = jnp.where(is_boundary, jnp.float32(1.0), gamma_acc)

        # Observation (with one-hot)
        obs, ee_pos, current_q = build_obs(state, prev_ee_pos, frame_dt, step_idx)

        # MLP → action
        q_dot_target, finger_target = mlp_forward(mlp_params, obs)

        # Safety
        safe_q_dot = safe_action(q_dot_target, current_q)

        # Build ctrl
        ctrl = home_ctrl.at[:N_JOINTS].set(safe_q_dot)
        ctrl = ctrl.at[7].set(finger_target)

        # Physics substeps
        def substep_fn(s, _):
            s = s.replace(ctrl=ctrl)
            s = mjx.step(mjx_model, s)
            return s, None

        state, _ = jax.lax.scan(substep_fn, state, None, length=N_SUBSTEPS)

        # Segment-dependent reward
        rew = select_reward(state, step_idx, init_cube_pos)

        total_reward = total_reward + gamma_acc * rew
        gamma_acc = gamma_acc * GAMMA

        return (state, ee_pos, total_reward, gamma_acc), None

    init_carry = (
        mjx_data_init,
        mjx_data_init.site_xpos[EE_SITE_IDX],
        jnp.float64(0.0),
        jnp.float64(1.0),
    )

    (final_state, _, total_reward, _), _ = jax.lax.scan(
        frame_step_fn, init_carry, jnp.arange(N_TOTAL))

    return -total_reward

# ---------------------------------------------------------------------------
# Forward rollout (logging + rendering)
# ---------------------------------------------------------------------------
def combined_forward_fn(mlp_params):
    def frame_step_fn(carry, step_idx):
        state, prev_ee_pos = carry

        obs, ee_pos, current_q = build_obs(state, prev_ee_pos, frame_dt, step_idx)

        q_dot_target, finger_target = mlp_forward(mlp_params, obs)
        safe_q_dot = safe_action(q_dot_target, current_q)

        ctrl = home_ctrl.at[:N_JOINTS].set(safe_q_dot)
        ctrl = ctrl.at[7].set(finger_target)

        def substep_fn(s, _):
            s = s.replace(ctrl=ctrl)
            s = mjx.step(mjx_model, s)
            return s, s.qpos

        state, step_qpos = jax.lax.scan(substep_fn, state, None, length=N_SUBSTEPS)

        ee_new = state.site_xpos[EE_SITE_IDX]
        ee_vel_new = (ee_new - ee_pos) / frame_dt
        actual_qvel = state.qvel[:N_JOINTS]
        cube_pos = state.xpos[CUBE1_BODY_IDX]
        actual_rew = select_reward(state, step_idx, init_cube_pos)
        cube2_pos = state.xpos[CUBE2_BODY_IDX]

        step_info = jnp.concatenate([
            ee_new,                                    # 0:3
            ee_vel_new,                                # 3:6
            jnp.array([finger_target, actual_rew]),    # 6:8
            safe_q_dot,                                # 8:15
            actual_qvel,                               # 15:22
            cube_pos,                                  # 22:25
            cube2_pos, 
        ])

        return (state, ee_pos), (step_info, step_qpos)

    init_carry = (mjx_data_init, mjx_data_init.site_xpos[EE_SITE_IDX])

    (final_state, _), (all_step_info, all_qpos) = jax.lax.scan(
        frame_step_fn, init_carry, jnp.arange(N_TOTAL))

    return final_state, all_step_info, all_qpos

# ---------------------------------------------------------------------------
# JIT compile
# ---------------------------------------------------------------------------
print("JIT compiling combined loss + grad...")
t0 = time.time()
value_and_grad_fn = jax.jit(jax.value_and_grad(combined_loss_fn))
loss_val, grad_val = value_and_grad_fn(mlp_params)
loss_val.block_until_ready()
print(f"  JIT compile: {time.time()-t0:.1f}s")
print(f"  Initial loss: {float(loss_val):.6f}")
print(f"  Initial grad norm: {float(optax.global_norm(grad_val)):.4f}")

print("\nJIT compiling forward rollout...")
t0 = time.time()
jit_forward = jax.jit(combined_forward_fn)
_, info_test, qpos_test = jit_forward(mlp_params)
info_test.block_until_ready()
print(f"  JIT compile: {time.time()-t0:.1f}s")
print(f"  all_step_info shape: {info_test.shape}")
print(f"  all_qpos shape: {qpos_test.shape}")

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
ema_loss_history = [] 
seg0_dist_history = []
seg1_dist_history = []
seg2_dist_history = []
seg3_dist_history = []
seg4_dist_history = []

SEG_NAMES = ["PRE-GRASP", "DESCEND", "CLOSE"]

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

    # Log before update
    if iteration % 5 == 0 or iteration < 3:
        ema_loss = combined_loss_fn(ema_params)
        ema_loss_history.append(float(ema_loss))
        ema_loss_f = float(ema_loss)
        
        _, all_info, all_qpos = jit_forward(ema_params)
        info_np = np.array(all_info)

        # Segment end distances
        seg0_ee = info_np[SEG1_START-1, :3]
        seg0_cube = info_np[SEG1_START-1, 22:25]
        seg0_dist = np.linalg.norm(seg0_ee - seg0_cube)
        seg0_dist_history.append(seg0_dist)

        seg1_ee = info_np[SEG2_START-1, :3]
        seg1_cube = info_np[SEG2_START-1, 22:25]
        seg1_dist = np.linalg.norm(seg1_ee - seg1_cube)
        seg1_dist_history.append(seg1_dist)

        seg2_ee = info_np[SEG3_START-1, :3]       # düzeltildi
        seg2_cube = info_np[SEG3_START-1, 22:25]   # düzeltildi
        seg2_dist = np.linalg.norm(seg2_ee - seg2_cube)
        seg2_finger = info_np[SEG3_START-1, 6]     # düzeltildi
        seg2_dist_history.append(seg2_dist)

        seg3_cube1 = info_np[SEG4_START-1, 22:25]
        seg3_cube2 = info_np[SEG4_START-1, 25:28]
        stack_target = seg3_cube2 + np.array([0.0, 0.0, CUBE2_HEIGHT + STACK_OFFSET])
        seg3_stack_dist = np.linalg.norm(seg3_cube1 - stack_target)
        seg3_ee = info_np[-1, :3]
        seg3_ee_cube_dist = np.linalg.norm(seg3_ee - seg3_cube1)
        seg3_finger = info_np[-1, 6]
        seg3_dist_history.append(seg3_stack_dist)

        seg4_cube1 = info_np[-1, 22:25]
        seg4_cube2 = info_np[-1, 25:28]
        seg4_xy_err = np.linalg.norm(seg4_cube1[:2] - seg4_cube2[:2])
        seg4_z_target = seg4_cube2[2] + CUBE2_HEIGHT + STACK_OFFSET
        seg4_z_err = abs(seg4_cube1[2] - seg4_z_target)
        seg4_finger = info_np[-1, 6]
        seg4_dist_history.append(seg4_xy_err + seg4_z_err)

        print(f"\nIter {iteration:4d}:  loss(train)={loss_f:.4f}  loss(ema)={ema_loss_f:.4f}  grad_norm={grad_norm:.4f}")
        print(f"  Seg0 end: EE-Cube={seg0_dist:.4f}")
        print(f"  Seg1 end: EE-Cube={seg1_dist:.4f}")
        print(f"  Seg2 end: EE-Cube={seg2_dist:.4f}  finger={seg2_finger:.4f}")
        print(f"  Seg3 end: Cube-Tgt={seg3_stack_dist:.4f}  EE-Cube={seg3_ee_cube_dist:.4f}  finger={seg3_finger:.4f}")
        print(f"  Seg4 end: xy_err={seg4_xy_err:.4f}  z_err={seg4_z_err:.4f}  finger={seg4_finger:.4f}")

        # Per-step detail per segment
        segments = [
            ("PRE-GRASP", 0, SEG1_START),
            ("DESCEND",   SEG1_START, SEG2_START),
            ("CLOSE",     SEG2_START, SEG3_START),
            ("MOVE",      SEG3_START, SEG4_START),      # düzeltildi: N_TOTAL → SEG4_START
            ("RELEASE",   SEG4_START, N_TOTAL),          # eklendi
        ]
        for seg_name, s_start, s_end in segments:
            print(f"  --- {seg_name} (step {s_start}-{s_end-1}) ---")
            for s in range(s_start, s_end):
                ee = info_np[s, :3]
                v = info_np[s, 3:6]
                fng = info_np[s, 6]
                rew = info_np[s, 7]
                cube = info_np[s, 22:25]
                print(f"    step {s:2d}: EE=[{ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}]  "
                      f"cube=[{cube[0]:.3f},{cube[1]:.3f},{cube[2]:.3f}]  "
                      f"fng={fng:.4f}  rew={rew:.4f}")

        # Joint vel plot with segment boundaries
        fig, axes = plt.subplots(7, 1, figsize=(14, 22))
        for j in range(7):
            axes[j].plot(info_np[:, 8+j], label=f"target j{j}", linestyle="--")
            axes[j].plot(info_np[:, 15+j], label=f"actual j{j}")
            axes[j].axvline(x=SEG1_START, color='red', linestyle=':', alpha=0.7, label='seg1 start')
            axes[j].axvline(x=SEG2_START, color='green', linestyle=':', alpha=0.7, label='seg2 start')
            axes[j].axvline(x=SEG3_START, color='blue', linestyle=':', alpha=0.7, label='seg3')
            axes[j].axvline(x=SEG4_START, color='purple', linestyle=':', alpha=0.7, label='seg4')
            axes[j].set_ylabel("rad/s")
            axes[j].legend(fontsize=7)
            axes[j].grid(True, alpha=0.3)
        axes[-1].set_xlabel("Step")
        fig.suptitle(f"Joint Vel — Iter {iteration} | Seg0:{N_SEG0} Seg1:{N_SEG1} Seg2:{N_SEG2}")
        fig.tight_layout()
        fig.savefig(f"joint_vel_iter_{iteration:04d}.png", dpi=100)
        plt.close('all')
        import gc; gc.collect()

        render_trajectory(all_qpos, skip=1)
    else:
        seg0_dist_history.append(seg0_dist_history[-1] if seg0_dist_history else 0)
        seg1_dist_history.append(seg1_dist_history[-1] if seg1_dist_history else 0)
        seg2_dist_history.append(seg2_dist_history[-1] if seg2_dist_history else 0)
        seg3_dist_history.append(seg3_dist_history[-1] if seg3_dist_history else 0)
        seg4_dist_history.append(seg4_dist_history[-1] if seg4_dist_history else 0)
        ema_loss_history.append(ema_loss_history[-1] if ema_loss_history else 0)

    # Update
    updates, opt_state = optimizer.update(grad_val, opt_state, mlp_params)
    mlp_params = optax.apply_updates(mlp_params, updates)

    ema_params = jax.tree.map(
        lambda ema, cur: EMA_TAU * ema + (1.0 - EMA_TAU) * cur,
        ema_params, mlp_params
    )

train_time = time.time() - train_start
print(f"\nTraining done: {train_time:.1f}s ({len(loss_history)} iters, "
      f"{train_time/max(1,len(loss_history)):.3f}s/iter)")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

import pickle
with open("mlp_params_cube_stack.pkl", "wb") as f:
    pickle.dump(jax.tree.map(lambda x: np.array(x), ema_params), f)  # ← ema_params kaydet

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"SONUÇLAR")
print(f"{'='*60}")
print(f"  Initial loss:      {loss_history[0]:.4f}")
print(f"  Final loss:        {loss_history[-1]:.4f}")
print(f"  Seg0 final dist:   {seg0_dist_history[-1]:.4f}")
print(f"  Seg1 final dist:   {seg1_dist_history[-1]:.4f}")
print(f"  Seg2 final dist:   {seg2_dist_history[-1]:.4f}")
print(f"  Seg3 final dist:   {seg3_dist_history[-1]:.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(7, 1, figsize=(10, 18))

axes[0].plot(loss_history, linewidth=1.5)
axes[0].set_ylabel("Loss (-reward)")
axes[0].set_title("Combined Loss")
axes[0].grid(True, alpha=0.3)

axes[1].plot(loss_history, linewidth=1.0, alpha=0.4, label="train (mlp_params)")
axes[1].plot(ema_loss_history, linewidth=2.0, label="EMA")
axes[1].legend()
axes[1].set_ylabel("Loss (-reward)")
axes[1].set_title("Combined Loss")
axes[1].grid(True, alpha=0.3)

axes[2].plot(seg0_dist_history, linewidth=1.5)
axes[2].set_ylabel("Distance (m)")
axes[2].set_title("Seg0 (Pre-grasp): EE → Cube")
axes[2].grid(True, alpha=0.3)

axes[3].plot(seg1_dist_history, linewidth=1.5)
axes[3].set_ylabel("Distance (m)")
axes[3].set_title("Seg1 (Descend): EE → Cube")
axes[3].grid(True, alpha=0.3)

axes[4].plot(seg2_dist_history, linewidth=1.5)
axes[4].set_ylabel("Distance (m)")
axes[4].set_title("Seg2 (Close): EE → Cube")
axes[4].grid(True, alpha=0.3)

axes[5].plot(seg3_dist_history, linewidth=1.5)
axes[5].set_ylabel("Distance (m)")
axes[5].set_xlabel("Iteration")
axes[5].set_title("Seg3 (Move): Cube → Target")
axes[5].grid(True, alpha=0.3)

axes[6].plot(seg4_dist_history, linewidth=1.5)
axes[6].set_ylabel("Error (m)")
axes[6].set_xlabel("Iteration")
axes[6].set_title("Seg4 (Release): Align + Release Error")
axes[6].grid(True, alpha=0.3)

fig.suptitle(f"BPTT 4-Segment — {N_SEG0}+{N_SEG1}+{N_SEG2}+{N_SEG3}, {len(loss_history)} iters")
fig.tight_layout()
plot_path = "bptt_3seg_results.png"
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