"""
MJX BPTT — 5-Segment Cube Stacking + Depth Camera CNN

Differences from the current code:
  - In `build_obs`, `diff_render` → CNN → latent is used instead of `state.xpos`
  - Two cameras: `overhead_cam` and `overhead_cam2` (from XML)
  - CNN: (32,32,2) depth → 32-dimensional latent
  - Proprioception (joint angles, velocities, EE pos/quat) is fed directly into the obs
  - Reward functions are the SAME — still using state.xpos (supervision signal)

Usage:
    python mjx_franka_bptt_depth_cnn.py
    python mjx_franka_bptt_depth_cnn.py --iters 900 --lr 5e-4 --no-viewer
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
parser.add_argument("--seg0", type=int, default=16, help="Pre-grasp steps")
parser.add_argument("--seg1", type=int, default=16, help="Descend steps")
parser.add_argument("--seg2", type=int, default=6, help="Close steps")
parser.add_argument("--seg3", type=int, default=16, help="Move steps")
parser.add_argument("--seg4", type=int, default=8, help="Align+Release steps")
parser.add_argument("--substeps", type=int, default=4)
parser.add_argument("--iters", type=int, default=1500)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.999)
parser.add_argument("--solver-iters", type=int, default=6)
parser.add_argument("--grad-clip", type=float, default=300.0)
parser.add_argument("--no-viewer", action="store_true")
parser.add_argument("--img-res", type=int, default=64, help="Depth image resolution")
parser.add_argument("--xml", type=str,
                    default="/home/emre/newton_diff_RL/MJX_Franka_DiffSim/old_files/franka_emika_panda_depth/mjx_single_cube.xml")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_SEG0      = args.seg0
N_SEG1      = args.seg1
N_SEG2      = args.seg2
N_SEG3      = args.seg3
N_SEG4      = args.seg4
N_TOTAL     = N_SEG0 + N_SEG1 + N_SEG2 + N_SEG3 + N_SEG4
SEG1_START  = N_SEG0
SEG2_START  = N_SEG0 + N_SEG1
SEG3_START  = N_SEG0 + N_SEG1 + N_SEG2
SEG4_START  = N_SEG0 + N_SEG1 + N_SEG2 + N_SEG3
N_SUBSTEPS  = args.substeps
BATCH_SIZE  = 15
TRAIN_ITERS = args.iters
TRAIN_LR    = args.lr
GAMMA       = args.gamma
GRAD_CLIP   = args.grad_clip
N_JOINTS    = 7

FINGER_OPEN        = 0.04
GRASP_Z_OFFSET     = 0.0
PRE_GRASP_Z_OFFSET = 0.15
CUBE2_HEIGHT       = 0.05
STACK_OFFSET       = 0.01

Q_LO = jnp.array([-2.897, -1.763, -2.897, -3.072, -2.897, -0.018, -2.897])
Q_HI = jnp.array([ 2.897,  1.763,  2.897, -0.070,  2.897,  3.752,  2.897])
VEL_LIMITS = jnp.array([2.17, 2.17, 2.17, 2.17, 2.61, 2.61, 2.61])

TARGET_POS  = jnp.array([0.4, 0.15, 0.55])
TARGET_QUAT = jnp.array([0.0, 0.7071068, 0.7071068, 0.0])

# ---------------------------------------------------------------------------
# Depth Renderer Config
# ---------------------------------------------------------------------------
IMG_W = IMG_H = args.img_res
MAX_DEPTH = 1.5
FOVY_RAD = jnp.deg2rad(60.0)

# Box half-size (from XML: size="0.025 0.025 0.025")
BOX_HALF = jnp.array([0.025, 0.025, 0.025])

# ---------------------------------------------------------------------------
# Network dimensions
# ---------------------------------------------------------------------------
LATENT_DIM   = 64
PROPRIO_DIM  = 3 + 3 + 1 + 7 + 7 + 4  # ee_pos, ee_vel, finger, q, qdot, ee_quat = 25
OBS_DIM      = LATENT_DIM + PROPRIO_DIM  # 32 + 25 = 57
ACT_DIM      = N_JOINTS + 1
HIDDEN_TRUNK = 128
HIDDEN_HEAD  = 32

# CNN flatten dim: 32 channels × (IMG_W/8) × (IMG_H/8)
CNN_FLAT_DIM = 32 * (IMG_W // 8) * (IMG_H // 8)

# ---------------------------------------------------------------------------
# Differentiable Depth Renderer
# ---------------------------------------------------------------------------
def quat_to_rot(q):
    """Quaternion [w, x, y, z] → 3×3 rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def generate_rays_from_mat(cam_pos, cam_mat_flat, fovy, img_w, img_h):
    """MuJoCo cam_mat (9,) → ray directions (H, W, 3).
    cam_mat row-major 3×3: row0=right, row1=up, row2=z_axis.
    Camera looks along -z_axis.
    """
    cam_mat = cam_mat_flat.reshape(3,3)

    right   = -cam_mat[:, 0]
    up      = cam_mat[:, 1]
    forward = -cam_mat[:, 2]

    aspect = img_w / img_h
    half_h = jnp.tan(fovy / 2.0)
    half_w = half_h * aspect

    u = (jnp.arange(img_w) + 0.5) / img_w * 2.0 - 1.0
    v = ((jnp.arange(img_h) + 0.5) / img_h * 2.0 - 1.0)[::-1]
    uu, vv = jnp.meshgrid(u, v)

    dirs = (forward[None, None, :]
            + uu[:, :, None] * half_w * right[None, None, :]
            + vv[:, :, None] * half_h * up[None, None, :])
    dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs

def generate_local_rays(fovy, img_w, img_h):
    """Kamera-lokal ray yönleri (forward=-z, right=+x, up=+y). Bir kez hesaplanır."""
    aspect = img_w / img_h
    half_h = jnp.tan(fovy / 2.0)
    half_w = half_h * aspect

    u = (jnp.arange(img_w) + 0.5) / img_w * 2.0 - 1.0
    v = ((jnp.arange(img_h) + 0.5) / img_h * 2.0 - 1.0)[::-1]
    uu, vv = jnp.meshgrid(u, v)

    # Lokal frame: forward=(0,0,-1), right=(1,0,0), up=(0,1,0)
    dirs = jnp.stack([-uu * half_w, vv * half_h, -jnp.ones_like(uu)], axis=-1)
    dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs  # (H, W, 3)

def ray_plane_intersect(ray_o, ray_d):
    """Ray → z=0 plane intersection."""
    t = -ray_o[2] / ray_d[:, :, 2]
    valid = (ray_d[:, :, 2] < -1e-8) & (t > 0)
    return jnp.where(valid, t, MAX_DEPTH)


def ray_box_intersect(ray_o, ray_d, box_pos, box_quat, box_half):
    """Ray → rotated box intersection via slab method."""
    R_inv = quat_to_rot(box_quat).T
    local_o = R_inv @ (ray_o - box_pos)
    local_d = jnp.einsum('ij,hwj->hwi', R_inv, ray_d)
    inv_d = 1.0 / jnp.where(jnp.abs(local_d) < 1e-8,
                              jnp.sign(local_d) * 1e-8 + 1e-10, local_d)
    t1 = (-box_half[None, None, :] - local_o[None, None, :]) * inv_d
    t2 = ( box_half[None, None, :] - local_o[None, None, :]) * inv_d
    t_near = jnp.minimum(t1, t2)
    t_far  = jnp.maximum(t1, t2)
    t_enter = jnp.max(t_near, axis=-1)
    t_exit  = jnp.min(t_far, axis=-1)
    hit = (t_enter < t_exit) & (t_exit > 0)
    t_box = jnp.where(hit, jnp.maximum(t_enter, 0.0), MAX_DEPTH)
    return t_box


def render_single_camera(cam_pos, ray_dirs, cube1_pos, cube1_quat, cube2_pos, cube2_quat):
    """Tek kamera: zemin + 2 küp depth render."""
    t_ground = ray_plane_intersect(cam_pos, ray_dirs)
    t_box1 = ray_box_intersect(cam_pos, ray_dirs, cube1_pos, cube1_quat, BOX_HALF)
    t_box2 = ray_box_intersect(cam_pos, ray_dirs, cube2_pos, cube2_quat, BOX_HALF)
    return jnp.minimum(t_ground, jnp.minimum(t_box1, t_box2))


def render_both_cameras(state):
    c1_pos  = state.xpos[CUBE1_BODY_IDX]
    c1_quat = state.xquat[CUBE1_BODY_IDX]
    c2_pos  = state.xpos[CUBE2_BODY_IDX]
    c2_quat = state.xquat[CUBE2_BODY_IDX]

    # Overhead: küpler + zemin (finger'lar uzakta, gereksiz)
    depth1 = render_single_camera(OVERHEAD_CAM_POS, OVERHEAD_RAYS,
                                   c1_pos, c1_quat, c2_pos, c2_quat)

    # Wrist: küpler + zemin + finger'lar
    wrist_pos = state.cam_xpos[WRIST_CAM_ID]
    wrist_mat = state.cam_xmat[WRIST_CAM_ID].reshape(3, 3)
    wrist_rays = jnp.einsum('ij,hwj->hwi', wrist_mat, WRIST_LOCAL_RAYS)

    t_scene = render_single_camera(wrist_pos, wrist_rays,
                                    c1_pos, c1_quat, c2_pos, c2_quat)

    # Finger'ları ekle
    t_lf = ray_box_intersect(wrist_pos, wrist_rays,
                              state.xpos[LEFT_FINGER_BODY_IDX],
                              state.xquat[LEFT_FINGER_BODY_IDX],
                              FINGER_HALF)
    t_rf = ray_box_intersect(wrist_pos, wrist_rays,
                              state.xpos[RIGHT_FINGER_BODY_IDX],
                              state.xquat[RIGHT_FINGER_BODY_IDX],
                              FINGER_HALF)

    depth2 = jnp.minimum(t_scene, jnp.minimum(t_lf, t_rf))

    return jnp.stack([depth1, depth2], axis=-1)


# ---------------------------------------------------------------------------
# CNN: (H, W, 2) depth → latent (32)
# ---------------------------------------------------------------------------
def init_cnn(key):
    keys = jax.random.split(key, 5)
    params = {
        # Conv: (C_out, C_in, kH, kW) — JAX lax.conv format
        'conv1_w': (jax.random.normal(keys[0], (8, 2, 3, 3)) * 0.1).astype(jnp.float32),
        'conv1_b': jnp.zeros(8, dtype=jnp.float32),

        'conv2_w': (jax.random.normal(keys[1], (16, 8, 3, 3)) * 0.1).astype(jnp.float32),
        'conv2_b': jnp.zeros(16, dtype=jnp.float32),

        'conv3_w': (jax.random.normal(keys[2], (32, 16, 3, 3)) * 0.1).astype(jnp.float32),
        'conv3_b': jnp.zeros(32, dtype=jnp.float32),

        'fc1_w': (jax.random.normal(keys[3], (CNN_FLAT_DIM, 64)) * jnp.sqrt(2.0/CNN_FLAT_DIM)).astype(jnp.float32),
        'fc1_b': jnp.zeros(64, dtype=jnp.float32),

        'fc2_w': (jax.random.normal(keys[4], (64, LATENT_DIM)) * 0.01).astype(jnp.float32),
        'fc2_b': jnp.zeros(LATENT_DIM, dtype=jnp.float32),
    }
    return params


def cnn_forward(cnn_params, depth_2ch):
    """depth_2ch: (H, W, 2) → latent (LATENT_DIM,)"""
    # Normalize depth: [0.3, 3.0] → [-1, 1]
    overhead_raw = depth_2ch[:, :, 0:1].astype(jnp.float32)
    overhead = (jnp.clip(overhead_raw, 0.0, 1.5) - 0.75) / 0.75

    wrist_raw = depth_2ch[:, :, 1:2].astype(jnp.float32)
    wrist = (jnp.clip(wrist_raw, 0.0, 1.0) - 0.5) / 0.5

    x = jnp.concatenate([overhead, wrist], axis=-1)

    # (H, W, 2) → (1, 2, H, W) — NCHW format
    x = jnp.transpose(x, (2, 0, 1))[None, :, :, :]

    # Conv1: 2→8, stride=2, SAME
    x = jax.lax.conv_general_dilated(x, cnn_params['conv1_w'],
                                      window_strides=(2, 2), padding='SAME')
    x = x + cnn_params['conv1_b'][None, :, None, None]
    x = jax.nn.relu(x)

    # Conv2: 8→16, stride=2
    x = jax.lax.conv_general_dilated(x, cnn_params['conv2_w'],
                                      window_strides=(2, 2), padding='SAME')
    x = x + cnn_params['conv2_b'][None, :, None, None]
    x = jax.nn.relu(x)

    # Conv3: 16→32, stride=2
    x = jax.lax.conv_general_dilated(x, cnn_params['conv3_w'],
                                      window_strides=(2, 2), padding='SAME')
    x = x + cnn_params['conv3_b'][None, :, None, None]
    x = jax.nn.relu(x)

    # Flatten
    x = x.reshape(-1)

    # FC1 → FC2
    x = jnp.tanh(x @ cnn_params['fc1_w'] + cnn_params['fc1_b'])
    x = x @ cnn_params['fc2_w'] + cnn_params['fc2_b']

    return x  # (LATENT_DIM,)


# ---------------------------------------------------------------------------
# Shaped distance function
# ---------------------------------------------------------------------------
def shaped_distance(a, b, s, t):
    dist = jnp.sqrt(jnp.dot(a - b, a - b) + 1e-6)
    scale = 1.8318 / jnp.maximum(s, 1e-6)
    shaped = (1.0 - jnp.tanh(dist * scale)) ** 2
    return jnp.where(dist < t, 1.0, shaped)

# ---------------------------------------------------------------------------
# MLP — trunk + 5 segment heads
# ---------------------------------------------------------------------------
def init_all_params(key):
    keys = jax.random.split(key, 8)

    cnn = init_cnn(keys[0])

    trunk = {
        'w1': (jax.random.normal(keys[1], (OBS_DIM, HIDDEN_TRUNK)) * jnp.sqrt(2.0 / OBS_DIM)).astype(jnp.float32),
        'b1': jnp.zeros(HIDDEN_TRUNK, dtype=jnp.float32),
        'w2': (jax.random.normal(keys[2], (HIDDEN_TRUNK, HIDDEN_TRUNK)) * jnp.sqrt(2.0 / HIDDEN_TRUNK)).astype(jnp.float32),
        'b2': jnp.zeros(HIDDEN_TRUNK, dtype=jnp.float32),
    }

    def make_head(k, finger_bias):
        k1, k2 = jax.random.split(k)
        return {
            'w3': (jax.random.normal(k1, (HIDDEN_TRUNK, HIDDEN_HEAD)) * jnp.sqrt(2.0 / HIDDEN_TRUNK)).astype(jnp.float32),
            'b3': jnp.zeros(HIDDEN_HEAD, dtype=jnp.float32),
            'w4': (jax.random.normal(k2, (HIDDEN_HEAD, ACT_DIM)) * 0.01).astype(jnp.float32),
            'b4': jnp.array([0.0]*7 + [finger_bias], dtype=jnp.float32),
        }

    return {
        'cnn':  cnn,
        'trunk': trunk,
        'seg0': make_head(keys[3], 2.0),
        'seg1': make_head(keys[4], 2.0),
        'seg2': make_head(keys[5], 0.0),
        'seg3': make_head(keys[6], 0.0),
        'seg4': make_head(keys[7], 0.0),
    }


def mlp_forward_head(trunk_features, head_params):
    tf32 = trunk_features.astype(jnp.float32)
    x = jnp.tanh(tf32 @ head_params['w3'] + head_params['b3'])
    x = x @ head_params['w4'] + head_params['b4']
    q_dot_target = jnp.tanh(x[:N_JOINTS]) * VEL_LIMITS
    finger = jax.nn.sigmoid(x[N_JOINTS]) * FINGER_OPEN
    return q_dot_target, finger


def mlp_forward(all_params, obs, step_idx):
    obs32 = obs.astype(jnp.float32)
    x = jnp.tanh(obs32 @ all_params['trunk']['w1'] + all_params['trunk']['b1'])
    x = jnp.tanh(x @ all_params['trunk']['w2'] + all_params['trunk']['b2'])

    qd0, f0 = mlp_forward_head(x, all_params['seg0'])
    qd1, f1 = mlp_forward_head(x, all_params['seg1'])
    qd2, f2 = mlp_forward_head(x, all_params['seg2'])
    qd3, f3 = mlp_forward_head(x, all_params['seg3'])
    qd4, f4 = mlp_forward_head(x, all_params['seg4'])

    qd = jnp.where(step_idx < SEG1_START, qd0,
         jnp.where(step_idx < SEG2_START, qd1,
         jnp.where(step_idx < SEG3_START, qd2,
         jnp.where(step_idx < SEG4_START, qd3, qd4))))
    f = jnp.where(step_idx < SEG1_START, f0,
        jnp.where(step_idx < SEG2_START, f1,
        jnp.where(step_idx < SEG3_START, f2,
        jnp.where(step_idx < SEG4_START, f3, f4))))
    return qd, f

# ---------------------------------------------------------------------------
# Obs Normalization
# ---------------------------------------------------------------------------
class ObsRMS:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, batch):
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        self.var = (self.var * self.count + batch_var * batch_count +
                    delta**2 * self.count * batch_count / total) / total
        self.count = total

    def get_jnp(self):
        mean = jnp.array(self.mean, dtype=jnp.float32)
        std = jnp.sqrt(jnp.array(self.var, dtype=jnp.float32)) + 1e-8
        return mean, std

obs_rms = ObsRMS(PROPRIO_DIM)

# ---------------------------------------------------------------------------
# Observation builder — DEPTH CNN + Proprioception
# ---------------------------------------------------------------------------
HAND_BODY_IDX = 9  # hand body index in Franka model

def build_obs_depth(all_params, state, prev_ee_pos, frame_dt):
    """Depth render → CNN → latent + proprioception → obs."""
    # Render iki kameradan depth
    depth_2ch = render_both_cameras(state)  # (H, W, 2)

    # CNN → latent
    cnn_latent = cnn_forward(all_params['cnn'], depth_2ch)  # (LATENT_DIM,)

    # Proprioception (robot'un kendi sensörleri)
    ee_pos = state.site_xpos[EE_SITE_IDX]
    ee_vel = (ee_pos - prev_ee_pos) / frame_dt
    finger_width = state.qpos[7] + state.qpos[8]
    current_q = state.qpos[:N_JOINTS]
    current_q_dot = state.qvel[:N_JOINTS]
    ee_quat = state.xquat[HAND_BODY_IDX]

    obs = jnp.concatenate([
        cnn_latent,                    # 32 — depth'ten öğrenilen
        ee_pos,                        #  3 — end-effector pozisyonu
        ee_vel,                        #  3 — end-effector hızı
        jnp.array([finger_width]),     #  1 — gripper açıklığı
        current_q,                     #  7 — eklem açıları
        current_q_dot,                 #  7 — eklem hızları
        ee_quat,                       #  4 — end-effector orientasyonu
    ])                                 # = 57 toplam

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
# Reward functions — AYNI, değişmedi
# ---------------------------------------------------------------------------
def reward_seg0_pregrasp(state, init_cube_pos):
    ee_pos = state.site_xpos[EE_SITE_IDX]
    cube_pos = state.xpos[CUBE1_BODY_IDX]
    ee_pos_z = ee_pos[2]

    pre_grip_pos = cube_pos + jnp.array([0.0, 0.0, PRE_GRASP_Z_OFFSET])

    # XY ve Z ayrı
    xy_dist = jnp.sqrt(jnp.dot(ee_pos[:2] - pre_grip_pos[:2], 
                                ee_pos[:2] - pre_grip_pos[:2]) + 1e-6)
    z_dist = jnp.abs(ee_pos[2] - pre_grip_pos[2])

    r_xy = shaped_distance(ee_pos[:2], pre_grip_pos[:2], 0.3, 0.001)
    r_z  = shaped_distance(jnp.array([ee_pos[2]]), jnp.array([pre_grip_pos[2]]), 0.3, 0.001)

    # XY ağırlığı çok daha yüksek — önce hizalan, sonra in
    r_approach = r_xy * 1.0 - xy_dist * 1.0 + r_z * 0.50 - z_dist * 0.50

    ee_quat = state.xquat[9]
    quat_dot = jnp.dot(ee_quat, TARGET_QUAT)
    ori_loss = (1.0 - quat_dot * quat_dot) * r_approach

    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_open = finger_width * 5.0

    ground_penalty = jnp.where(ee_pos_z < 0.175, 5.0 * (0.175 - ee_pos_z) / 0.175, 0.0)

    return r_approach - ori_loss * 5.0 - ground_penalty + r_finger_open


def reward_seg1_descend(state, init_cube_pos):
    ee_pos = state.site_xpos[EE_SITE_IDX]
    cube_pos = state.xpos[CUBE1_BODY_IDX]
    cube_holding_pos = cube_pos + jnp.array([0.0, 0.0, -0.01])

    # XY ve Z ayrı — seg1'de z ağırlıklı, xy koruma
    r_xy = shaped_distance(ee_pos[:2], cube_holding_pos[:2], 0.1, 0.001)
    r_z  = shaped_distance(jnp.array([ee_pos[2]]), jnp.array([cube_holding_pos[2]]), 0.1, 0.001)

    # Z ağır — in. XY hafif — hizalamayı koru, bozma
    r_approach = r_z * 1.0 + r_xy * 2.0

    ee_quat = state.xquat[9]
    quat_dot = jnp.dot(ee_quat, TARGET_QUAT)
    ori_loss = (1.0 - quat_dot * quat_dot) * r_approach

    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_open = finger_width * 1.0

    ee_pos_z = ee_pos[2]
    ground_penalty = jnp.where(ee_pos_z < 0.04, 4.0 * (0.04 - ee_pos_z) / 0.04, 0.0)

    return r_approach - ori_loss * 5.0 - ground_penalty + r_finger_open

def reward_seg2_close(state, init_cube_pos):
    ee_pos = state.site_xpos[EE_SITE_IDX]
    cube_pos = state.xpos[CUBE1_BODY_IDX]
    cube_holding_pos = cube_pos + jnp.array([0.0, 0.0, -0.01])

    r_proximity = shaped_distance(ee_pos, cube_holding_pos, 0.1, 0.001)

    finger_width = state.qpos[7] + state.qpos[8]
    r_grip_close = -finger_width * 2.0

    ee_quat = state.xquat[9]
    quat_dot = jnp.dot(ee_quat, TARGET_QUAT)
    ori_loss = (1.0 - quat_dot * quat_dot) * r_proximity

    ee_pos_z = ee_pos[2]
    ground_penalty = jnp.where(ee_pos_z < 0.04, 4.0 * (0.04 - ee_pos_z) / 0.04, 0.0)

    return r_proximity * 3.0 + r_grip_close - ori_loss * 5.0 - ground_penalty

def reward_seg3_move(state, init_cube_pos):
    ee_pos = state.site_xpos[EE_SITE_IDX]
    cube1_pos = state.xpos[CUBE1_BODY_IDX]
    cube2_pos = state.xpos[CUBE2_BODY_IDX]
    cube1_z = state.xpos[CUBE1_BODY_IDX][2]

    stack_target = cube2_pos + jnp.array([0.0, 0.0, CUBE2_HEIGHT + STACK_OFFSET])
    min_height = 0.05
    max_height = 0.075 + 0.01

    lift_weight = jnp.clip((cube1_z - min_height) / (max_height - min_height), 0.0, 1.0)

    r_cube_to_target = (shaped_distance(cube1_pos, stack_target, 0.5, 0.001) + shaped_distance(cube1_pos, stack_target, 0.1, 0.001)) / 2
    r_cube_to_target = jnp.clip(r_cube_to_target, 0.0, 1.0) * lift_weight

    finger_width = state.qpos[7] + state.qpos[8]
    r_grip_close = -finger_width * 2.0

    ee_quat = state.xquat[9]
    quat_dot = jnp.dot(ee_quat, TARGET_QUAT)
    ori_loss = 1.0 - quat_dot * quat_dot

    ee_pos_z = ee_pos[2]
    ground_penalty = jnp.where(ee_pos_z < 0.04, 4.0 * (0.04 - ee_pos_z) / 0.04, 0.0)

    return r_cube_to_target * 1.0 + lift_weight + r_grip_close - ori_loss * 3.0 - ground_penalty

def reward_seg4_release(state, init_cube_pos):
    ee_pos = state.site_xpos[EE_SITE_IDX]
    cube1_pos = state.xpos[CUBE1_BODY_IDX]
    cube2_pos = state.xpos[CUBE2_BODY_IDX]

    stack_target = cube2_pos + jnp.array([0.0, 0.0, CUBE2_HEIGHT + 0.005])

    r_align = shaped_distance(cube1_pos, stack_target, 0.1, 0.001)

    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_open = finger_width * r_align

    ee_quat = state.xquat[9]
    quat_dot = jnp.dot(ee_quat, TARGET_QUAT)
    ori_loss = 1.0 - quat_dot * quat_dot

    r_away = shaped_distance(ee_pos, cube1_pos, 0.1, 0.001)

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

# Kamera ID'leri
OVERHEAD_CAM_ID = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
WRIST_CAM_ID    = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
print(f"  Cameras: overhead_cam={OVERHEAD_CAM_ID}, wrist_cam={WRIST_CAM_ID}")

# Overhead kamera: sabit, ray'leri bir kere hesapla
key_id = mj_model.keyframe("home").id
mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
mujoco.mj_forward(mj_model, mj_data)
OVERHEAD_CAM_POS = jnp.array(mj_data.cam_xpos[OVERHEAD_CAM_ID])
OVERHEAD_CAM_MAT = jnp.array(mj_data.cam_xmat[OVERHEAD_CAM_ID]).reshape(-1)
OVERHEAD_RAYS = generate_rays_from_mat(OVERHEAD_CAM_POS, OVERHEAD_CAM_MAT,
                                        FOVY_RAD, IMG_W, IMG_H)

# Wrist kamera: fovy farklı olabilir
WRIST_FOVY_RAD = jnp.deg2rad(float(mj_model.cam_fovy[WRIST_CAM_ID]))
WRIST_LOCAL_RAYS = generate_local_rays(WRIST_FOVY_RAD, IMG_W, IMG_H)
print(f"  Overhead pos: {OVERHEAD_CAM_POS}")
print(f"  Wrist fovy: {float(jnp.rad2deg(WRIST_FOVY_RAD)):.1f}°")

LEFT_FINGER_BODY_IDX = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
RIGHT_FINGER_BODY_IDX = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
FINGER_HALF = jnp.array([0.005, 0.005, 0.02])  # yaklaşık finger boyutu

sim_dt = mj_model.opt.timestep
frame_dt = N_SUBSTEPS * sim_dt
total_time = N_TOTAL * frame_dt
total_physics = N_TOTAL * N_SUBSTEPS

print(f"  sim_dt={sim_dt}, frame_dt={frame_dt:.4f}s")
print(f"  Segments: {N_SEG0}+{N_SEG1}+{N_SEG2}+{N_SEG3}+{N_SEG4} = {N_TOTAL} steps")
print(f"  Total: {N_TOTAL} × {N_SUBSTEPS} substeps = {total_physics} physics ({total_time:.3f}s)")
print(f"  Batch: {BATCH_SIZE}, Depth: {IMG_W}×{IMG_H}, Latent: {LATENT_DIM}")
print(f"  EE home: {mj_data.site_xpos[EE_SITE_IDX]}")
print(f"  Cube1: {mj_data.xpos[CUBE1_BODY_IDX]}")
print(f"  Cube2: {mj_data.xpos[CUBE2_BODY_IDX]}")

home_ctrl = jnp.array(mj_data.ctrl)
init_cube_pos = jnp.array(mj_data.xpos[CUBE1_BODY_IDX])

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

# --- Debug: depth render görselleştirme ---
mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
mujoco.mj_forward(mj_model, mj_data)
d_test = mjx.put_data(mj_model, mj_data)
d_test = mjx.forward(mjx_model, d_test)

depth_2ch = render_both_cameras(d_test)  # (H, W, 2)
depth_np = np.array(depth_2ch)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
im0 = axes[0].imshow(depth_np[:, :, 0], cmap='viridis', vmin=0, vmax=MAX_DEPTH)
axes[0].set_title("Overhead Camera")
plt.colorbar(im0, ax=axes[0], label='depth (m)')

im1 = axes[1].imshow(depth_np[:, :, 1], cmap='viridis', vmin=0, vmax=MAX_DEPTH)
axes[1].set_title("Wrist Camera")
plt.colorbar(im1, ax=axes[1], label='depth (m)')

fig.suptitle(f"Depth Render — {IMG_W}×{IMG_H}")
fig.tight_layout()
fig.savefig("debug_depth_cameras.png", dpi=150)
plt.close(fig)
print("✓ Debug depth saved: debug_depth_cameras.png")

# ---------------------------------------------------------------------------
# Batch data — randomized cube positions
# ---------------------------------------------------------------------------
N_BATCH_SETS = 1
N_ACCUM = N_BATCH_SETS  # her set'i kullan, sonra güncelle
print(f"\nCreating {N_BATCH_SETS} batch sets × {BATCH_SIZE} envs...")

def make_randomized_data(rng_key):
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    noise = jax.random.uniform(rng_key, (2,), minval=-0.05, maxval=0.05)
    mj_data.qpos[9] += float(noise[0])   # cube1 x
    mj_data.qpos[10] += float(noise[1])  # cube1 y
    mujoco.mj_forward(mj_model, mj_data)
    d = mjx.put_data(mj_model, mj_data)
    d = mjx.forward(mjx_model, d)
    return d

batch_sets = []
init_cube_pos_sets = []

for s in range(N_BATCH_SETS):
    batch_keys = jax.random.split(jax.random.PRNGKey(s * 1000), BATCH_SIZE)
    batch_data_list = [make_randomized_data(k) for k in batch_keys]
    batch = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *batch_data_list)
    batch_sets.append(batch)
    init_cube_pos_sets.append(batch.xpos[:, CUBE1_BODY_IDX])
    print(f"  Set {s}: {BATCH_SIZE} envs")

# Logging için set 0, env 0
mjx_data_single0 = jax.tree.map(lambda x: x[0], batch_sets[0])
init_cube_pos_single0 = init_cube_pos_sets[0][0]
# ---------------------------------------------------------------------------
# Initialize params
# ---------------------------------------------------------------------------
rng = jax.random.PRNGKey(42)
all_params = init_all_params(rng)

n_cnn_params = sum(p.size for p in jax.tree.leaves(all_params['cnn']))
n_mlp_params = sum(p.size for p in jax.tree.leaves({k: v for k, v in all_params.items() if k != 'cnn'}))
n_total_params = n_cnn_params + n_mlp_params

optimizer = optax.chain(
    optax.clip_by_global_norm(GRAD_CLIP),
    optax.adam(TRAIN_LR),
)
opt_state = optimizer.init(all_params)

print(f"\n{'='*60}")
print(f"BPTT 5-SEG + DEPTH CNN + Batch={BATCH_SIZE}")
print(f"  CNN: (32,32,2)→8→16→32→flat({CNN_FLAT_DIM})→64→{LATENT_DIM} ({n_cnn_params} params)")
print(f"  MLP: {OBS_DIM}→{HIDDEN_TRUNK}→{HIDDEN_TRUNK}→{HIDDEN_HEAD}→{ACT_DIM} × 5 heads ({n_mlp_params} params)")
print(f"  Total params: {n_total_params}")
print(f"  Obs: {LATENT_DIM} (CNN) + {PROPRIO_DIM} (proprio) = {OBS_DIM}")
print(f"  Segments: {N_SEG0}+{N_SEG1}+{N_SEG2}+{N_SEG3}+{N_SEG4} = {N_TOTAL} steps")
print(f"  LR: {TRAIN_LR}, Gamma: {GAMMA}, Grad clip: {GRAD_CLIP}")
print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# Single env loss
# ---------------------------------------------------------------------------
def single_env_loss_fn(all_params, obs_mean, obs_std, mjx_data_single, init_cube_single):
    def frame_step_fn(carry, step_idx):
        state, prev_ee_pos, total_reward, gamma_acc = carry

        is_boundary = ((step_idx == SEG1_START) | (step_idx == SEG2_START) |
                       (step_idx == SEG3_START) | (step_idx == SEG4_START))
        state = jax.lax.cond(is_boundary,
                             lambda s: jax.lax.stop_gradient(s),
                             lambda s: s, state)
        prev_ee_pos = jax.lax.cond(is_boundary,
                                    lambda p: jax.lax.stop_gradient(p),
                                    lambda p: p, prev_ee_pos)
        gamma_acc = jnp.where(is_boundary, jnp.float32(1.0), gamma_acc)

        # Depth render → CNN → obs (ana değişiklik burada)
        obs, ee_pos, current_q = build_obs_depth(all_params, state, prev_ee_pos, frame_dt)

        # Normalize
        obs_f32 = obs.astype(jnp.float32)
        proprio_norm = (obs_f32[LATENT_DIM:] - obs_mean) / obs_std
        obs_norm = jnp.concatenate([obs_f32[:LATENT_DIM], proprio_norm])

        q_dot_target, finger_target = mlp_forward(all_params, obs_norm, step_idx)
        safe_q_dot = safe_action(q_dot_target, current_q)

        ctrl = home_ctrl.at[:N_JOINTS].set(safe_q_dot)
        ctrl = ctrl.at[7].set(finger_target)

        def substep_fn(s, _):
            s = s.replace(ctrl=ctrl)
            s = mjx.step(mjx_model, s)
            return s, None

        state, _ = jax.lax.scan(substep_fn, state, None, length=N_SUBSTEPS)

        rew = select_reward(state, step_idx, init_cube_single)
        total_reward = total_reward + gamma_acc * rew
        gamma_acc = gamma_acc * GAMMA

        return (state, ee_pos, total_reward, gamma_acc), obs

    init_carry = (
        mjx_data_single,
        mjx_data_single.site_xpos[EE_SITE_IDX],
        jnp.float64(0.0),
        jnp.float64(1.0),
    )

    (_, _, total_reward, _), all_obs = jax.lax.scan(
        frame_step_fn, init_carry, jnp.arange(N_TOTAL))

    return -total_reward, all_obs

# ---------------------------------------------------------------------------
# Batch loss
# ---------------------------------------------------------------------------
def batch_loss_fn(all_params, obs_mean, obs_std, mjx_data_batch, init_cube_pos_batch):
    losses, all_obs_batch = jax.vmap(
        single_env_loss_fn,
        in_axes=(None, None, None, 0, 0)
    )(all_params, obs_mean, obs_std, mjx_data_batch, init_cube_pos_batch)
    return jnp.mean(losses), (all_obs_batch, losses)

# ---------------------------------------------------------------------------
# Forward rollout for logging (batch[0])
# ---------------------------------------------------------------------------
def single_forward_fn(all_params, obs_mean, obs_std):
    def frame_step_fn(carry, step_idx):
        state, prev_ee_pos = carry

        obs, ee_pos, current_q = build_obs_depth(all_params, state, prev_ee_pos, frame_dt)
        obs_f32 = obs.astype(jnp.float32)
        proprio_norm = (obs_f32[LATENT_DIM:] - obs_mean) / obs_std
        obs_norm = jnp.concatenate([obs_f32[:LATENT_DIM], proprio_norm])

        q_dot_target, finger_target = mlp_forward(all_params, obs_norm, step_idx)
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
        cube_pos = state.xpos[CUBE1_BODY_IDX]
        actual_rew = select_reward(state, step_idx, init_cube_pos_single0)
        cube2_pos = state.xpos[CUBE2_BODY_IDX]

        step_info = jnp.concatenate([
            ee_new,                                    # 0:3
            ee_vel_new,                                # 3:6
            jnp.array([finger_target, actual_rew]),    # 6:8
            safe_q_dot,                                # 8:15
            state.qvel[:N_JOINTS],                     # 15:22
            cube_pos,                                  # 22:25
            cube2_pos,                                 # 25:28
        ])

        return (state, ee_pos), (step_info, step_qpos)

    init_carry = (mjx_data_single0, mjx_data_single0.site_xpos[EE_SITE_IDX])

    (final_state, _), (all_step_info, all_qpos) = jax.lax.scan(
        frame_step_fn, init_carry, jnp.arange(N_TOTAL))

    return final_state, all_step_info, all_qpos

# ---------------------------------------------------------------------------
# JIT compile
# ---------------------------------------------------------------------------
print("JIT compiling batch loss + grad...")
t0 = time.time()
value_and_grad_fn = jax.jit(jax.value_and_grad(batch_loss_fn, has_aux=True))
# Warmup — set 0 ile
obs_mean, obs_std = obs_rms.get_jnp()
(loss_val, (all_obs_init, _)), grad_val = value_and_grad_fn(all_params, obs_mean, obs_std,
                                                        batch_sets[0], init_cube_pos_sets[0])
loss_val.block_until_ready()
print(f"  JIT compile: {time.time()-t0:.1f}s")
print(f"  Initial loss: {float(loss_val):.6f}")

# Seed obs_rms
obs_rms.update(np.array(all_obs_init).reshape(-1, OBS_DIM)[:, LATENT_DIM:])
obs_mean, obs_std = obs_rms.get_jnp()
print(f"  Obs RMS seeded: mean range [{float(obs_mean.min()):.3f}, {float(obs_mean.max()):.3f}]")

print("\nJIT compiling forward rollout...")
t0 = time.time()
jit_forward = jax.jit(single_forward_fn)
_, info_test, qpos_test = jit_forward(all_params, obs_mean, obs_std)
info_test.block_until_ready()
print(f"  JIT compile: {time.time()-t0:.1f}s")
print(f"  all_step_info shape: {info_test.shape}")

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
seg0_dist_history = []
seg1_dist_history = []
seg2_dist_history = []
seg3_dist_history = []
seg4_dist_history = []
grad_norm_history = []
per_env_history = []

print("Training başlıyor...\n")
train_start = time.time()

for iteration in range(TRAIN_ITERS):
    if USE_VIEWER and not viewer_handle.is_running():
        print("\nViewer kapatıldı.")
        break
    
    obs_mean, obs_std = obs_rms.get_jnp()

    accum_loss = 0.0
    accum_grad = jax.tree.map(jnp.zeros_like, all_params)

    for a in range(N_ACCUM):
        batch_idx = (iteration * N_ACCUM + a) % N_BATCH_SETS
        (loss_val, (all_obs_batch, per_env_losses)), grad_val = value_and_grad_fn(
            all_params, obs_mean, obs_std,
            batch_sets[batch_idx], init_cube_pos_sets[batch_idx])

        accum_loss += float(loss_val) / N_ACCUM
        accum_grad = jax.tree.map(lambda acc, g: acc + g / N_ACCUM, accum_grad, grad_val)
        per_env_history.append(np.array(-per_env_losses))
        obs_rms.update(np.array(all_obs_batch).reshape(-1, OBS_DIM)[:, LATENT_DIM:])

    loss_f = accum_loss
    grad_val = accum_grad

    if np.isnan(loss_f):
        print(f"\n⚠ NaN at iter {iteration}!")
        break

    loss_history.append(loss_f)
    grad_norm = float(optax.global_norm(grad_val))
    grad_norm_history.append(grad_norm)

    if iteration % 5 == 0 or iteration < 3:
        _, all_info, all_qpos = jit_forward(all_params, obs_mean, obs_std)
        info_np = np.array(all_info)

        seg0_ee = info_np[SEG1_START-1, :3]
        seg0_cube = info_np[SEG1_START-1, 22:25]
        seg0_dist = np.linalg.norm(seg0_ee - seg0_cube)
        seg0_dist_history.append(seg0_dist)

        seg1_ee = info_np[SEG2_START-1, :3]
        seg1_cube = info_np[SEG2_START-1, 22:25]
        seg1_dist = np.linalg.norm(seg1_ee - seg1_cube)
        seg1_dist_history.append(seg1_dist)

        seg2_ee = info_np[SEG3_START-1, :3]
        seg2_cube = info_np[SEG3_START-1, 22:25]
        seg2_dist = np.linalg.norm(seg2_ee - seg2_cube)
        seg2_finger = info_np[SEG3_START-1, 6]
        seg2_dist_history.append(seg2_dist)

        seg3_cube1 = info_np[SEG4_START-1, 22:25]
        seg3_cube2 = info_np[SEG4_START-1, 25:28]
        stack_target = seg3_cube2 + np.array([0.0, 0.0, CUBE2_HEIGHT + STACK_OFFSET])
        seg3_stack_dist = np.linalg.norm(seg3_cube1 - stack_target)
        seg3_ee = info_np[SEG4_START-1, :3]
        seg3_ee_cube_dist = np.linalg.norm(seg3_ee - seg3_cube1)
        seg3_finger = info_np[SEG4_START-1, 6]
        seg3_dist_history.append(seg3_stack_dist)

        seg4_cube1 = info_np[-1, 22:25]
        seg4_cube2 = info_np[-1, 25:28]
        seg4_xy_err = np.linalg.norm(seg4_cube1[:2] - seg4_cube2[:2])
        seg4_z_target = seg4_cube2[2] + CUBE2_HEIGHT + STACK_OFFSET
        seg4_z_err = abs(seg4_cube1[2] - seg4_z_target)
        seg4_finger = info_np[-1, 6]
        seg4_dist_history.append(seg4_xy_err + seg4_z_err)

        print(f"\nIter {iteration:4d}:  loss={loss_f:.4f}  grad_norm={grad_norm:.4f}")
        print(f"  Seg0 end: EE-Cube={seg0_dist:.4f}")
        print(f"  Seg1 end: EE-Cube={seg1_dist:.4f}")
        print(f"  Seg2 end: EE-Cube={seg2_dist:.4f}  finger={seg2_finger:.4f}")
        print(f"  Seg3 end: Cube-Tgt={seg3_stack_dist:.4f}  EE-Cube={seg3_ee_cube_dist:.4f}  finger={seg3_finger:.4f}")
        print(f"  Seg4 end: xy_err={seg4_xy_err:.4f}  z_err={seg4_z_err:.4f}  finger={seg4_finger:.4f}")

        segments = [
            ("PRE-GRASP", 0, SEG1_START),
            ("DESCEND",   SEG1_START, SEG2_START),
            ("CLOSE",     SEG2_START, SEG3_START),
            ("MOVE",      SEG3_START, SEG4_START),
            ("RELEASE",   SEG4_START, N_TOTAL),
        ]
        for seg_name, s_start, s_end in segments:
            print(f"  --- {seg_name} (step {s_start}-{s_end-1}) ---")
            for s in range(s_start, s_end):
                ee = info_np[s, :3]
                fng = info_np[s, 6]
                rew = info_np[s, 7]
                cube = info_np[s, 22:25]
                print(f"    step {s:2d}: EE=[{ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}]  "
                      f"cube=[{cube[0]:.3f},{cube[1]:.3f},{cube[2]:.3f}]  "
                      f"fng={fng:.4f}  rew={rew:.4f}")

        render_trajectory(all_qpos, skip=1)
    else:
        seg0_dist_history.append(seg0_dist_history[-1] if seg0_dist_history else 0)
        seg1_dist_history.append(seg1_dist_history[-1] if seg1_dist_history else 0)
        seg2_dist_history.append(seg2_dist_history[-1] if seg2_dist_history else 0)
        seg3_dist_history.append(seg3_dist_history[-1] if seg3_dist_history else 0)
        seg4_dist_history.append(seg4_dist_history[-1] if seg4_dist_history else 0)

    updates, opt_state = optimizer.update(grad_val, opt_state, all_params)
    all_params = optax.apply_updates(all_params, updates)

train_time = time.time() - train_start
print(f"\nTraining done: {train_time:.1f}s ({len(loss_history)} iters, "
      f"{train_time/max(1,len(loss_history)):.3f}s/iter)")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
import pickle
with open("depth_cnn_params.pkl", "wb") as f:
    pickle.dump(jax.tree.map(lambda x: np.array(x), all_params), f)
print("Model saved: depth_cnn_params.pkl")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"SONUÇLAR — DEPTH CNN")
print(f"{'='*60}")
print(f"  Depth resolution:  {IMG_W}×{IMG_H}")
print(f"  CNN latent:        {LATENT_DIM}")
print(f"  Total params:      {n_total_params} (CNN: {n_cnn_params}, MLP: {n_mlp_params})")
print(f"  Batch size:        {BATCH_SIZE}")
print(f"  Initial loss:      {loss_history[0]:.4f}")
print(f"  Final loss:        {loss_history[-1]:.4f}")
print(f"  Seg0 final dist:   {seg0_dist_history[-1]:.4f}")
print(f"  Seg1 final dist:   {seg1_dist_history[-1]:.4f}")
print(f"  Seg2 final dist:   {seg2_dist_history[-1]:.4f}")
print(f"  Seg3 final dist:   {seg3_dist_history[-1]:.4f}")
print(f"  Seg4 final dist:   {seg4_dist_history[-1]:.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(7, 1, figsize=(10, 21))

per_env_arr = np.array(per_env_history)
for env_i in range(BATCH_SIZE):
    axes[0].plot(per_env_arr[:, env_i], alpha=0.15, linewidth=0.5, color='gray')
axes[0].plot([-l for l in loss_history], linewidth=2, color='blue', label='mean reward')
axes[0].set_ylabel("Reward")
axes[0].set_title("Per-env Reward (gray) + Mean (blue)")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(grad_norm_history, linewidth=1.5, color='red')
axes[1].set_ylabel("Grad Norm")
axes[1].set_title("Gradient Norm (before clip)")
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
axes[5].set_title("Seg3 (Move): Cube → Target")
axes[5].grid(True, alpha=0.3)

axes[6].plot(seg4_dist_history, linewidth=1.5)
axes[6].set_ylabel("Error (m)")
axes[6].set_xlabel("Iteration")
axes[6].set_title("Seg4 (Release): Align + Release Error")
axes[6].grid(True, alpha=0.3)

fig.suptitle(f"BPTT 5-Seg + DEPTH CNN — Batch={BATCH_SIZE}, {IMG_W}×{IMG_H}, "
             f"latent={LATENT_DIM}, {len(loss_history)} iters")
fig.tight_layout()
plot_path = "bptt_depth_cnn_results.png"
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