"""
Cabinet door opening environment — hinge cabinet sol kapıyı aç.

4 segment:
  Seg 0: Approach handle
  Seg 1: Grasp handle
  Seg 2: Pull door open
  Seg 3: Release

qpos layout: [7 arm, 2 finger, 1 left_hinge, 1 right_hinge] = 11
Hedef: left door hinge → -1.2 rad (~69° açık)
"""
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx

# ---------------------------------------------------------------------------
# Task sabitleri
# ---------------------------------------------------------------------------
N_JOINTS    = 7
FINGER_OPEN = 0.04

Q_LO = jnp.array([-2.897, -1.763, -2.897, -3.072, -2.897, -0.018, -2.897])
Q_HI = jnp.array([ 2.897,  1.763,  2.897, -0.070,  2.897,  3.752,  2.897])
VEL_LIMITS = jnp.array([2.17, 2.17, 2.17, 2.17, 2.61, 2.61, 2.61])

# Gripper yandan yaklaşım — drawer handle ile benzer
TARGET_QUAT = jnp.array([0.3430, 0.6184, 0.6183, 0.3431])

HINGE_OPEN_TARGET = -1.2   # sol kapı hedef açı (0 = kapalı, -1.57 = tam açık)
LEFT_HINGE_QPOS_IDX = 9

OBS_DIM = 36
ACT_DIM = N_JOINTS + 1

N_SEGMENTS = 4
PHASE_NAMES = ['approach', 'grasp', 'pull', 'release']


# ---------------------------------------------------------------------------
# Index discovery
# ---------------------------------------------------------------------------
def discover_indices(mj_model):
    """Body/site index'lerini bul."""
    return {
        'ee_site_idx':     mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper"),
        'handle_site_idx': mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE,
                                              "leftdoor_handle"),
        'hand_body_idx':   mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand"),
        'door_body_idx':   mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY,
                                              "hingeleftdoor"),
    }


def make_task_cfg(indices):
    """Reward/obs'a geçilen config dict."""
    return {
        **indices,
        'target_quat':       TARGET_QUAT,
        'hinge_qpos_idx':    LEFT_HINGE_QPOS_IDX,
        'hinge_open_target': HINGE_OPEN_TARGET,
    }


# ---------------------------------------------------------------------------
# Observation builder (36-dim)
# ---------------------------------------------------------------------------
def build_obs(state, prev_ee_pos, frame_dt, step_idx, cfg):
    """
    İçerik:
        ee_pos(3), ee_vel(3), finger_width(1),
        q(7), qdot(7),
        ee_quat(4), quat_err(4),
        handle_pos(3), ee_to_handle(3),
        hinge_angle(1)
    Total = 36
    """
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    ee_vel = (ee_pos - prev_ee_pos) / frame_dt
    finger_width = state.qpos[7] + state.qpos[8]
    current_q = state.qpos[:N_JOINTS]
    current_q_dot = state.qvel[:N_JOINTS]
    ee_quat = state.xquat[cfg['hand_body_idx']]
    quat_err = TARGET_QUAT - ee_quat

    handle_pos = state.site_xpos[cfg['handle_site_idx']]
    ee_to_handle = handle_pos - ee_pos
    hinge_angle = state.qpos[cfg['hinge_qpos_idx']]

    obs = jnp.concatenate([
        ee_pos,                         #  3
        ee_vel,                         #  3
        jnp.array([finger_width]),      #  1
        current_q,                      #  7
        current_q_dot,                  #  7
        ee_quat,                        #  4
        quat_err,                       #  4
        handle_pos,                     #  3
        ee_to_handle,                   #  3
        jnp.array([hinge_angle]),       #  1
    ])                                  # = 36
    return obs, ee_pos, current_q


# ---------------------------------------------------------------------------
# Batch data
# ---------------------------------------------------------------------------
def make_randomized_data(rng_key, mj_model, mj_data, mjx_model, key_id,
                         indices):
    """Arm joint'lere küçük gürültü ekle (dolap sabit)."""
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    noise = jax.random.uniform(rng_key, (N_JOINTS,),
                               minval=-0.02, maxval=0.02)
    for i in range(N_JOINTS):
        mj_data.qpos[i] += float(noise[i])
    mujoco.mj_forward(mj_model, mj_data)
    d = mjx.put_data(mj_model, mj_data)
    d = mjx.forward(mjx_model, d)
    return d


def create_batch(batch_size, mj_model, mj_data, mjx_model, key_id, indices,
                 seed=0):
    """Randomized batch oluştur."""
    batch_keys = jax.random.split(jax.random.PRNGKey(seed), batch_size)
    batch_data_list = [
        make_randomized_data(k, mj_model, mj_data, mjx_model, key_id, indices)
        for k in batch_keys
    ]
    mjx_data_batch = jax.tree.map(
        lambda *xs: jnp.stack(xs, axis=0), *batch_data_list)
    return mjx_data_batch


# ---------------------------------------------------------------------------
# Progressive training config
# ---------------------------------------------------------------------------
PHASE_CRITERIA = [
    ("dist",   0.04),   # 0: EE → handle
    ("dist",   0.04),   # 1: EE → handle (finger closing)
    ("hinge", -0.8),    # 2: hinge_angle < -0.8  (~46° açık)
    ("always", 0.0),    # 3: release
]

PATIENCE = 4


def evaluate_phase_metric(info_np, phase, phase_boundaries, cfg):
    """Phase metriğini hesapla.

    info_np layout:
        0:3 ee, 3:6 ee_vel, 6 finger, 7 rew,
        8:11 handle_pos, 11 hinge_angle
    """
    crit_type, _ = PHASE_CRITERIA[phase]
    seg_end = phase_boundaries[phase] - 1

    if crit_type == "dist":
        ee = info_np[seg_end, :3]
        handle = info_np[seg_end, 8:11]
        return np.linalg.norm(ee - handle)
    elif crit_type == "hinge":
        return info_np[seg_end, 11]  # hinge angle (negatif = açık)
    elif crit_type == "always":
        return -999.0
    return 999.0


def metric_satisfied(metric, phase):
    """Phase kriteri karşılandı mı?"""
    crit_type, threshold = PHASE_CRITERIA[phase]
    if crit_type == "always":
        return True
    elif crit_type == "hinge":
        # hinge_angle < threshold → kapı yeterince açık
        return metric <= threshold
    else:
        return metric < threshold