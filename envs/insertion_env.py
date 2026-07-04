"""
Peg insertion environment — peg'i slot'a sok.

7 segment:
  Seg 0: Pre-grasp peg
  Seg 1: Descend to peg
  Seg 2: Grasp peg
  Seg 3: Lift peg
  Seg 4: Align over slot
  Seg 5: Insert
  Seg 6: Release

qpos layout: [7 arm, 2 finger, 7 peg] = 16
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

PRE_GRASP_Z_OFFSET = 0.075   # peg grip üzerinden ne kadar yukarı
LIFT_Z_TARGET      = 0.1    # peg merkezi hedef z (peg_bottom ≈ 0.12, slot_entry = 0.09)

Q_LO = jnp.array([-2.897, -1.763, -2.897, -3.072, -2.897, -0.018, -2.897])
Q_HI = jnp.array([ 2.897,  1.763,  2.897, -0.070,  2.897,  3.752,  2.897])
VEL_LIMITS = jnp.array([2.17, 2.17, 2.17, 2.17, 2.61, 2.61, 2.61])

# Gripper aşağı bakıyor (peg'i yukarıdan kavra)
TARGET_QUAT = jnp.array([0.0, 0.7071068, 0.7071068, 0.0])

OBS_DIM = 48
ACT_DIM = N_JOINTS + 1

N_SEGMENTS = 7
PEG_QPOS_START = 9   # qpos[9:16] = peg freejoint

PHASE_NAMES = ['pre-grasp', 'descend', 'grasp', 'lift', 'align', 'insert', 'release']


# ---------------------------------------------------------------------------
# Index discovery
# ---------------------------------------------------------------------------
def discover_indices(mj_model):
    """Body/site index'lerini bul."""
    return {
        'ee_site_idx':          mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper"),
        'peg_body_idx':         mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "peg"),
        'peg_bottom_site_idx':  mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "peg_bottom"),
        'peg_grip_site_idx':    mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "peg_grip"),
        'slot_target_site_idx': mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "slot_target"),
        'slot_entry_site_idx':  mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "slot_entry"),
        'hand_body_idx':        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand"),
    }


def make_task_cfg(indices):
    """Reward/obs'a geçilen config dict."""
    return {
        **indices,
        'target_quat':         TARGET_QUAT,
        'pre_grasp_z_offset':  PRE_GRASP_Z_OFFSET,
        'lift_z_target':       LIFT_Z_TARGET,
    }


# ---------------------------------------------------------------------------
# Observation builder (48-dim)
# ---------------------------------------------------------------------------
def build_obs(state, prev_ee_pos, frame_dt, step_idx, cfg):
    """
    İçerik:
        ee_pos(3), ee_vel(3), finger_width(1),
        q(7), qdot(7), ee_quat(4), quat_err(4),
        peg_pos(3), peg_quat(4),
        peg_bottom(3), slot_entry(3), slot_target(3),
        peg_bottom_to_slot_target(3)
    Total = 48
    """
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    ee_vel = (ee_pos - prev_ee_pos) / frame_dt
    finger_width = state.qpos[7] + state.qpos[8]
    current_q = state.qpos[:N_JOINTS]
    current_q_dot = state.qvel[:N_JOINTS]
    ee_quat = state.xquat[cfg['hand_body_idx']]
    quat_err = TARGET_QUAT - ee_quat

    peg_pos = state.xpos[cfg['peg_body_idx']]
    peg_quat = state.xquat[cfg['peg_body_idx']]
    peg_bottom = state.site_xpos[cfg['peg_bottom_site_idx']]
    slot_entry = state.site_xpos[cfg['slot_entry_site_idx']]
    slot_target = state.site_xpos[cfg['slot_target_site_idx']]
    peg_bottom_to_slot_target = slot_target - peg_bottom

    obs = jnp.concatenate([
        ee_pos,                      #  3
        ee_vel,                      #  3
        jnp.array([finger_width]),   #  1
        current_q,                   #  7
        current_q_dot,               #  7
        ee_quat,                     #  4
        quat_err,                    #  4
        peg_pos,                     #  3
        peg_quat,                    #  4
        peg_bottom,                  #  3
        slot_entry,                  #  3
        slot_target,                 #  3
        peg_bottom_to_slot_target,   #  3
    ])                               # = 48
    return obs, ee_pos, current_q


# ---------------------------------------------------------------------------
# Batch data
# ---------------------------------------------------------------------------
def make_randomized_data(rng_key, mj_model, mj_data, mjx_model, key_id,
                         indices):
    """Peg XY pozisyonunu randomize et."""
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    noise = jax.random.uniform(rng_key, (2,), minval=-0.05, maxval=0.05)
    mj_data.qpos[PEG_QPOS_START]     += float(noise[0])   # peg x
    mj_data.qpos[PEG_QPOS_START + 1] += float(noise[1])   # peg y
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
    ("dist_above", 0.03),   # 0: EE → above peg
    ("dist_grip",  0.03),   # 1: EE → peg grip
    ("dist_grip",  0.03),   # 2: EE → peg (finger closing)
    ("lift_z",     0.15),   # 3: peg center z > threshold
    ("xy_align",   0.02),   # 4: peg_bottom XY → slot_entry < 2cm
    ("insert_z",   0.03),   # 5: peg_bottom → slot_target dist < 3cm
    ("always",     0.0),    # 6: release
]

PATIENCE = 4


def evaluate_phase_metric(info_np, phase, phase_boundaries, cfg):
    """Phase metriğini hesapla.

    info_np layout:
        0:3 ee, 3:6 ee_vel, 6 finger, 7 rew,
        8:11 peg_pos, 11:15 peg_quat,
        15:18 peg_bottom, 18:21 slot_entry, 21:24 slot_target
    """
    crit_type, _ = PHASE_CRITERIA[phase]
    seg_end = phase_boundaries[phase] - 1

    if crit_type == "dist_above":
        ee = info_np[seg_end, :3]
        peg_pos = info_np[seg_end, 8:11]
        peg_grip = peg_pos + np.array([0, 0, 0.04])
        above = peg_grip + np.array([0, 0, PRE_GRASP_Z_OFFSET])
        return np.linalg.norm(ee - above)

    elif crit_type == "dist_grip":
        ee = info_np[seg_end, :3]
        peg_pos = info_np[seg_end, 8:11] 
        peg_grip = peg_pos + np.array([0, 0, 0.04])
        return np.linalg.norm(ee - peg_grip)

    elif crit_type == "lift_z":
        peg_z = info_np[seg_end, 10]  # peg_pos z
        return -peg_z  # metrik negatif peg_z → satisfied when peg_z > threshold

    elif crit_type == "xy_align":
        peg_bottom = info_np[seg_end, 15:18]
        slot_entry = info_np[seg_end, 18:21]
        return np.sqrt((peg_bottom[0] - slot_entry[0])**2 +
                       (peg_bottom[1] - slot_entry[1])**2)

    elif crit_type == "insert_z":
        peg_bottom = info_np[seg_end, 15:18]
        slot_target = info_np[seg_end, 21:24]
        return np.linalg.norm(peg_bottom - slot_target)

    elif crit_type == "always":
        return -999.0

    return 999.0


def metric_satisfied(metric, phase):
    """Phase kriteri karşılandı mı?"""
    crit_type, threshold = PHASE_CRITERIA[phase]
    if crit_type == "always":
        return True
    elif crit_type == "lift_z":
        # metric = -peg_z, satisfied when peg_z > threshold → -peg_z < -threshold
        return metric < -threshold
    else:
        return metric < threshold