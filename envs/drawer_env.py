"""
Drawer + Cube environment bileşenleri — 10 segment.

Segment 0-2: Handle approach → grasp → pull drawer
Segment 3:   Release handle + lift
Segment 4-6: Cube approach → descend → grasp
Segment 7:   Place cube in drawer
Segment 8:   Re-approach handle
Segment 9:   Push drawer closed
"""
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

# ---------------------------------------------------------------------------
# Task sabitleri
# ---------------------------------------------------------------------------
N_JOINTS    = 7
FINGER_OPEN = 0.04

LIFT_Z_TARGET        = 0.20
ABOVE_CUBE_Z         = 0.1
DRAWER_OPEN_TARGET   = -0.24
DRAWER_CLOSED_TARGET = 0.0
CUBE_QPOS_START      = 12    # cube qpos offset in qpos vector

Q_LO = jnp.array([-2.897, -1.763, -2.897, -3.072, -2.897, -0.018, -2.897])
Q_HI = jnp.array([ 2.897,  1.763,  2.897, -0.070,  2.897,  3.752,  2.897])
VEL_LIMITS = jnp.array([2.17, 2.17, 2.17, 2.17, 2.61, 2.61, 2.61])

TARGET_QUAT      = jnp.array([0.3430, 0.6184, 0.6183, 0.3431])
TARGET_QUAT_CUBE = jnp.array([0.0, 0.7071068, 0.7071068, 0.0])

OBS_DIM = 46
ACT_DIM = N_JOINTS + 1


def discover_indices(mj_model):
    """Drawer task body/site index'lerini bul."""
    return {
        'hand_body_idx':          mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand"),
        'drawer_body_idx':        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "studyTable_Drawer"),
        'cube_body_idx':          mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "cube"),
        'ee_site_idx':            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper"),
        'handle_site_idx':        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "drawer_handle"),
        'drawer_inside_site_idx': mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "drawer_inside"),
        'drawer_joint_qpos_idx':  11,
    }


def make_task_cfg(indices):
    """Reward ve obs fonksiyonlarına geçilecek config dict."""
    return {
        **indices,
        'target_quat':      TARGET_QUAT,
        'target_quat_cube': TARGET_QUAT_CUBE,
        'lift_z_target':    LIFT_Z_TARGET,
        'above_cube_z':     ABOVE_CUBE_Z,
        'drawer_open_target':   DRAWER_OPEN_TARGET,
        'drawer_closed_target': DRAWER_CLOSED_TARGET,
    }


# ---------------------------------------------------------------------------
# Observation builder (46-dim)
# ---------------------------------------------------------------------------
def build_obs(state, prev_ee_pos, frame_dt, step_idx, cfg):
    """Drawer task observation (46-dim).

    İçerik:
        ee_pos(3), handle_pos(3), ee_to_handle(3), ee_vel(3),
        finger_width(1), q(7), q_dot(7), ee_quat(4), quat_err(4),
        drawer_joint_pos(1), cube_pos(3), ee_to_cube(3), cube_quat(4)
    """
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    handle_pos = state.site_xpos[cfg['handle_site_idx']]
    ee_to_handle = handle_pos - ee_pos
    ee_vel = (ee_pos - prev_ee_pos) / frame_dt
    finger_width = state.qpos[7] + state.qpos[8]
    current_q = state.qpos[:N_JOINTS]
    current_q_dot = state.qvel[:N_JOINTS]
    ee_quat = state.xquat[cfg['hand_body_idx']]
    quat_err = TARGET_QUAT - ee_quat
    drawer_joint_pos = state.qpos[cfg['drawer_joint_qpos_idx']]
    cube_pos = state.xpos[cfg['cube_body_idx']]
    ee_to_cube = cube_pos - ee_pos
    cube_quat = state.xquat[cfg['cube_body_idx']]

    obs = jnp.concatenate([
        ee_pos,                             #  3
        handle_pos,                         #  3
        ee_to_handle,                       #  3
        ee_vel,                             #  3
        jnp.array([finger_width]),          #  1
        current_q,                          #  7
        current_q_dot,                      #  7
        ee_quat,                            #  4
        quat_err,                           #  4
        jnp.array([drawer_joint_pos]),      #  1
        cube_pos,                           #  3
        ee_to_cube,                         #  3
        cube_quat,                          #  4
    ])                                      # = 46
    return obs, ee_pos, current_q


# ---------------------------------------------------------------------------
# Batch data — randomized cube positions
# ---------------------------------------------------------------------------
def make_randomized_data(rng_key, mj_model, mj_data, mjx_model, key_id):
    """Tek bir randomized env verisi oluştur (cube pos noise)."""
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    noise_cube = jax.random.uniform(rng_key, (2,), minval=-0.03, maxval=0.03)
    mj_data.qpos[CUBE_QPOS_START]     += float(noise_cube[0])
    mj_data.qpos[CUBE_QPOS_START + 1] += float(noise_cube[1])
    mujoco.mj_forward(mj_model, mj_data)
    d = mjx.put_data(mj_model, mj_data)
    d = mjx.forward(mjx_model, d)
    return d


def create_batch(batch_size, mj_model, mj_data, mjx_model, key_id, seed=0):
    """Randomized cube pozisyonlarıyla batch oluştur."""
    batch_keys = jax.random.split(jax.random.PRNGKey(seed), batch_size)
    batch_data_list = [
        make_randomized_data(k, mj_model, mj_data, mjx_model, key_id)
        for k in batch_keys
    ]
    mjx_data_batch = jax.tree.map(
        lambda *xs: jnp.stack(xs, axis=0), *batch_data_list)
    return mjx_data_batch


# ---------------------------------------------------------------------------
# Progressive training config
# ---------------------------------------------------------------------------
PHASE_CRITERIA = [
    ("dist",   0.04),   # 0: EE→handle
    ("dist",   0.04),   # 1: EE→handle
    ("drawer", -0.10),  # 2: drawer_pos
    ("dist",   0.06),   # 3: EE→lift_target
    ("dist",   0.05),   # 4: EE→above_cube
    ("dist",   0.05),   # 5: EE→cube
    ("dist",   0.05),   # 6: EE→cube
    ("dist",   0.10),   # 7: cube→inside
    ("dist",   0.05),   # 8: EE→handle
    ("always", 0.0),    # 9: final
]

PATIENCE = 4


def evaluate_phase_metric(info_np, phase, seg_boundaries, cfg):
    """Phase tamamlanma metriğini hesapla."""
    import numpy as np
    crit_type, _ = PHASE_CRITERIA[phase]
    phase_bounds = seg_boundaries + [seg_boundaries[-1] + 1]  # dummy end
    seg_end = (seg_boundaries[phase] if phase < len(seg_boundaries)
               else seg_boundaries[-1]) - 1

    if crit_type == "dist":
        if phase in (0, 1, 8):
            ee = info_np[seg_end, :3]
            handle = info_np[seg_end, 3:6]
            return np.linalg.norm(ee - handle)
        elif phase == 3:
            ee = info_np[seg_end, :3]
            handle = info_np[seg_end, 3:6]
            lift_target = handle + np.array([0, 0, LIFT_Z_TARGET])
            return np.linalg.norm(ee - lift_target)
        elif phase == 4:
            ee = info_np[seg_end, :3]
            cube = info_np[seg_end, 23:26]
            above_target = cube + np.array([0, 0, ABOVE_CUBE_Z])
            return np.linalg.norm(ee - above_target)
        elif phase in (5, 6):
            ee = info_np[seg_end, :3]
            cube = info_np[seg_end, 23:26]
            return np.linalg.norm(ee - cube)
        elif phase in (7, 8):
            cube = info_np[seg_end, 23:26]
            inside = info_np[seg_end, 26:29]
            return np.linalg.norm(cube - inside)
    elif crit_type == "drawer":
        return info_np[seg_end, 22]
    elif crit_type == "always":
        return -999.0
    return 999.0


def metric_satisfied(metric, phase):
    """Metrik threshold'u geçti mi?"""
    crit_type, threshold = PHASE_CRITERIA[phase]
    if crit_type == "always":
        return True
    elif crit_type == "drawer":
        return metric <= threshold
    else:
        return metric < threshold