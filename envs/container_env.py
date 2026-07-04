"""
Container sorting environment — 3 nesneyi kutuya koy.

Sıra: Cube → Ball → Prism
Her nesne için 5 segment: pre-grasp, descend, grasp, move, release
Toplam: 15 segment

Nesneler:
  - cube: box 5×5×5cm
  - ball: sphere r=2.5cm
  - prism: box 8×5×4cm (yatay)
  - container: statik, 25×25cm, container_inside site
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

PRE_GRASP_Z_OFFSET = 0.075   # nesne merkezinden ne kadar yukarı
PLACE_Z_OFFSET     = 0.08    # container_inside üzerinde bırakma yüksekliği

Q_LO = jnp.array([-2.897, -1.763, -2.897, -3.072, -2.897, -0.018, -2.897])
Q_HI = jnp.array([ 2.897,  1.763,  2.897, -0.070,  2.897,  3.752,  2.897])
VEL_LIMITS = jnp.array([2.17, 2.17, 2.17, 2.17, 2.61, 2.61, 2.61])

TARGET_QUAT = jnp.array([0.0, 0.7071068, 0.7071068, 0.0])

OBS_DIM = 41
ACT_DIM = N_JOINTS + 1

# Nesne sırası (segment grupları)
OBJECT_ORDER = ['ball', 'prism', 'cube']
PHASES_PER_OBJECT = 5   # pre-grasp, descend, grasp, move, release
N_SEGMENTS = len(OBJECT_ORDER) * PHASES_PER_OBJECT  # 15

PHASE_NAMES = ['pre-grasp', 'descend', 'grasp', 'move', 'release']


def discover_indices(mj_model):
    """Body/site index'lerini bul."""
    return {
        'ee_site_idx': mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper"),
        'container_inside_site_idx': mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "container_inside"),
        'cube_body_idx':  mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "cube"),
        'ball_body_idx':  mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "ball"),
        'prism_body_idx': mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "prism"),
        'hand_body_idx':  mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand"),
    }


def make_task_cfg(indices):
    """Config dict oluştur."""
    cfg = {
        **indices,
        'target_quat': TARGET_QUAT,
        'pre_grasp_z_offset': PRE_GRASP_Z_OFFSET,
        'place_z_offset': PLACE_Z_OFFSET,
    }
    # Segment → object body idx mapping (her 5 segment bir nesne)
    obj_keys = ['cube_body_idx', 'ball_body_idx', 'prism_body_idx']
    cfg['seg_to_obj_idx'] = []
    for obj_key in obj_keys:
        for _ in range(PHASES_PER_OBJECT):
            cfg['seg_to_obj_idx'].append(indices[obj_key])
    return cfg


def get_object_idx_for_segment(seg_idx, cfg):
    """Segment numarasına göre hedef nesnenin body index'ini döndür."""
    obj_group = seg_idx // PHASES_PER_OBJECT  # 0, 1, 2
    obj_keys = ['cube_body_idx', 'ball_body_idx', 'prism_body_idx']
    return cfg[obj_keys[min(obj_group, 2)]]


# ---------------------------------------------------------------------------
# Observation builder (41-dim)
# ---------------------------------------------------------------------------
def build_obs(state, prev_ee_pos, frame_dt, step_idx, cfg):
    """
    İçerik:
        ee_pos(3), ee_vel(3), finger_width(1),
        q(7), q_dot(7), ee_quat(4), quat_err(4),
        cube_pos(3), ball_pos(3), prism_pos(3),
        container_inside_pos(3)
    """
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    ee_vel = (ee_pos - prev_ee_pos) / frame_dt
    finger_width = state.qpos[7] + state.qpos[8]
    current_q = state.qpos[:N_JOINTS]
    current_q_dot = state.qvel[:N_JOINTS]
    ee_quat = state.xquat[cfg['hand_body_idx']]
    quat_err = TARGET_QUAT - ee_quat

    cube_pos = state.xpos[cfg['cube_body_idx']]
    ball_pos = state.xpos[cfg['ball_body_idx']]
    prism_pos = state.xpos[cfg['prism_body_idx']]
    container_pos = state.site_xpos[cfg['container_inside_site_idx']]

    obs = jnp.concatenate([
        ee_pos,                         #  3
        ee_vel,                         #  3
        jnp.array([finger_width]),      #  1
        current_q,                      #  7
        current_q_dot,                  #  7
        ee_quat,                        #  4
        quat_err,                       #  4
        cube_pos,                       #  3
        ball_pos,                       #  3
        prism_pos,                      #  3
        container_pos,                  #  3
    ])                                  # = 41
    return obs, ee_pos, current_q


# ---------------------------------------------------------------------------
# Batch data
# ---------------------------------------------------------------------------
def make_randomized_data(rng_key, mj_model, mj_data, mjx_model, key_id, indices):
    """Nesne pozisyonlarını randomize et."""
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    # Cube, ball, prism qpos offset'lerini bul
    # qpos: [7arm, 2finger, 7cube, 7ball, 7prism] = 30
    k1, k2, k3 = jax.random.split(rng_key, 3)
    for i, offset in enumerate([9, 16, 23]):  # cube, ball, prism qpos start
        noise = jax.random.uniform(
            [k1, k2, k3][i], (2,), minval=-0.05, maxval=0.05)
        mj_data.qpos[offset]     += float(noise[0])
        mj_data.qpos[offset + 1] += float(noise[1])
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
# Her segment için kriter: (type, threshold)
# 5 phase per object × 3 objects = 15
PHASE_CRITERIA = []
for _ in range(3):  # cube, ball, prism — aynı kriterler
    PHASE_CRITERIA.extend([
        ("dist_above", 0.04),  # pre-grasp: EE → above object
        ("dist_obj",   0.04),  # descend: EE → object
        ("dist_obj",   0.04),  # grasp: EE → object (finger closing)
        ("xy_inside",  0.10),  # move: object XY container merkezine < 10cm
        ("always",     0.0),   # release: always pass
    ])

PATIENCE = 4


def evaluate_phase_metric(info_np, phase, phase_boundaries, cfg):
    """Phase metriğini hesapla.

    info_np layout:
        0:3 ee, 3:6 ee_vel, 6 finger, 7 rew,
        8:11 cube_pos, 11:14 ball_pos, 14:17 prism_pos,
        17:20 container_inside_pos
    """
    import numpy as np
    crit_type, _ = PHASE_CRITERIA[phase]
    seg_end = phase_boundaries[phase] - 1

    # OBJECT_ORDER'a göre doğru info_np slice'ını bul
    INFO_OBJ_OFFSETS = {'cube': 8, 'ball': 11, 'prism': 14}
    obj_name = OBJECT_ORDER[phase // PHASES_PER_OBJECT]
    obj_start = INFO_OBJ_OFFSETS[obj_name]
    obj_slice = slice(obj_start, obj_start + 3)

    if crit_type == "dist_above":
        ee = info_np[seg_end, :3]
        obj_pos = info_np[seg_end, obj_slice]
        above = obj_pos + np.array([0, 0, PRE_GRASP_Z_OFFSET])
        return np.linalg.norm(ee - above)
    elif crit_type == "dist_obj":
        ee = info_np[seg_end, :3]
        obj_pos = info_np[seg_end, obj_slice]
        return np.linalg.norm(ee - obj_pos)
    elif crit_type == "xy_inside":
        obj_pos = info_np[seg_end, obj_slice]
        container = info_np[seg_end, 17:20]
        return np.sqrt((obj_pos[0] - container[0])**2 +
                       (obj_pos[1] - container[1])**2)
    elif crit_type == "always":
        return -999.0
    return 999.0


def metric_satisfied(metric, phase):
    crit_type, threshold = PHASE_CRITERIA[phase]
    if crit_type == "always":
        return True
    return metric < threshold