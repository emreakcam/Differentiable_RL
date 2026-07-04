"""
Container sorting reward fonksiyonları — 15 segment.

3 nesne × 5 aşama: pre-grasp, descend, grasp, move, release
Generic fonksiyonlar nesne body_idx'e göre parametrize edilmiş.
"""
import jax.numpy as jnp
from helpers.mjx_utils import shaped_distance

# Container iç yarıçapı — XY mesafe bunun altındaysa nesne "içeride" sayılır
# Container 25×25cm (half=0.125), küçük margin ile
INSIDE_RADIUS = 0.125


# ---------------------------------------------------------------------------
# Generic reward fonksiyonları (nesne bağımsız)
# ---------------------------------------------------------------------------
def _reward_pregrasp(state, cfg, obj_body_idx):
    """EE → nesnenin üstüne git."""
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    obj_pos = state.xpos[obj_body_idx]
    pre_grip_pos = obj_pos + jnp.array([0.0, 0.0, cfg['pre_grasp_z_offset']])
    delta = ee_pos - pre_grip_pos
    r_approach = (-jnp.sqrt(jnp.dot(delta, delta) + 1e-6) * 1.0 +
                  shaped_distance(ee_pos, pre_grip_pos, 0.3, 0.001) * 1.0)

    ee_quat = state.xquat[cfg['hand_body_idx']]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat'])
    ori_loss = (1.0 - quat_dot * quat_dot) * jnp.maximum(r_approach, 0.0)

    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_open = finger_width * 5.0

    ee_z = ee_pos[2]
    ground_penalty = jnp.where(ee_z < 0.1, 5.0 * (0.1 - ee_z) / 0.1, 0.0)

    return r_approach * 2.0 - ori_loss * 7.5 - ground_penalty + r_finger_open


def _reward_descend(state, cfg, obj_body_idx):
    """EE → nesneye in."""
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    obj_pos = state.xpos[obj_body_idx]
    hold_pos = obj_pos + jnp.array([0.0, 0.0, -0.01])

    r_approach =  shaped_distance(ee_pos, obj_pos, 0.1, 0.001)
    r_linear = -jnp.sqrt(jnp.dot(ee_pos - obj_pos, ee_pos - obj_pos) + 1e-6) * 1.0

    ee_quat = state.xquat[cfg['hand_body_idx']]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat'])
    ori_loss = (1.0 - quat_dot * quat_dot) * r_approach

    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_open = finger_width * 1.0

    ee_z = ee_pos[2]
    ground_penalty = jnp.where(ee_z < 0.04, 4.0 * (0.04 - ee_z) / 0.04, 0.0)

    return r_approach * 2.0 - ori_loss * 7.5 - ground_penalty + r_finger_open + r_linear * 2.0


def _reward_grasp(state, cfg, obj_body_idx):
    """Finger'ları kapat, nesneye yakın kal."""
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    obj_pos = state.xpos[obj_body_idx]
    hold_pos = obj_pos + jnp.array([0.0, 0.0, -0.01])

    r_proximity = shaped_distance(ee_pos, obj_pos, 0.1, 0.001)

    finger_width = state.qpos[7] + state.qpos[8]
    r_grip_close = -finger_width * 2.0

    ee_quat = state.xquat[cfg['hand_body_idx']]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat'])
    ori_loss = (1.0 - quat_dot * quat_dot) * r_proximity

    ee_z = ee_pos[2]
    ground_penalty = jnp.where(ee_z < 0.04, 4.0 * (0.04 - ee_z) / 0.04, 0.0)

    return r_proximity * 3.0 + r_grip_close - ori_loss * 7.5 - ground_penalty


def _reward_move(state, cfg, obj_body_idx):
    """Nesneyi container'ın üstüne taşı (XY radius < INSIDE_RADIUS ise tam ödül)."""
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    obj_pos = state.xpos[obj_body_idx]
    container_pos = state.site_xpos[cfg['container_inside_site_idx']]

    place_target = container_pos + jnp.array([
        0.0, 0.0, cfg['place_z_offset']])

    obj_z = obj_pos[2]
    min_height = 0.05
    max_height = cfg['place_z_offset']
    lift_weight = jnp.clip(
        (obj_z - min_height) / (max_height - min_height + 1e-6), 0.0, 1.0)

    # XY mesafe — radius içindeyse tam ödül
    xy_dist = jnp.sqrt((obj_pos[0] - container_pos[0])**2 +
                       (obj_pos[1] - container_pos[1])**2 + 1e-6)
    r_obj_to_target = jnp.where(
        xy_dist < INSIDE_RADIUS,
        1.0,
        shaped_distance(obj_pos, place_target, 0.5, 0.001))
    r_obj_to_target = r_obj_to_target * lift_weight

    # Lineer mesafe terimi — uzaktayken sabit gradient sağlar
    delta = obj_pos - place_target
    r_linear = -jnp.sqrt(jnp.dot(delta, delta) + 1e-6) * lift_weight

    finger_width = state.qpos[7] + state.qpos[8]
    r_grip_close = -finger_width * 2.0

    ee_quat = state.xquat[cfg['hand_body_idx']]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat'])
    ori_loss = 1.0 - quat_dot * quat_dot

    ee_z = ee_pos[2]
    ground_penalty = jnp.where(ee_z < 0.04, 4.0 * (0.04 - ee_z) / 0.04, 0.0)

    return (r_obj_to_target * 1.0 + r_linear * 1.0 + lift_weight + r_grip_close
            - ori_loss * 3.0 - ground_penalty)


def _reward_release(state, cfg, obj_body_idx):
    """Nesneyi container içine bırak (XY radius içindeyse tam ödül)."""
    obj_pos = state.xpos[obj_body_idx]
    container_pos = state.site_xpos[cfg['container_inside_site_idx']]

    xy_dist = jnp.sqrt((obj_pos[0] - container_pos[0])**2 +
                       (obj_pos[1] - container_pos[1])**2 + 1e-6)
    r_align = jnp.where(
        xy_dist < INSIDE_RADIUS,
        1.0,
        shaped_distance(obj_pos, container_pos, 0.1, 0.001))

    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_open = finger_width * r_align

    ee_quat = state.xquat[cfg['hand_body_idx']]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat'])
    ori_loss = 1.0 - quat_dot * quat_dot

    return r_align * 2.0 + r_finger_open - ori_loss * 3.0


# ---------------------------------------------------------------------------
# Reward dispatch — 15 segment
# ---------------------------------------------------------------------------
_GENERIC_FNS = [
    _reward_pregrasp, _reward_descend, _reward_grasp,
    _reward_move, _reward_release,
]

_OBJ_KEYS = ['ball_body_idx', 'prism_body_idx', 'cube_body_idx']


def select_reward(state, step_idx, cfg, seg_boundaries):
    """15 segmentten doğru reward'ı seç.

    seg_boundaries: 14 element list
    """
    # 15 reward hesapla: 3 nesne × 5 phase
    rews = []
    for obj_key in _OBJ_KEYS:
        obj_idx = cfg[obj_key]
        for fn in _GENERIC_FNS:
            rews.append(fn(state, cfg, obj_idx))

    # Nested where: son segment'ten başa
    rew = rews[-1]
    for i in range(len(seg_boundaries) - 1, -1, -1):
        rew = jnp.where(step_idx < seg_boundaries[i], rews[i], rew)
    return rew