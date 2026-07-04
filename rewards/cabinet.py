"""
Cabinet door opening reward fonksiyonları — 4 segment.

Seg 0: Approach handle   (EE → handle, finger açık)
Seg 1: Grasp handle      (finger kapat)
Seg 2: Pull door open    (hinge → target angle, EE handle'ı takip)
Seg 3: Release           (finger aç, hinge açık kalsın)
"""
import jax.numpy as jnp
from helpers.mjx_utils import shaped_distance


# ---------------------------------------------------------------------------
# Segment 0: Approach handle
# ---------------------------------------------------------------------------
def _reward_approach(state, cfg):
    """EE → handle, finger açık."""
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    handle_pos = state.site_xpos[cfg['handle_site_idx']]

    delta = ee_pos - handle_pos
    r_approach = (-jnp.sqrt(jnp.dot(delta, delta) + 1e-6) * 1.0 +
                  shaped_distance(ee_pos, handle_pos, 0.3, 0.001) * 1.0)

    ee_quat = state.xquat[cfg['hand_body_idx']]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat'])
    ori_loss = (1.0 - quat_dot * quat_dot) * jnp.maximum(r_approach, 0.0)

    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_open = finger_width * 5.0

    ee_z = ee_pos[2]
    ground_penalty = jnp.where(ee_z < 0.1, 5.0 * (0.1 - ee_z) / 0.1, 0.0)

    return r_approach * 2.0 + r_finger_open - ground_penalty


# ---------------------------------------------------------------------------
# Segment 1: Grasp handle
# ---------------------------------------------------------------------------
def _reward_grasp(state, cfg):
    """Finger kapat, handle'a yakın kal."""
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    handle_pos = state.site_xpos[cfg['handle_site_idx']]

    r_proximity = shaped_distance(ee_pos, handle_pos, 0.1, 0.001)

    finger_width = state.qpos[7] + state.qpos[8]
    r_grip_close = -finger_width * 2.0

    ee_quat = state.xquat[cfg['hand_body_idx']]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat'])
    ori_loss = (1.0 - quat_dot * quat_dot) * r_proximity

    ee_z = ee_pos[2]
    ground_penalty = jnp.where(ee_z < 0.05, 4.0 * (0.05 - ee_z) / 0.05, 0.0)

    return r_proximity * 3.0 + r_grip_close - ground_penalty


# ---------------------------------------------------------------------------
# Segment 2: Pull door open
# ---------------------------------------------------------------------------
def _reward_pull(state, cfg):
    """Hinge'i hedef açıya aç, EE handle'ı takip etsin."""
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    handle_pos = state.site_xpos[cfg['handle_site_idx']]
    hinge_angle = state.qpos[cfg['hinge_qpos_idx']]
    target = cfg['hinge_open_target']

    # Hinge progress — shaped + linear
    # Hinge negatife gidiyor (0 → -1.2), clamp progress
    progress = jnp.clip(hinge_angle / target, 0.0, 1.0)
    r_hinge = (shaped_distance(
        jnp.array([hinge_angle]), jnp.array([target]), 1.5, 0.01) +
        progress) / 2.0

    # EE handle'a yakın kalsın (handle ark çiziyor, site otomatik güncellenir)
    r_grip = shaped_distance(ee_pos, handle_pos, 0.1, 0.001)

    # Finger kapalı
    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_close = -finger_width * 2.0

    # Orientation
    ee_quat = state.xquat[cfg['hand_body_idx']]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat'])
    ori_loss = 1.0 - quat_dot * quat_dot

    return (r_hinge * 3.0 + r_grip * 1.5 + r_finger_close
            )


# ---------------------------------------------------------------------------
# Segment 3: Release
# ---------------------------------------------------------------------------
def _reward_release(state, cfg):
    """Finger aç, hinge açık kalsın."""
    hinge_angle = state.qpos[cfg['hinge_qpos_idx']]
    target = cfg['hinge_open_target']

    r_hinge = shaped_distance(
        jnp.array([hinge_angle]), jnp.array([target]), 1.5, 0.01)

    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_open = finger_width * r_hinge

    ee_quat = state.xquat[cfg['hand_body_idx']]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat'])
    ori_loss = 1.0 - quat_dot * quat_dot

    return r_hinge * 2.0 + r_finger_open


# ---------------------------------------------------------------------------
# Reward dispatch — 4 segment
# ---------------------------------------------------------------------------
_REWARD_FNS = [
    _reward_approach,   # 0
    _reward_grasp,      # 1
    _reward_pull,       # 2
    _reward_release,    # 3
]


def select_reward(state, step_idx, cfg, seg_boundaries):
    """4 segmentten doğru reward'ı seç.

    seg_boundaries: 3 element list.
    """
    rews = [fn(state, cfg) for fn in _REWARD_FNS]

    rew = rews[-1]
    for i in range(len(seg_boundaries) - 1, -1, -1):
        rew = jnp.where(step_idx < seg_boundaries[i], rews[i], rew)
    return rew