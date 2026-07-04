"""
Peg insertion reward fonksiyonları — 7 segment.

Seg 0: Pre-grasp peg     (EE → peg üstüne)
Seg 1: Descend to peg    (EE → peg grip noktası)
Seg 2: Grasp peg         (finger kapat)
Seg 3: Lift peg          (peg'i kaldır)
Seg 4: Align over slot   (peg_bottom → slot_entry hizala)
Seg 5: Insert            (peg_bottom → slot_target aşağı it)
Seg 6: Release           (finger aç)
"""
import jax.numpy as jnp
from helpers.mjx_utils import shaped_distance

# Peg dik durmalı — identity quaternion
PEG_UPRIGHT_QUAT = jnp.array([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Generic reward helpers
# ---------------------------------------------------------------------------
def _peg_orientation_loss(state, cfg):
    """Peg'in dikliğini ölç — 0 = dik, 1 = yatık."""
    peg_quat = state.xquat[cfg['peg_body_idx']]
    quat_dot = jnp.dot(peg_quat, PEG_UPRIGHT_QUAT)
    return 1.0 - quat_dot * quat_dot


def _gripper_orientation_loss(state, cfg, weight_by=None):
    """Gripper orientation hatası. weight_by varsa çarpılır."""
    ee_quat = state.xquat[cfg['hand_body_idx']]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat'])
    loss = 1.0 - quat_dot * quat_dot
    if weight_by is not None:
        loss = loss * jnp.maximum(weight_by, 0.0)
    return loss


def _ground_penalty(state, cfg, threshold=0.04):
    """EE yere çok yakınsa ceza."""
    ee_z = state.site_xpos[cfg['ee_site_idx']][2]
    return jnp.where(ee_z < threshold,
                     5.0 * (threshold - ee_z) / threshold, 0.0)


# ---------------------------------------------------------------------------
# Segment 0: Pre-grasp peg
# ---------------------------------------------------------------------------
def _reward_pregrasp(state, cfg):
    """EE → peg grip noktasının üstüne git."""
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    peg_grip = state.site_xpos[cfg['peg_grip_site_idx']]
    pre_grip_pos = peg_grip + jnp.array([0.0, 0.0, cfg['pre_grasp_z_offset']])

    delta = ee_pos - pre_grip_pos
    r_approach = (-jnp.sqrt(jnp.dot(delta, delta) + 1e-6) * 1.0 +
                  shaped_distance(ee_pos, pre_grip_pos, 0.3, 0.001) * 1.0)

    ori_loss = _gripper_orientation_loss(state, cfg, weight_by=r_approach)

    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_open = finger_width * 5.0

    ground_pen = _ground_penalty(state, cfg, 0.1)

    return r_approach * 2.0 - ori_loss * 5.0 - ground_pen + r_finger_open


# ---------------------------------------------------------------------------
# Segment 1: Descend to peg
# ---------------------------------------------------------------------------
def _reward_descend(state, cfg):
    """EE → peg grip noktasına in."""
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    peg_grip = state.site_xpos[cfg['peg_grip_site_idx']]

    r_approach = shaped_distance(ee_pos, peg_grip, 0.1, 0.001)

    ori_loss = _gripper_orientation_loss(state, cfg, weight_by=r_approach)

    delta = ee_pos - peg_grip
    r_dist = jnp.sqrt(jnp.dot(delta, delta) + 1e-6)

    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_open = finger_width * 1.0

    ground_pen = _ground_penalty(state, cfg, 0.04)

    return r_approach * 2.0 - ori_loss * 5.0 - ground_pen + r_finger_open - r_dist * 1.0


# ---------------------------------------------------------------------------
# Segment 2: Grasp peg
# ---------------------------------------------------------------------------
def _reward_grasp(state, cfg):
    """Finger kapat, peg'e yakın kal."""
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    peg_grip = state.site_xpos[cfg['peg_grip_site_idx']]

    r_proximity = shaped_distance(ee_pos, peg_grip, 0.1, 0.001)

    finger_width = state.qpos[7] + state.qpos[8]
    r_grip_close = -finger_width * 2.0

    ori_loss = _gripper_orientation_loss(state, cfg, weight_by=r_proximity)
    ground_pen = _ground_penalty(state, cfg, 0.04)

    return r_proximity * 3.0 + r_grip_close - ori_loss * 5.0 - ground_pen


# ---------------------------------------------------------------------------
# Segment 3: Lift peg
# ---------------------------------------------------------------------------
def _reward_lift(state, cfg):
    """Peg'i kaldırırken slot_entry üzerine götür."""
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    peg_bottom = state.site_xpos[cfg['peg_bottom_site_idx']]
    peg_grip = state.site_xpos[cfg['peg_grip_site_idx']]
    slot_entry = state.site_xpos[cfg['slot_entry_site_idx']]

    # Lift gating — yeterince kalkmadan XY ödül verme
    peg_z = peg_bottom[2]
    min_z = 0.0
    target_z = slot_entry[2] + 0.03  # entry'nin biraz üstü
    lift_weight = jnp.clip(
        (peg_z - min_z) / (target_z - min_z + 1e-6), 0.0, 1.0)

    # peg_bottom → slot_entry (XY + Z birlikte, lift_weight ile gated)
    lift_target = jnp.array([slot_entry[0], slot_entry[1], target_z])
    r_to_target = (
        shaped_distance(peg_bottom, lift_target, 0.5, 0.001) +
        shaped_distance(peg_bottom, lift_target, 0.1, 0.001)
    ) / 2.0
    r_to_target = jnp.clip(r_to_target, 0.0, 1.0) * lift_weight
    delta = peg_bottom - lift_target
    r_dist = jnp.sqrt(jnp.dot(delta, delta) + 1e-6) * lift_weight

    # EE peg'e yakın kalsın
    r_ee_peg = shaped_distance(ee_pos, peg_grip, 0.1, 0.001)

    # Finger kapalı
    finger_width = state.qpos[7] + state.qpos[8]
    r_grip_close = -finger_width * 2.0

    # Peg dik kalsın
    peg_ori_loss = _peg_orientation_loss(state, cfg)
    ori_loss = _gripper_orientation_loss(state, cfg)

    return (r_to_target * 1.0 + lift_weight + r_ee_peg * 1.0 + r_grip_close
            - peg_ori_loss * 3.0 - ori_loss * 3.0 - r_dist * 1.0)


# ---------------------------------------------------------------------------
# Segment 4: Align over slot
# ---------------------------------------------------------------------------
def _reward_align(state, cfg):
    """peg_bottom'ı slot_entry üzerine hizala."""
    peg_bottom = state.site_xpos[cfg['peg_bottom_site_idx']]
    slot_entry = state.site_xpos[cfg['slot_entry_site_idx']]
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    peg_grip = state.site_xpos[cfg['peg_grip_site_idx']]

    # XY hizalama (kritik — 3mm clearance)
    xy_target = jnp.array([slot_entry[0], slot_entry[1], peg_bottom[2]])
    r_xy = (shaped_distance(peg_bottom, xy_target, 0.15, 0.001) +
            shaped_distance(peg_bottom, xy_target, 0.05, 0.001)) / 2.0

    # Z: peg_bottom slot_entry seviyesinde veya biraz üstünde
    z_target = jnp.array([peg_bottom[0], peg_bottom[1], slot_entry[2] + 0.005])
    r_z = shaped_distance(peg_bottom, z_target, 0.1, 0.001)

    delta = peg_bottom - slot_entry
    r_dist = jnp.sqrt(jnp.dot(delta, delta) + 1e-6)

    # EE peg'e yakın
    r_ee_peg = shaped_distance(ee_pos, peg_grip, 0.1, 0.001)

    # Finger kapalı
    finger_width = state.qpos[7] + state.qpos[8]
    r_grip_close = -finger_width * 2.0

    # Peg dik
    peg_ori_loss = _peg_orientation_loss(state, cfg)
    ori_loss = _gripper_orientation_loss(state, cfg)

    return (r_xy * 3.0 + r_z * 1.0 + r_ee_peg * 0.5 + r_grip_close
            - peg_ori_loss * 5.0 - ori_loss * 3.0 - r_dist * 2.0)


# ---------------------------------------------------------------------------
# Segment 5: Insert
# ---------------------------------------------------------------------------
def _reward_insert(state, cfg):
    """peg_bottom'ı slot_target'a doğru aşağı it."""
    peg_bottom = state.site_xpos[cfg['peg_bottom_site_idx']]
    slot_target = state.site_xpos[cfg['slot_target_site_idx']]
    slot_entry = state.site_xpos[cfg['slot_entry_site_idx']]
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    peg_grip = state.site_xpos[cfg['peg_grip_site_idx']]

    # XY hizalama sürdür (çok kritik — duvarla çarpışma riski)
    xy_dist = jnp.sqrt((peg_bottom[0] - slot_target[0])**2 +
                       (peg_bottom[1] - slot_target[1])**2 + 1e-6)
    r_xy = shaped_distance(
        peg_bottom[:2], slot_target[:2], 0.05, 0.001)

    # Z ilerleme — slot_entry'den slot_target'a
    z_progress = jnp.clip(
        (slot_entry[2] - peg_bottom[2]) /
        (slot_entry[2] - slot_target[2] + 1e-6), 0.0, 1.0)

    # Toplam insertion reward — XY iyi olmalı ki Z ödül kazansın
    r_insert = shaped_distance(peg_bottom, slot_target, 0.1, 0.001) * r_xy
    delta = peg_bottom - slot_target
    r_dist = jnp.sqrt(jnp.dot(delta, delta) + 1e-6) * r_xy

    # EE peg'e yakın
    r_ee_peg = shaped_distance(ee_pos, peg_grip, 0.1, 0.001)

    # Finger kapalı
    finger_width = state.qpos[7] + state.qpos[8]
    r_grip_close = -finger_width * 2.0

    # Peg dik
    peg_ori_loss = _peg_orientation_loss(state, cfg)
    ori_loss = _gripper_orientation_loss(state, cfg)

    return (r_insert * 3.0 + z_progress * 1.0 + r_xy * 1.0
            + r_ee_peg * 0.5 + r_grip_close
            - peg_ori_loss * 5.0 - ori_loss * 3.0 - r_dist * 3.0)


# ---------------------------------------------------------------------------
# Segment 6: Release
# ---------------------------------------------------------------------------
def _reward_release(state, cfg):
    """Peg slot'ta, finger aç."""
    peg_bottom = state.site_xpos[cfg['peg_bottom_site_idx']]
    slot_target = state.site_xpos[cfg['slot_target_site_idx']]

    r_in_slot = shaped_distance(peg_bottom, slot_target, 0.1, 0.001)

    # Peg dik kalsın
    peg_ori_loss = _peg_orientation_loss(state, cfg)

    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_open = finger_width * r_in_slot

    ori_loss = _gripper_orientation_loss(state, cfg)

    return r_in_slot * 2.0 + r_finger_open - peg_ori_loss * 3.0 - ori_loss * 1.0


# ---------------------------------------------------------------------------
# Reward dispatch — 7 segment
# ---------------------------------------------------------------------------
_REWARD_FNS = [
    _reward_pregrasp,   # 0
    _reward_descend,    # 1
    _reward_grasp,      # 2
    _reward_lift,       # 3
    _reward_align,      # 4
    _reward_insert,     # 5
    _reward_release,    # 6
]


def select_reward(state, step_idx, cfg, seg_boundaries):
    """7 segmentten doğru reward'ı seç.

    seg_boundaries: 6 element list (seg0-seg1, ..., seg5-seg6 sınırları).
    """
    rews = [fn(state, cfg) for fn in _REWARD_FNS]

    # Nested where: son segment'ten başa
    rew = rews[-1]
    for i in range(len(seg_boundaries) - 1, -1, -1):
        rew = jnp.where(step_idx < seg_boundaries[i], rews[i], rew)
    return rew