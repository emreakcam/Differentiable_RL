"""
Cube stacking reward fonksiyonları — 5 segment.

Segment 0: Pre-grasp — küpün üstüne git
Segment 1: Descend  — küpe in
Segment 2: Close    — finger kapat
Segment 3: Move     — küpü hedefe taşı
Segment 4: Release  — hizala ve bırak

Her fonksiyon (state, cfg) alır, cfg dict task sabitlerini içerir:
  ee_site_idx, cube1_body_idx, cube2_body_idx,
  target_quat, pre_grasp_z_offset, cube2_height, stack_offset
"""
import jax.numpy as jnp
from helpers.mjx_utils import shaped_distance


def reward_seg0_pregrasp(state, cfg):
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    cube_pos = state.xpos[cfg['cube1_body_idx']]

    pre_grip_pos = cube_pos + jnp.array([0.0, 0.0, cfg['pre_grasp_z_offset']])
    delta = ee_pos - pre_grip_pos
    r_approach = (-jnp.sqrt(jnp.dot(delta, delta) + 1e-6) * 1.0 +
                  shaped_distance(ee_pos, pre_grip_pos, 0.3, 0.001) * 1.0)

    ee_quat = state.xquat[9]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat'])
    ori_loss = (1.0 - quat_dot * quat_dot) * r_approach

    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_open = finger_width * 5.0

    ee_z = ee_pos[2]
    ground_penalty = jnp.where(ee_z < 0.1, 5.0 * (0.1 - ee_z) / 0.1, 0.0)

    return r_approach * 2.0 - ori_loss * 5.0 - ground_penalty * 1.0 + r_finger_open


def reward_seg1_descend(state, cfg):
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    cube_pos = state.xpos[cfg['cube1_body_idx']]
    cube_holding_pos = cube_pos + jnp.array([0.0, 0.0, -0.01])

    r_approach = shaped_distance(ee_pos, cube_holding_pos, 0.1, 0.001)

    ee_quat = state.xquat[9]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat'])
    ori_loss = (1.0 - quat_dot * quat_dot) * r_approach

    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_open = finger_width * 1.0

    ee_z = ee_pos[2]
    ground_penalty = jnp.where(ee_z < 0.04, 4.0 * (0.04 - ee_z) / 0.04, 0.0)

    return r_approach * 2.0 - ori_loss * 5.0 - ground_penalty + r_finger_open


def reward_seg2_close(state, cfg):
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    cube_pos = state.xpos[cfg['cube1_body_idx']]
    cube_holding_pos = cube_pos + jnp.array([0.0, 0.0, -0.01])

    r_proximity = shaped_distance(ee_pos, cube_holding_pos, 0.1, 0.001)

    finger_width = state.qpos[7] + state.qpos[8]
    r_grip_close = -finger_width * 2.0

    ee_quat = state.xquat[9]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat'])
    ori_loss = (1.0 - quat_dot * quat_dot) * r_proximity

    ee_z = ee_pos[2]
    ground_penalty = jnp.where(ee_z < 0.04, 4.0 * (0.04 - ee_z) / 0.04, 0.0)

    return r_proximity * 3.0 + r_grip_close - ori_loss * 5.0 - ground_penalty


def reward_seg3_move(state, cfg):
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    cube1_pos = state.xpos[cfg['cube1_body_idx']]
    cube2_pos = state.xpos[cfg['cube2_body_idx']]
    cube1_z = cube1_pos[2]

    stack_target = cube2_pos + jnp.array([0.0, 0.0,
                                          cfg['cube2_height'] + cfg['stack_offset']])
    min_height = 0.05
    max_height = 0.075 + 0.01

    lift_weight = jnp.clip((cube1_z - min_height) / (max_height - min_height),
                           0.0, 1.0)

    r_cube_to_target = (
        shaped_distance(cube1_pos, stack_target, 0.5, 0.001) +
        shaped_distance(cube1_pos, stack_target, 0.1, 0.001)
    ) / 2
    r_cube_to_target = jnp.clip(r_cube_to_target, 0.0, 1.0) * lift_weight

    finger_width = state.qpos[7] + state.qpos[8]
    r_grip_close = -finger_width * 2.0

    ee_quat = state.xquat[9]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat'])
    ori_loss = 1.0 - quat_dot * quat_dot

    ee_z = ee_pos[2]
    ground_penalty = jnp.where(ee_z < 0.04, 4.0 * (0.04 - ee_z) / 0.04, 0.0)

    return (r_cube_to_target * 1.0 + lift_weight + r_grip_close
            - ori_loss * 3.0 - ground_penalty)


def reward_seg4_release(state, cfg):
    cube1_pos = state.xpos[cfg['cube1_body_idx']]
    cube2_pos = state.xpos[cfg['cube2_body_idx']]

    stack_target = cube2_pos + jnp.array([0.0, 0.0,
                                          cfg['cube2_height'] + 0.005])

    r_align = shaped_distance(cube1_pos, stack_target, 0.1, 0.001)

    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_open = finger_width * r_align

    ee_quat = state.xquat[9]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat'])
    ori_loss = 1.0 - quat_dot * quat_dot

    return r_align * 2.0 + r_finger_open - ori_loss * 3.0


def select_reward(state, step_idx, cfg, seg_boundaries):
    """Segment'e göre doğru reward fonksiyonunu seç.

    seg_boundaries: [SEG1_START, SEG2_START, SEG3_START, SEG4_START]
    """
    rew0 = reward_seg0_pregrasp(state, cfg)
    rew1 = reward_seg1_descend(state, cfg)
    rew2 = reward_seg2_close(state, cfg)
    rew3 = reward_seg3_move(state, cfg)
    rew4 = reward_seg4_release(state, cfg)

    rew = jnp.where(step_idx < seg_boundaries[0], rew0,
          jnp.where(step_idx < seg_boundaries[1], rew1,
          jnp.where(step_idx < seg_boundaries[2], rew2,
          jnp.where(step_idx < seg_boundaries[3], rew3, rew4))))
    return rew