"""
Drawer + Cube reward fonksiyonları — 10 segment.

Segment 0: Approach handle     Segment 5: Descend to cube
Segment 1: Grasp handle        Segment 6: Grasp cube
Segment 2: Pull drawer         Segment 7: Place cube in drawer
Segment 3: Release + lift      Segment 8: Release cube + re-approach handle
Segment 4: Above cube          Segment 9: Push drawer closed

Her fonksiyon (state, cfg) alır.
"""
import jax.numpy as jnp
from helpers.mjx_utils import shaped_distance


def reward_seg0_approach(state, cfg):
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    handle_pos = state.site_xpos[cfg['handle_site_idx']]
    delta = ee_pos - handle_pos
    r_approach = (shaped_distance(ee_pos, handle_pos, 0.3, 0.001) -
                  jnp.sqrt(jnp.dot(delta, delta) + 1e-6))
    ee_quat = state.xquat[cfg['hand_body_idx']]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat'])
    ori_loss = (1.0 - quat_dot * quat_dot) * jnp.maximum(r_approach, 0.0)
    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_open = finger_width * 5.0
    ee_z = ee_pos[2]
    ground_penalty = jnp.where(ee_z < 0.1, 5.0 * (0.1 - ee_z) / 0.1, 0.0)
    return r_approach * 2.0 - ori_loss * 5.0 + r_finger_open - ground_penalty


def reward_seg1_grasp(state, cfg):
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
    return r_proximity * 3.0 + r_grip_close - ori_loss * 5.0 - ground_penalty


def reward_seg2_pull(state, cfg):
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    handle_pos = state.site_xpos[cfg['handle_site_idx']]
    drawer_pos = state.qpos[cfg['drawer_joint_qpos_idx']]
    clamped = jnp.maximum(drawer_pos, -0.12)
    r_open = shaped_distance(jnp.array([clamped]), jnp.array([-0.12]), 0.3, 0.001)
    r_grip = shaped_distance(ee_pos, handle_pos, 0.1, 0.001)
    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_close = -finger_width * 2.0
    ee_quat = state.xquat[cfg['hand_body_idx']]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat'])
    ori_loss = 1.0 - quat_dot * quat_dot
    return r_open * 2.0 + r_grip * 1.0 + r_finger_close - ori_loss * 3.0


def reward_seg3_release_lift(state, cfg):
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    handle_pos = state.site_xpos[cfg['handle_site_idx']]
    drawer_pos = state.qpos[cfg['drawer_joint_qpos_idx']]
    clamped = jnp.maximum(drawer_pos, -0.12)
    r_open = shaped_distance(jnp.array([clamped]), jnp.array([-0.12]), 0.3, 0.001)
    lift_target = handle_pos + jnp.array([0.0, 0.0, cfg['lift_z_target']])
    r_lift = shaped_distance(ee_pos, lift_target, 0.4, 0.001) * r_open
    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_open = finger_width * 5.0
    ee_quat = state.xquat[cfg['hand_body_idx']]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat'])
    ori_loss = (1.0 - quat_dot * quat_dot) * r_lift
    return r_lift * 2.0 + r_finger_open - ori_loss * 5.0 + r_open * 0.5


def reward_seg4_above_cube(state, cfg):
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    cube_pos = state.xpos[cfg['cube_body_idx']]
    above_target = cube_pos + jnp.array([0.0, 0.0, cfg['above_cube_z']])
    delta = ee_pos - above_target
    r_above = (shaped_distance(ee_pos, above_target, 0.5, 0.001) +
               shaped_distance(ee_pos, above_target, 0.3, 0.001) -
               jnp.sqrt(jnp.dot(delta, delta) + 1e-6))
    ee_quat = state.xquat[cfg['hand_body_idx']]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat_cube'])
    ori_loss = (1.0 - quat_dot * quat_dot) * jnp.maximum(r_above, 0.0)
    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_open = finger_width * 5.0
    drawer_pos = state.qpos[cfg['drawer_joint_qpos_idx']]
    clamped = jnp.maximum(drawer_pos, -0.12)
    r_open = shaped_distance(jnp.array([clamped]), jnp.array([-0.12]), 0.3, 0.001)
    r_above = r_above * r_open
    return r_above * 2.0 - ori_loss * 5.0 + r_finger_open + r_open * 0.5


def reward_seg5_descend_cube(state, cfg):
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    cube_pos = state.xpos[cfg['cube_body_idx']]
    r_approach = shaped_distance(ee_pos, cube_pos, 0.1, 0.001)
    ee_quat = state.xquat[cfg['hand_body_idx']]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat_cube'])
    ori_loss = (1.0 - quat_dot * quat_dot) * r_approach
    finger_width = state.qpos[7] + state.qpos[8]
    r_finger_open = finger_width * 5.0
    drawer_pos = state.qpos[cfg['drawer_joint_qpos_idx']]
    clamped = jnp.maximum(drawer_pos, -0.12)
    r_open = shaped_distance(jnp.array([clamped]), jnp.array([-0.12]), 0.3, 0.001)
    return r_approach * 2.0 - ori_loss * 5.0 + r_finger_open + r_open * 0.5


def reward_seg6_grasp_cube(state, cfg):
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    cube_pos = state.xpos[cfg['cube_body_idx']]
    r_proximity = shaped_distance(ee_pos, cube_pos, 0.1, 0.001)
    finger_width = state.qpos[7] + state.qpos[8]
    r_grip_close = -finger_width * 2.0
    ee_quat = state.xquat[cfg['hand_body_idx']]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat_cube'])
    ori_loss = (1.0 - quat_dot * quat_dot) * r_proximity
    return r_proximity * 3.0 + r_grip_close - ori_loss * 5.0


def reward_seg7_place_cube(state, cfg):
    cube_pos = state.xpos[cfg['cube_body_idx']]
    inside_pos = state.site_xpos[cfg['drawer_inside_site_idx']]
    target_pos = jnp.array([inside_pos[0], inside_pos[1], cube_pos[2] + 0.03])
    r_place = shaped_distance(cube_pos, target_pos, 0.3, 0.001)
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    r_ee_cube = shaped_distance(ee_pos, cube_pos, 0.1, 0.001)
    finger_width = state.qpos[7] + state.qpos[8]
    r_finger = -finger_width * 2.0
    ee_quat = state.xquat[cfg['hand_body_idx']]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat_cube'])
    ori_loss = 1.0 - quat_dot * quat_dot
    drawer_pos = state.qpos[cfg['drawer_joint_qpos_idx']]
    clamped = jnp.maximum(drawer_pos, -0.12)
    r_open = shaped_distance(jnp.array([clamped]), jnp.array([-0.12]), 0.3, 0.001)
    return r_place * 3.0 + r_ee_cube * 1.0 + r_finger - ori_loss * 3.0 + r_open * 0.5


def reward_seg8_release_cube(state, cfg):
    cube_pos = state.xpos[cfg['cube_body_idx']]
    inside_pos = state.site_xpos[cfg['drawer_inside_site_idx']]
    r_place = shaped_distance(cube_pos, inside_pos, 0.3, 0.001)
    finger_width = state.qpos[7] + state.qpos[8]
    r_finger = finger_width * 2.0
    ee_quat = state.xquat[cfg['hand_body_idx']]
    quat_dot = jnp.dot(ee_quat, cfg['target_quat_cube'])
    ori_loss = 1.0 - quat_dot * quat_dot
    return r_place * 3.0 + r_finger - ori_loss * 1.0


def reward_seg9_close_drawer(state, cfg):
    drawer_pos = state.qpos[cfg['drawer_joint_qpos_idx']]
    cube_pos = state.xpos[cfg['cube_body_idx']]
    inside_pos = state.site_xpos[cfg['drawer_inside_site_idx']]
    r_place = shaped_distance(cube_pos, inside_pos, 0.3, 0.001)
    r_close = shaped_distance(
        jnp.array([drawer_pos]),
        jnp.array([cfg['drawer_closed_target']]), 0.3, 0.001)
    r_close = r_close * r_place
    return r_close * 3.0


# ---------------------------------------------------------------------------
# Reward dispatch
# ---------------------------------------------------------------------------
_REWARD_FNS = [
    reward_seg0_approach, reward_seg1_grasp, reward_seg2_pull,
    reward_seg3_release_lift, reward_seg4_above_cube,
    reward_seg5_descend_cube, reward_seg6_grasp_cube,
    reward_seg7_place_cube, reward_seg8_release_cube,
    reward_seg9_close_drawer,
]


def select_reward(state, step_idx, cfg, seg_boundaries):
    """Segment'e göre doğru reward fonksiyonunu seç.

    seg_boundaries: 9 element list
        [SEG1_START, SEG2_START, ..., SEG9_START]
    """
    rews = [fn(state, cfg) for fn in _REWARD_FNS]

    # Nested where: son segment'ten başa doğru
    rew = rews[-1]
    for i in range(len(seg_boundaries) - 1, -1, -1):
        rew = jnp.where(step_idx < seg_boundaries[i], rews[i], rew)
    return rew