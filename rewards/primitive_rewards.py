import jax.numpy as jnp
from helpers.mjx_utils import shaped_distance

# ═══════════════════════════════════════════════════════════════════════════
# REWARD FUNCTIONS (inline, robosuite finger convention)
# ═══════════════════════════════════════════════════════════════════════════
def _finger_width(state):
    """Robosuite gripper opening: qpos[7] - qpos[8]."""
    return state.qpos[7] - state.qpos[8]


def reward_reach(state, target_pos, target_quat, ee_site_idx, hand_body_idx):
    ee_pos = state.site_xpos[ee_site_idx]
    delta = ee_pos - target_pos
    r_approach = (-jnp.sqrt(jnp.dot(delta, delta) + 1e-6) +
                  shaped_distance(ee_pos, target_pos, 0.1, 0.001))

    ee_quat = state.xquat[hand_body_idx]
    quat_dot = jnp.dot(ee_quat, target_quat)
    ori_loss = (1.0 - quat_dot * quat_dot) * jnp.maximum(r_approach, 0.0)

    finger = _finger_width(state)
    r_finger = finger * 5.0

    ee_z = ee_pos[2]
    ground_pen = jnp.where(ee_z < 0.1, 5.0 * (0.1 - ee_z) / 0.1, 0.0)

    return r_approach * 2.0 - ori_loss * 5.0 + r_finger - ground_pen


def reward_descend(state, target_pos, target_quat, ee_site_idx, hand_body_idx):
    ee_pos = state.site_xpos[ee_site_idx]
    delta = ee_pos - target_pos
    r_approach = (-jnp.sqrt(jnp.dot(delta, delta) + 1e-6) +
                  shaped_distance(ee_pos, target_pos, 0.1, 0.001))

    ee_quat = state.xquat[hand_body_idx]
    quat_dot = jnp.dot(ee_quat, target_quat)
    ori_loss = (1.0 - quat_dot * quat_dot) * jnp.maximum(r_approach, 0.0)

    finger = _finger_width(state)
    r_finger = finger * 1.0

    ee_z = ee_pos[2]
    ground_pen = jnp.where(ee_z < 0.04, 4.0 * (0.04 - ee_z) / 0.04, 0.0)

    return r_approach * 2.0 - ori_loss * 5.0 + r_finger - ground_pen


def reward_grasp(state, target_pos, target_quat, ee_site_idx, hand_body_idx):
    ee_pos = state.site_xpos[ee_site_idx]
    r_prox = shaped_distance(ee_pos, target_pos, 0.1, 0.001)

    finger = _finger_width(state)
    r_grip = -finger * 2.0

    ee_quat = state.xquat[hand_body_idx]
    quat_dot = jnp.dot(ee_quat, target_quat)
    ori_loss = (1.0 - quat_dot * quat_dot) * jnp.maximum(r_prox, 0.0)

    ee_z = ee_pos[2]
    ground_pen = jnp.where(ee_z < 0.04, 4.0 * (0.04 - ee_z) / 0.04, 0.0)

    return r_prox * 3.0 - ori_loss * 5.0 + r_grip - ground_pen


def reward_move_generic(state, obj_pos, target_pos, ee_site_idx,
                        hand_body_idx, target_quat):
    ee_pos = state.site_xpos[ee_site_idx]

    obj_z = obj_pos[2]
    target_z = target_pos[2]
    ground_z = 0.8
    lift_weight = jnp.clip(
        (obj_z - ground_z) / (target_z - ground_z + 1e-6), 0.0, 1.0)

    r_shaped = (
        shaped_distance(obj_pos, target_pos, 0.5, 0.001) +
        shaped_distance(obj_pos, target_pos, 0.1, 0.001)
    ) / 2.0
    r_shaped = jnp.clip(r_shaped, 0.0, 1.0) * lift_weight

    delta = obj_pos - target_pos
    r_linear = -jnp.sqrt(jnp.dot(delta, delta) + 1e-6) * lift_weight

    finger = _finger_width(state)
    r_grip = -finger * 2.0

    ee_quat = state.xquat[hand_body_idx]
    quat_dot = jnp.dot(ee_quat, target_quat)
    ori_loss = (1.0 - quat_dot * quat_dot) * jnp.maximum(r_shaped, 0.0)

    ee_z = ee_pos[2]
    ground_pen = jnp.where(ee_z < 0.04, 4.0 * (0.04 - ee_z) / 0.04, 0.0)

    return (r_shaped * 2.0 + r_linear * 1.0 + lift_weight + r_grip
            - ori_loss * 5.0 - ground_pen)


def reward_release_generic(state, target_pos, ee_site_idx,
                           hand_body_idx, target_quat):
    ee_pos = state.site_xpos[ee_site_idx]
    r_stay = shaped_distance(ee_pos, target_pos, 0.1, 0.001)

    finger = _finger_width(state)
    r_finger = finger * 5.0

    ee_quat = state.xquat[hand_body_idx]
    quat_dot = jnp.dot(ee_quat, target_quat)
    ori_loss = 1.0 - quat_dot * quat_dot

    return r_stay * 1.0 + r_finger - ori_loss * 3.0

def reward_slide(state, handle_site_idx, qpos_idx, target,
                 init, ee_site_idx, hand_body_idx):
    """Slide joint toward target — works for both pull and push."""
    ee_pos = state.site_xpos[ee_site_idx]
    handle_pos = state.site_xpos[handle_site_idx]
    drawer_pos = state.qpos[qpos_idx]

    total_travel = target - init
    # Fraction of the way from `init` to `target`: 0 at the start, 1 at the
    # target, and falling again past it so an overshoot costs exactly what
    # stopping the same distance short costs.
    #
    # Clipping at 1.0 instead made every overshoot score identically to a
    # perfect stop, and since shaped_distance below has already decayed to ~0 by
    # then, the term plateaued: a dial spun a full radian past its target scored
    # the same as one spun 0.5 past, with no gradient pulling it back — while
    # success_fn, which measures |q - target|, counted both as failures.
    #
    # This changes nothing for drawer/window/door: their targets sit ON a joint
    # limit, so the hard stop keeps `raw` at 1.0 and the branch never fires.
    # The dial is the one articulated task whose target is mid-range.
    raw = (drawer_pos - init) / (total_travel + 1e-8)
    progress = jnp.clip(jnp.where(raw > 1.0, 2.0 - raw, raw), 0.0, 1.0)
    r_joint = (shaped_distance(
        jnp.array([drawer_pos]), jnp.array([target]), 0.3, 0.001) +
        progress) / 2.0

    r_grip = shaped_distance(ee_pos, handle_pos, 0.1, 0.001)

    finger = state.qpos[7] - state.qpos[8]
    r_finger = -finger * 2.0

    return r_joint * 2.0 + r_grip * 1.5 + r_finger

def reward_push_object(state, obj_pos, target_pos):
    """Lean push reward — ONLY the object-to-target gap.

    Deliberately minimal compared with reward_slide (3 terms) or
    reward_move_generic (6): no grip term, no orientation term, no ground
    penalty. The linear -dist supplies gradient everywhere, and the shaped
    term sharpens the last few centimetres.

    Still a *dense* reward: it is a smooth function of distance at every
    timestep, not a sparse 0/1 on success. A truly sparse reward would give
    zero gradient almost everywhere, which BPTT through the simulator cannot
    learn from at all.
    """
    delta = obj_pos - target_pos
    dist = jnp.sqrt(jnp.dot(delta, delta) + 1e-6)
    return -dist + shaped_distance(obj_pos, target_pos, 0.1, 0.001)


# ═══════════════════════════════════════════════════════════════════════════
# PEG INSERTION
# ═══════════════════════════════════════════════════════════════════════════
# The peg's long axis is its local z and it spawns unrotated, so "upright" is
# simply the identity quaternion — no calibrated constant needed.
PEG_UPRIGHT_QUAT = jnp.array([1.0, 0.0, 0.0, 0.0])


def _ori_loss(quat, target_quat):
    """1 - cos^2 of the angle between two quaternions (sign-agnostic)."""
    d = jnp.dot(quat, target_quat)
    return 1.0 - d * d


def reward_move_peg(state, obj_pos, target_pos, peg_body_idx, ee_site_idx,
                    hand_body_idx, target_quat, ori_weight=5.0):
    """reward_move_generic plus a peg-upright penalty.

    reward_move_generic only constrains the *end-effector* orientation, which
    leaves the peg free to be carried tilted — and with ~5 mm of clearance the
    align that follows cannot recover from that. Uprightness has to be held
    during transport, not corrected at the hole.
    """
    base = reward_move_generic(state, obj_pos, target_pos, ee_site_idx,
                               hand_body_idx, target_quat)
    peg_ori = _ori_loss(state.xquat[peg_body_idx], PEG_UPRIGHT_QUAT)
    return base - peg_ori * ori_weight


def reward_align(state, peg_bottom_idx, slot_entry_idx, peg_grip_idx,
                 peg_body_idx, ee_site_idx, hand_body_idx, target_quat):
    """Bring the peg tip over the hole and stand the peg up.

    Drives peg_bottom's *xy* onto the slot entry while easing z down to just
    above it, so the peg hovers ready to drop straight in. The hole has only
    ~5 mm clearance per side, so the upright penalty carries real weight here:
    a few degrees of tilt makes the insert that follows impossible.
    """
    peg_bottom = state.site_xpos[peg_bottom_idx]
    slot_entry = state.site_xpos[slot_entry_idx]
    ee_pos = state.site_xpos[ee_site_idx]
    peg_grip = state.site_xpos[peg_grip_idx]

    xy_target = jnp.array([slot_entry[0], slot_entry[1], peg_bottom[2]])
    r_xy = (shaped_distance(peg_bottom, xy_target, 0.15, 0.001) +
            shaped_distance(peg_bottom, xy_target, 0.05, 0.001)) / 2.0

    z_target = jnp.array([peg_bottom[0], peg_bottom[1], slot_entry[2] + 0.005])
    r_z = shaped_distance(peg_bottom, z_target, 0.1, 0.001)

    delta = peg_bottom - slot_entry
    r_dist = -jnp.sqrt(jnp.dot(delta, delta) + 1e-6)

    r_hold = shaped_distance(ee_pos, peg_grip, 0.1, 0.001)
    r_grip = -_finger_width(state) * 2.0

    peg_ori = _ori_loss(state.xquat[peg_body_idx], PEG_UPRIGHT_QUAT)
    ee_ori = _ori_loss(state.xquat[hand_body_idx], target_quat)

    return (r_xy * 3.0 + r_z * 1.0 + r_dist * 2.0 + r_hold * 0.5 + r_grip
            - peg_ori * 5.0 - ee_ori * 3.0)


def reward_insert(state, peg_bottom_idx, slot_target_idx, slot_entry_idx,
                  peg_grip_idx, peg_body_idx, ee_site_idx, hand_body_idx,
                  target_quat):
    """Drive the peg tip down to the hole floor, gated on xy alignment.

    r_xy multiplies the descent terms deliberately: without that gate the
    policy can collect reward for pushing the peg downward while it is over a
    wall, which is the classic way a shaped insertion reward gets gamed.
    """
    peg_bottom = state.site_xpos[peg_bottom_idx]
    slot_target = state.site_xpos[slot_target_idx]
    slot_entry = state.site_xpos[slot_entry_idx]
    ee_pos = state.site_xpos[ee_site_idx]
    peg_grip = state.site_xpos[peg_grip_idx]

    r_xy = shaped_distance(peg_bottom[:2], slot_target[:2], 0.05, 0.001)

    z_progress = jnp.clip(
        (slot_entry[2] - peg_bottom[2]) /
        (slot_entry[2] - slot_target[2] + 1e-6), 0.0, 1.0)

    r_insert = shaped_distance(peg_bottom, slot_target, 0.1, 0.001) * r_xy

    delta = peg_bottom - slot_target
    r_dist = -jnp.sqrt(jnp.dot(delta, delta) + 1e-6) * r_xy

    r_hold = shaped_distance(ee_pos, peg_grip, 0.1, 0.001)
    r_grip = -_finger_width(state) * 2.0

    peg_ori = _ori_loss(state.xquat[peg_body_idx], PEG_UPRIGHT_QUAT)
    ee_ori = _ori_loss(state.xquat[hand_body_idx], target_quat)

    return (r_insert * 3.0 + z_progress * 1.0 + r_xy * 1.0 + r_dist * 3.0
            + r_hold * 0.5 + r_grip
            - peg_ori * 5.0 - ee_ori * 3.0)
