# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# warp_kinematics.py
#
# Warp-native forward kinematics and analytical position Jacobian for
# a serial revolute chain, with a fixed end-effector (TCP) offset.
#
# Matches the conventions used by helpers/geom_diff_helpers.py and
# helpers/kinematic_helpers.py (from the existing torch-side code):
#
#   - Quaternions are [w, x, y, z]
#   - Joint k: first apply relative link transform (link_pos[k], link_quat[k]),
#     then rotate by angle q[k] around the local joint axis joint_axes[k].
#   - After the last revolute joint, a rigid EE offset (ee_offset_p, ee_offset_q)
#     is applied to reach the TCP frame.
#
# All kernels are fully differentiable — they only use operations with
# Warp adjoints (add, cross, normalize, transform_vector, quat_rotate, etc.)
# and contain no data-dependent control flow.
#
# Launch convention:
#   fk_kernel         : dim=1 (single chain)
#   jacobian_kernel   : dim=N_JOINTS (one thread per joint column)
#   damped_pinv_kernel: dim=1 (single 3x3 solve)
#   integrate_q_kernel: dim=N_JOINTS (one thread per joint)
###########################################################################

import warp as wp

# The Franka FR3 arm has 7 revolute joints. We hard-code this so kernel
# signatures can use fixed-size types (vec7 doesn't exist, so we use
# wp.array parameters and rely on launch dim for iteration).
N_JOINTS = 7


# ---------------------------------------------------------------------------
# Forward Kinematics kernel
# ---------------------------------------------------------------------------

@wp.kernel
def fk_tcp_kernel(
    q:              wp.array(dtype=float),          # [N_JOINTS]  joint angles
    joint_axes:     wp.array(dtype=wp.vec3),        # [N_JOINTS]  axis in joint local frame
    link_pos:       wp.array(dtype=wp.vec3),        # [N_JOINTS]  translation into joint k frame
    link_quats:     wp.array(dtype=wp.quat),        # [N_JOINTS]  rotation into joint k frame [x,y,z,w]*
    ee_offset_p:    wp.vec3,                         # translation from last joint to TCP
    ee_offset_q:    wp.quat,                         # rotation from last joint to TCP
    # outputs:
    ee_pos:         wp.array(dtype=wp.vec3),        # [1]
    joint_pos_w:    wp.array(dtype=wp.vec3),        # [N_JOINTS]  world position of joint k origin
    joint_axis_w:   wp.array(dtype=wp.vec3),        # [N_JOINTS]  world axis of joint k
):
    """Forward kinematics for the full chain. Launched with dim=1.

    Accumulates the running (pos, quat) transform link-by-link, applies
    the joint rotation at each link, and finally applies the rigid TCP offset.

    Also stores each joint's world position and world axis — these are
    needed by the Jacobian kernel to avoid recomputing FK inside it.

    NOTE on wp.quat convention: Warp uses [x, y, z, w] internally. We
    accept quaternions as wp.quat (which IS [x,y,z,w]). The caller is
    responsible for feeding quaternions in [x,y,z,w] order — see
    load_chain_to_warp() in this module.
    """
    # Running accumulator: (accum_pos, accum_quat) = transform from world to
    # joint k's frame (before the joint rotation is applied).
    accum_pos = wp.vec3(0.0, 0.0, 0.0)
    accum_quat = wp.quat_identity()

    for k in range(N_JOINTS):
        # Apply the relative link transform into joint k's frame.
        accum_pos = accum_pos + wp.quat_rotate(accum_quat, link_pos[k])
        accum_quat =  accum_quat * link_quats[k]

        # Record joint k's world-frame origin and rotation axis BEFORE
        # applying its own rotation — the axis is fixed in this frame.
        joint_pos_w[k] = accum_pos
        joint_axis_w[k] = wp.quat_rotate(accum_quat, joint_axes[k])

        # Apply the joint rotation itself: rotation of q[k] around local axis.
        half = q[k] * 0.5
        s = wp.sin(half)
        c = wp.cos(half)
        axis = joint_axes[k]
        joint_q = wp.quat(s * axis[0], s * axis[1], s * axis[2], c)
        accum_quat = accum_quat * joint_q

    # Apply the rigid TCP offset after the last revolute joint.
    tcp_pos = accum_pos + wp.quat_rotate(accum_quat, ee_offset_p)
    # (We don't need the final TCP orientation for a position-only Jacobian,
    #  but we could propagate it if needed.)
    ee_pos[0] = tcp_pos


# ---------------------------------------------------------------------------
# Analytical position Jacobian kernel
# ---------------------------------------------------------------------------

@wp.kernel
def jacobian_p_tcp_kernel(
    joint_pos_w:   wp.array(dtype=wp.vec3),    # [N_JOINTS]   from fk_tcp_kernel
    joint_axis_w:  wp.array(dtype=wp.vec3),    # [N_JOINTS]   from fk_tcp_kernel
    ee_pos:        wp.array(dtype=wp.vec3),    # [1]          from fk_tcp_kernel
    # output: stored column-major as flat array of length 3*N_JOINTS
    # J_p_flat[3*k + i] = J_p[i, k]
    J_p_flat:      wp.array(dtype=float),
):
    """Geometric position Jacobian: column k = axis_k_world × (p_ee − p_k_world).

    Launched with dim=N_JOINTS, one thread per joint column.

    This is the textbook geometric Jacobian for revolute joints:
        ∂p_ee / ∂q_k = ω_k × r_{k→ee}
    where ω_k is the joint's world-frame rotation axis and r_{k→ee} is the
    lever arm from the joint origin to the EE, both in world frame.
    """
    k = wp.tid()
    lever = ee_pos[0] - joint_pos_w[k]
    col = wp.cross(joint_axis_w[k], lever)
    J_p_flat[3 * k + 0] = col[0]
    J_p_flat[3 * k + 1] = col[1]
    J_p_flat[3 * k + 2] = col[2]


# ---------------------------------------------------------------------------
# Damped pseudo-inverse: q̇ = Jᵀ (J Jᵀ + λI)⁻¹ v
# ---------------------------------------------------------------------------

@wp.kernel
def damped_pinv_3xN_kernel(
    J_p_flat:  wp.array(dtype=float),      # [3 * N_JOINTS]
    v_target:  wp.array(dtype=float),      # [3] — passed as array so gradients flow
    damping:   float,
    # output:
    q_dot:     wp.array(dtype=float),      # [N_JOINTS]
):
    """Damped least-squares solve for q_dot given desired EE velocity v_target.

    Launched with dim=1 (all the arithmetic is small: 3x3 matrix inverse).

    Formula:
        A  = J Jᵀ + λI                  (3×3, symmetric positive definite)
        y  = A⁻¹ v                      (3-vector)
        q̇  = Jᵀ y                       (N-vector)

    v_target is passed as an array (not wp.vec3) so the Warp tape can
    propagate gradients back to the learnable parameter.

    Kernel uses wp.mat33 + wp.inverse for the 3×3 solve — all operations
    have proper Warp adjoints, so gradients flow cleanly through the pinv.
    """
    # Build J Jᵀ (3×3). J_p_flat stores column k at offset 3*k.
    A00 = float(0.0)
    A01 = float(0.0)
    A02 = float(0.0)
    A11 = float(0.0)
    A12 = float(0.0)
    A22 = float(0.0)

    for k in range(N_JOINTS):
        c0 = J_p_flat[3 * k + 0]
        c1 = J_p_flat[3 * k + 1]
        c2 = J_p_flat[3 * k + 2]
        A00 += c0 * c0
        A01 += c0 * c1
        A02 += c0 * c2
        A11 += c1 * c1
        A12 += c1 * c2
        A22 += c2 * c2

    # Add damping λI. Symmetric matrix — mat33 constructor below is row-major.
    A = wp.mat33(
        A00 + damping, A01,           A02,
        A01,           A11 + damping, A12,
        A02,           A12,           A22 + damping,
    )
    A_inv = wp.inverse(A)

    # Read v_target from array and build vec3 (keeps tape alive).
    v = wp.vec3(v_target[0], v_target[1], v_target[2])

    # y = A_inv @ v
    y = A_inv * v

    # q_dot[k] = Jᵀ[k] · y = (column k of J) · y
    for k in range(N_JOINTS):
        c0 = J_p_flat[3 * k + 0]
        c1 = J_p_flat[3 * k + 1]
        c2 = J_p_flat[3 * k + 2]
        q_dot[k] = c0 * y[0] + c1 * y[1] + c2 * y[2]


# ---------------------------------------------------------------------------
# Integrate: q_target = q_current + q_dot * dt
# ---------------------------------------------------------------------------

@wp.kernel
def integrate_q_kernel(
    q_current:    wp.array(dtype=float),    # [N_JOINTS_MODEL] (reads only first N_JOINTS)
    q_dot:        wp.array(dtype=float),    # [N_JOINTS]
    dt:           float,
    # output: full joint_target_pos buffer of the Newton control (length 9 for Franka+hand)
    # We only write the first N_JOINTS entries; finger slots keep their defaults.
    joint_target: wp.array(dtype=float),
):
    """Forward-Euler integrate q_dot to produce the position target."""
    k = wp.tid()
    joint_target[k] = q_current[k] + q_dot[k] * dt


# ---------------------------------------------------------------------------
# Helper: load kinematic chain parameters into Warp arrays
# ---------------------------------------------------------------------------

def load_chain_to_warp(chain_dict, device="cuda:0"):
    """Convert the dict returned by parse_urdf_kinematic_chain (torch tensors)
    into Warp arrays with the right dtype/shape for the kernels above.

    Handles the quaternion convention: the torch helpers use [w,x,y,z],
    Warp's wp.quat is [x,y,z,w]. We reorder here.

    Returns dict with keys:
        joint_axes   : wp.array of wp.vec3, shape (N_JOINTS,)
        link_pos     : wp.array of wp.vec3, shape (N_JOINTS,)
        link_quats   : wp.array of wp.quat, shape (N_JOINTS,)
        ee_offset_p  : wp.vec3 (scalar)
        ee_offset_q  : wp.quat (scalar)
    """
    import numpy as np

    joint_axes_np = chain_dict["joint_axes"].detach().cpu().numpy().astype(np.float32)
    link_pos_np   = chain_dict["p_rel"].detach().cpu().numpy().astype(np.float32)
    link_quats_wxyz = chain_dict["q_rel"].detach().cpu().numpy().astype(np.float32)
    ee_offset_p_np  = chain_dict["ee_offset_p"].detach().cpu().numpy().astype(np.float32)
    ee_offset_q_wxyz = chain_dict["ee_offset_q"].detach().cpu().numpy().astype(np.float32)

    assert joint_axes_np.shape[0] == N_JOINTS, (
        f"Expected {N_JOINTS} joints in chain, got {joint_axes_np.shape[0]}"
    )

    # Reorder [w,x,y,z] -> [x,y,z,w] for Warp.
    link_quats_xyzw = link_quats_wxyz[:, [1, 2, 3, 0]]
    ee_offset_q_xyzw = ee_offset_q_wxyz[[1, 2, 3, 0]]

    return {
        "joint_axes":  wp.array(joint_axes_np,   dtype=wp.vec3, device=device),
        "link_pos":    wp.array(link_pos_np,     dtype=wp.vec3, device=device),
        "link_quats":  wp.array(link_quats_xyzw, dtype=wp.quat, device=device),
        "ee_offset_p": wp.vec3(float(ee_offset_p_np[0]),
                               float(ee_offset_p_np[1]),
                               float(ee_offset_p_np[2])),
        "ee_offset_q": wp.quat(float(ee_offset_q_xyzw[0]),
                               float(ee_offset_q_xyzw[1]),
                               float(ee_offset_q_xyzw[2]),
                               float(ee_offset_q_xyzw[3])),
    }