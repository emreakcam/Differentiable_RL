import torch

from helpers.geom_diff_helpers import (
    analytic_to_geometric_jacobian,
    quat_L,
    quat_error_to_rotvec,
    rpy_to_quat_wxyz,
    transform_by_quat_diff,
    transform_quat_by_quat_diff,
)

import xml.etree.ElementTree as ET
import math

# Unbatched FK: q (n_joints,) -> pos (3,), quat (4,)
def forward_kinematics(q, arm_joint_axes, arm_link_pos, arm_link_quats, return_all=False):
    
    accum_pos = torch.zeros(3, device=q.device)
    accum_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=q.device)    

    positions = []
    quaternions = []

    for i in range(q.shape[0]):
        half = q[i] / 2
        joint_quat = torch.cat([torch.cos(half).unsqueeze(0), arm_joint_axes[i] * torch.sin(half)])

        accum_pos = accum_pos + transform_by_quat_diff(arm_link_pos[i], accum_quat)                
        accum_quat = transform_quat_by_quat_diff(accum_quat, arm_link_quats[i]) 
        accum_quat = transform_quat_by_quat_diff(accum_quat, joint_quat) 

        if return_all:
            positions.append(accum_pos)
            quaternions.append(accum_quat)
    
    if return_all:
        return torch.stack(positions), torch.stack(quaternions)
    else:
        return accum_pos, accum_quat

# Batched FK via vmap: q_batch (B, n_joints) -> pos (B, 3), quat (B, 4)
def forward_kinematics_batch(q_batch, arm_joint_axes, arm_link_pos, arm_link_quats, return_all=False):    
    fn = lambda q: forward_kinematics(q, arm_joint_axes, arm_link_pos, arm_link_quats, return_all=return_all)
    return torch.vmap(fn)(q_batch)

def forward_kinematics_jacobian_analytical(q, arm_joint_axes, arm_link_pos, arm_link_quats):
    """
    Analytical Jacobians of FK end-effector pose w.r.t. joint angles.

    Position column k:    J_p[:, :, k] = axis_world_k × (p_ee - p_k)_world  (geometric Jacobian)
    Quaternion column k:  J_q[:, :, k] = ½ Ω(axis_world_k) @ q_final        (quaternion kinematics)

    Args:
        q:              [B, N] or [N] joint angles
        arm_joint_axes: [N, 3] local joint rotation axes
        arm_link_pos:   [N, 3] relative link translations
        arm_link_quats: [N, 4] relative link rotations [w, x, y, z]
    Returns:
        J_p: [B, 3, N] or [3, N]
        J_q: [B, 4, N] or [4, N]
    """
    unbatched = q.dim() == 1
    if unbatched:
        q = q.unsqueeze(0)
    B, N = q.shape
    device, dtype = q.device, q.dtype

    # forward pass: store Q_pre[k] (after link), Q_post[k] (after joint)
    # Build these with lists to avoid in-place indexed writes (vmap-safe).
    Q_pre_list = []
    Q_post_list = []

    acc = torch.cat([
        torch.ones(B, 1, device=device, dtype=dtype),
        torch.zeros(B, 3, device=device, dtype=dtype),
    ], dim=1)
    for k in range(N):
        acc = transform_quat_by_quat_diff(acc, arm_link_quats[k].expand(B, -1))
        Q_pre_list.append(acc)
        half = q[:, k] / 2
        jk = torch.cat([torch.cos(half).unsqueeze(1),
                        torch.sin(half).unsqueeze(1) * arm_joint_axes[k].expand(B, -1)], dim=1)
        acc = transform_quat_by_quat_diff(acc, jk)
        Q_post_list.append(acc)

    Q_pre = torch.stack(Q_pre_list, dim=1)
    Q_post = torch.stack(Q_post_list, dim=1)
    Q_final = acc  # [B, 4]

    def _qconj(x):
        return torch.cat([x[..., :1], -x[..., 1:]], dim=-1)

    def _cross3(a, b):
        return torch.stack([
            a[:, 1]*b[:, 2] - a[:, 2]*b[:, 1],
            a[:, 2]*b[:, 0] - a[:, 0]*b[:, 2],
            a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0],
        ], dim=1)

    # backward pass: p_local = lever from joint k to EE expressed in frame Q_post[k]
    J_p_cols = []
    J_q_cols = []
    p_local = torch.zeros(B, 3, device=device, dtype=dtype)

    for k in range(N - 1, -1, -1):
        axis_w  = transform_by_quat_diff(arm_joint_axes[k].expand(B, -1), Q_pre[:, k])
        lever_w = transform_by_quat_diff(p_local, Q_post[:, k])

        # geometric position Jacobian: axis × lever
        J_p_cols.append(_cross3(axis_w, lever_w))

        # orientation Jacobian: dq/dα_k = ½ [0, axis_w] ⊗ q_final  (left multiplication)
        omega_quat = torch.cat([torch.zeros(B, 1, device=device, dtype=dtype), axis_w], dim=-1)
        J_q_cols.append(0.5 * (quat_L(omega_quat) @ Q_final.unsqueeze(-1)).squeeze(-1))

        # p_local recursion: rotate lever into frame Q_post[k-1]
        if k > 0:
            local_rot = transform_quat_by_quat_diff(_qconj(Q_post[:, k - 1]), Q_post[:, k])
            p_local = arm_link_pos[k].expand(B, -1) + transform_by_quat_diff(p_local, local_rot)

    # We built columns in reverse order (N-1 -> 0); flip to recover [0 .. N-1].
    J_p = torch.stack(J_p_cols[::-1], dim=-1)
    J_q = torch.stack(J_q_cols[::-1], dim=-1)

    if unbatched:
        return J_p.squeeze(0), J_q.squeeze(0)
    return J_p, J_q


#    q: (n_joints,) or (B, n_joints)
#    returns: J_pos (3, n_joints) or (B, 3, n_joints)
#             J_quat (4, n_joints) or (B, 4, n_joints)    
@torch.compile
def forward_kinematics_jacobian(q, arm_joint_axes, arm_link_pos, arm_link_quat):
    
    def fk_flat(q_):
        pos, quat = forward_kinematics(q_, arm_joint_axes, arm_link_pos, arm_link_quat)
        return torch.cat([pos, quat], dim=-1)  # (7,)

    jac_fn = torch.func.jacrev(fk_flat)
    if q.dim() == 1:
        J = jac_fn(q)  # (7, n_joints)
    else:
        J = torch.vmap(jac_fn)(q)  # (B, 7, n_joints)
    return J[..., :3, :], J[..., 3:, :]  # J_pos, J_quat



# ---------------------------------------------------------------------------
# IK solver: Levenberg-Marquardt
# ---------------------------------------------------------------------------

def solve_ik(
    target_pos:   torch.Tensor,           # [3]   desired EE position
    target_quat:  torch.Tensor,           # [4]   desired EE orientation [w,x,y,z]
    joint_axes:   torch.Tensor,           # [N, 3]
    link_pos:     torch.Tensor,           # [N, 3]
    link_quats:   torch.Tensor,           # [N, 4]
    ee_offset_p:  torch.Tensor,           # [3]
    ee_offset_q:  torch.Tensor,           # [4]
    joint_limits: torch.Tensor | None,    # [N, 2] or None
    q_init:       torch.Tensor | None = None,
    n_iters:      int   = 10,
    lam:          float = 0.2,
    lam_decay:    float = 0.95,
    pos_weight:   float = 1.0,
    rot_weight:   float = 0.0,
    max_step_norm: float = 0.12,
) -> torch.Tensor:
    """
    Levenberg-Marquardt IK with 6-DOF task space (position + orientation).

    Position error   : w_p * (target_pos  - pos_ee)               [3]
    Orientation error: w_r * log( q_target ⊗ q_ee^* )             [3]

    The position Jacobian is analytically corrected for the fixed EE offset
    using the cross-product extension:
        J_p_ee = J_p - skew(R·ee_offset) @ J_omega
    where J_omega is obtained from the analytical quaternion Jacobian.

    Returns:
        q : [N] solved joint angles (clamped to joint_limits if provided)
    """
    device = joint_axes.device
    N = joint_axes.shape[0]

    q = q_init.clone() if q_init is not None else torch.zeros(N, device=device)
    eye = torch.eye(N, device=device, dtype=q.dtype)

    for _ in range(n_iters):
        # Forward kinematics
        pos_fk, quat_fk = forward_kinematics(q, joint_axes, link_pos, link_quats)
        rot_offset_w    = transform_by_quat_diff(ee_offset_p, quat_fk)  # [3]
        pos_ee          = pos_fk  + rot_offset_w
        quat_ee         = transform_quat_by_quat_diff(quat_fk, ee_offset_q)

        # Analytical Jacobians at the last revolute joint frame
        J_p, J_q = forward_kinematics_jacobian_analytical(
            q.unsqueeze(0), joint_axes, link_pos, link_quats
        )
        J_p = J_p.squeeze(0)   # [3, N]
        J_q = J_q.squeeze(0)   # [4, N]

        # Convert analytical quaternion Jacobian to geometric angular velocity.
        J_omega = analytic_to_geometric_jacobian(
            J_q.unsqueeze(0), quat_fk.unsqueeze(0)
        ).squeeze(0)   # [3, N]

        # Extend position Jacobian to account for the EE offset lever arm:
        #   dp_ee/dq_k = J_p[:,k] + axis_k_w × rot_offset_w
        #              = J_p[:,k] - skew(rot_offset_w) @ J_omega[:,k]
        rx, ry, rz = rot_offset_w.unbind(-1)
        zero = torch.zeros_like(rx)
        skew_r = torch.stack([
            torch.stack([zero, -rz,  ry], dim=-1),
            torch.stack([rz,  zero, -rx], dim=-1),
            torch.stack([-ry,   rx, zero], dim=-1),
        ], dim=-2)  # [3, 3]
        J_p_ee = J_p - skew_r @ J_omega   # [3, N]

        # 6-D error vector
        err_p = target_pos - pos_ee
        if rot_weight > 0.0:
            err_r = quat_error_to_rotvec(target_quat, quat_ee)
        else:
            err_r = torch.zeros(3, device=device, dtype=q.dtype)

        err = torch.cat([pos_weight * err_p, rot_weight * err_r])   # [6]

        # 6 × N stacked Jacobian
        J = torch.cat([pos_weight * J_p_ee, rot_weight * J_omega], dim=0)  # [6, N]

        # LM update: dq = (J^T J + λI)^{-1} J^T err
        JtJ = J.T @ J
        dq  = torch.linalg.solve(JtJ + lam * eye, J.T @ err)
        dq_norm = dq.norm()
        if dq_norm > max_step_norm:
            dq = dq * (max_step_norm / dq_norm)
        q   = q + dq
        lam *= lam_decay

        if joint_limits is not None:
            q = torch.clamp(q, joint_limits[:, 0], joint_limits[:, 1])

        if err_p.norm().item() < 5e-4 and err_r.norm().item() < 5e-3:
            break

    return q


def solve_ik_batch(
    target_pos: torch.Tensor,           # [B, 3]
    target_quat: torch.Tensor,          # [B, 4]
    joint_axes: torch.Tensor,           # [N, 3]
    link_pos: torch.Tensor,             # [N, 3]
    link_quats: torch.Tensor,           # [N, 4]
    ee_offset_p: torch.Tensor,          # [3]
    ee_offset_q: torch.Tensor,          # [4]
    joint_limits: torch.Tensor | None,  # [N, 2] or None
    q_init: torch.Tensor | None = None, # [B, N]
    n_iters: int = 10,
    lam: float = 0.2,
    lam_decay: float = 0.95,
    pos_weight: float = 1.0,
    rot_weight: float = 0.0,
    max_step_norm: float = 0.12,
) -> torch.Tensor:
    """Batched Levenberg-Marquardt IK with 6-DOF task space."""
    batch_size = target_pos.shape[0]
    num_joints = joint_axes.shape[0]
    device = target_pos.device
    dtype = target_pos.dtype

    if q_init is None:
        q = torch.zeros((batch_size, num_joints), device=device, dtype=dtype)
    else:
        q = q_init.clone()

    eye = torch.eye(num_joints, device=device, dtype=dtype).expand(batch_size, -1, -1)

    for _ in range(n_iters):
        pos_fk, quat_fk = forward_kinematics_batch(q, joint_axes, link_pos, link_quats)
        rot_offset_w = transform_by_quat_diff(ee_offset_p.expand(batch_size, -1), quat_fk)
        pos_ee = pos_fk + rot_offset_w
        quat_ee = transform_quat_by_quat_diff(quat_fk, ee_offset_q.expand(batch_size, -1))

        J_p, J_q = forward_kinematics_jacobian_analytical(q, joint_axes, link_pos, link_quats)
        J_omega = analytic_to_geometric_jacobian(J_q, quat_fk)

        rx, ry, rz = rot_offset_w.unbind(-1)
        zero = torch.zeros_like(rx)
        skew_r = torch.stack(
            [
                torch.stack([zero, -rz, ry], dim=-1),
                torch.stack([rz, zero, -rx], dim=-1),
                torch.stack([-ry, rx, zero], dim=-1),
            ],
            dim=-2,
        )
        J_p_ee = J_p - skew_r @ J_omega

        err_p = target_pos - pos_ee
        if rot_weight > 0.0:
            err_r = quat_error_to_rotvec(target_quat, quat_ee)
        else:
            err_r = torch.zeros((batch_size, 3), device=device, dtype=dtype)

        err = torch.cat([pos_weight * err_p, rot_weight * err_r], dim=-1)
        J = torch.cat([pos_weight * J_p_ee, rot_weight * J_omega], dim=-2)

        Jt = J.transpose(-1, -2)
        dq = torch.linalg.solve(Jt @ J + lam * eye, Jt @ err.unsqueeze(-1)).squeeze(-1)

        dq_norm = dq.norm(dim=-1, keepdim=True)
        dq = torch.where(
            dq_norm > max_step_norm,
            dq * (max_step_norm / dq_norm.clamp_min(1.0e-8)),
            dq,
        )

        q = q + dq
        lam *= lam_decay

        if joint_limits is not None:
            q = torch.clamp(q, joint_limits[:, 0].unsqueeze(0), joint_limits[:, 1].unsqueeze(0))

    return q

# MUJUCO parser
def parse_kinematic_chain(xml_root, start_link_name,link_names, device):

    target_link_name = link_names[-1]

    def recurse_bodies(body, current_path, target_link_name):
        
        new_path = current_path + [body] # use list concatenation to trigger copy (else all recursions write in same list)
        
        if body.attrib["name"] == target_link_name:            
            # if the compute graph contains the target link, return the current body list as the kinematic chain
            return new_path
        
        for child in body.findall("body"):
            # kick of recursive search in child bodies
            result = recurse_bodies(child, new_path, target_link_name)            
            if result is not None:
                return result
            
        return None
    
    def get_relative_joint_transform(delta_pos_list, delta_quat_list):
        rel_pos = torch.tensor([0, 0, 0], device=device)
        rel_quat = torch.tensor([1, 0, 0, 0], device=device)
        for i in range(len(delta_pos_list)):
            rel_pos = rel_pos + transform_by_quat_diff(torch.tensor(delta_pos_list[i], device=device), rel_quat)
            rel_quat = transform_quat_by_quat_diff(rel_quat, torch.tensor(delta_quat_list[i], device=device))            

        return rel_pos.tolist(), rel_quat.tolist()
    
    start_body = None
    for body in xml_root.iter("body"):
        if body.attrib["name"] == start_link_name:
            start_body = body
            break
            
    link_chain = recurse_bodies(start_body, [], target_link_name)
    
    list_p_rel = []
    list_q_rel = []
    list_joint_axes = []
    list_joint_ranges = []

    # loop over kinematic chain
    accum_pos = []
    accum_quat = []
    for body in link_chain:
        print(f"Parsing link: {body.attrib['name']}")

        # check body transformations
        link_pos = [float(x) for x in body.attrib.get("pos", "0 0 0").split()]
        link_quat = [float(x) for x in body.attrib.get("quat", "1 0 0 0").split()]
        
        # check if body has joint attached
        if body.find("joint") is not None:
            joint = body.find("joint")
            if joint.attrib["type"] != "free":
                if joint.attrib["type"] == "slide":
                    raise NotImplementedError("Prismatic joints not supported yet")
                elif joint.attrib["type"] == "hinge":

                    # extract joint information
                    joint_axis = [float(x) for x in joint.attrib.get("axis").split()]
                    joint_range = [float(x) for x in joint.attrib.get("range").split()]            

                    # use accumulated transformations to compute the transformation from previous joint / link to current joint / link, and add to list
                    accum_pos.append(link_pos)
                    accum_quat.append(link_quat)

                    p_prev_to_current, q_prev_to_current = get_relative_joint_transform(accum_pos, accum_quat)

                    # assign
                    list_p_rel.append(p_prev_to_current)
                    list_q_rel.append(q_prev_to_current)
                    list_joint_axes.append(joint_axis)
                    list_joint_ranges.append(joint_range)

                    # reset lists
                    accum_pos = []
                    accum_quat = []

        else:
            accum_pos.append(link_pos)
            accum_quat.append(link_quat)    

    return {
        "p_rel": list_p_rel,
        "q_rel": list_q_rel,
        "joint_axes": list_joint_axes,
        "joint_ranges": list_joint_ranges
    }


# ---------------------------------------------------------------------------
# URDF kinematic chain parser
# ---------------------------------------------------------------------------

def parse_urdf_kinematic_chain(urdf_path: str, root_link: str, ee_link: str, device):
    """
    Parse a URDF and extract kinematic chain parameters compatible with
    ``forward_kinematics`` / ``forward_kinematics_jacobian_analytical``.

    Fixed joints between revolute joints are accumulated into the next
    revolute joint's relative transform.  Fixed joints that trail *after*
    the last revolute joint (e.g. flange / hand mounts) are returned as a
    rigid EE offset so that the caller can evaluate the true TCP pose.

    Args:
        urdf_path : path to the URDF file.
        root_link : name of the base link   (e.g. "fr3_link0").
        ee_link   : name of the target link (e.g. "fr3_link7").
        device    : torch device for all returned tensors.

    Returns dict with keys:
        p_rel        : Tensor [N, 3]   relative joint-origin positions
        q_rel        : Tensor [N, 4]   relative joint-origin orientations [w,x,y,z]
        joint_axes   : Tensor [N, 3]   rotation axes in the joint's own frame
        joint_ranges : Tensor [N, 2]   joint limits [lower, upper]
        joint_names  : list[str]       joint names in chain order
        ee_offset_p  : Tensor [3]      fixed translation after the last revolute joint
        ee_offset_q  : Tensor [4]      fixed rotation    after the last revolute joint
    """
    tree = ET.parse(urdf_path)
    root_elem = tree.getroot()

    # parent_link → list[(joint_element, child_link)]
    parent_to_children: dict[str, list] = {}
    for joint in root_elem.findall("joint"):
        parent_link = joint.find("parent").attrib["link"]
        child_link  = joint.find("child").attrib["link"]
        parent_to_children.setdefault(parent_link, []).append((joint, child_link))

    # DFS: ordered list of (joint, child_link) from root_link to ee_link.
    def find_chain(current: str, target: str, path: list):
        if current == target:
            return path
        for joint, child in parent_to_children.get(current, []):
            result = find_chain(child, target, path + [(joint, child)])
            if result is not None:
                return result
        return None

    chain = find_chain(root_link, ee_link, [])
    if chain is None:
        raise ValueError(f"No kinematic chain found from '{root_link}' to '{ee_link}'")

    def _parse_origin(joint_elem):
        origin = joint_elem.find("origin")
        if origin is None:
            return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        xyz  = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
        rpy  = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()]
        return xyz, rpy_to_quat_wxyz(*rpy)

    def _compose(acc_p, acc_q, delta_p, delta_q):
        """Compose transform (acc_p, acc_q) ∘ (delta_p, delta_q)."""
        a_p = torch.tensor(acc_p,   dtype=torch.float64, device=device)
        a_q = torch.tensor(acc_q,   dtype=torch.float64, device=device)
        d_p = torch.tensor(delta_p, dtype=torch.float64, device=device)
        d_q = torch.tensor(delta_q, dtype=torch.float64, device=device)
        new_p = a_p + transform_by_quat_diff(d_p, a_q)
        new_q = transform_quat_by_quat_diff(a_q, d_q)
        return new_p.tolist(), new_q.tolist()

    p_rel_list   = []
    q_rel_list   = []
    joint_axes   = []
    joint_ranges = []
    joint_names  = []

    # Accumulate fixed transforms leading up to the next revolute joint.
    acc_p = [0.0, 0.0, 0.0]
    acc_q = [1.0, 0.0, 0.0, 0.0]

    for joint, _child_link in chain:
        jtype = joint.attrib.get("type", "fixed")
        xyz, q_wxyz = _parse_origin(joint)

        if jtype in ("revolute", "continuous"):
            composed_p, composed_q = _compose(acc_p, acc_q, xyz, q_wxyz)

            axis_elem = joint.find("axis")
            axis = [float(v) for v in axis_elem.attrib.get("xyz", "0 0 1").split()] \
                   if axis_elem is not None else [0.0, 0.0, 1.0]

            limit_elem = joint.find("limit")
            if limit_elem is not None:
                lower = float(limit_elem.attrib.get("lower", f"-{math.pi}"))
                upper = float(limit_elem.attrib.get("upper",  f"{math.pi}"))
            else:
                lower, upper = -math.pi, math.pi

            p_rel_list.append(composed_p)
            q_rel_list.append(composed_q)
            joint_axes.append(axis)
            joint_ranges.append([lower, upper])
            joint_names.append(joint.attrib["name"])

            # Reset accumulator for the next segment.
            acc_p = [0.0, 0.0, 0.0]
            acc_q = [1.0, 0.0, 0.0, 0.0]

        else:  # fixed joint → fold into the accumulator
            acc_p, acc_q = _compose(acc_p, acc_q, xyz, q_wxyz)

    # Remaining fixed transforms become the rigid EE offset.
    ee_offset_p = torch.tensor(acc_p, dtype=torch.float32, device=device)
    ee_offset_q = torch.tensor(acc_q, dtype=torch.float32, device=device)

    return {
        "p_rel":        torch.tensor(p_rel_list,   dtype=torch.float32, device=device),
        "q_rel":        torch.tensor(q_rel_list,   dtype=torch.float32, device=device),
        "joint_axes":   torch.tensor(joint_axes,   dtype=torch.float32, device=device),
        "joint_ranges": torch.tensor(joint_ranges, dtype=torch.float32, device=device),
        "joint_names":  joint_names,
        "ee_offset_p":  ee_offset_p,
        "ee_offset_q":  ee_offset_q,
    }