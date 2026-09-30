import torch


def rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> list[float]:
    """Convert roll-pitch-yaw angles (radians) to quaternion [w, x, y, z]."""
    cr, sr = torch.cos(torch.tensor(roll / 2)), torch.sin(torch.tensor(roll / 2))
    cp, sp = torch.cos(torch.tensor(pitch / 2)), torch.sin(torch.tensor(pitch / 2))
    cy, sy = torch.cos(torch.tensor(yaw / 2)), torch.sin(torch.tensor(yaw / 2))
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [float(w), float(x), float(y), float(z)]


def quat_omega_dot(q, omega):
    """
    q: [N,4] quaternions (w,x,y,z)
    omega: [N,3] angular velocity in B frame
    returns: [N,4] dq/dt
    """
    w, x, y, z = q.unbind(dim=-1)
    omega_x, omega_y, omega_z = omega.unbind(dim=-1)

    dq = torch.stack(
        [
            -x * omega_x - y * omega_y - z * omega_z,
            w * omega_x + y * omega_z - z * omega_y,
            w * omega_y - x * omega_z + z * omega_x,
            w * omega_z + x * omega_y - y * omega_x,
        ],
        dim=-1,
    )
    return 0.5 * dq

def skew(a):
    ax, ay, az = a.unbind(dim=-1)
    zero = torch.zeros_like(ax)
    row0 = torch.stack([zero, -az, ay], dim=-1)
    row1 = torch.stack([az, zero, -ax], dim=-1)
    row2 = torch.stack([-ay, ax, zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)

def quat_G(omega):
    # omega: (...,3)
    ox, oy, oz = omega.unbind(dim=-1)
    zero = torch.zeros_like(ox)
    row0 = torch.stack([zero, -ox, -oy, -oz], dim=-1)
    row1 = torch.stack([ox, zero, oz, -oy], dim=-1)
    row2 = torch.stack([oy, -oz, zero, ox], dim=-1)
    row3 = torch.stack([oz, oy, -ox, zero], dim=-1)
    return torch.stack([row0, row1, row2, row3], dim=-2)


def _expand_eye3(ref: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(3, device=ref.device, dtype=ref.dtype)
    return eye.expand(*ref.shape[:-1], 3, 3)

def quat_K(q):
    # q: (...,4) -> [qw, qx, qy, qz]
    qw = q[..., 0:1]
    qv = q[..., 1:]
    eye = _expand_eye3(q)
    top = -qv.unsqueeze(dim=-2)
    bottom = qw.unsqueeze(dim=-1) * eye + skew(qv)
    return torch.cat([top, bottom], dim=-2)

def quat_L(q): # quaternion left multiplication matrix
    w = q[..., 0:1]
    v = q[..., 1:4]
    eye = _expand_eye3(q)

    top_row = torch.cat([w, -v], dim=-1).unsqueeze(dim=-2)
    bottom_left = v.unsqueeze(dim=-1)
    bottom_right = w.unsqueeze(dim=-1) * eye + skew(v)
    bottom_block = torch.cat([bottom_left, bottom_right], dim=-1)
    return torch.cat([top_row, bottom_block], dim=-2)

def quat_R(q): # quaternion right multiplication matrix
    w = q[..., 0:1]
    v = q[..., 1:4]
    eye = _expand_eye3(q)

    top_row = torch.cat([w, -v], dim=-1).unsqueeze(dim=-2)
    bottom_left = v.unsqueeze(dim=-1)
    bottom_right = w.unsqueeze(dim=-1) * eye - skew(v)
    bottom_block = torch.cat([bottom_left, bottom_right], dim=-1)
    return torch.cat([top_row, bottom_block], dim=-2)
    
# q = [w,vx,vy,vz] = [w, v], a = [ax,ay,az]
def d_rot_d_quat(q, a):
    w = q[..., 0:1]
    v = q[..., 1:4]
    eye = _expand_eye3(q)

    Jw = w * a + torch.cross(v, a, dim=-1)
    vv = v.unsqueeze(dim=-1) * a.unsqueeze(dim=-2)
    av = a.unsqueeze(dim=-1) * v.unsqueeze(dim=-2)
    dot_va = (v * a).sum(dim=-1, keepdim=True).unsqueeze(dim=-1)
    Jv = dot_va * eye + vv - av - w.unsqueeze(dim=-1) * skew(a)
    J = 2 * torch.concat([Jw.unsqueeze(dim=-1), Jv], dim=-1)
    return J


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    q_normalized = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q_normalized.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r00 = 1 - 2 * (yy + zz)
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = 1 - 2 * (xx + zz)
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = 1 - 2 * (xx + yy)
    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )

# attention: this is directly flipped wrt to transform_quat_by_quat from genesis (no idea why they chose the reverse order)
def transform_quat_by_quat_diff(q_left, q_right):
    """
    Hamilton product (left-to-right composition).

    Computes:
        q_out = q_left ⊗ q_right

    Meaning:
        If q_left = orientation of B in W
           q_right = orientation of E in B
        then q_out = orientation of E in W

    Both inputs are [..., 4] quaternions in [w, x, y, z] convention.
    """
    lw, lx, ly, lz = q_left[..., 0:1], q_left[..., 1:2], q_left[..., 2:3], q_left[..., 3:4]
    rw, rx, ry, rz = q_right[..., 0:1], q_right[..., 1:2], q_right[..., 2:3], q_right[..., 3:4]

    w = lw * rw - lx * rx - ly * ry - lz * rz
    x = lw * rx + lx * rw + ly * rz - lz * ry
    y = lw * ry - lx * rz + ly * rw + lz * rx
    z = lw * rz + lx * ry - ly * rx + lz * rw

    return torch.cat([w, x, y, z], dim=-1)


def quat_mul(q_left: torch.Tensor, q_right: torch.Tensor) -> torch.Tensor:
    """Quaternion product [w, x, y, z] ⊗ [w, x, y, z]."""
    return transform_quat_by_quat_diff(q_left, q_right)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate for [w, x, y, z]."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_error_to_rotvec(target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """Shortest-arc rotation-vector error taking `current` to `target`."""
    q_err = quat_mul(target, quat_conjugate(current))
    q_err = torch.where(q_err[..., :1] < 0.0, -q_err, q_err)

    imag = q_err[..., 1:]
    imag_norm = imag.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(imag_norm, q_err[..., :1].clamp(-1.0, 1.0))
    scale = torch.where(imag_norm > 1.0e-8, angle / imag_norm, torch.zeros_like(imag_norm))
    return imag * scale


def quat_to_ang_vel_jacobian(q):
    """
    Convert analytic quaternion Jacobian to angular velocity Jacobian.
    J_omega = 2 * Q(q)^T @ J_quat
    where Q maps omega -> q_dot via q_dot = 1/2 * Q @ omega.

    q: (..., 4) [w, x, y, z]
    returns: Q_T: (..., 3, 4)  such that  J_omega = 2 * Q_T @ J_quat
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # Q is (4,3): Q = [[-x,-y,-z],[w,-z,y],[z,w,-x],[-y,x,w]]
    # Q^T is (3,4)
    Q_T = torch.stack([
        torch.stack([-x,  w,  z, -y], dim=-1),
        torch.stack([-y, -z,  w,  x], dim=-1),
        torch.stack([-z,  y, -x,  w], dim=-1),
    ], dim=-2)  # (..., 3, 4)
    return Q_T


def analytic_to_geometric_jacobian(J_quat, q):
    """
    Convert analytic quat Jacobian to geometric angular velocity Jacobian.
    J_quat: (..., 4, n_joints)
    q:      (..., 4)
    returns J_omega: (..., 3, n_joints)
    """
    Q_T = quat_to_ang_vel_jacobian(q)  # (..., 3, 4)
    return 2.0 * (Q_T @ J_quat)  # (..., 3, n_joints)


def transform_by_quat_diff(v, quat, out: torch.Tensor | None = None):
    """
    Differentiable version of transform_by_quat without in-place operations.
    
    Args:
        v: vectors to transform [..., 3]
        quat: quaternions (w, x, y, z) [..., 4]
        out: unused compatibility argument. Kept for API stability.
    
    Returns:
        Transformed vectors [..., 3]
    """
    q_w, q_x, q_y, q_z = quat[..., :1], quat[..., 1:2], quat[..., 2:3], quat[..., 3:]
    q_ww, q_wx, q_wy, q_wz = q_w * q_w, q_w * q_x, q_w * q_y, q_w * q_z
    q_xx, q_xy, q_xz = q_x * q_x, q_x * q_y, q_x * q_z
    q_yy, q_yz = q_y * q_y, q_y * q_z
    q_zz = q_z**2

    vs = v / (q_ww + q_xx + q_yy + q_zz).clamp(min=1e-8)
    v_x, v_y, v_z = vs[..., :1], vs[..., 1:2], vs[..., 2:]

    u_x = v_x * (q_xx + q_ww - q_yy - q_zz) + v_y * (2.0 * q_xy - 2.0 * q_wz) + v_z * (2.0 * q_xz + 2.0 * q_wy)
    u_y = v_x * (2.0 * q_wz + 2.0 * q_xy) + v_y * (q_ww - q_xx + q_yy - q_zz) + v_z * (2.0 * q_yz - 2.0 * q_wx)
    u_z = v_x * (2.0 * q_xz - 2.0 * q_wy) + v_y * (2.0 * q_wx + 2.0 * q_yz) + v_z * (q_ww - q_xx - q_yy + q_zz)

    result = torch.cat([u_x, u_y, u_z], dim=-1)
    
    return result
