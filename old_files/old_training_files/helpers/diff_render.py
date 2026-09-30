"""
Differentiable Depth Renderer — JAX ray casting.

Overhead kameralar: XML'den cam_xpos/cam_xmat okunur (sabit).
  → ground + scene boxes + hand box render edilir.
Wrist kamera: hand body'ye bağlı, her step dinamik ray hesaplama.
  → ground + scene boxes + finger boxes render edilir.

Desteklenen geometriler: ground plane + N adet box.
"""
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Quaternion → Rotation
# ---------------------------------------------------------------------------
def quat_to_rot(q):
    """Quaternion [w, x, y, z] → 3×3 rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


# ---------------------------------------------------------------------------
# Ray Generation
# ---------------------------------------------------------------------------
def generate_rays_from_mat(cam_pos, cam_mat_flat, fovy, img_w, img_h):
    """MuJoCo cam_mat (9,) → ray directions (H, W, 3).
    cam_mat row-major 3×3: row0=right, row1=up, row2=z_axis.
    Camera looks along -z_axis.
    """
    cam_mat = cam_mat_flat.reshape(3, 3)
    right   = -cam_mat[:, 0]
    up      = cam_mat[:, 1]
    forward = -cam_mat[:, 2]

    aspect = img_w / img_h
    half_h = jnp.tan(fovy / 2.0)
    half_w = half_h * aspect

    u = (jnp.arange(img_w) + 0.5) / img_w * 2.0 - 1.0
    v = ((jnp.arange(img_h) + 0.5) / img_h * 2.0 - 1.0)[::-1]
    uu, vv = jnp.meshgrid(u, v)

    dirs = (forward[None, None, :]
            + uu[:, :, None] * half_w * right[None, None, :]
            + vv[:, :, None] * half_h * up[None, None, :])
    dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs


def generate_local_rays(fovy, img_w, img_h):
    """Kamera-lokal ray yönleri (forward=-z, right=+x, up=+y).
    Bir kez hesaplanır, her step'te kamera matrisiyle dönüştürülür.
    """
    aspect = img_w / img_h
    half_h = jnp.tan(fovy / 2.0)
    half_w = half_h * aspect

    u = (jnp.arange(img_w) + 0.5) / img_w * 2.0 - 1.0
    v = ((jnp.arange(img_h) + 0.5) / img_h * 2.0 - 1.0)[::-1]
    uu, vv = jnp.meshgrid(u, v)

    dirs = jnp.stack([-uu * half_w, vv * half_h, -jnp.ones_like(uu)],
                      axis=-1)
    dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs


# ---------------------------------------------------------------------------
# Camera Setup — XML'den oku
# ---------------------------------------------------------------------------
def setup_cameras(mj_model, mj_data, cam_names, fovy, img_w, img_h):
    """XML kameralarından pos + ray hesapla (sabit)."""
    import mujoco

    cameras = []
    for name in cam_names:
        cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        cam_pos = jnp.array(mj_data.cam_xpos[cam_id])
        cam_mat = jnp.array(mj_data.cam_xmat[cam_id]).reshape(-1)
        rays = generate_rays_from_mat(cam_pos, cam_mat, fovy, img_w, img_h)
        cameras.append({'pos': cam_pos, 'rays': rays, 'cam_id': cam_id})
    return cameras


def setup_wrist_camera(mj_model, fovy, img_w, img_h):
    """Wrist kamera: lokal ray'ler + cam_id."""
    import mujoco
    cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA,
                                "wrist_cam")
    local_rays = generate_local_rays(fovy, img_w, img_h)
    return {
        'cam_id': cam_id,
        'local_rays': local_rays,
    }


# ---------------------------------------------------------------------------
# Ray-Primitive Intersections
# ---------------------------------------------------------------------------
def ray_plane_intersect(ray_o, ray_d, max_depth):
    """Ray → z=0 ground plane intersection."""
    t = -ray_o[2] / ray_d[:, :, 2]
    valid = (ray_d[:, :, 2] < -1e-8) & (t > 0)
    return jnp.where(valid, t, max_depth)


def ray_box_intersect(ray_o, ray_d, box_pos, box_quat, box_half, max_depth):
    """Ray → rotated box intersection via slab method."""
    R_inv = quat_to_rot(box_quat).T
    local_o = R_inv @ (ray_o - box_pos)
    local_d = jnp.einsum('ij,hwj->hwi', R_inv, ray_d)
    inv_d = 1.0 / jnp.where(jnp.abs(local_d) < 1e-8,
                              jnp.sign(local_d) * 1e-8 + 1e-10, local_d)
    t1 = (-box_half[None, None, :] - local_o[None, None, :]) * inv_d
    t2 = ( box_half[None, None, :] - local_o[None, None, :]) * inv_d
    t_near = jnp.minimum(t1, t2)
    t_far  = jnp.maximum(t1, t2)
    t_enter = jnp.max(t_near, axis=-1)
    t_exit  = jnp.min(t_far, axis=-1)
    hit = (t_enter < t_exit) & (t_exit > 0)
    return jnp.where(hit, jnp.maximum(t_enter, 0.0), max_depth)


# ---------------------------------------------------------------------------
# Combined Render — 2 overhead + 1 wrist → (H, W, 3)
# ---------------------------------------------------------------------------
def render_all_cameras(state, overhead_cams, wrist_info,
                       box_body_indices, box_halfs,
                       finger_body_indices, finger_half,
                       hand_body_idx, hand_half,
                       overhead_max_depth, wrist_max_depth):
    """Tüm kameralardan depth render → (H, W, 3).

    Overhead: ground + scene boxes + hand box
    Wrist:    ground + scene boxes + finger boxes
    """
    # Scene box pozisyonları (küpler)
    scene_positions = [state.xpos[idx] for idx in box_body_indices]
    scene_quats = [state.xquat[idx] for idx in box_body_indices]

    # Hand box pozisyonu
    hand_pos = state.xpos[hand_body_idx]
    hand_quat = state.xquat[hand_body_idx]

    depths = []

    # ── Overhead kameralar: ground + küpler + hand ──
    for cam in overhead_cams:
        cam_pos = cam['pos']
        ray_dirs = cam['rays']

        depth = ray_plane_intersect(cam_pos, ray_dirs, overhead_max_depth)

        # Scene boxes (küpler)
        for pos, quat, half in zip(scene_positions, scene_quats, box_halfs):
            t = ray_box_intersect(cam_pos, ray_dirs, pos, quat, half,
                                   overhead_max_depth)
            depth = jnp.minimum(depth, t)

        for fidx in finger_body_indices:
            t = ray_box_intersect(cam_pos, ray_dirs,
                                   state.xpos[fidx], state.xquat[fidx],
                                   finger_half, overhead_max_depth)
            depth = jnp.minimum(depth, t)

        depths.append(depth)

    # ── Wrist kamera: ground + küpler + finger'lar ──
    if wrist_info is not None:
        wrist_cam_id = wrist_info['cam_id']
        local_rays = wrist_info['local_rays']

        wrist_pos = state.cam_xpos[wrist_cam_id]
        wrist_mat = state.cam_xmat[wrist_cam_id].reshape(3, 3)
        world_rays = jnp.einsum('ij,hwj->hwi', wrist_mat, local_rays)

        depth = ray_plane_intersect(wrist_pos, world_rays, wrist_max_depth)

        # Scene boxes (küpler)
        for pos, quat, half in zip(scene_positions, scene_quats, box_halfs):
            t = ray_box_intersect(wrist_pos, world_rays, pos, quat, half,
                                   wrist_max_depth)
            depth = jnp.minimum(depth, t)

        # Finger boxes
        for fidx in finger_body_indices:
            t = ray_box_intersect(wrist_pos, world_rays,
                                   state.xpos[fidx], state.xquat[fidx],
                                   finger_half, wrist_max_depth)
            depth = jnp.minimum(depth, t)

        depths.append(depth)

    return jnp.stack(depths, axis=-1)