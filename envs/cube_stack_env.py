"""
Cube stacking environment bileşenleri:
  - Task config sabitleri
  - Body/site index keşfi
  - Observation builder
  - Batch data oluşturma (randomized cube positions)
"""
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

# ---------------------------------------------------------------------------
# Task sabitleri
# ---------------------------------------------------------------------------
N_JOINTS           = 7
FINGER_OPEN        = 0.04
GRASP_Z_OFFSET     = 0.0
PRE_GRASP_Z_OFFSET = 0.075
CUBE2_HEIGHT       = 0.05
STACK_OFFSET       = 0.01

Q_LO = jnp.array([-2.897, -1.763, -2.897, -3.072, -2.897, -0.018, -2.897])
Q_HI = jnp.array([ 2.897,  1.763,  2.897, -0.070,  2.897,  3.752,  2.897])
VEL_LIMITS = jnp.array([2.17, 2.17, 2.17, 2.17, 2.61, 2.61, 2.61])

TARGET_QUAT = jnp.array([0.0, 0.7071068, 0.7071068, 0.0])

OBS_DIM  = 49
ACT_DIM  = N_JOINTS + 1


def discover_indices(mj_model):
    """MuJoCo model'den body/site index'lerini bul.

    Returns:
        dict: {'cube1_body_idx', 'cube2_body_idx', 'ee_site_idx'}
    """
    return {
        'cube1_body_idx': mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "box"),
        'cube2_body_idx': mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "box2"),
        'ee_site_idx': 0,
    }


def make_task_cfg(indices):
    """Reward ve obs fonksiyonlarına geçilecek config dict oluştur."""
    return {
        **indices,
        'target_quat': TARGET_QUAT,
        'pre_grasp_z_offset': PRE_GRASP_Z_OFFSET,
        'cube2_height': CUBE2_HEIGHT,
        'stack_offset': STACK_OFFSET,
    }


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------
def build_obs(state, prev_ee_pos, frame_dt, step_idx, cfg):
    """State'ten observation vektörü oluştur (49-dim).

    İçerik:
        ee_pos(3), cube_pos(3), ee_to_cube(3), ee_vel(3), finger_width(1),
        q(7), q_dot(7), ee_quat(4), quat_err(4), cube_quat(4),
        cube2_pos(3), cube2_quat(4), cube1_to_cube2(3)
    """
    ee_pos = state.site_xpos[cfg['ee_site_idx']]
    ee_vel = (ee_pos - prev_ee_pos) / frame_dt
    cube_pos = state.xpos[cfg['cube1_body_idx']]
    cube_quat = state.xquat[cfg['cube1_body_idx']]
    ee_to_cube = cube_pos - ee_pos
    finger_width = state.qpos[7] + state.qpos[8]
    current_q = state.qpos[:N_JOINTS]
    current_q_dot = state.qvel[:N_JOINTS]
    ee_quat = state.xquat[9]
    quat_err = TARGET_QUAT - ee_quat
    cube2_pos = state.xpos[cfg['cube2_body_idx']]
    cube2_quat = state.xquat[cfg['cube2_body_idx']]
    cube1_to_cube2 = cube2_pos - cube_pos

    obs = jnp.concatenate([
        ee_pos, cube_pos, ee_to_cube, ee_vel,
        jnp.array([finger_width]), current_q, current_q_dot,
        ee_quat, quat_err, cube_quat,
        cube2_pos, cube2_quat, cube1_to_cube2,
    ])
    return obs, ee_pos, current_q


# ---------------------------------------------------------------------------
# Batch data — randomized cube positions
# ---------------------------------------------------------------------------
def make_randomized_data(rng_key, mj_model, mj_data, mjx_model, key_id):
    """Tek bir randomized env verisi oluştur."""
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    k1, k2 = jax.random.split(rng_key)
    noise_x = jax.random.uniform(k1, (), minval=-0.15, maxval=0.05)
    noise_y = jax.random.uniform(k2, (), minval=-0.1, maxval=0.1)
    mj_data.qpos[9] += float(noise_x)
    mj_data.qpos[10] += float(noise_y)
    mujoco.mj_forward(mj_model, mj_data)
    d = mjx.put_data(mj_model, mj_data)
    d = mjx.forward(mjx_model, d)
    return d


def create_batch(batch_size, mj_model, mj_data, mjx_model, key_id, seed=0):
    """Randomized cube pozisyonlarıyla batch oluştur.

    Returns:
        mjx_data_batch: vmapped MJX data (leading dim = batch_size)
        init_cube_pos_batch: (batch_size, 3)
    """
    batch_keys = jax.random.split(jax.random.PRNGKey(seed), batch_size)
    batch_data_list = [
        make_randomized_data(k, mj_model, mj_data, mjx_model, key_id)
        for k in batch_keys
    ]
    mjx_data_batch = jax.tree.map(
        lambda *xs: jnp.stack(xs, axis=0), *batch_data_list)

    cube1_idx = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, "box")
    init_cube_pos_batch = mjx_data_batch.xpos[:, cube1_idx]

    return mjx_data_batch, init_cube_pos_batch