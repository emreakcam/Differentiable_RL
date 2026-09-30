"""
PickPlaceEnv — pick a cube up and carry it to a randomised marker.
==================================================================
cube_stacking's sequence minus the release: reach → descend → grasp → move.
Success is scored while the cube is still held, because there is no segment in
which to let go.

The cube is found visually; the place marker is fed in through the proprio move
goal slot, since it is too small on a 4x4 patch grid to localise from pixels.
Both the reward and the policy therefore read the marker from the same place in
the state.
"""
import jax
import jax.numpy as jnp
import numpy as np
import mujoco

from envs.base import VisionWarpEnv
from rewards.primitive_rewards import (reward_reach, reward_descend,
                                       reward_grasp, reward_move_generic)


class PickPlaceEnv(VisionWarpEnv):

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        t = self.tcfg
        self.cube_qpos_idx = int(t['cube_qpos_idx'])
        self.cube_xy_noise = float(t['cube_xy_noise'])
        self.mocap_body = t['mocap_body']
        self.place_center = jnp.array(t['place_center'])
        self.place_range = jnp.array(t['place_range'])
        self.success_thresh = float(t['success_thresh'])

    # ── indices ──
    def discover_indices(self, mj_model):
        N = mujoco.mj_name2id
        idx = super().discover_indices(mj_model)
        body_id = N(mj_model, mujoco.mjtObj.mjOBJ_BODY, self.mocap_body)
        assert body_id >= 0, f"mocap body '{self.mocap_body}' not found"
        idx.update({
            'cube1': N(mj_model, mujoco.mjtObj.mjOBJ_BODY, "box"),
            'mocap_idx': mj_model.body_mocapid[body_id],
            'left_finger': N(mj_model, mujoco.mjtObj.mjOBJ_BODY,
                             "gripper0_right_leftfinger"),
            'right_finger': N(mj_model, mujoco.mjtObj.mjOBJ_BODY,
                              "gripper0_right_rightfinger"),
        })
        for k, v in idx.items():
            assert v >= 0, f"Index discovery failed: {k} not found in XML"
            print(f"  {k}: {v}")
        return idx

    def make_task_cfg(self, idx):
        cfg = super().make_task_cfg(idx)
        cfg.update(cube1_body_idx=idx['cube1'], mocap_idx=idx['mocap_idx'])
        return cfg

    # ── proprio: the place target rides in the move goal slot ──
    def goal_slots(self, state):
        return jnp.concatenate([
            jnp.zeros(3),                             # reach goal — cube is visual
            state.mocap_pos[self.idx['mocap_idx']],   # move goal — place target
        ])

    # ── reward ──
    def segment_rewards(self, state, task_params):
        c = self.task_cfg
        ee, hand, tq = c['ee_site_idx'], c['hand_body_idx'], c['target_quat']
        cube1_pos = state.xpos[c['cube1_body_idx']]
        place_pos = state.mocap_pos[c['mocap_idx']]

        return [
            reward_reach(state, cube1_pos + jnp.array([0.0, 0.0, self.pre_grasp_z]),
                         tq, ee, hand),
            reward_descend(state, cube1_pos + jnp.array([0.0, 0.0, self.descend_z]),
                           tq, ee, hand),
            reward_grasp(state, cube1_pos, tq, ee, hand),
            reward_move_generic(state, cube1_pos, place_pos, ee, hand, tq),
        ]

    # ── reset ──
    def randomize_one(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        cube_noise = jax.random.uniform(k1, (2,), minval=-self.cube_xy_noise,
                                        maxval=self.cube_xy_noise)
        qpos = self.home_data.qpos.at[self.cube_qpos_idx].add(cube_noise[0])
        qpos = qpos.at[self.cube_qpos_idx + 1].add(cube_noise[1])
        qpos = self.with_arm_noise(qpos, k2)

        place = self.place_center + jax.random.uniform(
            k3, (3,), minval=-1.0, maxval=1.0) * self.place_range
        mocap_pos = self.home_data.mocap_pos.at[self.idx['mocap_idx']].set(place)
        return self.settle(qpos, mocap_pos=mocap_pos)

    # ── metrics ──
    def compute_metrics(self, all_states_batch, all_fng_batch):
        idx = self.idx
        s_ee = np.array(all_states_batch.site_xpos[:, :, idx['ee_site']])
        s_c1 = np.array(all_states_batch.xpos[:, :, idx['cube1']])
        s_tg = np.array(all_states_batch.mocap_pos[:, :, idx['mocap_idx']])
        s_fng = np.array(all_fng_batch)

        se = self.seg_ends
        all_metrics = np.zeros((self.batch_size, 4))
        all_metrics[:, 0] = np.linalg.norm(
            s_ee[:, se[0]] - (s_c1[:, se[0]] + [0, 0, self.pre_grasp_z]), axis=-1)
        all_metrics[:, 1] = np.linalg.norm(
            s_ee[:, se[1]] - (s_c1[:, se[1]] + [0, 0, self.descend_z]), axis=-1)
        all_metrics[:, 2] = np.linalg.norm(
            s_ee[:, se[2]] - s_c1[:, se[2]], axis=-1)
        # move: how far the cube ended from the place target
        all_metrics[:, 3] = np.linalg.norm(
            s_c1[:, se[3]] - s_tg[:, se[3]], axis=-1)
        return all_metrics, s_fng, {}

    def success_fn(self, all_metrics, extras):
        """Placed = cube within success_thresh of the marker, while still held."""
        count = int(np.sum(all_metrics[:, -1] < self.success_thresh))
        return count, count / self.batch_size
