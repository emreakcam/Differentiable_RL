"""
CubeStackingEnv — pick up one cube and stack it on another.
===========================================================
The canonical five-primitive sequence: reach → descend → grasp → move →
release.  Every other pick-style task is a variation on it, which is why these
five heads see gradient from the most tasks.

Both cubes are found visually — nothing is fed through the proprio goal slots.
Only the carried cube's start position is randomised; the target cube stays put,
so the policy has a stable landmark to stack against.
"""
import jax
import jax.numpy as jnp
import numpy as np
import mujoco

from envs.base import VisionWarpEnv
from rewards.primitive_rewards import (reward_reach, reward_descend,
                                       reward_grasp, reward_move_generic,
                                       reward_release_generic)


class CubeStackingEnv(VisionWarpEnv):

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        t = self.tcfg
        self.cube2_height = float(t['cube2_height'])
        self.stack_offset = float(t['stack_offset'])
        self.cube_qpos_idx = int(t['cube_qpos_idx'])
        self.cube_xy_noise = float(t['cube_xy_noise'])
        self.success_xy_tol = float(t['success_xy_tol'])

    # ── indices ──
    def discover_indices(self, mj_model):
        N = mujoco.mj_name2id
        idx = super().discover_indices(mj_model)
        idx.update({
            'cube1': N(mj_model, mujoco.mjtObj.mjOBJ_BODY, "box"),
            'cube2': N(mj_model, mujoco.mjtObj.mjOBJ_BODY, "box2"),
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
        cfg.update(cube1_body_idx=idx['cube1'], cube2_body_idx=idx['cube2'])
        return cfg

    # ── reward ──
    def segment_rewards(self, state, task_params):
        c = self.task_cfg
        ee, hand, tq = c['ee_site_idx'], c['hand_body_idx'], c['target_quat']
        cube1_pos = state.xpos[c['cube1_body_idx']]
        cube2_pos = state.xpos[c['cube2_body_idx']]

        # Reach hovers above the cube; descend drops onto it; move carries it to
        # a point one cube-height above the base cube; release opens there.
        stack_target = cube2_pos + jnp.array(
            [0.0, 0.0, self.cube2_height + self.stack_offset])

        return [
            reward_reach(state, cube1_pos + jnp.array([0.0, 0.0, self.pre_grasp_z]),
                         tq, ee, hand),
            reward_descend(state, cube1_pos + jnp.array([0.0, 0.0, self.descend_z]),
                           tq, ee, hand),
            reward_grasp(state, cube1_pos, tq, ee, hand),
            reward_move_generic(state, cube1_pos, stack_target, ee, hand, tq),
            reward_release_generic(state, stack_target, ee, hand, tq),
        ]

    # ── reset ──
    def randomize_one(self, key):
        k1, k2 = jax.random.split(key)
        cube_noise = jax.random.uniform(k1, (2,), minval=-self.cube_xy_noise,
                                        maxval=self.cube_xy_noise)
        qpos = self.home_data.qpos.at[self.cube_qpos_idx].add(cube_noise[0])
        qpos = qpos.at[self.cube_qpos_idx + 1].add(cube_noise[1])
        qpos = self.with_arm_noise(qpos, k2)
        return self.settle(qpos)

    # ── metrics ──
    def compute_metrics(self, all_states_batch, all_fng_batch):
        """Per segment: how close the tracked body got to that segment's goal."""
        idx = self.idx
        B = self.batch_size
        s_ee = np.array(all_states_batch.site_xpos[:, :, idx['ee_site']])
        s_c1 = np.array(all_states_batch.xpos[:, :, idx['cube1']])
        s_c2 = np.array(all_states_batch.xpos[:, :, idx['cube2']])
        s_fng = np.array(all_fng_batch)

        se = self.seg_ends
        stack_off = np.array([0, 0, self.cube2_height + self.stack_offset])
        all_metrics = np.zeros((B, 5))
        # reach / descend: ee vs a point relative to the carried cube
        all_metrics[:, 0] = np.linalg.norm(
            s_ee[:, se[0]] - (s_c1[:, se[0]] + [0, 0, self.pre_grasp_z]), axis=-1)
        all_metrics[:, 1] = np.linalg.norm(
            s_ee[:, se[1]] - (s_c1[:, se[1]] + [0, 0, self.descend_z]), axis=-1)
        # grasp: ee vs the cube itself
        all_metrics[:, 2] = np.linalg.norm(
            s_ee[:, se[2]] - s_c1[:, se[2]], axis=-1)
        # move: the cube vs the stack point
        all_metrics[:, 3] = np.linalg.norm(
            s_c1[:, se[3]] - (s_c2[:, se[3]] + stack_off), axis=-1)
        # release: xy miss + z miss, so a cube resting beside the base scores badly
        xy_err = np.linalg.norm(
            s_c1[:, se[4], :2] - s_c2[:, se[4], :2], axis=-1)
        z_err = np.abs(s_c1[:, se[4], 2] - (s_c2[:, se[4], 2] + stack_off[2]))
        all_metrics[:, 4] = xy_err + z_err
        return all_metrics, s_fng, {'c1': s_c1, 'c2': s_c2}

    def success_fn(self, all_metrics, extras):
        """Stacked = cube1 within success_xy_tol of cube2 in xy at the last step."""
        se_final = self.seg_ends[-1]
        c1, c2 = extras['c1'][:, se_final], extras['c2'][:, se_final]
        ok = ((np.abs(c1[:, 0] - c2[:, 0]) < self.success_xy_tol) &
              (np.abs(c1[:, 1] - c2[:, 1]) < self.success_xy_tol))
        count = int(ok.sum())
        return count, count / self.batch_size
