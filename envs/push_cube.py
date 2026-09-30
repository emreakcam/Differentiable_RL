"""
PushCubeEnv — slide a cube across the table to a marker, without grasping.
==========================================================================
Two segments: reach a standoff point behind the cube, then push.  The `grasp`
head never runs; `push_object` holds the fingers shut so the closed fist makes a
solid pushing face.

Marked `retired: true` in the config — it stays runnable by name but is left out
of `--tasks all`, since non-prehensile pushing is a much harder credit-
assignment problem than the prehensile tasks and is not worth the shared
training budget right now.
"""
import jax
import jax.numpy as jnp
import numpy as np
import mujoco

from envs.base import VisionWarpEnv
from rewards.primitive_rewards import reward_reach, reward_push_object


class PushCubeEnv(VisionWarpEnv):

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        t = self.tcfg
        self.cube_qpos_idx = int(t['cube_qpos_idx'])
        self.cube_xy_noise = float(t['cube_xy_noise'])
        self.push_standoff = float(t['push_standoff'])
        self.push_offset = float(t['push_offset'])
        self.mocap_body = t['mocap_body']
        self.goal_center = jnp.array(t['goal_center'])
        self.goal_range = jnp.array(t['goal_range'])
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

    # ── geometry ──
    def _standoff(self, cube_pos, goal_pos):
        """Point behind the cube, opposite the goal, at cube height.

        Direction is taken in xy only so the approach stays level with the cube.
        The 1e-6 floor keeps the normalise finite if the goal ever coincides
        with the cube — otherwise a NaN would propagate through the whole BPTT
        graph.
        """
        d_xy = goal_pos[:2] - cube_pos[:2]
        n = jnp.sqrt(jnp.dot(d_xy, d_xy) + 1e-6)
        back = d_xy / n * self.push_standoff
        return jnp.array([cube_pos[0] - back[0],
                          cube_pos[1] - back[1],
                          cube_pos[2] + self.push_offset])

    # ── reward ──
    def segment_rewards(self, state, task_params):
        c = self.task_cfg
        ee, hand, tq = c['ee_site_idx'], c['hand_body_idx'], c['target_quat']
        cube1_pos = state.xpos[c['cube1_body_idx']]
        goal_pos = state.mocap_pos[c['mocap_idx']]

        return [
            reward_reach(state, self._standoff(cube1_pos, goal_pos),
                         tq, ee, hand),
            reward_push_object(state, cube1_pos, goal_pos),
        ]

    # ── reset ──
    def randomize_one(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        cube_noise = jax.random.uniform(k1, (2,), minval=-self.cube_xy_noise,
                                        maxval=self.cube_xy_noise)
        qpos = self.home_data.qpos.at[self.cube_qpos_idx].add(cube_noise[0])
        qpos = qpos.at[self.cube_qpos_idx + 1].add(cube_noise[1])
        qpos = self.with_arm_noise(qpos, k2)

        goal = self.goal_center + jax.random.uniform(
            k3, (3,), minval=-1.0, maxval=1.0) * self.goal_range
        mocap_pos = self.home_data.mocap_pos.at[self.idx['mocap_idx']].set(goal)
        return self.settle(qpos, mocap_pos=mocap_pos)

    # ── metrics ──
    def compute_metrics(self, all_states_batch, all_fng_batch):
        idx = self.idx
        s_ee = np.array(all_states_batch.site_xpos[:, :, idx['ee_site']])
        s_c1 = np.array(all_states_batch.xpos[:, :, idx['cube1']])
        s_tg = np.array(all_states_batch.mocap_pos[:, :, idx['mocap_idx']])
        s_fng = np.array(all_fng_batch)

        se0, se1 = self.seg_ends
        all_metrics = np.zeros((self.batch_size, 2))
        # Reach: did the gripper line up behind the cube?
        c, g = s_c1[:, se0], s_tg[:, se0]
        d = g[:, :2] - c[:, :2]
        back = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-6) \
            * self.push_standoff
        stand = np.stack([c[:, 0] - back[:, 0], c[:, 1] - back[:, 1], c[:, 2]],
                         axis=-1)
        all_metrics[:, 0] = np.linalg.norm(s_ee[:, se0] - stand, axis=-1)
        # Push: how far the cube ended from the goal
        all_metrics[:, 1] = np.linalg.norm(s_c1[:, se1] - s_tg[:, se1], axis=-1)
        return all_metrics, s_fng, {}

    def success_fn(self, all_metrics, extras):
        """Pushed = cube within success_thresh of the goal at the final step."""
        count = int(np.sum(all_metrics[:, -1] < self.success_thresh))
        return count, count / self.batch_size
