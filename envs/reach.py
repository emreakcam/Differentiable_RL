"""
ReachEnv — drive the gripper to a randomised free-space target.
===============================================================
The simplest task, and the one that trains the `reach` head every other task
depends on.  One segment, one primitive, no object.

The target is a mocap body sampled per episode.  It is fed to the policy as
coordinates in the proprio goal slot rather than being found visually: a 2.5 cm
marker covers roughly 20 of 4096 pixels against a 4x4 patch grid, too little
signal to localise.  It is still rendered — hence the extra mocap render fields
in the config — so the scene the camera sees matches the state.
"""
import jax
import jax.numpy as jnp
import numpy as np
import mujoco

from envs.base import VisionWarpEnv
from rewards.primitive_rewards import reward_reach


class ReachEnv(VisionWarpEnv):

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        t = self.tcfg
        self.mocap_body = t['mocap_body']
        self.target_center = jnp.array(t['target_center'])
        self.target_range = jnp.array(t['target_range'])
        self.success_thresh = float(t['success_thresh'])

    # ── indices ──
    def discover_indices(self, mj_model):
        idx = super().discover_indices(mj_model)
        body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, self.mocap_body)
        assert body_id >= 0, f"mocap body '{self.mocap_body}' not found"
        idx['mocap_idx'] = mj_model.body_mocapid[body_id]
        for k, v in idx.items():
            assert v >= 0, f"Index discovery failed: {k} not found in XML"
            print(f"  {k}: {v}")
        return idx

    def make_task_cfg(self, idx):
        cfg = super().make_task_cfg(idx)
        cfg['mocap_idx'] = idx['mocap_idx']
        return cfg

    # ── proprio: the target rides in the reach goal slot ──
    def goal_slots(self, state):
        return jnp.concatenate([
            state.mocap_pos[self.idx['mocap_idx']],   # 3  reach goal
            jnp.zeros(3),                             # 3  move goal (unused)
        ])

    # ── reward ──
    def segment_rewards(self, state, task_params):
        c = self.task_cfg
        target_pos = state.mocap_pos[c['mocap_idx']]
        return [reward_reach(state, target_pos, c['target_quat'],
                             c['ee_site_idx'], c['hand_body_idx'])]

    # ── reset ──
    def randomize_one(self, key):
        k1, k2 = jax.random.split(key)
        noise = jax.random.uniform(k1, (3,), minval=-1.0, maxval=1.0)
        target = self.target_center + noise * self.target_range
        mocap_pos = self.home_data.mocap_pos.at[self.idx['mocap_idx']].set(target)
        qpos = self.with_arm_noise(self.home_data.qpos, k2)
        return self.settle(qpos, mocap_pos=mocap_pos)

    # ── metrics ──
    def compute_metrics(self, all_states_batch, all_fng_batch):
        """|ee - target| at the end of the episode."""
        idx = self.idx
        s_ee = np.array(all_states_batch.site_xpos[:, :, idx['ee_site']])
        # the goal rides in the state as a mocap body, so it is available here too
        s_tgt = np.array(all_states_batch.mocap_pos[:, :, idx['mocap_idx']])
        s_fng = np.array(all_fng_batch)

        se = self.seg_ends[0]
        all_metrics = np.zeros((self.batch_size, 1))
        all_metrics[:, 0] = np.linalg.norm(s_ee[:, se] - s_tgt[:, se], axis=-1)
        return all_metrics, s_fng, {}

    def success_fn(self, all_metrics, extras):
        """Solved = |ee - target| below success_thresh at the final step."""
        count = int(np.sum(all_metrics[:, -1] < self.success_thresh))
        return count, count / self.batch_size
