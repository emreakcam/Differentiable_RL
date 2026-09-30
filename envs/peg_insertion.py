"""
PegInsertionEnv — grasp a peg, carry it over a hole and seat it.
================================================================
The longest sequence: reach → descend → grasp → move → align → insert →
release.  Two of those heads exist only for this task, and both want a smaller
learning rate than the approach primitives (see `training.lr_prim` in the
config): the tight peg/slot fit makes their gradients larger and spikier.

Two different points are tracked as the episode progresses.  Approach and grasp
follow `peg_grip`, where the gripper takes the peg; move, align and insert
follow `peg_bottom`, the tip that actually has to find the hole.  Tracking the
tip is what makes alignment a well-posed objective — the peg can be gripped
anywhere along its length, so the gripper pose alone does not determine where
the tip is.
"""
import jax
import jax.numpy as jnp
import numpy as np
import mujoco

from envs.base import VisionWarpEnv
from rewards.primitive_rewards import (reward_reach, reward_descend,
                                       reward_grasp, reward_move_peg,
                                       reward_align, reward_insert,
                                       reward_release_generic)


class PegInsertionEnv(VisionWarpEnv):

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        t = self.tcfg
        self.sites = dict(t['sites'])
        self.peg_qpos_idx = int(t['peg_qpos_idx'])
        self.peg_xy_noise = float(t['peg_xy_noise'])
        self.success_thresh = float(t['success_thresh'])

    # ── indices ──
    def discover_indices(self, mj_model):
        N = mujoco.mj_name2id
        idx = super().discover_indices(mj_model)
        s = self.sites
        idx.update({
            'peg_body':    N(mj_model, mujoco.mjtObj.mjOBJ_BODY, s['peg_body']),
            'peg_bottom':  N(mj_model, mujoco.mjtObj.mjOBJ_SITE, s['peg_bottom']),
            'peg_grip':    N(mj_model, mujoco.mjtObj.mjOBJ_SITE, s['peg_grip']),
            'slot_entry':  N(mj_model, mujoco.mjtObj.mjOBJ_SITE, s['slot_entry']),
            'slot_target': N(mj_model, mujoco.mjtObj.mjOBJ_SITE, s['slot_target']),
        })
        for k, v in idx.items():
            assert v >= 0, f"Index discovery failed: {k} not found in XML"
            print(f"  {k}: {v}")
        return idx

    def make_task_cfg(self, idx):
        cfg = super().make_task_cfg(idx)
        cfg.update(peg_body_idx=idx['peg_body'],
                   peg_bottom_idx=idx['peg_bottom'],
                   peg_grip_idx=idx['peg_grip'],
                   slot_entry_idx=idx['slot_entry'],
                   slot_target_idx=idx['slot_target'])
        return cfg

    # ── reward ──
    def segment_rewards(self, state, task_params):
        c = self.task_cfg
        ee, hand, tq = c['ee_site_idx'], c['hand_body_idx'], c['target_quat']

        peg_grip = state.site_xpos[c['peg_grip_idx']]
        peg_bottom = state.site_xpos[c['peg_bottom_idx']]
        slot_entry = state.site_xpos[c['slot_entry_idx']]

        return [
            # approach / grasp track peg_grip — where the gripper takes the peg
            reward_reach(state, peg_grip + jnp.array([0.0, 0.0, self.pre_grasp_z]),
                         tq, ee, hand),
            reward_descend(state, peg_grip, tq, ee, hand),
            reward_grasp(state, peg_grip, tq, ee, hand),
            # move / align / insert track peg_bottom — the tip that finds the hole
            reward_move_peg(state, peg_bottom, slot_entry, c['peg_body_idx'],
                            ee, hand, tq),
            reward_align(state, c['peg_bottom_idx'], c['slot_entry_idx'],
                         c['peg_grip_idx'], c['peg_body_idx'], ee, hand, tq),
            reward_insert(state, c['peg_bottom_idx'], c['slot_target_idx'],
                          c['slot_entry_idx'], c['peg_grip_idx'],
                          c['peg_body_idx'], ee, hand, tq),
            reward_release_generic(state, peg_grip, ee, hand, tq),
        ]

    # ── reset ──
    def randomize_one(self, key):
        k1, k2 = jax.random.split(key)
        peg_noise = jax.random.uniform(k1, (2,), minval=-self.peg_xy_noise,
                                       maxval=self.peg_xy_noise)
        qpos = self.home_data.qpos.at[self.peg_qpos_idx].add(peg_noise[0])
        qpos = qpos.at[self.peg_qpos_idx + 1].add(peg_noise[1])
        qpos = self.with_arm_noise(qpos, k2)
        return self.settle(qpos)

    # ── metrics ──
    def compute_metrics(self, all_states_batch, all_fng_batch):
        idx = self.idx
        sx = all_states_batch.site_xpos
        s_ee = np.array(sx[:, :, idx['ee_site']])
        s_pg = np.array(sx[:, :, idx['peg_grip']])
        s_pb = np.array(sx[:, :, idx['peg_bottom']])
        s_se = np.array(sx[:, :, idx['slot_entry']])
        s_st = np.array(sx[:, :, idx['slot_target']])
        s_fng = np.array(all_fng_batch)

        e = self.seg_ends
        m = np.zeros((self.batch_size, len(self.seg_steps)))
        # reach: ee → above peg_grip
        m[:, 0] = np.linalg.norm(
            s_ee[:, e[0]] - (s_pg[:, e[0]] + [0, 0, self.pre_grasp_z]), axis=-1)
        # descend: ee → peg_grip
        m[:, 1] = np.linalg.norm(
            s_ee[:, e[1]] - (s_pg[:, e[1]] + [0, 0, self.descend_z]), axis=-1)
        # grasp: ee → peg_grip
        m[:, 2] = np.linalg.norm(s_ee[:, e[2]] - s_pg[:, e[2]], axis=-1)
        # move: peg tip → slot entry
        m[:, 3] = np.linalg.norm(s_pb[:, e[3]] - s_se[:, e[3]], axis=-1)
        # align: peg tip xy vs the hole centre — height is the insert's job
        m[:, 4] = np.linalg.norm(
            s_pb[:, e[4], :2] - s_se[:, e[4], :2], axis=-1)
        # insert / release: peg tip → hole floor
        for si in range(5, len(e)):
            m[:, si] = np.linalg.norm(s_pb[:, e[si]] - s_st[:, e[si]], axis=-1)
        return m, s_fng, {}

    def success_fn(self, all_metrics, extras):
        """Seated = peg tip within success_thresh of the hole floor."""
        count = int(np.sum(all_metrics[:, -1] < self.success_thresh))
        return count, count / self.batch_size
