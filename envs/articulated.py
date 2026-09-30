"""
ArticulatedEnv — one hinge or slide joint, driven by its handle.
================================================================
Covers drawer_open, drawer_close, window_open, window_close, dial_turn and
door_open.  The six were separate scripts with identical structure; all that
ever differed was the joint name, the handle site, the endpoints and the
approach pose — every one of which now comes from configs/tasks.yaml:

    joint             MuJoCo joint whose qpos defines progress
    handle_site       what the gripper grabs
    joint_target      where it must end up
    joint_init_range  the start is drawn uniformly from this; a task and its
                      opposite overlap here on purpose (see randomize_one)

Segments are one or more approach primitives (reach / descend / grasp, in any
order the task declares) followed by a terminal one — pull_drawer, push_drawer,
slide_open, slide_close, turn or swing.  `prim_seq` names them, and the count
follows from it, so a task can stage its approach over two segments or three.
The terminal primitive is a distinct head per task family so that pulling a
drawer and rotating a dial do not share one set of weights, even though both
use the same `reward_slide` shaping in joint space.

The start position is randomised per episode, so it is handed to the reward as
a task parameter rather than baked into the config: `reward_slide` measures
progress from where this episode actually began.
"""
import jax
import jax.numpy as jnp
import numpy as np
import mujoco

from envs.base import VisionWarpEnv
from rewards.primitive_rewards import (reward_reach, reward_descend,
                                       reward_grasp, reward_slide)

#: Approach primitives an articulated task may use before its terminal segment.
#: All of them shape the same thing — get the gripper onto the handle — and
#: differ in how hard they push the fingers open and how close to the ground
#: they tolerate, so a task can stage its approach over as many of them as it
#: wants. The terminal segment is always reward_slide in joint space.
APPROACH_REWARDS = {
    'reach':   reward_reach,
    'descend': reward_descend,
    'grasp':   reward_grasp,
}


class ArticulatedEnv(VisionWarpEnv):

    #: the per-episode joint start travels alongside the state
    has_task_params = True

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        t = self.tcfg
        self.joint_name = t['joint']
        self.handle_name = t['handle_site']
        self.joint_target = float(t['joint_target'])
        self.init_lo, self.init_hi = (float(v) for v in t['joint_init_range'])
        self.success_thresh = float(t['success_thresh'])

        # Every segment but the last is an approach; the last drives the joint.
        self.prim_seq = list(t['prim_seq'])
        unknown = [p for p in self.prim_seq[:-1] if p not in APPROACH_REWARDS]
        if unknown:
            raise SystemExit(
                f"{self.name}: articulated tasks can only use "
                f"{sorted(APPROACH_REWARDS)} before the terminal segment; "
                f"got {unknown}")

        # Which approach offset each primitive uses. pre_grasp_z / descend_z
        # come from the shared `approach:` block (see VisionWarpEnv), so the
        # dial hovers the same distance above its site that cube_stacking
        # hovers above the cube.
        self.approach_dz = {
            'reach':   self.pre_grasp_z,
            'descend': self.descend_z,
            'grasp':   0.0,
        }

    # ── indices ──
    def discover_indices(self, mj_model):
        idx = super().discover_indices(mj_model)
        jnt_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_JOINT, self.joint_name)
        assert jnt_id >= 0, f"joint '{self.joint_name}' not found"
        idx['handle_site'] = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SITE, self.handle_name)
        idx['joint_qpos_idx'] = mj_model.jnt_qposadr[jnt_id]
        for k, v in idx.items():
            assert v >= 0, f"Index discovery failed: {k} not found"
            print(f"  {k}: {v}")
        return idx

    def make_task_cfg(self, idx):
        cfg = super().make_task_cfg(idx)
        cfg.update(handle_site_idx=idx['handle_site'],
                   joint_qpos_idx=idx['joint_qpos_idx'],
                   joint_target=self.joint_target)
        return cfg

    # ── reward ──
    def segment_rewards(self, state, task_params):
        c = self.task_cfg
        ee, hand, tq = c['ee_site_idx'], c['hand_body_idx'], c['target_quat']
        handle_site = c['handle_site_idx']
        handle_pos = state.site_xpos[handle_site]

        rewards = [
            APPROACH_REWARDS[p](
                state, handle_pos + jnp.array([0.0, 0.0, self.approach_dz[p]]),
                tq, ee, hand)
            for p in self.prim_seq[:-1]]
        # `task_params` is this episode's randomised joint start, so the
        # shaping measures progress from where the episode actually began.
        rewards.append(reward_slide(state, handle_site, c['joint_qpos_idx'],
                                    c['joint_target'], task_params, ee, hand))
        return rewards

    # ── reset ──
    def randomize_one(self, key):
        # Sampled uniformly over an explicit range rather than init +- noise
        # then clipped. The old form put `init` on a range boundary, so half the
        # noise was clamped away and 50% of episodes started at one exact value
        # — and the open/close variants of a task ended up with disjoint,
        # near-degenerate start clusters. A shared `reach` head could then score
        # full reward by memorising one cluster, which is the local minimum this
        # range is drawn to break: the ranges of a task and its opposite overlap,
        # so the same handle position occurs in both and only servoing to the
        # observed handle earns the reward.
        k1, k2 = jax.random.split(key, 2)
        joint_init = jax.random.uniform(k1, (), minval=self.init_lo,
                                        maxval=self.init_hi)

        qpos = self.with_arm_noise(self.home_data.qpos, k2)
        qpos = qpos.at[self.idx['joint_qpos_idx']].set(joint_init)
        return self.settle(qpos), joint_init

    # ── metrics ──
    def compute_metrics(self, all_states_batch, all_fng_batch):
        """Approach segments: ee-to-handle distance. Terminal: |joint - target|.

        Sized from seg_ends rather than a fixed 3, so a task can stage its
        approach over as many segments as its prim_seq declares.
        """
        idx = self.idx
        B = self.batch_size
        s_ee = np.array(all_states_batch.site_xpos[:, :, idx['ee_site']])
        s_handle = np.array(all_states_batch.site_xpos[:, :, idx['handle_site']])
        s_q = np.array(all_states_batch.qpos[:, :, idx['joint_qpos_idx']])
        s_fng = np.array(all_fng_batch)

        all_metrics = np.zeros((B, len(self.seg_ends)))
        # Approach segments: distance to that segment's own target, which is the
        # handle plus its height offset. Measuring every segment against the bare
        # handle would score `reach` as failing precisely when it is doing what
        # it is rewarded for — hovering above.
        for i, e in enumerate(self.seg_ends[:-1]):
            dz = self.approach_dz[self.prim_seq[i]]
            all_metrics[:, i] = np.linalg.norm(
                s_ee[:, e] - (s_handle[:, e] + [0, 0, dz]), axis=-1)
        # Terminal: distance from the target joint position
        all_metrics[:, -1] = np.abs(s_q[:, self.seg_ends[-1]] - self.joint_target)
        return all_metrics, s_fng, {}

    def success_fn(self, all_metrics, extras):
        """Solved = |joint - target| below success_thresh at the final step."""
        count = int(np.sum(all_metrics[:, -1] < self.success_thresh))
        return count, count / self.batch_size
