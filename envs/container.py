"""
ContainerEnv — drop three objects into a container, one after another.
======================================================================
cube_stacking's five-primitive sequence repeated once per object, so a 15
segment episode driving the same five heads three times over.  The episode
structure is generated from the `per_object` block in configs/tasks.yaml rather
than written out: change the object list there and the segments, labels and
criteria follow.

Ball first: a sphere with no rolling friction is the hardest of the three to
grasp, so it gets the cleanest scene.

Success is scored with partial credit — each object inside the container radius
in xy is worth 1/n — because an all-or-nothing score gives no gradient signal
about the difference between placing one object and placing none.  Height is
deliberately ignored: every object targets the same drop point, so they settle
where they land and z carries no information about task completion.
"""
import jax
import jax.numpy as jnp
import numpy as np
import mujoco

from envs.base import VisionWarpEnv
from rewards.primitive_rewards import (reward_reach, reward_descend,
                                       reward_grasp, reward_move_generic,
                                       reward_release_generic)


class ContainerEnv(VisionWarpEnv):

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        t = self.tcfg
        po = t['per_object']
        self.objects = list(po['objects'])
        self.obj_qpos = dict(po['qpos_idx'])
        self.place_z_offset = float(t['place_z_offset'])
        self.container_site = t['container_site']
        self.container_radius = float(t['container_radius'])
        self.table_z = float(t['table_z'])
        self.obj_xy_noise = float(t['obj_xy_noise'])

    # ── episode structure: the per-object block, repeated ──
    def resolve_seg_steps(self):
        po = self.tcfg['per_object']
        return list(po['seg_steps']) * len(po['objects'])

    def resolve_seg_labels(self):
        po = self.tcfg['per_object']
        return [f"{p}-{l}" for p in po['label_prefixes'] for l in po['seg_labels']]

    def resolve_criteria(self):
        po = self.tcfg['per_object']
        return [c for _ in po['objects'] for c in po['criteria']]

    # ── indices ──
    def discover_indices(self, mj_model):
        idx = super().discover_indices(mj_model)
        idx['container'] = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SITE, self.container_site)
        for nm in self.objects:
            idx[nm] = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, nm)
        for k, v in idx.items():
            assert v >= 0, f"Index discovery failed: {k} not found in XML"
            print(f"  {k}: {v}")
        return idx

    def make_task_cfg(self, idx):
        cfg = super().make_task_cfg(idx)
        cfg.update(container_idx=idx['container'],
                   obj_idxs=[idx[nm] for nm in self.objects])
        return cfg

    # ── reward: five per object, in object order ──
    def segment_rewards(self, state, task_params):
        c = self.task_cfg
        ee, hand, tq = c['ee_site_idx'], c['hand_body_idx'], c['target_quat']
        container_pos = state.site_xpos[c['container_idx']]
        place_target = container_pos + jnp.array([0.0, 0.0, self.place_z_offset])

        rewards = []
        for bi in c['obj_idxs']:
            obj_pos = state.xpos[bi]
            rewards += [
                reward_reach(state, obj_pos + jnp.array([0.0, 0.0, self.pre_grasp_z]),
                             tq, ee, hand),
                reward_descend(state, obj_pos + jnp.array([0.0, 0.0, self.descend_z]),
                               tq, ee, hand),
                reward_grasp(state, obj_pos, tq, ee, hand),
                reward_move_generic(state, obj_pos, place_target, ee, hand, tq),
                reward_release_generic(state, place_target, ee, hand, tq),
            ]
        return rewards

    # ── reset: every object jittered, then the arm ──
    def randomize_one(self, key):
        keys = jax.random.split(key, len(self.objects) + 1)
        qpos = self.home_data.qpos
        for j, nm in enumerate(self.objects):
            off = self.obj_qpos[nm]
            n = jax.random.uniform(keys[j], (2,), minval=-self.obj_xy_noise,
                                   maxval=self.obj_xy_noise)
            qpos = qpos.at[off].add(n[0])
            qpos = qpos.at[off + 1].add(n[1])
        qpos = self.with_arm_noise(qpos, keys[-1])
        return self.settle(qpos)

    # ── metrics ──
    def compute_metrics(self, all_states_batch, all_fng_batch):
        idx = self.idx
        B, n_obj = self.batch_size, len(self.objects)
        s_ee = np.array(all_states_batch.site_xpos[:, :, idx['ee_site']])
        s_ct = np.array(all_states_batch.site_xpos[:, :, idx['container']])
        s_ob = {nm: np.array(all_states_batch.xpos[:, :, idx[nm]])
                for nm in self.objects}
        s_fng = np.array(all_fng_batch)

        se = self.seg_ends
        all_metrics = np.zeros((B, len(self.seg_steps)))
        for oi, nm in enumerate(self.objects):
            o = s_ob[nm]
            for k in range(5):
                si = oi * 5 + k
                e = se[si]
                place = s_ct[:, e] + np.array([0, 0, self.place_z_offset])
                if k == 0:      # reach: ee → above the object
                    all_metrics[:, si] = np.linalg.norm(
                        s_ee[:, e] - (o[:, e] + [0, 0, self.pre_grasp_z]), axis=-1)
                elif k == 1:    # descend: ee → at the object
                    all_metrics[:, si] = np.linalg.norm(
                        s_ee[:, e] - (o[:, e] + [0, 0, self.descend_z]), axis=-1)
                elif k == 2:    # grasp: ee → the object
                    all_metrics[:, si] = np.linalg.norm(s_ee[:, e] - o[:, e], axis=-1)
                elif k == 3:    # move: object → the drop point
                    all_metrics[:, si] = np.linalg.norm(o[:, e] - place, axis=-1)
                else:           # release: xy radius from the container
                    all_metrics[:, si] = np.linalg.norm(
                        o[:, e][:, :2] - s_ct[:, e][:, :2], axis=-1)

        # final-step xy radius per object — what success is scored on
        se_f = se[-1]
        radii = np.stack([
            np.linalg.norm(s_ob[nm][:, se_f, :2] - s_ct[:, se_f, :2], axis=-1)
            for nm in self.objects], axis=1)                     # (B, n_obj)
        # An object is "lost" if it ended below the table surface — the ball is
        # a sphere with no rolling friction, so a bad nudge sends it off the
        # edge. Checked per object at its own release step.
        fallen = np.stack([
            s_ob[nm][:, se[oi * 5 + 4], 2] < self.table_z
            for oi, nm in enumerate(self.objects)], axis=1)      # (B, n_obj)
        return all_metrics, s_fng, {'radii': radii, 'fallen': fallen}

    def success_fn(self, all_metrics, extras):
        """Partial credit: each object inside the container radius scores 1/n.

        An env with one of three objects placed counts as 0.33 rather than 0.
        """
        n_obj = len(self.objects)
        n_in = (extras['radii'] < self.container_radius).sum(axis=1)
        per_env = n_in / n_obj
        count = round(float(per_env.sum()), 2)      # "effective" solved envs
        return count, float(per_env.mean())
