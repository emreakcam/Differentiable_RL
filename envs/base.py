"""
VisionWarpEnv — shared machinery for every robosuite vision-primitive task.
===========================================================================
Everything the twelve standalone training scripts used to duplicate lives here
exactly once:

  - MuJoCo / MJX model loading, visual-geom filtering, home keyframe
  - MJWarp GPU render context and the render-field copy
  - chunked feature collection (render → DINOv3 → vmapped physics)
  - the BPTT episode loss with per-segment gradient cuts
  - the metrics rollout
  - curriculum bookkeeping (phase bounds, segment ends)

A subclass supplies only what genuinely differs per task:

    discover_indices(mj_model)         → dict of MuJoCo ids
    make_task_cfg(idx)                 → what the reward closure needs
    goal_slots(state)                  → the 6 goal dims of proprio (default 0)
    segment_rewards(state, params)     → one reward per segment, in order
    randomize_one(key)                 → an initial mjx.Data (+ task params)
    compute_metrics(states, fingers)   → (metrics[B, n_seg], fingers, extras)
    success_fn(metrics, extras)        → (count, rate)

Numbers — segment lengths, thresholds, targets, noise — never appear in a
subclass; they come from configs/tasks.yaml through `self.tcfg`.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx
from mujoco.mjx import get_rgb

from helpers.mjx_utils import safe_action


def _to_f32(tree):
    return jax.tree.map(
        lambda x: x.astype(jnp.float32)
        if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating)
        else x, tree)


def _cumsum(steps):
    out, acc = [], 0
    for s in steps:
        acc += s
        out.append(acc)
    return out


class VisionWarpEnv:
    """One robosuite task: physics, rendering, rollout, loss and metrics.

    Construct with the task's config slice, then call `build()` to get the
    bundle the trainer and the eval script consume.
    """

    #: set True by subclasses whose reset returns per-env parameters alongside
    #: the state (the articulated tasks hand out each episode's joint start).
    has_task_params = False

    # ═══════════════════════════════════════════════════════════════════
    # CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════════
    def __init__(self, name, cfg, tcfg, backbone, policy_forward,
                 xml_path, batch_size, img_size, substeps, chunk_size, gamma,
                 full_chain=False):
        self.name = name
        self.cfg = cfg                  # whole config (robot, camera, physics)
        self.tcfg = tcfg                # this task's slice of `tasks:`
        self.backbone = backbone
        self.policy_forward = policy_forward
        self.xml_path = xml_path

        self.batch_size = batch_size
        self.img_size = img_size
        self.n_substeps = substeps
        self.chunk_size = chunk_size
        self.gamma = gamma
        # Ablation: carry gradients across segment boundaries instead of cutting
        # them. See the stop_gradient site in episode_loss.
        self.full_chain = full_chain

        # ── robot ──
        rb = cfg['robot']
        self.n_joints = rb['n_joints']
        self.proprio_dim = rb['proprio_dim']
        self.q_lo = jnp.array(rb['q_lo'])
        self.q_hi = jnp.array(rb['q_hi'])
        self.joint_reset_noise = rb['joint_reset_noise']
        self.ee_site_name = rb['ee_site']
        self.hand_body_name = rb['hand_body']
        g = rb['gripper']
        self.ctrl_pos_idx = g['ctrl_pos_idx']
        self.ctrl_neg_idx = g['ctrl_neg_idx']
        self.qpos_pos_idx = g['qpos_pos_idx']
        self.qpos_neg_idx = g['qpos_neg_idx']

        # ── episode structure ──
        self.seg_steps = list(self.resolve_seg_steps())
        self.seg_labels = list(self.resolve_seg_labels())
        self.criteria = [tuple(c) for c in self.resolve_criteria()]
        assert len(self.seg_steps) == len(self.seg_labels) == len(self.criteria), (
            f"{name}: seg_steps/seg_labels/criteria disagree "
            f"({len(self.seg_steps)}/{len(self.seg_labels)}/{len(self.criteria)})")

        self.n_total = sum(self.seg_steps)
        self.phase_bounds = _cumsum(self.seg_steps)
        self.seg_boundaries = self.phase_bounds[:-1]
        self.seg_ends = [b - 1 for b in self.phase_bounds]

        self.n_chunks = self.n_total // chunk_size
        self.n_feat_slots = self.n_chunks + 1

        # ── camera / render ──
        self.cam_name = cfg['camera']['name']
        self.n_cameras = cfg['camera']['count']
        self.render_fields = (list(cfg['render_fields'])
                              + list(tcfg.get('extra_render_fields', [])))

        self.target_quat = jnp.array(tcfg['target_quat']) \
            if 'target_quat' in tcfg else None

        # ── approach geometry, shared by every task ──
        # Where `reach` and `descend` aim RELATIVE to whatever target their task
        # tracks — the cube for cube_stacking, the dial site for dial_turn, the
        # peg grip for peg_insertion. One constant rather than a copy per task,
        # because it describes the arm's approach, not any one object. A task
        # may still override it if its geometry genuinely differs.
        # The hover applies only to tasks staged reach → descend → grasp, i.e.
        # those whose target sits on the table and is approached from above.
        # A task that goes straight reach → grasp aims at its target directly:
        # the drawer, window and door handles are on vertical faces, so there is
        # nothing above them to hover at, and no descend segment to close a gap
        # if one were introduced.
        ap = cfg.get('approach', {})
        staged = 'descend' in tcfg.get('prim_seq', [])
        self.pre_grasp_z = float(
            tcfg.get('pre_grasp_z', ap.get('pre_grasp_z', 0.0) if staged else 0.0))
        self.descend_z = float(tcfg.get('descend_z', ap.get('descend_z', 0.0)))

    def _check_no_shadowed_methods(self):
        """Fail here if a subclass stored config over one of our methods.

        Subclasses set config as plain attributes, so a name that collides with
        a method on this class silently replaces it — and the failure surfaces
        much later as a `'float' object is not callable` inside a jitted trace,
        pointing at base.py rather than at the subclass that caused it.
        """
        shadowed = [n for n in dir(type(self))
                    if not n.startswith('__')
                    and callable(getattr(type(self), n, None))
                    and not callable(self.__dict__.get(n, print))]
        if shadowed:
            raise TypeError(
                f"{type(self).__name__} overwrote method(s) {shadowed} with "
                f"config values; rename the attribute in the subclass")

    # ── config helpers, overridable where a task derives its structure ──
    def resolve_seg_steps(self):
        return self.tcfg['seg_steps']

    def resolve_seg_labels(self):
        return self.tcfg['seg_labels']

    def resolve_criteria(self):
        return self.tcfg['criteria']

    # ═══════════════════════════════════════════════════════════════════
    # SUBCLASS HOOKS
    # ═══════════════════════════════════════════════════════════════════
    def discover_indices(self, mj_model):
        """MuJoCo name → id lookups this task needs. Base resolves the arm."""
        return {
            'ee_site': mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name),
            'hand_body': mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_BODY, self.hand_body_name),
        }

    def make_task_cfg(self, idx):
        """The dict the reward closure reads. Base supplies the arm + pose."""
        return {
            'ee_site_idx':   idx['ee_site'],
            'hand_body_idx': idx['hand_body'],
            'target_quat':   self.target_quat,
        }

    def goal_slots(self, state):
        """The last 6 proprio dims: (reach_goal, move_goal).

        Zero unless the task feeds a goal in as coordinates — a small marker
        covers too few pixels on a 4x4 patch grid to be localised visually.
        """
        return jnp.zeros(6)

    def segment_rewards(self, state, task_params):
        """One reward per segment, in segment order. Length == len(seg_steps)."""
        raise NotImplementedError

    def randomize_one(self, key):
        """One randomised initial state; with task params if has_task_params."""
        raise NotImplementedError

    def compute_metrics(self, all_states_batch, all_fng_batch):
        """(batch, T, ...) rollout → (metrics[B, n_seg], fingers[B, T], extras)."""
        raise NotImplementedError

    def success_fn(self, all_metrics, extras):
        """→ (solved count, solved fraction) under this task's own criterion."""
        raise NotImplementedError

    # ── shared reset pieces subclasses compose ──
    # Named `arm_*` rather than `joint_*`: a task's own articulation (a drawer
    # slide, a door hinge) is also a joint, and a subclass storing its noise as
    # `self.joint_noise` would silently shadow a method called that.
    def arm_noise(self, key):
        """Uniform arm perturbation applied at every reset."""
        n = self.joint_reset_noise
        return jax.random.uniform(key, (self.n_joints,), minval=-n, maxval=n)

    def with_arm_noise(self, qpos, key):
        return qpos.at[:self.n_joints].add(self.arm_noise(key))

    def settle(self, qpos, **replace):
        """qpos (+ overrides) → a forward-solved mjx.Data from the home state."""
        d = self.home_data.replace(qpos=qpos, **replace)
        return mjx.forward(self.mjx_model, d)

    # ═══════════════════════════════════════════════════════════════════
    # PROPRIOCEPTION — identical 25-dim prefix, task-specific goal slots
    # ═══════════════════════════════════════════════════════════════════
    def build_proprio(self, state, prev_ee_pos):
        """ee(3)+vel(3)+finger(1)+q(7)+qdot(7)+ee_quat(4)+goals(6) = 31."""
        idx = self.idx
        ee_pos = state.site_xpos[idx['ee_site']]
        ee_vel = (ee_pos - prev_ee_pos) / self.frame_dt
        # Robosuite: finger1 in [0, 0.04], finger2 in [-0.04, 0]
        finger_width = state.qpos[self.qpos_pos_idx] - state.qpos[self.qpos_neg_idx]
        current_q = state.qpos[:self.n_joints]
        current_q_dot = state.qvel[:self.n_joints]
        ee_quat = state.xquat[idx['hand_body']]

        proprio = jnp.concatenate([
            ee_pos,                       # 3
            ee_vel,                       # 3
            jnp.array([finger_width]),    # 1
            current_q,                    # 7
            current_q_dot,                # 7
            ee_quat,                      # 4
            self.goal_slots(state),       # 6
        ])
        return proprio, ee_pos, current_q

    def select_reward(self, state, step_idx, task_params):
        """Nested where over the segment boundaries — last segment falls through."""
        rewards = self.segment_rewards(state, task_params)
        rew = rewards[-1]
        for i in range(len(self.seg_boundaries) - 1, -1, -1):
            rew = jnp.where(step_idx < self.seg_boundaries[i], rewards[i], rew)
        return rew

    def make_ctrl(self, qd, f, current_q):
        """Joint velocities + finger target → the full robosuite ctrl vector."""
        safe_qd = safe_action(qd, current_q, self.q_lo, self.q_hi)
        ctrl = self.home_ctrl.at[:self.n_joints].set(safe_qd)
        ctrl = ctrl.at[self.ctrl_pos_idx].set(f)
        ctrl = ctrl.at[self.ctrl_neg_idx].set(-f)
        return ctrl

    # ═══════════════════════════════════════════════════════════════════
    # BUILD — everything below was copy-pasted across the task scripts
    # ═══════════════════════════════════════════════════════════════════
    def build(self):
        """Compile the task and return the bundle the trainers consume."""
        self._check_no_shadowed_methods()
        self._load_model()
        self._make_renderer()
        self._make_rollout()
        return self._bundle()

    # ── 1. MuJoCo / MJX ──
    def _load_model(self):
        print(f"\n[1] Loading MuJoCo model: {self.xml_path}")
        mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        mj_model.opt.iterations = self.cfg['physics']['solver_iterations']
        mj_model.opt.ls_iterations = self.cfg['physics']['solver_ls_iterations']

        # Collision-only geoms are hidden from the camera: they carry no
        # material and would otherwise occlude the scene the policy sees.
        for i in range(mj_model.ngeom):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            is_visual = 'vis' in name or mj_model.geom_matid[i] >= 0
            if not is_visual:
                mj_model.geom_group[i] = 3

        mj_data = mujoco.MjData(mj_model)

        self.mj_model, self.mj_data = mj_model, mj_data
        self.idx = self.discover_indices(mj_model)
        self.task_cfg = self.make_task_cfg(self.idx)

        # Which keyframe the episode starts from. randomize_one() overwrites the
        # articulated joint anyway, so for the *policy* this only supplies the
        # arm pose — but it is also what the viewer resets its display to, and
        # a close-task showing a closed drawer before training starts is
        # actively misleading. Tasks that begin from the far end of their travel
        # name `home2`.
        self.key_id = mj_model.keyframe(self.tcfg.get('keyframe', 'home')).id
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, self.key_id)
        mujoco.mj_forward(mj_model, mj_data)

        self.home_ctrl = jnp.array(mj_data.ctrl)
        self.frame_dt = self.n_substeps * mj_model.opt.timestep
        self.mjx_model = mjx.put_model(mj_model)
        self.home_data = mjx.forward(self.mjx_model,
                                     mjx.put_data(mj_model, mj_data))

        print(f"  nq={mj_model.nq}, nu={mj_model.nu}, nsite={mj_model.nsite}")
        print(f"  Steps: {self.n_total}, Chunks: {self.n_chunks}, "
              f"Chunk size: {self.chunk_size}")
        print(f"  Feature slots: {self.n_feat_slots}, frame_dt: {self.frame_dt:.4f}s")

    # ── 2. MJWarp GPU renderer ──
    def _make_renderer(self):
        print("\n[2] Creating MJWarp GPU renderer...")
        mj_model, mj_data = self.mj_model, self.mj_data
        for i in range(mj_model.ncam):
            mj_model.cam_resolution[i] = [self.img_size, self.img_size]

        self.mx_warp = _to_f32(mjx.put_model(mj_model, impl='warp'))
        d_warp_template = _to_f32(mjx.put_data(mj_model, mj_data, impl='warp'))
        self.rc = mjx.create_render_context(mj_model, nworld=self.batch_size)
        self.rc_pytree = self.rc.pytree()

        @jax.jit
        @jax.vmap
        def warp_batched_forward(qpos):
            return mjx.forward(self.mx_warp, d_warp_template.replace(qpos=qpos))

        @jax.jit
        def warp_render(mx_w, d_w, rc_pt):
            d_w = mjx.refit_bvh(mx_w, d_w, rc_pt)
            pixels, _ = mjx.render(mx_w, d_w, rc_pt)
            return pixels

        self.warp_batched_forward = warp_batched_forward
        self.warp_render = warp_render

        self.warp_cam_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_CAMERA, self.cam_name)
        assert self.warp_cam_id >= 0, f"Camera '{self.cam_name}' not in XML"
        print(f"  Camera: {self.cam_name} → cam_id: {self.warp_cam_id}")
        print(f"  Render context: nworld={self.batch_size}, "
              f"{self.img_size}x{self.img_size}")

    def warp_render_to_rgb(self, states_batched):
        """JAX state → copy render fields (f32) → warp render. No forward."""
        d_render = self.d_warp_batched_template
        for field in self.render_fields:
            if hasattr(states_batched, field):
                val = getattr(states_batched, field).astype(jnp.float32)
                d_render = d_render.replace(**{field: val})

        pixels = self.warp_render(self.mx_warp, d_render, self.rc_pytree)
        jax.block_until_ready(pixels)
        return np.array(get_rgb(self.rc_pytree, self.warp_cam_id, pixels))

    def make_initial_state(self, seed=0):
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.key_id)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        d = mjx.put_data(self.mj_model, self.mj_data)
        return mjx.forward(self.mjx_model, d)

    # ── 3. rollout, loss, metrics ──
    def _make_rollout(self):
        CHUNK = self.chunk_size
        N_SUB = self.n_substeps
        N_TOTAL = self.n_total
        GAMMA = self.gamma
        BOUNDS = self.seg_boundaries
        FULL_CHAIN = self.full_chain
        ee_site = self.idx['ee_site']

        def rebatch_jax(rng_key, batch_size):
            keys = jax.random.split(rng_key, batch_size)
            return jax.vmap(self.randomize_one)(keys)

        self.rebatch_jit = jax.jit(rebatch_jax, static_argnums=(1,))

        def step_action(policy_params, vis_list, state, prev_ee,
                        obs_mean, obs_std, step_idx):
            """One control decision: proprio → policy → robosuite ctrl."""
            proprio, ee_pos, current_q = self.build_proprio(state, prev_ee)
            proprio_norm = (proprio.astype(jnp.float32) - obs_mean) / obs_std
            qd, f = self.policy_forward(policy_params, vis_list,
                                        proprio_norm, step_idx, BOUNDS)
            return self.make_ctrl(qd, f, current_q), ee_pos, proprio, f

        # ── forward chunk (no gradient): advances physics between renders ──
        def forward_chunk_single(policy_params, vis_list, state, prev_ee,
                                 obs_mean, obs_std, chunk_start, active_steps):
            def step_fn(carry, step_offset):
                state, prev_ee_pos = carry
                step_idx = chunk_start + step_offset
                ctrl, ee_pos, _, _ = step_action(
                    policy_params, vis_list, state, prev_ee_pos,
                    obs_mean, obs_std, step_idx)

                def do_sub(s):
                    def sub(s, _):
                        s = mjx.step(self.mjx_model, s.replace(ctrl=ctrl))
                        return s, s.qpos
                    return jax.lax.scan(sub, s, None, length=N_SUB)

                def skip_sub(s):
                    return s, jnp.broadcast_to(s.qpos, (N_SUB,) + s.qpos.shape)

                state, sub_qpos = jax.lax.cond(
                    step_idx < active_steps, do_sub, skip_sub, state)
                return (state, ee_pos), sub_qpos

            (state, ee), all_qpos = jax.lax.scan(
                step_fn, (state, prev_ee), jnp.arange(CHUNK))
            return state, ee, all_qpos

        def forward_chunk_batched(policy_params, vis_batch, states, prev_ees,
                                  obs_mean, obs_std, chunk_start, active_steps):
            def single(vis_arr, state, prev_ee):
                return forward_chunk_single(
                    policy_params, [vis_arr[0]], state, prev_ee,
                    obs_mean, obs_std, chunk_start, active_steps)
            return jax.vmap(single)(vis_batch, states, prev_ees)

        self.forward_chunk_batched_jit = jax.jit(forward_chunk_batched)

        # ── BPTT episode loss ──
        def episode_loss(policy_params, vision_array, initial_state,
                         obs_mean, obs_std, active_steps, task_params=None):
            @jax.checkpoint
            def frame_step(carry, step_idx):
                state, prev_ee, total_rew, gamma_acc = carry

                # Cut the gradient at every segment boundary: each primitive is
                # credited only for its own segment, and the discount restarts.
                #
                # Under `full_chain` the cut is removed and only the cut: the
                # discount still restarts per segment, the rewards are unchanged
                # and every other term is identical, so a full-chain run differs
                # from the default in exactly one thing. That is the point — it
                # is an ablation, not a second training mode.
                is_boundary = jnp.zeros((), dtype=bool)
                for b in BOUNDS:
                    is_boundary = is_boundary | (step_idx == b)
                if not FULL_CHAIN:
                    state = jax.lax.cond(is_boundary, jax.lax.stop_gradient,
                                         lambda s: s, state)
                    prev_ee = jax.lax.cond(is_boundary, jax.lax.stop_gradient,
                                           lambda p: p, prev_ee)
                gamma_acc = jnp.where(is_boundary, jnp.float64(1.0), gamma_acc)

                vis_list = [vision_array[step_idx // CHUNK, 0]]
                ctrl, ee_pos, proprio, _ = step_action(
                    policy_params, vis_list, state, prev_ee,
                    obs_mean, obs_std, step_idx)

                @jax.checkpoint
                def substep(s, _):
                    return mjx.step(self.mjx_model, s.replace(ctrl=ctrl)), None

                def do_substeps(s):
                    s, _ = jax.lax.scan(substep, s, None, length=N_SUB)
                    return s

                state = jax.lax.cond(step_idx < active_steps,
                                     do_substeps, lambda s: s, state)

                rew = self.select_reward(state, step_idx, task_params)
                rew = jnp.where(step_idx < active_steps, rew, 0.0)
                total_rew = total_rew + gamma_acc * rew
                gamma_acc = gamma_acc * GAMMA

                return (state, ee_pos, total_rew, gamma_acc), proprio

            init = (initial_state, initial_state.site_xpos[ee_site],
                    jnp.float64(0.0), jnp.float64(1.0))
            (final_state, _, total_rew, _), all_proprio = jax.lax.scan(
                frame_step, init, jnp.arange(N_TOTAL))
            return -total_rew, (all_proprio, final_state)

        if self.has_task_params:
            def batch_episode_loss(policy_params, all_vision, all_states,
                                   obs_mean, obs_std, active_steps, task_params):
                losses, aux = jax.vmap(
                    episode_loss, in_axes=(None, 0, 0, None, None, None, 0)
                )(policy_params, all_vision, all_states,
                  obs_mean, obs_std, active_steps, task_params)
                return jnp.mean(losses), aux
        else:
            def batch_episode_loss(policy_params, all_vision, all_states,
                                   obs_mean, obs_std, active_steps):
                losses, aux = jax.vmap(
                    episode_loss, in_axes=(None, 0, 0, None, None, None)
                )(policy_params, all_vision, all_states,
                  obs_mean, obs_std, active_steps)
                return jnp.mean(losses), aux

        # ── metrics rollout (no gradient) ──
        def forward_for_metrics(policy_params, vision_array, initial_state,
                                obs_mean, obs_std, active_steps):
            def frame_step(carry, step_idx):
                state, prev_ee = carry
                vis_list = [vision_array[step_idx // CHUNK, 0]]
                ctrl, ee_pos, _, f = step_action(
                    policy_params, vis_list, state, prev_ee,
                    obs_mean, obs_std, step_idx)

                def do_sub(s):
                    def sub(s, _):
                        s = mjx.step(self.mjx_model, s.replace(ctrl=ctrl))
                        return s, s.qpos
                    return jax.lax.scan(sub, s, None, length=N_SUB)

                def skip_sub(s):
                    return s, jnp.broadcast_to(s.qpos, (N_SUB,) + s.qpos.shape)

                state, step_qpos = jax.lax.cond(
                    step_idx < active_steps, do_sub, skip_sub, state)
                return (state, ee_pos), (state, step_qpos, f)

            init = (initial_state, initial_state.site_xpos[ee_site])
            (_, _), out = jax.lax.scan(frame_step, init, jnp.arange(N_TOTAL))
            return out

        def batch_forward_metrics(policy_params, all_vision, all_states,
                                  obs_mean, obs_std, active_steps):
            return jax.vmap(
                forward_for_metrics, in_axes=(None, 0, 0, None, None, None)
            )(policy_params, all_vision, all_states,
              obs_mean, obs_std, active_steps)

        self.batch_episode_loss = batch_episode_loss
        self.batch_forward_metrics = batch_forward_metrics
        self.value_and_grad_fn = jax.jit(
            jax.value_and_grad(batch_episode_loss, has_aux=True))
        self.jit_batch_metrics = jax.jit(batch_forward_metrics)

        # Warp render template — warp_render_to_rgb() reads this every chunk.
        tpl = self.rebatch_jit(jax.random.PRNGKey(8888), self.batch_size)
        tpl = tpl[0] if isinstance(tpl, tuple) else tpl
        self.d_warp_batched_template = self.warp_batched_forward(
            tpl.qpos.astype(jnp.float32))
        jax.block_until_ready(self.d_warp_batched_template.qpos)

    # ── 4. chunked feature collection ──
    def collect_features_batched(self, policy_params, batched_states,
                                 obs_mean, obs_std, active_steps,
                                 collect_qpos=False):
        """Roll the batch forward chunk by chunk, caching DINOv3 features.

        Rendering and the backbone are torch/GPU work outside JAX, so the
        episode is split into chunks: render once per chunk, then advance the
        physics under those features.  The BPTT pass later replays the same
        episode against this cache, which is what keeps the backward graph free
        of the renderer.
        """
        import torch

        n_active_chunks = int(np.ceil(float(active_steps) / self.chunk_size)) + 1
        n_active_chunks = min(n_active_chunks, self.n_chunks + 1)

        all_vision = np.zeros(
            (self.batch_size, self.n_feat_slots, self.n_cameras,
             self.backbone.grid_h, self.backbone.grid_w, 768), dtype=np.float32)
        all_qpos_env0 = []

        states = batched_states
        prev_ees = batched_states.site_xpos[:, self.idx['ee_site']]
        t_sync = t_render = t_dino = t_forward = 0.0

        for chunk_idx in range(self.n_chunks + 1):
            if chunk_idx >= n_active_chunks:
                break

            t0 = time.time()
            jax.block_until_ready(states.qpos)
            t_sync += time.time() - t0

            t0 = time.time()
            rgb_batch = self.warp_render_to_rgb(states)      # (B, H, W, 3)
            t_render += time.time() - t0

            t0 = time.time()
            flat_rgb = []
            for env_i in range(self.batch_size):
                img = rgb_batch[env_i]
                if img.dtype in (np.float32, np.float64):
                    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                flat_rgb.append(img)
            feats_flat = self.backbone.extract_batch(np.stack(flat_rgb, axis=0))
            torch.cuda.synchronize()
            feats_all = feats_flat.reshape(
                self.batch_size, 1, self.backbone.grid_h,
                self.backbone.grid_w, 768)
            t_dino += time.time() - t0

            all_vision[:, chunk_idx] = feats_all

            if chunk_idx < self.n_chunks and chunk_idx < n_active_chunks - 1:
                t0 = time.time()
                states, prev_ees, chunk_qpos = self.forward_chunk_batched_jit(
                    policy_params, jnp.array(feats_all), states, prev_ees,
                    obs_mean, obs_std, jnp.int32(chunk_idx * self.chunk_size),
                    active_steps)
                if collect_qpos:
                    all_qpos_env0.append(np.array(chunk_qpos[0]))
                t_forward += time.time() - t0

        return (jnp.array(all_vision), all_qpos_env0,
                t_sync, t_render, t_dino, t_forward)

    # ── 5. the bundle ──
    def _bundle(self):
        return dict(
            env=self,
            has_task_params=self.has_task_params,
            name=self.name, xml=self.xml_path,
            mj_model=self.mj_model, mj_data=self.mj_data,
            mjx_model=self.mjx_model, idx=self.idx, cfg=self.task_cfg,
            home_data=self.home_data, home_ctrl=self.home_ctrl,
            frame_dt=self.frame_dt, key_id=self.key_id,
            seg_steps=self.seg_steps, seg_boundaries=self.seg_boundaries,
            seg_labels=self.seg_labels, seg_ends=self.seg_ends,
            n_total=self.n_total, phase_bounds=self.phase_bounds,
            criteria=self.criteria,
            n_chunks=self.n_chunks, n_feat_slots=self.n_feat_slots,
            batch_size=self.batch_size,
            grid_h=self.backbone.grid_h, grid_w=self.backbone.grid_w,
            n_cameras=self.n_cameras, warp_cam_id=self.warp_cam_id,
            rebatch_jit=self.rebatch_jit,
            collect_features_batched=self.collect_features_batched,
            forward_chunk_batched_jit=self.forward_chunk_batched_jit,
            batch_episode_loss=self.batch_episode_loss,
            batch_forward_metrics=self.batch_forward_metrics,
            value_and_grad_fn=self.value_and_grad_fn,
            jit_batch_metrics=self.jit_batch_metrics,
            compute_metrics=self.compute_metrics,
            success_fn=self.success_fn,
            make_initial_state=self.make_initial_state,
            warp_render_to_rgb=self.warp_render_to_rgb,
            warp_batched_forward=self.warp_batched_forward,
            warp_render=self.warp_render, mx_warp=self.mx_warp,
            rc=self.rc, rc_pytree=self.rc_pytree,
            d_warp_batched_template=self.d_warp_batched_template,
        )
