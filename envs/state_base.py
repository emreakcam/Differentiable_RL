"""
StateWarpEnv — the state-observation counterpart of VisionWarpEnv.
==================================================================
Same tasks, same rewards, same primitive heads, same curriculum.  The single
difference is what reaches the policy:

    VisionWarpEnv   camera → DINOv3 patches → spatial softmax → trunk
    StateWarpEnv    exact object coordinates from the simulator → trunk

That one change removes a surprising amount of machinery.  The vision pipeline
splits an episode into chunks because the renderer and the backbone are
torch/GPU work *outside* JAX: it renders once per chunk, caches the features,
and replays the episode against that cache so the backward graph stays free of
the renderer.  Nothing here lives outside JAX, so an episode is one pure scan —
no render context, no feature cache, no chunking, and `collect_features_batched`
has no counterpart at all.

WHAT THIS CLASS OVERRIDES, AND NOTHING ELSE
-------------------------------------------
It is mixed in *in front of* the existing task classes (see envs/state_tasks.py):

    class StateCubeStackingEnv(StateWarpEnv, CubeStackingEnv): ...

so index discovery, reward composition, state randomisation, metrics and the
success criterion are inherited from the task class unchanged — those were
never vision-specific.  What this class replaces is the observation
(`build_obs`), the rollout (`_make_rollout`), the build sequence (`build`) and
the bundle handed to the trainer.

THE OBSERVATION
---------------
    [ ee(3) vel(3) finger(1) q(7) qdot(7) ee_quat(4) | slot0(3) … slotN(3) ]
      └────────────── the same 25-dim prefix as the vision pipeline ──────┘

The slots are a FIXED-WIDTH, zero-padded array declared per task in
`state_slots:`.  Fixed width is not incidental: one trunk and one ObsRMS serve
every task in a run, so a per-task-sized observation would mean a per-task
trunk and the end of cross-task primitive sharing.  See the `obs:` block in
configs/tasks_state.yaml for the slot kinds.
"""
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from envs.base import VisionWarpEnv

#: How each slot kind is resolved to an id at build time, and read out of a
#: state at rollout time.  Adding a kind means adding an entry here and to the
#: `obs:` documentation in the config — nothing else.
SLOT_KINDS = ('body', 'site', 'mocap', 'joint')


def parse_slot_specs(tcfg):
    """`state_slots:` → [(kind, name), …].

    Each YAML entry is a single-key mapping, e.g. {body: box} or
    {joint: drawer_slide}.  Raised as SystemExit rather than asserted: this is
    config the user wrote, and the message has to say which task is wrong.
    """
    specs = []
    for entry in tcfg.get('state_slots', []):
        if not isinstance(entry, dict) or len(entry) != 1:
            raise SystemExit(
                f"state_slots entries must be single-key mappings like "
                f"{{body: box}}; got {entry!r}")
        kind, name = next(iter(entry.items()))
        if kind not in SLOT_KINDS:
            raise SystemExit(
                f"unknown state_slots kind {kind!r} for {name!r}; "
                f"choose from {list(SLOT_KINDS)}")
        specs.append((kind, str(name)))
    return specs


class StateWarpEnv(VisionWarpEnv):

    # ═══════════════════════════════════════════════════════════════════
    # CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════════
    def __init__(self, *a, **kw):
        # Runs BEFORE the task class's __init__ (it is next in the MRO), which
        # in turn runs before VisionWarpEnv's — so self.tcfg is only readable
        # after the super() call has unwound.
        super().__init__(*a, **kw)

        obs = self.cfg['obs']
        self.n_slots = int(obs['n_slots'])
        self.slot_dim = int(obs['slot_dim'])
        self.prefix_dim = int(self.cfg['robot']['proprio_prefix_dim'])
        self.obs_dim = self.prefix_dim + self.n_slots * self.slot_dim

        self.slot_specs = parse_slot_specs(self.tcfg)
        if len(self.slot_specs) > self.n_slots:
            raise SystemExit(
                f"{self.name}: {len(self.slot_specs)} state_slots declared but "
                f"obs.n_slots is {self.n_slots}")

        # A `joint` slot reports [q, target, target - q]; the target is the
        # task's own joint_target, so the two can never disagree.
        self.slot_joint_target = {}
        for i, (kind, name) in enumerate(self.slot_specs):
            if kind == 'joint':
                if 'joint_target' not in self.tcfg:
                    raise SystemExit(
                        f"{self.name}: state slot {{joint: {name}}} needs the "
                        f"task's joint_target, which is not set")
                self.slot_joint_target[i] = float(self.tcfg['joint_target'])

    # ═══════════════════════════════════════════════════════════════════
    # SLOTS
    # ═══════════════════════════════════════════════════════════════════
    def discover_indices(self, mj_model):
        """The task's own ids, plus one id per declared slot."""
        idx = super().discover_indices(mj_model)
        N = mujoco.mj_name2id
        O = mujoco.mjtObj
        for i, (kind, name) in enumerate(self.slot_specs):
            if kind == 'body':
                v = N(mj_model, O.mjOBJ_BODY, name)
            elif kind == 'site':
                v = N(mj_model, O.mjOBJ_SITE, name)
            elif kind == 'mocap':
                body = N(mj_model, O.mjOBJ_BODY, name)
                if body < 0:
                    raise SystemExit(
                        f"{self.name}: state slot mocap body '{name}' not in XML")
                v = mj_model.body_mocapid[body]
            else:                                   # 'joint'
                jnt = N(mj_model, O.mjOBJ_JOINT, name)
                if jnt < 0:
                    raise SystemExit(
                        f"{self.name}: state slot joint '{name}' not in XML")
                v = mj_model.jnt_qposadr[jnt]
            if v < 0:
                raise SystemExit(
                    f"{self.name}: state slot {kind} '{name}' not in XML")
            idx[f'slot{i}'] = int(v)
            print(f"  slot{i} ({kind} {name}): {int(v)}")
        return idx

    def task_state(self, state):
        """The declared slots, concatenated and zero-padded to a fixed width.

        Padding with real zeros rather than leaving the slots out is what lets
        every task share one trunk: the observation has the same layout and the
        same width everywhere, and a task that uses fewer objects simply reads
        zero where another reads a coordinate.
        """
        parts = []
        for i, (kind, _name) in enumerate(self.slot_specs):
            k = self.idx[f'slot{i}']
            if kind == 'body':
                parts.append(state.xpos[k])
            elif kind == 'site':
                parts.append(state.site_xpos[k])
            elif kind == 'mocap':
                parts.append(state.mocap_pos[k])
            else:                                   # 'joint'
                q = state.qpos[k]
                tgt = self.slot_joint_target[i]
                # Remaining travel is handed over explicitly rather than left
                # for the trunk to subtract: it is what every terminal
                # articulated primitive is actually servoing on.
                parts.append(jnp.array([q, tgt, tgt - q]))
        parts += [jnp.zeros(self.slot_dim)] * (self.n_slots - len(self.slot_specs))
        return jnp.concatenate(parts)

    # ── the vision pipeline's 6 goal dims have no counterpart here ──
    # Both tasks that used them (reach's mocap target, pick_place's place
    # marker) now declare that same coordinate as a slot instead.
    def goal_slots(self, state):
        raise NotImplementedError(
            "goal_slots belongs to the vision observation; state tasks declare "
            "their coordinates in state_slots")

    # ═══════════════════════════════════════════════════════════════════
    # OBSERVATION
    # ═══════════════════════════════════════════════════════════════════
    def build_obs(self, state, prev_ee_pos):
        """ee(3)+vel(3)+finger(1)+q(7)+qdot(7)+ee_quat(4) + slots = obs_dim."""
        idx = self.idx
        ee_pos = state.site_xpos[idx['ee_site']]
        ee_vel = (ee_pos - prev_ee_pos) / self.frame_dt
        # Robosuite: finger1 in [0, 0.04], finger2 in [-0.04, 0]
        finger_width = state.qpos[self.qpos_pos_idx] - state.qpos[self.qpos_neg_idx]
        current_q = state.qpos[:self.n_joints]
        current_q_dot = state.qvel[:self.n_joints]
        ee_quat = state.xquat[idx['hand_body']]

        obs = jnp.concatenate([
            ee_pos,                       # 3
            ee_vel,                       # 3
            jnp.array([finger_width]),    # 1
            current_q,                    # 7
            current_q_dot,                # 7
            ee_quat,                      # 4
            self.task_state(state),       # n_slots x slot_dim
        ])
        return obs, ee_pos, current_q

    # ═══════════════════════════════════════════════════════════════════
    # BUILD — no renderer, no backbone
    # ═══════════════════════════════════════════════════════════════════
    def build(self):
        self._check_no_shadowed_methods()
        self._load_model()
        self._check_obs_dim()
        self._make_rollout()
        return self._bundle()

    def _check_obs_dim(self):
        """Trace one observation and confirm it matches robot.proprio_dim.

        Cheap here, and the alternative is a shape error deep inside a jitted
        BPTT trace — or worse, a silent mismatch against a checkpoint trained at
        a different width.
        """
        s = self.home_data
        obs, _, _ = self.build_obs(s, s.site_xpos[self.idx['ee_site']])
        if obs.shape[0] != self.obs_dim:
            raise SystemExit(
                f"{self.name}: observation is {obs.shape[0]} dims but the config "
                f"declares {self.obs_dim} "
                f"({self.prefix_dim} prefix + {self.n_slots}x{self.slot_dim})")
        print(f"  obs dim: {self.obs_dim} = {self.prefix_dim} proprio + "
              f"{len(self.slot_specs)}/{self.n_slots} slots x {self.slot_dim}")

    # ── rollout, loss, metrics — base's, minus the vision cache ──
    def _make_rollout(self):
        N_SUB = self.n_substeps
        N_TOTAL = self.n_total
        GAMMA = self.gamma
        BOUNDS = self.seg_boundaries
        ee_site = self.idx['ee_site']

        def rebatch_jax(rng_key, batch_size):
            keys = jax.random.split(rng_key, batch_size)
            return jax.vmap(self.randomize_one)(keys)

        self.rebatch_jit = jax.jit(rebatch_jax, static_argnums=(1,))

        # ── single environment step, for the PPO baseline ────────────────────
        # The DiffRL rollout below is one lax.scan over the whole episode: the
        # policy is inside the graph and only the episode return comes out.
        # PPO needs the opposite — an action supplied from outside, one step at
        # a time, with that step's reward.  This provides it from the SAME
        # pieces (build_obs, make_ctrl, select_reward, the same substep count
        # and the same active_steps freeze), so the two arms differ in the
        # learning rule rather than in the environment.
        #
        # `qd`/`f` arrive already squashed: the torch policy emits raw logits
        # and the squash lives in the wrapper, which is where the JAX policy's
        # head_forward would have applied it.
        def step_env(state, prev_ee, qd, f, step_idx, active_steps,
                     task_params=None):
            _, ee_pos, current_q = self.build_obs(state, prev_ee)
            ctrl = self.make_ctrl(qd, f, current_q)

            def substep(s, _):
                return mjx.step(self.mjx_model, s.replace(ctrl=ctrl)), None

            def do_substeps(s):
                s, _ = jax.lax.scan(substep, s, None, length=N_SUB)
                return s

            # Steps past active_steps are frozen exactly as in episode_loss, so
            # the curriculum means the same thing to both arms.
            new_state = jax.lax.cond(step_idx < active_steps,
                                     do_substeps, lambda s: s, state)
            rew = self.select_reward(new_state, step_idx, task_params)
            rew = jnp.where(step_idx < active_steps, rew, 0.0)
            next_obs, _, _ = self.build_obs(new_state, ee_pos)
            return new_state, ee_pos, next_obs, rew

        self.step_env_jit = jax.jit(jax.vmap(
            step_env, in_axes=(0, 0, 0, 0, None, None)))

        def reset_obs(states):
            """Observation at t=0. prev_ee seeds from the state itself, so the
            first ee velocity is zero rather than a jump off a stale value."""
            ee = states.site_xpos[:, ee_site]
            obs, _, _ = jax.vmap(self.build_obs)(states, ee)
            return obs, ee

        self.reset_obs_jit = jax.jit(reset_obs)

        def step_action(policy_params, state, prev_ee, obs_mean, obs_std,
                        step_idx):
            """One control decision: observation → policy → robosuite ctrl."""
            obs, ee_pos, current_q = self.build_obs(state, prev_ee)
            obs_norm = (obs.astype(jnp.float32) - obs_mean) / obs_std
            qd, f = self.policy_forward(policy_params, obs_norm, step_idx, BOUNDS)
            return self.make_ctrl(qd, f, current_q), ee_pos, obs, f

        # ── BPTT episode loss ──
        # The whole episode, in one scan. The vision version had to replay a
        # cached feature array here; there is nothing to cache.
        def episode_loss(policy_params, initial_state, obs_mean, obs_std,
                         active_steps, task_params=None):
            @jax.checkpoint
            def frame_step(carry, step_idx):
                state, prev_ee, total_rew, gamma_acc = carry

                # Cut the gradient at every segment boundary: each primitive is
                # credited only for its own segment, and the discount restarts.
                is_boundary = jnp.zeros((), dtype=bool)
                for b in BOUNDS:
                    is_boundary = is_boundary | (step_idx == b)
                state = jax.lax.cond(is_boundary, jax.lax.stop_gradient,
                                     lambda s: s, state)
                prev_ee = jax.lax.cond(is_boundary, jax.lax.stop_gradient,
                                       lambda p: p, prev_ee)
                gamma_acc = jnp.where(is_boundary, jnp.float64(1.0), gamma_acc)

                ctrl, ee_pos, obs, _ = step_action(
                    policy_params, state, prev_ee, obs_mean, obs_std, step_idx)

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

                return (state, ee_pos, total_rew, gamma_acc), obs

            init = (initial_state, initial_state.site_xpos[ee_site],
                    jnp.float64(0.0), jnp.float64(1.0))
            (final_state, _, total_rew, _), all_obs = jax.lax.scan(
                frame_step, init, jnp.arange(N_TOTAL))
            return -total_rew, (all_obs, final_state)

        if self.has_task_params:
            def batch_episode_loss(policy_params, all_states, obs_mean, obs_std,
                                   active_steps, task_params):
                losses, aux = jax.vmap(
                    episode_loss, in_axes=(None, 0, None, None, None, 0)
                )(policy_params, all_states, obs_mean, obs_std,
                  active_steps, task_params)
                return jnp.mean(losses), aux
        else:
            def batch_episode_loss(policy_params, all_states, obs_mean, obs_std,
                                   active_steps):
                losses, aux = jax.vmap(
                    episode_loss, in_axes=(None, 0, None, None, None)
                )(policy_params, all_states, obs_mean, obs_std, active_steps)
                return jnp.mean(losses), aux

        # ── metrics rollout (no gradient) ──
        def forward_for_metrics(policy_params, initial_state, obs_mean, obs_std,
                                active_steps):
            def frame_step(carry, step_idx):
                state, prev_ee = carry
                ctrl, ee_pos, _, f = step_action(
                    policy_params, state, prev_ee, obs_mean, obs_std, step_idx)

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

        def batch_forward_metrics(policy_params, all_states, obs_mean, obs_std,
                                  active_steps):
            return jax.vmap(
                forward_for_metrics, in_axes=(None, 0, None, None, None)
            )(policy_params, all_states, obs_mean, obs_std, active_steps)

        self.batch_episode_loss = batch_episode_loss
        self.batch_forward_metrics = batch_forward_metrics
        self.value_and_grad_fn = jax.jit(
            jax.value_and_grad(batch_episode_loss, has_aux=True))
        self.jit_batch_metrics = jax.jit(batch_forward_metrics)

    # ── the bundle ──
    def _bundle(self):
        """What the state trainer and eval consume.

        Deliberately a different shape from VisionWarpEnv's: every render key is
        gone, and so is `collect_features_batched`. A state bundle handed to the
        vision trainer fails immediately on the missing key rather than
        half-working.
        """
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
            batch_size=self.batch_size,
            obs_dim=self.obs_dim, n_slots=self.n_slots,
            slot_specs=self.slot_specs,
            rebatch_jit=self.rebatch_jit,
            # PPO baseline only; the DiffRL trainers never touch these.
            step_env_jit=self.step_env_jit,
            reset_obs_jit=self.reset_obs_jit,
            batch_episode_loss=self.batch_episode_loss,
            batch_forward_metrics=self.batch_forward_metrics,
            value_and_grad_fn=self.value_and_grad_fn,
            jit_batch_metrics=self.jit_batch_metrics,
            compute_metrics=self.compute_metrics,
            success_fn=self.success_fn,
            make_initial_state=self.make_initial_state,
        )
