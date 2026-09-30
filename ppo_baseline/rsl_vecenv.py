"""
rsl_rl VecEnv over the MJX state environment.
=============================================
Bridges a JAX/MJX task to rsl_rl's PyTorch PPO without leaving the GPU: actions
cross as dlpack capsules, so there is no host round-trip and no numpy copy in
the inner loop.

WHAT IS SHARED WITH THE DiffRL ARM, AND IS THEREFORE NOT REIMPLEMENTED HERE
  - the physics            `step_env_jit`, built from the same mjx_model
  - the observation        `build_obs` (25-dim proprio prefix + object slots)
  - the reward             `select_reward`, per segment, per step
  - the action mapping     `make_ctrl`, including safe_action's joint limits
  - the curriculum         active_steps freeze + pass_frac advancement
  - the metrics            `compute_metrics` / `success_fn` / `criteria`
  - observation scaling    StateObsRMS with the config's var_floor

WHAT DIFFERS, BY NECESSITY
  - the policy is stochastic (PPO samples; DiffRL is deterministic)
  - the action squash lives here rather than in the network, so PPO's Gaussian
    is over unbounded logits — see ppo_baseline/torch_policy.py
  - rewards reach PPO raw per-step and GAE discounts them with its own gamma;
    the DiffRL rollout instead restarts its own discount at each segment
    boundary, which is a credit-assignment device rather than a task property

EPISODES
Fixed length (n_total steps), no early termination — which is what the task
already does. `done` fires only on the final step, and the environment resets
every env at once.
"""
import jax
import jax.numpy as jnp
import types

import numpy as np
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv


def _to_torch(x, device):
    """JAX array -> torch tensor, on the GPU, with its own storage.

    The `.clone()` is not optional.  dlpack hands torch a VIEW of the JAX
    buffer, and every caller here passes a temporary (`rew.astype(...)`,
    `jnp.concatenate(...)`) whose refcount hits zero as the expression ends —
    so the buffer can be freed while torch still points into it.  Reading it
    back then yields garbage, which surfaces as NaN observations, and writing
    near it corrupts the heap ("corrupted double-linked list", core dump).
    Cloning costs a copy of a few hundred KB against a physics step; keeping
    the alias costs correctness.
    """
    x = jax.device_get(x) if device == 'cpu' else x
    return torch.from_dlpack(x).clone()


def _to_jax(x):
    """torch tensor -> JAX array, with its own storage (see _to_torch)."""
    return jnp.from_dlpack(x.detach().contiguous()).copy()


class MJXStateVecEnv(VecEnv):
    """One task, `num_envs` parallel episodes, rsl_rl's interface."""

    def __init__(self, task_bundle, obs_rms, n_joints, vel_limits, finger_open,
                 prim_ids, n_prims, patience, pass_frac, device='cuda:0',
                 curriculum=True, seed=0):
        t = self.t = task_bundle
        self.device = device
        self.num_envs = t['batch_size']
        self.num_actions = n_joints + 1
        self.max_episode_length = t['n_total']
        self.cfg = {}
        self.obs_rms = obs_rms

        self.n_joints = n_joints
        self.vel_limits = jnp.asarray(vel_limits)
        self.finger_open = float(finger_open)

        # Which primitive drives each step — the same step_idx -> primitive map
        # the JAX policy resolves internally, precomputed here because the
        # network now receives it as a one-hot input instead.
        # The FULL primitive vocabulary, not just this task's slice: the
        # policy carries a head per vocabulary entry exactly as the DiffRL one
        # does, so the one-hot has to index the same space the heads do. A task
        # that uses five of fourteen simply never activates the other nine.
        self.n_prims = int(n_prims)
        bounds = list(t['seg_boundaries'])
        self.step_prim = np.empty(t['n_total'], dtype=np.int64)
        for s in range(t['n_total']):
            seg = sum(1 for b in bounds if s >= b)
            self.step_prim[s] = prim_ids[min(seg, len(prim_ids) - 1)]

        # ── curriculum, identical in rule to the DiffRL trainer ──
        self.curriculum = curriculum
        self.phase = 0
        self.patience_cnt = 0
        self.PATIENCE = patience
        self.PASS_FRAC = pass_frac
        pb = t['phase_bounds']
        self.active_steps = (int(pb[min(1, len(pb) - 1)]) if curriculum
                             else int(t['n_total']))

        self._key = jax.random.PRNGKey(seed)
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=device)
        self._step = 0
        self._traj = []          # states over the episode, for the metrics
        self.viewer = None       # set by the trainer; replays env 0
        self.viewer_every = 1
        self._episodes = 0
        self.last_qpos0 = None   # env 0's trajectory, for the viewer
        self.n_nan_envs = 0      # env-steps rolled back, read at episode end
        self._nan_count = jnp.int32(0)   # stays on device; no per-step sync
        self._reset_jax()

    # ── rsl_rl interface ────────────────────────────────────────────────────
    def get_observations(self):
        return self._obs_td()

    def step(self, actions):
        qd, f = self._squash(_to_jax(actions.to(torch.float32)))
        st, ee, obs, rew = self.t['step_env_jit'](
            self._state, self._prev_ee, qd, f,
            jnp.int32(self._step), jnp.int32(self.active_steps))
        obs, rew = self._guard(obs, rew)
        self._state, self._prev_ee, self._obs = st, ee, obs
        # Only what compute_metrics reads, not the whole MJX state: a state is
        # ~24KB per env, so keeping 100 of them at 1500 envs is ~3.6GB and ran
        # the GPU out of memory. These two fields are ~0.9KB per env per step.
        self._traj.append((st.site_xpos, st.xpos,
                           st.qpos[0] if self.viewer is not None else None))
        self._step += 1
        self.episode_length_buf += 1

        rew_t = _to_torch(rew.astype(jnp.float32), self.device).to(self.device)
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        extras = {}
        if self._step >= self.max_episode_length:
            done[:] = True
            extras = self._end_of_episode()
            self._reset_jax()
        return self._obs_td(), rew_t, done, extras

    def reset(self):
        self._reset_jax()
        return self._obs_td(), {}

    # ── internals ───────────────────────────────────────────────────────────
    def _guard(self, obs, rew):
        """Keep non-finite physics out of the learner, as cheaply as possible.

        An earlier version rolled the offending env's WHOLE state back to its
        episode start with a `jnp.where` over every field of the MJX state.
        That was correct and unaffordable: it ran 1500-env `where`s across
        ~30 arrays every step and dropped throughput from 107k to 7k
        env-steps/s. Since an episode resets all envs every n_total steps, a
        diverged env is already bounded in how long it can misbehave — so the
        cheap version just replaces the non-finite VALUES the learner would
        otherwise choke on, and lets the reset clean up the state.

        Counted on device; the total is read at episode end where a sync is
        already being paid. Reading it per step would itself cost ~13x.
        """
        finite = jnp.isfinite(obs).all(axis=-1) & jnp.isfinite(rew)
        self._nan_count = self._nan_count + (~finite).sum()
        return jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0), \
            jnp.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)

    def _squash(self, raw):
        """The bound the JAX head_forward applies, moved out of the network.

        PPO's Gaussian is over `raw`; the environment is what turns it into a
        joint velocity and a finger target, so the obs -> ctrl mapping matches
        the DiffRL arm end to end.
        """
        qd = jnp.tanh(raw[:, :self.n_joints]) * self.vel_limits
        f = jax.nn.sigmoid(raw[:, self.n_joints]) * self.finger_open
        return qd, f

    def _obs_td(self):
        """Normalised observation + the active primitive's one-hot.

        Normalisation uses the run's own StateObsRMS, floor included; rsl_rl's
        EmpiricalNormalization is left disabled so the padded constant slots are
        scaled the same way in both arms.
        """
        mean, std = self.obs_rms.get_jnp()
        obs_n = (self._obs.astype(jnp.float32) - mean) / std
        oh = jax.nn.one_hot(
            int(self.step_prim[min(self._step, self.max_episode_length - 1)]),
            self.n_prims, dtype=jnp.float32)
        oh = jnp.broadcast_to(oh, (self.num_envs, self.n_prims))
        policy = _to_torch(jnp.concatenate([obs_n, oh], -1), self.device)

        # ── critic-only time channel ────────────────────────────────────────
        # The actor deliberately sees no clock: it must stay identical to the
        # DiffRL policy, which has none either. The critic has no DiffRL
        # counterpart, so it is free to see more — and it needs to, for two
        # reasons the one-hot cannot cover:
        #
        #   phase   the one-hot names the primitive but not the position within
        #           it. With seg_steps [25,20,10,35,10] every step of 0..24
        #           looks identical, though the return at step 0 discounts 100
        #           steps of reward and at step 24 only 76.
        #   live    beyond active_steps the physics is frozen and the reward is
        #           zero, yet step_prim still announces grasp/move/release. The
        #           critic would otherwise see "grasp, reward 0" now and
        #           "grasp, reward real" after the curriculum advances, from an
        #           indistinguishable input.
        st = min(self._step, self.max_episode_length)
        n = float(self.max_episode_length)
        time_ch = jnp.array([st / n, max(0.0, self.active_steps - st) / n],
                            dtype=jnp.float32)
        time_ch = jnp.broadcast_to(time_ch, (self.num_envs, 2))
        critic_extra = _to_torch(time_ch, self.device)
        return TensorDict({"policy": policy.to(self.device),
                           "time": critic_extra.to(self.device)},
                          batch_size=[self.num_envs])

    def _reset_jax(self):
        self._key, k = jax.random.split(self._key)
        self._state = self.t['rebatch_jit'](k, self.num_envs)
        self._obs, self._prev_ee = self.t['reset_obs_jit'](self._state)
        self._nan_count = jnp.int32(0) if not hasattr(self, "_nan_count") else self._nan_count
        self._step = 0
        self._traj = []
        self.episode_length_buf.zero_()

    def _end_of_episode(self):
        """Task metrics + the curriculum tick, on the same rule as training.

        rsl_rl only ever sees per-step rewards and dones; the segment criteria
        are trajectory quantities (`compute_metrics` reads states at seg_ends),
        so they are computed here and reported through `extras`.
        """
        # The one place a sync is already being paid, so the guard's counter is
        # read here rather than per step.
        self._episodes += 1
        if self.viewer is not None and self._episodes % self.viewer_every == 0:
            # Only env 0, and only on viewer episodes: this is a host transfer
            # of (steps, nq) and there is no reason to pay it otherwise.
            self.last_qpos0 = np.asarray(
                jnp.stack([x[2] for x in self._traj]))
            # Blocks for about n_total*frame_dt seconds, same as the DiffRL
            # trainers' viewer does after a log line. Raise --viewer-every to
            # pay it less often.
            if self.viewer.running:
                self.viewer.replay(self.last_qpos0)

        _n = int(self._nan_count)
        if _n > self.n_nan_envs:
            print(f"  ! guard rolled back {_n - self.n_nan_envs} env-steps "
                  f"of non-finite physics this episode ({_n} total)")
        self.n_nan_envs = _n
        self.obs_rms.update(np.asarray(self._obs, np.float64))
        # compute_metrics indexes .site_xpos and .xpos as (env, step, ...), so a
        # namespace with those two fields is all it needs.
        states_b = types.SimpleNamespace(
            site_xpos=jnp.stack([x[0] for x in self._traj], axis=1),
            xpos=jnp.stack([x[1] for x in self._traj], axis=1))
        fng = jnp.zeros((self.num_envs, len(self._traj)))
        metrics, _s_fng, ex = self.t['compute_metrics'](states_b, fng)
        n_ok, rate = self.t['success_fn'](metrics, ex)

        frac = np.ones(len(self.t['criteria']))
        for si, (ctype, thr) in enumerate(self.t['criteria']):
            col = metrics[:, si]
            frac[si] = (1.0 if ctype == "always"
                        else float((col > thr).mean()) if ctype in ("lift", "angle")
                        else float((col < thr).mean()))

        full = self.active_steps >= self.t['n_total']
        if self.curriculum and self.phase < len(self.t['phase_bounds']) - 1:
            ok = frac[self.phase] >= self.PASS_FRAC
            self.patience_cnt = self.patience_cnt + 1 if ok else 0
            if self.patience_cnt >= self.PATIENCE:
                self.phase += 1
                self.patience_cnt = 0
                pb = self.t['phase_bounds']
                self.active_steps = int(pb[min(self.phase + 1, len(pb) - 1)])
        return {"episode": {
            "success_rate": float(rate) if full else 0.0,
            "raw_success_rate": float(rate),
            "phase": float(self.phase),
            "active_steps": float(self.active_steps),
            **{f"seg_frac/{l}": float(frac[i])
               for i, l in enumerate(self.t['seg_labels'])},
            **{f"seg_mean/{l}": float(metrics[:, i].mean())
               for i, l in enumerate(self.t['seg_labels'])},
        }}
