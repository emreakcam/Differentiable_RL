"""
PPO baseline for the state-observation experiments (rsl_rl).
============================================================
The counterpart to train_robosuit/train_primitives_state.py: same task, same
network, same reward, same curriculum — a different learning rule.  DiffRL
differentiates through the simulator; this estimates the gradient from sampled
returns.

WHAT IS SHARED, BY CALLING THE SAME CODE RATHER THAN COPYING IT
    physics/observation/reward   envs/state_base.py via step_env_jit
    network                      the trunk + per-primitive heads, ported to
                                 torch and checked to 1.3e-07 against the JAX
                                 original by verify_policy_match.py
    curriculum                   the same pass_frac/patience rule
    metrics, criteria, success   the task's own compute_metrics / success_fn
    observation scaling          StateObsRMS, variance floor included

WHAT NECESSARILY DIFFERS
    stochastic policy     PPO samples; the DiffRL policy is deterministic
    value function        an addition PPO requires
    action squash         moved from the network into the environment, so the
                          Gaussian is over unbounded logits (see torch_policy)
    discount              raw per-step rewards, discounted by GAE's own gamma;
                          the DiffRL rollout restarts its discount per segment

Default 2048 envs: the rollout carries no gradient tape, so the batch that fits
is two orders of magnitude larger than the BPTT trainer's 20.  Measured
throughput on this GPU is ~107k env-steps/s at 2048 against ~358/s for DiffRL,
so the two are closer in wall-clock than in sample count.

    python ppo_baseline/train_ppo.py --task cube_stacking
    python ppo_baseline/train_ppo.py --task cube_stacking --num-envs 1024
"""
import sys, os, argparse, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from helpers.xla_env import apply_xla_defaults
apply_xla_defaults()

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")

import torch

import envs
from envs.state_registry import load_state_config, make_state_env
from models.state_policy import make_state_forward
from models.vision_policy import N_JOINTS, VEL_LIMITS, FINGER_OPEN
from helpers.state_obs import StateObsRMS
from helpers.mjx_utils import patch_solver

import pickle

from rsl_rl.runners import OnPolicyRunner


class ResumableRunner(OnPolicyRunner):
    """OnPolicyRunner that also persists the ENVIRONMENT's training state.

    rsl_rl's checkpoint holds the actor, the critic, the optimizer and the
    iteration counter — everything it owns. It does not know about the
    curriculum phase, the patience counter, active_steps or the ObsRMS, because
    those live in the VecEnv wrapper here. Resuming from the .pt alone would
    restore a trained network and then feed it observations normalised by fresh
    statistics, with the curriculum back at phase 0 — worse than not resuming,
    since the mismatch is silent.

    So each save writes a sidecar next to the .pt, and each load restores it.
    """

    def _sidecar(self, path):
        return path.replace(".pt", "_env.pkl")

    def save(self, path, infos=None):
        super().save(path, infos)
        e = self.env
        with open(self._sidecar(path), "wb") as f:
            pickle.dump({"phase": e.phase, "patience_cnt": e.patience_cnt,
                         "active_steps": e.active_steps,
                         "episodes": e._episodes,
                         "obs_rms": {"mean": e.obs_rms.mean,
                                     "var": e.obs_rms.var,
                                     "count": e.obs_rms.count}}, f)

    def load(self, path, *a, **kw):
        out = super().load(path, *a, **kw)
        side = self._sidecar(path)
        if not os.path.exists(side):
            print(f"  ! {side} missing — curriculum and ObsRMS restart from "
                  f"scratch, which will not match the loaded weights")
            return out
        with open(side, "rb") as f:
            d = pickle.load(f)
        e = self.env
        e.phase, e.patience_cnt = d["phase"], d["patience_cnt"]
        e.active_steps, e._episodes = d["active_steps"], d["episodes"]
        e.obs_rms.mean = d["obs_rms"]["mean"]
        e.obs_rms.var = d["obs_rms"]["var"]
        e.obs_rms.count = d["obs_rms"]["count"]
        print(f"  resumed env state: phase={e.phase} "
              f"active_steps={e.active_steps} obs_rms count={e.obs_rms.count:.0f}")
        return out

from ppo_baseline.rsl_vecenv import MJXStateVecEnv
from ppo_baseline.rsl_models import TrunkHeadActor

patch_solver()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="cube_stacking")
    p.add_argument("--config", default=None)
    p.add_argument("--num-envs", type=int, default=2048,
                   help="best measured throughput that fits in 8GB, verified "
                        "NaN-free over 1.8M env-steps. AVOID 256: MJX returns "
                        "NaN for every env from step 0 at exactly that size "
                        "(128/192/240/272/320/512/1024/1536/2048 are all fine). "
                        "Changing this costs a ~40s recompile.")
    p.add_argument("--num-steps-per-env", type=int, default=24,
                   help="transitions collected per env per PPO update")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.0,
                   help="OFF by default. At 0.005 the first run's sigma grew "
                        "0.50 -> 4.82: the critic fit the returns almost "
                        "exactly (value loss 0.0024), advantages collapsed to "
                        "~0, and with no surrogate signal left the entropy "
                        "bonus was the only force on the policy. sigma=4.8 "
                        "saturates tanh, so the policy became bang-bang random "
                        "and could not refine.")
    p.add_argument("--init-std", type=float, default=0.5)
    p.add_argument("--max-std", type=float, default=1.0,
                   help="hard ceiling on the policy's sigma. Independent of "
                        "--entropy-coef: even with the bonus off, a run that "
                        "drives sigma past ~1 has saturated the tanh squash "
                        "and is exploring with noise rather than with a "
                        "policy. Set high to disable.")
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--patience", type=int, default=2,
                   help="consecutive episodes meeting a segment's criterion "
                        "before the curriculum advances. Lower than the DiffRL "
                        "trainers' value because an episode here spans 1500+ "
                        "envs rather than 20, so one tick is already a much "
                        "larger sample and needs less repetition to trust.")
    p.add_argument("--viewer", action="store_true",
                   help="open a passive MuJoCo viewer replaying env 0 after "
                        "each --viewer-every episodes. Replay runs in real "
                        "time, so it costs roughly n_total*frame_dt seconds "
                        "per replay; keep --viewer-every high for long runs.")
    p.add_argument("--viewer-every", type=int, default=5,
                   help="replay every Nth completed episode")
    p.add_argument("--resume", type=str, default=None, metavar="CKPT",
                   help="continue from a model_N.pt. Restores actor, critic, "
                        "optimizer and iteration from the .pt, and the "
                        "curriculum phase plus ObsRMS from the _env.pkl beside "
                        "it. --iters is then how many MORE iterations to run.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="ppo_runs")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    if args.num_envs == 256:
        raise SystemExit(
            "--num-envs 256 is broken: MJX returns NaN for every env from the "
            "first step at exactly this batch size, deterministically, with "
            "zero actions and no torch involved. 240 and 272 are both fine. "
            "Pass --num-envs 2048 (the default) or any other size.")

    cfg = load_state_config(args.config)
    TR = cfg['training']
    POL = envs.policy_cfg(cfg)
    names = envs.primitive_names(cfg)
    fbs = envs.finger_biases(cfg)
    seq = envs.prim_seq(cfg, args.task)
    obs_dim = cfg['robot']['proprio_dim']

    print("=" * 74)
    print(f"PPO BASELINE (rsl_rl) — {args.task}, state observation")
    print(f"  envs={args.num_envs}  steps/env={args.num_steps_per_env}  "
          f"batch/update={args.num_envs * args.num_steps_per_env:,}")
    print(f"  curriculum={'off' if args.no_curriculum else 'on'}")
    print("=" * 74)

    env_obj = make_state_env(args.task, cfg, make_state_forward(seq, names),
                             batch_size=args.num_envs,
                             substeps=TR['substeps'])
    t = env_obj.build()

    obs_rms = StateObsRMS(obs_dim, var_floor=float(cfg['obs']['var_floor']))
    venv = MJXStateVecEnv(
        t, obs_rms, N_JOINTS, VEL_LIMITS, FINGER_OPEN,
        [names.index(pr) for pr in seq], len(names),
        args.patience, TR['curriculum']['pass_frac'],
        device=args.device, curriculum=not args.no_curriculum, seed=args.seed)

    actor_cfg = dict(
        class_name=TrunkHeadActor,
        state_obs_dim=obs_dim,
        n_prims=venv.n_prims,
        trunk_hidden=POL['trunk_hidden'],
        head_widths=[envs.head_widths(cfg)[pr] for pr in names],
        head_layers=POL.get('head_layers', 1),
        finger_biases=[fbs[pr] for pr in names],
        n_joints=N_JOINTS,
        distribution_cfg=dict(class_name="GaussianDistribution",
                              init_std=args.init_std, std_type="log",
                              std_range=(1e-3, args.max_std)),
    )

    train_cfg = {
        "algorithm": dict(
            class_name="PPO",
            num_learning_epochs=5, num_mini_batches=4,
            clip_param=args.clip, gamma=args.gamma, lam=args.lam,
            value_loss_coef=1.0, entropy_coef=args.entropy_coef,
            learning_rate=args.lr, max_grad_norm=1.0,
            use_clipped_value_loss=True, schedule="adaptive",
            desired_kl=0.01, normalize_advantage_per_mini_batch=False,
        ),
        "actor": actor_cfg,
        "critic": dict(class_name="MLPModel",
                       hidden_dims=[POL['trunk_hidden']] * 2,
                       activation="elu", obs_normalization=False),
        # The actor reads ONLY "policy", so its input is exactly the DiffRL
        # policy's. The critic also reads "time" — see rsl_vecenv._obs_td.
        "obs_groups": {"actor": ["policy"], "critic": ["policy", "time"]},
        "num_steps_per_env": args.num_steps_per_env,
        "save_interval": 50,
        "empirical_normalization": False,
        "logger": "tensorboard",
    }

    if args.viewer:
        from ppo_baseline.viewer import TrajectoryViewer
        venv.viewer = TrajectoryViewer(t, env_obj.frame_dt)
        venv.viewer_every = max(1, args.viewer_every)
        print(f"  viewer on: replaying env 0 every {venv.viewer_every} episodes")

    log_dir = os.path.join(args.out, f"{args.task}_{time.strftime('%m%d_%H%M')}")
    os.makedirs(log_dir, exist_ok=True)
    runner = ResumableRunner(venv, train_cfg, log_dir=log_dir,
                             device=args.device)
    if args.resume:
        print(f"\n  resuming from {args.resume}")
        runner.load(args.resume)
    print(f"\n  logging to {log_dir}\n")
    runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=False)


if __name__ == "__main__":
    main()
