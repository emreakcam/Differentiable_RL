"""
Joint State-Primitive Training — robosuite tasks, shared trunk + primitive heads.
=================================================================================
The state-observation counterpart to train_primitives_vison.py.  Same tasks,
same rewards, same shared-trunk/per-primitive-head policy, same curriculum and
the same checkpoint discipline.  The one difference is what the policy sees:
the exact coordinates of the objects that matter, instead of DINOv3 features of
a 64x64 camera image.

That removes the whole render pipeline, and with it the reason the vision
trainer had two phases.  There, an iteration was:

    phase 1   render each chunk → DINOv3 → advance physics under those features
    phase 2   replay the episode against the cached features, with gradients

Nothing here lives outside JAX, so an iteration is a rebatch and one backward
pass.  Expect it to be several times faster per iteration and to use far less
memory — there is no feature cache of shape (B, chunks, 1, 4, 4, 768) and no
torch context sharing the GPU.

Shared across tasks:
  - one trunk
  - one head per *primitive* — every task's `reach` drives the same `reach` head
  - one ObsRMS (the observation layout is task-agnostic: the same proprio prefix
    and the same fixed object slots everywhere)

Per task:
  - MjModel / rebatch / rollout — a different XML means different qpos shapes,
    so physics cannot be batched across tasks.
  - curriculum phase, active_steps, and success criterion.

One combined_loss summing every task's mean loss → a single value_and_grad →
one backward pass touching the shared trunk and all primitive heads at once.
The trunk gets one global-norm-clipped Adam; each primitive head gets its own
optimizer and its own clip.

    python train_primitives_state.py --tasks all --batch 20 --grad-groups 11
    python train_primitives_state.py --tasks drawer_open,window_open --batch 20
    python train_primitives_state.py --tasks cube_stacking          # opens the viewer
    python train_primitives_state.py --tasks drawer_open --no-viewer
    python train_primitives_state.py --config ../configs/my_state_tasks.yaml

Viewer: a single-task run opens a passive MuJoCo viewer replaying env 0, since
there is exactly one trajectory to show. Multi-task runs never do — suppress it
with --no-viewer.

Checkpoints: params, ObsRMS and a per-task reward plot are written every
--save-every iterations (50 by default) to --out-prefix, and again on exit —
including Ctrl+C and a closed viewer. Each save overwrites the same paths, so
what is on disk is always the newest good state. The one exception is a NaN
loss, which leaves the previous checkpoint alone rather than overwriting it with
poisoned weights.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# XLA memory/compile settings — MUST precede `import jax`, since the backend
# reads them once at initialisation. Anything already set in the shell wins.
from helpers.xla_env import apply_xla_defaults
apply_xla_defaults()

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")

# Persistent compilation cache.  Compiling every task's BPTT graph costs GBs of
# HOST RAM (not VRAM) and minutes of CPU; caching executables to disk makes
# every run after the first a cache hit.
#
# Pair this with --grad-groups equal to the task count: each compiled module
# then holds exactly one task, so its cache entry stays valid no matter which
# subset of tasks a later run selects.  With fewer groups the modules are
# group-specific and changing --tasks invalidates them.
#
# A separate cache directory from the vision trainer's: the graphs share no
# structure, and keeping them apart means clearing one never costs the other its
# warm cache.
_JIT_CACHE = os.environ.get(
    "MJX_STATE_JIT_CACHE", os.path.expanduser("~/.cache/mjx_diffrl_jit_state"))
os.makedirs(_JIT_CACHE, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _JIT_CACHE)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

import argparse, time, pickle, gc, random
import numpy as np
import jax.numpy as jnp
import optax
import mujoco
import matplotlib
matplotlib.use("Agg")          # headless: the reward plot is written, never shown
import matplotlib.pyplot as plt

from helpers.mjx_utils import patch_solver
from helpers.state_obs import StateObsRMS
from models.state_policy import (
    init_state_policy, make_state_forward, STATE_SHARED_KEYS,
)
import envs
from envs.state_registry import load_state_config, make_state_env

patch_solver()


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG + ARGS
# ═══════════════════════════════════════════════════════════════════════════
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config", type=str, default=None,
                  help="task config YAML (default: configs/tasks_state.yaml, "
                       "or $MJX_STATE_TASK_CONFIG)")
_cfg_args, _ = _pre.parse_known_args()

cfg = load_state_config(_cfg_args.config)
TR = cfg['training']
POL = envs.policy_cfg(cfg)
PRIM_NAMES = envs.primitive_names(cfg)
FINGER_BIASES = envs.finger_biases(cfg)

PATIENCE = TR['curriculum']['patience']
PASS_FRAC = TR['curriculum']['pass_frac']
SUCCESS_STREAK = TR['termination']['success_streak']
SUCCESS_RATE = TR['termination']['success_rate']

_lr_prim_default = ','.join(f"{k}={v:g}" for k, v in TR['lr_prim'].items())

parser = argparse.ArgumentParser(parents=[_pre])
parser.add_argument("--tasks", type=str, default=','.join(TR['default_tasks']),
                    help="comma-separated task names, or 'all'")
parser.add_argument("--batch", type=int, default=TR['batch'], help="envs per task")
parser.add_argument("--task-batch", type=str, default="",
                    help="per-task override, e.g. 'cube_stacking=10,dial_turn=8'")
parser.add_argument("--substeps", type=int, default=TR['substeps'])
parser.add_argument("--iters", type=int, default=TR['iters'])
parser.add_argument("--lr", type=float, default=TR['lr'], help="shared trunk")
parser.add_argument("--lr-head", type=float, default=TR['lr_head'],
                    help="per-primitive heads")
parser.add_argument("--lr-prim", type=str, default=_lr_prim_default,
                    help="per-primitive LR overrides, e.g. 'align=2e-5'. "
                         "Anything unlisted uses --lr-head. Defaults come from "
                         "training.lr_prim in the config.")
parser.add_argument("--gamma", type=float, default=TR['gamma'])
parser.add_argument("--grad-groups", type=int, default=TR['grad_groups'],
                    help="split the backward pass over N task groups, summing "
                         "the gradients before a single optimizer step. Peak "
                         "VRAM becomes the largest group instead of the sum "
                         "over all tasks; the resulting update is unchanged. "
                         "1 = one fused backward.")
parser.add_argument("--grad-clip", type=float, default=TR['grad_clip'],
                    help="clip for the shared trunk only; each head has its own "
                         "limit (training.grad_clip_head)")
parser.add_argument("--grad-clip-prim", type=str, default="",
                    help="per-primitive clip overrides, e.g. "
                         "'reach=15000,grasp=12000'. Anything unlisted keeps "
                         "its training.grad_clip_head value.")
parser.add_argument("--balance-tasks", type=float, default=0.0, metavar="BETA",
                    help="equalise how hard each task pushes the shared trunk. "
                         "Each task's gradient is rescaled to a common size "
                         "before the sum, using a running average of its own "
                         "norm with decay BETA (0.99 is a good start). Under a "
                         "plain sum a task's share follows its reward, so a "
                         "FAILING task contributes least exactly when it needs "
                         "most: drawer_close drove 0.4%% of the objective "
                         "against door_open's 20.5%%, a 51x gap. 0 (default) "
                         "keeps the plain sum. Needs --grad-groups >= n_tasks, "
                         "since a group holding several tasks cannot be "
                         "attributed to one of them.")
parser.add_argument("--balance-floor", type=float, default=1e-3, metavar="F",
                    help="with --balance-tasks, never scale a task's gradient "
                         "up by more than 1/F of the common size. A task whose "
                         "segments are all frozen by the curriculum has a "
                         "genuinely tiny gradient, and normalising that to full "
                         "size would amplify its noise rather than its signal.")
parser.add_argument("--task-minibatch", type=int, default=0, metavar="N",
                    help="update every N tasks instead of once per iteration. "
                         "Tasks are reshuffled each iteration and split into "
                         "chunks of ~N; each chunk is rolled out fresh, "
                         "back-propagated and stepped before the next starts. "
                         "Unset (0) keeps the current single update per "
                         "iteration. Chunks are built from --grad-groups "
                         "groups, so with one task per group this is exactly N "
                         "tasks; peak memory is unchanged either way.")
parser.add_argument("--no-mb-lr-compensate", dest="mb_lr_compensate",
                    action="store_false", default=True,
                    help="with --task-minibatch, do NOT divide each update by "
                         "the number of chunks that parameter appears in. "
                         "Compensation is on by default: `reach` is in every "
                         "task and so every chunk, `turn` in one, so without it "
                         "the shared stack and shared heads would take N steps "
                         "per iteration against a terminal head's one — an "
                         "Nx effective LR difference nobody chose.")
parser.add_argument("--trunk-hidden", type=int, default=POL['trunk_hidden'],
                    help="shared trunk width (policy.trunk_hidden)")
parser.add_argument("--head-layers", type=int,
                    default=POL.get('head_layers', 1),
                    help="hidden layers per head. Widths are set by "
                         "policy.head_hidden; measured head rank never passed "
                         "35%% of width, and width did not predict rank among "
                         "heads of equal workload, so depth is the axis that "
                         "adds usable capacity.")
parser.add_argument("--log-every", type=int, default=TR['log_every'])
parser.add_argument("--warmup-iters", type=int, default=TR['warmup_iters'],
                    help="iterations that only populate ObsRMS (no weight "
                         "update), so training never runs on a raw observation")
parser.add_argument("--no-viewer", action="store_true",
                    help="suppress the MuJoCo viewer in single-task runs "
                         "(multi-task runs never open one)")
parser.add_argument("--save-every", type=int, default=TR['save_every'],
                    help="checkpoint params, ObsRMS and the reward plot every "
                         "N iterations. 0 disables periodic saving; the final "
                         "save and the Ctrl+C save still happen.")
parser.add_argument("--normalise", "--normalize", dest="normalise",
                    action="store_true", default=TR['normalise_loss'],
                    help="divide each task's loss by its active_steps before "
                         "summing, so a task at phase 0 contributes as much to "
                         "the objective as one at its final phase. Off by "
                         "default: it changes what is optimised.")
parser.add_argument("--no-normalise", "--no-normalize", dest="normalise",
                    action="store_false",
                    help="force the plain unweighted sum (the default)")
parser.add_argument("--max-nan-skips", type=int, default=TR['max_nan_skips'],
                    help="stop after this many CONSECUTIVE non-finite "
                         "iterations; isolated ones are always skipped and "
                         "training continues")
parser.add_argument("--resume", type=str, default=None, metavar="PREFIX",
                    help="continue a run: loads PREFIX_params.pkl, "
                         "PREFIX_obs_rms.pkl and PREFIX_opt.pkl (optimizer "
                         "moments, curriculum phases and reward history). "
                         "Defaults to --out-prefix when given no value.")
parser.add_argument("--out-prefix", type=str, default="state_primitives")
args = parser.parse_args()

TASK_NAMES = (envs.task_names(cfg)
              if args.tasks.strip() == 'all'
              else [t.strip() for t in args.tasks.split(',') if t.strip()])
for t in TASK_NAMES:
    if t not in cfg['tasks']:
        raise SystemExit(
            f"unknown task {t!r}; choose from {sorted(cfg['tasks'])}")

task_batch = {t: args.batch for t in TASK_NAMES}
for item in filter(None, args.task_batch.split(',')):
    k, v = item.split('=')
    if k.strip() in task_batch:
        task_batch[k.strip()] = int(v)

OBS_DIM = cfg['robot']['proprio_dim']
VAR_FLOOR = float(cfg['obs'].get('var_floor', 1e-4))

# ── resume ────────────────────────────────────────────────────────────────
# Read the pickles up front so a bad path fails before the GPU work, but apply
# each piece where its variable is created: params after the policy is built,
# optimizer moments after the optimizers exist, curriculum after the tasks do.
RESUME, START_ITER = {}, 0
if args.resume is not None:
    _pfx = args.resume or args.out_prefix
    for _key, _suffix, _required in (('params', '_params.pkl', True),
                                     ('obs_rms', '_obs_rms.pkl', True),
                                     ('opt', '_opt.pkl', False)):
        _path = f"{_pfx}{_suffix}"
        if os.path.exists(_path):
            with open(_path, 'rb') as _f:
                RESUME[_key] = pickle.load(_f)
        elif _required:
            raise SystemExit(f"--resume: {_path} not found")
        else:
            print(f"! {_path} missing — optimizer moments and curriculum will "
                  f"restart (weights and ObsRMS still resume)")
    START_ITER = int(RESUME.get('opt', {}).get('iteration', 0)) + 1


# ── console history ────────────────────────────────────────────────────────
# Everything printed also goes to {out_prefix}_train.log. Terminal scrollback is
# finite and a long run does not fit in it; without this the only record of a run
# is the reward plot, and the per-head gradient tables are gone. Appended, not
# truncated, so a restart keeps the earlier runs.
class _Tee:
    def __init__(self, path):
        self.file = open(path, 'a', buffering=1)      # line-buffered

    def write(self, s):
        sys.__stdout__.write(s)
        self.file.write(s)
        return len(s)

    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()

    def isatty(self):
        return sys.__stdout__.isatty()


LOG_PATH = f"{args.out_prefix}_train.log"
sys.stdout = _Tee(LOG_PATH)
print(f"\n{'#' * 70}\n# run started {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
      f"# {' '.join(sys.argv)}\n{'#' * 70}")

# The viewer replays ONE env's qpos trajectory, so it only makes sense for a
# single task — with several running there is no one trajectory to show, and the
# sleep it needs to play back in real time would stall every other task.
USE_VIEWER = len(TASK_NAMES) == 1 and not args.no_viewer

print("=" * 70)
print("JOINT STATE-PRIMITIVE TRAINING — robosuite (no vision)")
print(f"  Tasks: {', '.join(TASK_NAMES)}")
print(f"  Batch: {task_batch}")
print(f"  Obs dim: {OBS_DIM} (shared trunk) — "
      f"{cfg['robot']['proprio_prefix_dim']} proprio + "
      f"{cfg['obs']['n_slots']}x{cfg['obs']['slot_dim']} object slots")
print(f"  Viewer: {'on' if USE_VIEWER else 'off'}"
      + ("" if USE_VIEWER else
         "  (multi-task run)" if len(TASK_NAMES) > 1 else "  (--no-viewer)"))
print(f"  Saving to {args.out_prefix}_*  every "
      + (f"{args.save_every} iters" if args.save_every > 0 else "run end only"))
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# SHARED POLICY
# ═══════════════════════════════════════════════════════════════════════════
print("\n[1] Shared state policy...")
policy_params = init_state_policy(
    jax.random.PRNGKey(42), PRIM_NAMES, FINGER_BIASES, obs_dim=OBS_DIM,
    trunk_hidden=args.trunk_hidden, head_widths=envs.head_widths(cfg),
    head_layers=args.head_layers)

if 'params' in RESUME:
    _want = jax.tree.map(lambda x: x.shape, policy_params)
    _got = jax.tree.map(lambda x: np.shape(x), RESUME['params'])
    if _want != _got:
        raise SystemExit(
            "--resume: checkpoint shapes do not match the current config. "
            "obs.n_slots / robot.proprio_dim / policy.trunk_hidden / "
            "head_hidden or the primitive vocabulary must have changed since "
            "it was written.")
    policy_params = jax.tree.map(jnp.asarray, RESUME['params'])
    print(f"  resumed weights from {args.resume or args.out_prefix}_params.pkl")


# ═══════════════════════════════════════════════════════════════════════════
# BUILD TASKS — each environment supplies its own bundle
# ═══════════════════════════════════════════════════════════════════════════
tasks = {}
for name in TASK_NAMES:
    prim_seq = envs.prim_seq(cfg, name)
    print(f"\n[2] Building task '{name}' "
          f"({cfg['tasks'][name]['env']} env, state obs) ...")
    env = make_state_env(
        name, cfg, make_state_forward(prim_seq, PRIM_NAMES),
        batch_size=task_batch[name], substeps=args.substeps, gamma=args.gamma)
    tasks[name] = env.build()
    tasks[name]['prim_seq'] = prim_seq
    print(f"    segments {tasks[name]['seg_steps']} → primitives {prim_seq}")
    print(f"    slots: " + ", ".join(f"{k}:{v}" for k, v in
                                     tasks[name]['slot_specs']))


# ═══════════════════════════════════════════════════════════════════════════
# COMBINED LOSS — one backward pass per task group
# ═══════════════════════════════════════════════════════════════════════════
def make_group_loss(group_names):
    """Σ over this group's tasks of that task's mean episode loss.

    With --normalise each task's loss is divided by its active_steps first.
    A task's episode reward grows with how many segments the curriculum has
    opened, so under a plain sum a task at phase 0 contributes a small fraction
    of the objective exactly when it needs the most help. Dividing by
    active_steps makes the contribution per-step rather than per-episode.

    Only the SUMMED value is scaled — `auxes` keeps each task's raw loss, so the
    per-task rew= in the log stays on the same scale as every previous run.
    """
    def group_loss(policy_params, obs_mean, obs_std, states, actives,
                   task_params):
        total = 0.0
        auxes = {}
        for name in group_names:
            t = tasks[name]
            call = (policy_params, states[name], obs_mean, obs_std,
                    actives[name])
            if t['has_task_params']:
                call = call + (task_params[name],)
            loss, aux = t['batch_episode_loss'](*call)
            auxes[name] = (loss, aux)          # raw, for reporting
            if args.normalise:
                # actives is traced, so this follows the curriculum without
                # forcing a recompile when a phase advances.
                loss = loss / jnp.maximum(actives[name].astype(jnp.float32), 1.0)
            total = total + loss
        return total, auxes
    return jax.jit(jax.value_and_grad(group_loss, has_aux=True))


def partition_tasks(names, n_groups):
    """Greedy longest-processing-time split, balancing batch x steps.

    Peak VRAM is the largest group's BPTT graph, not the sum over all tasks, so
    the split only has to make the heaviest group small enough.  Cost uses the
    task's FULL episode length rather than its current active_steps: the
    curriculum grows episodes over training, and a split sized on early
    active_steps would fit at iteration 0 and OOM once the phases mature.
    """
    cost = {n: tasks[n]['batch_size'] * sum(tasks[n]['seg_steps']) for n in names}
    groups = [[] for _ in range(n_groups)]
    load = [0] * n_groups
    for n in sorted(names, key=lambda k: -cost[k]):
        i = load.index(min(load))
        groups[i].append(n)
        load[i] += cost[n]
    keep = [(g, l) for g, l in zip(groups, load) if g]
    return [g for g, _ in keep], [l for _, l in keep]


GROUPS, GROUP_LOAD = partition_tasks(TASK_NAMES, min(args.grad_groups,
                                                     len(TASK_NAMES)))
group_fns = [make_group_loss(g) for g in GROUPS]
TASK_INDEX = {n: i for i, n in enumerate(TASK_NAMES)}


# ── per-task gradient balancing ──────────────────────────────────────────────
# Running mean of each group's gradient norm, restored across resumes so the
# scales do not restart from nothing mid-run. Only meaningful when a group
# holds one task; with fewer groups the norm mixes tasks and cannot be
# attributed, so the flag is refused rather than quietly doing the wrong thing.
BALANCE = args.balance_tasks > 0.0
if BALANCE and len(GROUPS) < len(TASK_NAMES):
    raise SystemExit(
        f"--balance-tasks needs one task per group: {len(TASK_NAMES)} tasks in "
        f"{len(GROUPS)} groups. Re-run with --grad-groups {len(TASK_NAMES)}.")
gscale = list(RESUME.get('opt', {}).get('gscale', [0.0] * len(GROUPS)))
if len(gscale) != len(GROUPS):          # task list changed across a resume
    gscale = [0.0] * len(GROUPS)


def balance_grad(gi, g):
    """Rescale group `gi`'s gradient toward the mean scale across all groups.

    The running average is updated on the RAW norm, so the estimate tracks what
    the task actually produces rather than what it was last scaled to. The
    target is the mean of every task's scale, which keeps the summed gradient on
    roughly the same magnitude as the unbalanced sum — the trunk's clip limit
    and learning rate stay meaningful instead of needing a second retune.
    """
    raw = float(optax.global_norm(g))
    # A non-finite norm must NEVER enter the running scale. The EMA has no way
    # back out of a NaN, and this runs BEFORE the caller's non-finite guard, so
    # one transient NaN would poison gscale permanently and multiply every
    # task's gradient by NaN from then on — a dead run from one bad batch.
    # Hand the gradient back untouched and let the guard skip the update.
    if not np.isfinite(raw):
        return g
    b = args.balance_tasks
    gscale[gi] = raw if gscale[gi] == 0.0 else b * gscale[gi] + (1.0 - b) * raw
    live = [s for s in gscale if np.isfinite(s) and s > 0.0]
    if not live:
        return g
    target = sum(live) / len(live)
    # Clamp the amplification: a task whose segments are all frozen has a
    # genuinely tiny gradient, and scaling that to full size amplifies noise.
    scale = min(target / (gscale[gi] + 1e-12), 1.0 / args.balance_floor)
    if not np.isfinite(scale) or scale <= 0.0:
        return g
    return jax.tree.map(lambda x: x * scale, g)


def make_chunks(iteration):
    """Group indices per optimizer step, reshuffled every iteration.

    Returns one chunk holding every group unless --task-minibatch is set, so
    the default path is the previous behaviour exactly: one rollout of
    everything, gradients summed, a single step.

    Chunks are built from GROUPS rather than from task names because each
    group has its own jitted value_and_grad closed over its task list —
    chunking arbitrary task subsets would compile a new one per subset per
    iteration. Regrouping existing groups reuses those compilations, and with
    one task per group (--grad-groups >= n_tasks) a chunk is exactly N tasks.
    """
    idx = list(range(len(GROUPS)))
    if args.task_minibatch <= 0:
        return [idx]
    rng = random.Random(iteration)
    rng.shuffle(idx)
    chunks, cur, n = [], [], 0
    for gi in idx:
        cur.append(gi)
        n += len(GROUPS[gi])
        if n >= args.task_minibatch:
            chunks.append(cur)
            cur, n = [], 0
    if cur:
        # A leftover tail becomes its own chunk when it is at least half a
        # chunk, otherwise it joins the previous one: a chunk of one or two
        # tasks would step the shared stack on a far noisier gradient than
        # every other step in the iteration.
        tail = sum(len(GROUPS[g]) for g in cur)
        if chunks and tail * 2 < args.task_minibatch:
            chunks[-1].extend(cur)
        else:
            chunks.append(cur)
    return chunks


if len(GROUPS) > 1:
    print(f"\n  Gradient accumulation over {len(GROUPS)} groups "
          f"(peak group = {max(GROUP_LOAD):,} env-steps, "
          f"total = {sum(GROUP_LOAD):,}):")
    for g, l in zip(GROUPS, GROUP_LOAD):
        print(f"    {l:>7,}  {', '.join(g)}")
    print("    grads are summed across groups, then clipped and stepped once — "
          "identical to one fused backward.")

if BALANCE:
    print(f"\n  Task balancing ON (beta={args.balance_tasks}, "
          f"floor={args.balance_floor:g}): each task's gradient is rescaled to "
          f"the mean scale before summing, so a failing task is not drowned out "
          f"by a solved one. Σrew in the log is no longer the objective.")


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZERS — shared trunk vs per-primitive heads
# ═══════════════════════════════════════════════════════════════════════════
shared_optimizer = optax.chain(
    optax.clip_by_global_norm(args.grad_clip), optax.adam(args.lr))
shared_opt_state = shared_optimizer.init(
    {k: policy_params[k] for k in STATE_SHARED_KEYS})

prim_lr = {p: args.lr_head for p in PRIM_NAMES}
for _item in filter(None, args.lr_prim.split(',')):
    _k, _v = _item.split('=')
    _k = _k.strip()
    if _k not in prim_lr:
        raise SystemExit(f"--lr-prim: unknown primitive {_k!r}; "
                         f"choose from {PRIM_NAMES}")
    prim_lr[_k] = float(_v)

prim_clip = envs.head_clips(cfg)
for _item in filter(None, args.grad_clip_prim.split(',')):
    _k, _v = _item.split('=')
    _k = _k.strip()
    if _k not in prim_clip:
        raise SystemExit(f"--grad-clip-prim: unknown primitive {_k!r}; "
                         f"choose from {PRIM_NAMES}")
    prim_clip[_k] = float(_v)

head_optimizers = {p: optax.adam(prim_lr[p]) for p in PRIM_NAMES}
head_opt_states = {p: head_optimizers[p].init(policy_params[p])
                   for p in PRIM_NAMES}

if 'opt' in RESUME:
    # Adam's moment estimates. Without these a resume is a warm restart: the
    # moments rebuild from zero over ~1/(1-beta2) steps and the first updates
    # are mis-scaled, which shows up as a transient dip right after loading.
    shared_opt_state = jax.tree.map(jnp.asarray, RESUME['opt']['shared_opt_state'])
    head_opt_states = jax.tree.map(jnp.asarray, RESUME['opt']['head_opt_states'])

# Shared across tasks: the observation layout is task-agnostic. The variance
# floor matters more here than in the vision pipeline — see helpers/state_obs.py.
obs_rms = StateObsRMS(OBS_DIM, var_floor=VAR_FLOOR)
if 'obs_rms' in RESUME:
    obs_rms.mean = RESUME['obs_rms']['mean']
    obs_rms.var = RESUME['obs_rms']['var']
    obs_rms.count = RESUME['obs_rms']['count']
    print(f"  resumed ObsRMS (count={obs_rms.count:.0f})")

# Only primitives exercised by the selected tasks receive updates.
ACTIVE_PRIMS = sorted({p for n in TASK_NAMES for p in tasks[n]['prim_seq']},
                      key=PRIM_NAMES.index)

# How many SEGMENTS across the selected tasks drive each head — not how many
# tasks, because container runs `reach` three times in one episode. This is the
# quantity a head's raw gradient magnitude should scale with, so printing it
# next to the measured norm shows whether that relationship actually holds.
HEAD_SEGMENTS = {p: sum(tasks[n]['prim_seq'].count(p) for n in TASK_NAMES)
                 for p in ACTIVE_PRIMS}
print(f"\n  Active primitives: {', '.join(ACTIVE_PRIMS)}")
_odd = {p: prim_lr[p] for p in ACTIVE_PRIMS if prim_lr[p] != args.lr_head}
print(f"  Head LR: {args.lr_head:.1e}"
      + (f"   overrides: " + "  ".join(f"{k}={v:.1e}" for k, v in _odd.items())
         if _odd else ""))


def rebatch(t, key):
    """→ (states, task_params|None), hiding the 6-arg/7-arg task difference."""
    if t['has_task_params']:
        return t['rebatch_jit'](key, t['batch_size'])
    return t['rebatch_jit'](key, t['batch_size']), None


# ═══════════════════════════════════════════════════════════════════════════
# CHECKPOINTING — params, ObsRMS and the reward plot
# ═══════════════════════════════════════════════════════════════════════════
# Written periodically rather than only at the end, because a run that is killed
# at iteration 900/1000 should not lose 900 iterations of work. Every save
# overwrites the same paths, so the newest checkpoint is always the one on disk.
hist = {n: [] for n in TASK_NAMES}          # per task: (iteration, reward)
if 'opt' in RESUME:
    for _n, _v in RESUME['opt'].get('hist', {}).items():
        if _n in hist:
            hist[_n] = list(_v)          # keeps the plot continuous across runs

# (iteration, total |g|, trunk |g|, {prim: |g|}, {prim: clip scale}).
# The per-head numbers were previously printed and then thrown away, so the
# clipping imbalance the log line describes could only be read one iteration at
# a time. Keeping them makes the gradient curves plottable over a whole run.
ghist = list(RESUME.get('opt', {}).get('ghist', []))

# Wall-clock seconds accumulated by EARLIER runs of this prefix. A resumed run
# restarts time.time() from zero, so reporting only this process's elapsed time
# would understate the cost of a job that was checkpointed and continued — and
# wall-clock is one of the numbers the experiments are meant to compare.
WALL_OFFSET = float(RESUME.get('opt', {}).get('wall', 0.0))


def wall_clock():
    """Total training seconds for this prefix, across resumes."""
    return WALL_OFFSET + (time.time() - train_start)


def fmt_wall(sec):
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h {m:02d}m {s:02d}s" if h else f"{m:d}m {s:02d}s"


def total_return():
    """Σ reward over tasks per iteration — the objective actually optimised.

    Restricted to iterations every task logged: a task added mid-run, or a
    resume whose history is shorter for some tasks, would otherwise put a step
    in the total that is bookkeeping rather than learning.
    """
    per = {n: dict(hist[n]) for n in TASK_NAMES}
    if not per or any(not d for d in per.values()):
        return [], []
    its = sorted(set.intersection(*(set(d) for d in per.values())))
    return its, [sum(per[n][i] for n in TASK_NAMES) for i in its]


def _smooth(y, k=9):
    """Centred moving average, for a trend line over the per-iteration noise."""
    if len(y) < k:
        return None
    pad = k // 2
    padded = np.r_[[y[0]] * pad, y, [y[-1]] * pad]
    return np.convolve(padded, np.ones(k) / k, mode='valid')


def save_learning_plot(path):
    """Total return across tasks, then reward-vs-iteration per task.

    Wall-clock is stamped in the top-right corner: a learning curve read on its
    own says nothing about what the iterations cost, and iteration count is not
    comparable across the vision and state-conditioned frameworks.
    """
    rows = len(TASK_NAMES) + 1
    fig, axes = plt.subplots(rows, 1, figsize=(10, 2.6 * rows), squeeze=False)

    ax = axes[0, 0]
    its, tot = total_return()
    if its:
        ax.plot(its, tot, linewidth=1.0, color='steelblue', alpha=0.45,
                label='Σ tasks')
        sm = _smooth(tot)
        if sm is not None:
            ax.plot(its, sm, linewidth=2.0, color='navy',
                    label='moving avg (9)')
        ax.axhline(0.0, color='grey', linewidth=0.8, linestyle=':')
        best = int(np.argmax(tot))
        ax.plot(its[best], tot[best], 'o', color='crimson', markersize=5)
        # The best point is usually the latest one, i.e. hard against the right
        # edge — label leftwards there so the text stays inside the axes.
        late = its[best] > its[0] + 0.75 * (its[-1] - its[0] + 1e-9)
        ax.annotate(f"best {tot[best]:.0f} @ {its[best]}",
                    (its[best], tot[best]), textcoords="offset points",
                    xytext=(-8, 6) if late else (8, 6),
                    ha='right' if late else 'left',
                    fontsize=8, color='crimson')
        ax.margins(y=0.15)
        ax.legend(fontsize=8, loc='upper left')
    ax.set_ylabel("Σ reward")
    ax.set_title(f"TOTAL RETURN — {len(TASK_NAMES)} tasks", fontsize=10,
                 fontweight='bold')
    ax.grid(True, alpha=0.3)

    for ax, name in zip(axes[1:, 0], TASK_NAMES):
        pts = hist[name]
        if pts:
            it, rew = zip(*pts)
            ax.plot(it, rew, linewidth=1.6, color='green')
        ax.set_ylabel("Reward")
        ax.set_title(f"{name}   ({' → '.join(tasks[name]['prim_seq'])})",
                     fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Iteration")
    fig.suptitle(f"State primitives — {len(TASK_NAMES)} tasks", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    # After tight_layout, so the corner stamp is placed against the final
    # figure edge rather than a pre-layout one.
    its_all = sorted({i for n in TASK_NAMES for i, _ in hist[n]})
    _wall = wall_clock()
    _per = f"{_wall / len(its_all):.1f}s/log" if its_all else "—"
    fig.text(0.995, 0.995,
             f"wall-clock {fmt_wall(_wall)}   ({_per})",
             ha='right', va='top', fontsize=9, fontweight='bold',
             color='#333333',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f2f2f2',
                       edgecolor='#999999', linewidth=0.6))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_gradient_plot(path):
    """Gradient norms over training — total, trunk, and per-primitive head.

    Separate from the learning curves: the two are read for different reasons
    (is it learning vs. is the optimisation healthy), they want different y
    scales, and stacking both in one figure made a file tall enough that
    neither was legible.
    """
    if not ghist:
        return
    its = [e[0] for e in ghist]
    total = np.array([e[1] for e in ghist], dtype=float)
    trunk = np.array([e[2] for e in ghist], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9.0), sharex=True)

    # ── total + trunk, log scale ──
    # Log scale because a NaN-adjacent spike is one or two orders of magnitude
    # above the run's normal band; on a linear axis it flattens everything else.
    ax = axes[0]
    ax.semilogy(its, np.maximum(total, 1e-8), linewidth=1.0, color='steelblue',
                alpha=0.5, label='total |g|')
    sm = _smooth(total)
    if sm is not None:
        ax.semilogy(its, np.maximum(sm, 1e-8), linewidth=2.0, color='navy',
                    label='total, moving avg (9)')
    ax.semilogy(its, np.maximum(trunk, 1e-8), linewidth=1.2, color='darkorange',
                alpha=0.8, label='trunk (shared) |g|')
    ax.axhline(args.grad_clip, color='crimson', linewidth=0.9, linestyle='--',
               label=f'trunk clip {args.grad_clip:g}')
    ax.set_ylabel("|g|")
    ax.set_title("GRADIENT NORM — total and shared trunk", fontsize=10,
                 fontweight='bold')
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3, which='both')

    # ── per-head norms against each head's own clip limit ──
    ax = axes[1]
    cmap = plt.get_cmap('tab10')
    for i, p in enumerate(ACTIVE_PRIMS):
        col = cmap(i % 10)
        gs = np.array([e[3].get(p, np.nan) for e in ghist], dtype=float)
        if np.all(np.isnan(gs)):
            continue
        ax.semilogy(its, np.maximum(gs, 1e-8), linewidth=1.3, color=col,
                    label=f"{p} (x{HEAD_SEGMENTS[p]})")
        ax.axhline(prim_clip[p], color=col, linewidth=0.7, linestyle=':',
                   alpha=0.7)
    ax.set_ylabel("head |g|")
    ax.set_title("PER-PRIMITIVE HEAD GRADIENT  (dotted = that head's clip "
                 "limit)", fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left', ncol=3)
    ax.grid(True, alpha=0.3, which='both')

    # ── clip scale: the fraction of each head's step that survived ──
    # 1.0 means the head trained at its nominal learning rate; 0.1 means nine
    # tenths of the step was thrown away, i.e. an effective lr 10x lower than
    # --lr-head claims. This is the panel that explains a head that will not
    # move.
    ax = axes[2]
    for i, p in enumerate(ACTIVE_PRIMS):
        ss = np.array([e[4].get(p, np.nan) for e in ghist], dtype=float)
        if np.all(np.isnan(ss)):
            continue
        ax.plot(its, ss, linewidth=1.3, color=cmap(i % 10), label=p)
    ax.axhline(1.0, color='grey', linewidth=0.8, linestyle=':')
    ax.set_ylim(-0.05, 1.10)
    ax.set_ylabel("clip scale")
    ax.set_xlabel("Iteration")
    ax.set_title("FRACTION OF HEAD UPDATE KEPT AFTER CLIPPING  "
                 "(1.0 = unclipped)", fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='lower left', ncol=3)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"State primitives — gradients, {len(TASK_NAMES)} tasks",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_checkpoint(iteration, tag=""):
    """Overwrite params / ObsRMS / plot.

    Catches BaseException, not Exception: a second Ctrl+C landing during the
    matplotlib render would otherwise escape mid-save and leave the plot stale
    relative to the weights beside it. A failed save must never take down a run
    that is otherwise fine either.
    """
    try:
        with open(f"{args.out_prefix}_params.pkl", "wb") as f:
            pickle.dump(jax.tree.map(lambda x: np.array(x), policy_params), f)
        with open(f"{args.out_prefix}_obs_rms.pkl", "wb") as f:
            pickle.dump({'mean': obs_rms.mean, 'var': obs_rms.var,
                         'count': obs_rms.count}, f)
        with open(f"{args.out_prefix}_opt.pkl", "wb") as f:
            pickle.dump({'iteration': iteration,
                         'shared_opt_state': jax.tree.map(np.asarray, shared_opt_state),
                         'head_opt_states': jax.tree.map(np.asarray, head_opt_states),
                         'phase': dict(phase), 'patience': dict(patience),
                         'streak': dict(streak),
                         'active_steps': {k: int(v) for k, v in active_steps.items()},
                         'hist': {k: list(v) for k, v in hist.items()},
                         'ghist': list(ghist),
                         'gscale': list(gscale),
                         'wall': wall_clock()}, f)
        save_learning_plot(f"{args.out_prefix}_learning.png")
        save_gradient_plot(f"{args.out_prefix}_gradients.png")
        print(f"  ✓ saved {args.out_prefix}_{{params,obs_rms,opt}}.pkl + "
              f"_learning.png + _gradients.png at iter {iteration}{tag}")
    except BaseException as e:                               # noqa: BLE001
        print(f"  ! checkpoint at iter {iteration} failed: {e!r}")


# ═══════════════════════════════════════════════════════════════════════════
# VIEWER — single-task runs only
# ═══════════════════════════════════════════════════════════════════════════
class TrajectoryViewer:
    """Passive MuJoCo viewer replaying env 0's qpos trajectory.

    Loads its own MjModel from the XML rather than reusing the task's: the
    task's copy has had collision geoms pushed to group 3, which does not suit a
    human.
    """

    def __init__(self, task):
        from mujoco import viewer as mj_viewer
        self.model = mujoco.MjModel.from_xml_path(task['xml'])
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, task['key_id'])
        mujoco.mj_forward(self.model, self.data)
        self.handle = mj_viewer.launch_passive(self.model, self.data)
        self.handle.opt.geomgroup[0] = False      # hide collision geoms
        self.handle.opt.geomgroup[1] = True       # show visual geoms
        self.handle.sync()
        self.sim_dt = self.model.opt.timestep
        print("✓ Viewer opened — close it to stop training (a checkpoint is "
              "written first)")

    @property
    def running(self):
        return self.handle.is_running()

    def replay(self, qpos_chunks, skip=2):
        """Play back (chunk, step, substep, nq) qpos in roughly real time."""
        if not self.running:
            return
        for chunk_qpos in qpos_chunks:
            for step in range(chunk_qpos.shape[0]):
                for sub in range(0, chunk_qpos.shape[1], skip):
                    if not self.running:
                        return
                    self.data.qpos[:] = chunk_qpos[step, sub]
                    mujoco.mj_forward(self.model, self.data)
                    self.handle.sync()
                    time.sleep(self.sim_dt * skip)

    def close(self):
        try:
            self.handle.close()
        except Exception:                                    # noqa: BLE001
            pass


viewer = None
if USE_VIEWER:
    try:
        viewer = TrajectoryViewer(tasks[TASK_NAMES[0]])
    except Exception as e:                                   # noqa: BLE001
        # No display (a headless box, ssh without X11) must not stop a run that
        # is otherwise fine — the viewer is a convenience, not a dependency.
        print(f"! could not open the viewer ({e!r}); training headless. "
              f"Pass --no-viewer to skip this attempt.")


# ═══════════════════════════════════════════════════════════════════════════
# CURRICULUM — per task
# ═══════════════════════════════════════════════════════════════════════════
phase = {n: 0 for n in TASK_NAMES}
patience = {n: 0 for n in TASK_NAMES}
streak = {n: 0 for n in TASK_NAMES}
active_steps = {
    n: jnp.int32(tasks[n]['phase_bounds'][min(1, len(tasks[n]['phase_bounds']) - 1)])
    for n in TASK_NAMES}

if 'opt' in RESUME:
    # Without this the curriculum silently restarts at phase 0, undoing every
    # advance the run had earned and shortening episodes back to their start.
    for _n in TASK_NAMES:
        if _n in RESUME['opt'].get('phase', {}):
            phase[_n] = int(RESUME['opt']['phase'][_n])
            patience[_n] = int(RESUME['opt']['patience'][_n])
            streak[_n] = int(RESUME['opt']['streak'][_n])
            active_steps[_n] = jnp.int32(RESUME['opt']['active_steps'][_n])
    print("  resumed curriculum: "
          + "  ".join(f"{n}=ph{phase[n]}/act{int(active_steps[n])}"
                      for n in TASK_NAMES))


def pass_fractions(name, all_metrics):
    """Fraction of envs meeting each segment's own criterion.

    A mean is unbounded: a few envs that fail badly (an object knocked off the
    table, an arm parked far from the target) drag it past any threshold even
    when most envs solved the segment.  A fraction is bounded and ignores how
    far the failures travelled.
    """
    crits = tasks[name]['criteria']
    frac = np.ones(len(crits))
    for si, (ctype, thr) in enumerate(crits):
        col = all_metrics[:, si]
        if ctype == "always":
            frac[si] = 1.0
        elif ctype in ("lift", "angle"):
            frac[si] = float((col > thr).mean())
        else:
            frac[si] = float((col < thr).mean())
    return frac


def advance(name, frac_ok):
    t = tasks[name]
    pb = t['phase_bounds']
    if phase[name] >= len(pb) - 1:
        return
    # advance on the fraction of envs that pass, not the mean
    ok = frac_ok[phase[name]] >= PASS_FRAC
    patience[name] = patience[name] + 1 if ok else 0
    if patience[name] >= PATIENCE:
        phase[name] += 1
        patience[name] = 0
        active_steps[name] = jnp.int32(pb[min(phase[name] + 1, len(pb) - 1)])
        print(f"    ──→ {name}: phase {phase[name]}, "
              f"active={int(active_steps[name])}")


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"Training — stop when all {len(TASK_NAMES)} tasks hold "
      f"≥{SUCCESS_RATE:.0%} success for {SUCCESS_STREAK} consecutive logs")
print("=" * 70)

train_start = time.time()
iteration = START_ITER
stop_reason = "reached --iters"
n_skipped = consecutive_skips = 0   # non-finite iterations survived

try:
    for iteration in range(START_ITER, args.iters):
        if viewer is not None and not viewer.running:
            stop_reason = "viewer closed"
            break

        obs_mean, obs_std = obs_rms.get_jnp()
        log_now = (iteration % args.log_every == 0 or iteration < 3)

        # ── chunks: which tasks are stepped together ─────────────────────────
        # One chunk holding everything unless --task-minibatch is set, so the
        # loop below reduces to the original single-update behaviour.
        chunks = make_chunks(iteration)
        # A parameter in several chunks is stepped once per chunk: `reach` is in
        # every task and so every chunk, `turn` in one. Dividing each update by
        # the chunk count it saw holds the per-iteration step size where it was,
        # leaving update FREQUENCY as the only thing that changed.
        chunk_count = {p: 0 for p in ACTIVE_PRIMS}
        for _ch in chunks:
            _seen = set()
            for _gi in _ch:
                for _n in GROUPS[_gi]:
                    _seen.update(tasks[_n]['prim_seq'])
            for _p in _seen:
                if _p in chunk_count:
                    chunk_count[_p] += 1
        _comp = args.mb_lr_compensate and len(chunks) > 1

        states, task_params, auxes = {}, {}, {}
        loss_f = t_reset = t_bptt = 0.0
        head_gnorm, head_scale, shared_gnorm = {}, {}, 0.0
        chunk_gnorms, chunk_skipped = [], False

        for ci, cidx in enumerate(chunks):
            cnames = [n for gi in cidx for n in GROUPS[gi]]

            # ── sample start states, this chunk only ──
            # The vision pipeline's whole "phase 1" (render every chunk, run
            # DINOv3, advance the physics under the cached features) has no
            # counterpart: the observation is read from the state inside the
            # rollout itself. States are still re-sampled per chunk, because
            # after a step the policy visits different ones.
            _t0 = time.time()
            for name in cnames:
                s, tp = rebatch(tasks[name], jax.random.PRNGKey(
                    iteration * 97 + TASK_INDEX[name]))
                jax.block_until_ready(s.qpos)
                states[name], task_params[name] = s, tp
            t_reset += time.time() - _t0

            # ── backward, one task group at a time ──
            # grad(Σ Lᵢ) = Σ grad(Lᵢ), so summing per-group gradients reproduces
            # the fused backward exactly. Groups within a chunk are summed, not
            # stepped: only whole chunks take a step.
            _t0 = time.time()
            grads, c_loss, c_auxes = None, 0.0, {}
            for gi in cidx:
                gnames = GROUPS[gi]
                sub = lambda d: {n: d[n] for n in gnames}
                (g_loss, g_aux), g = group_fns[gi](
                    policy_params, obs_mean, obs_std, sub(states),
                    sub(active_steps), sub(task_params))
                c_loss += float(g_loss)
                c_auxes.update(g_aux)
                if BALANCE:
                    g = balance_grad(gi, g)
                grads = g if grads is None else jax.tree.map(jnp.add, grads, g)
                del g      # release this group's graph before the next one runs
            t_bptt += time.time() - _t0
            loss_f += c_loss
            auxes.update(c_auxes)

            # ── non-finite guard (per chunk) ────────────────────────────────
            # A NaN reaches the weights only through the optimizer step, so
            # skipping that step leaves the policy clean and the next chunk
            # re-samples fresh states. Gradient clipping cannot do this job:
            # global_norm(NaN) is NaN and min(1, c/NaN) is NaN, so a NaN passes
            # straight through any clip. ObsRMS is skipped too: its running
            # mean/var never recover from one non-finite sample.
            _obs = np.concatenate(
                [np.array(c_auxes[n][1][0]).reshape(-1, OBS_DIM)
                 for n in cnames], axis=0)
            _bad = [nm for nm, ok in
                    (("loss", np.isfinite(c_loss)),
                     ("grads", bool(np.isfinite(float(optax.global_norm(grads))))),
                     ("obs", bool(np.isfinite(_obs).all()))) if not ok]
            if _bad:
                n_skipped += 1
                consecutive_skips += 1
                chunk_skipped = True
                _where = "" if len(chunks) == 1 else f" chunk {ci + 1}/{len(chunks)}"
                print(f"\n⚠ iter {iteration}{_where}: non-finite "
                      f"{'+'.join(_bad)} — update skipped "
                      f"({consecutive_skips} in a row, {n_skipped} total). "
                      f"Weights and ObsRMS left untouched.")
                if consecutive_skips >= args.max_nan_skips:
                    break
                continue
            consecutive_skips = 0
            obs_rms.update(_obs)

            # No weight update during ObsRMS warm-up; stats only.
            if iteration < args.warmup_iters:
                continue

            # ── shared trunk ──
            shared_params = {k: policy_params[k] for k in STATE_SHARED_KEYS}
            if log_now:
                shared_gnorm = max(shared_gnorm, float(optax.global_norm(
                    {k: grads[k] for k in STATE_SHARED_KEYS})))
            updates, shared_opt_state = shared_optimizer.update(
                {k: grads[k] for k in STATE_SHARED_KEYS}, shared_opt_state,
                shared_params)
            if _comp:
                _s = 1.0 / len(chunks)
                updates = jax.tree.map(lambda x: x * _s, updates)
            for k, v in optax.apply_updates(shared_params, updates).items():
                policy_params[k] = v

            # ── per-primitive heads (own clip + own optimizer) ──
            # Every head is clipped against its own absolute limit, but heads
            # differ wildly in how many segments feed them (`reach` collects
            # from nearly every task, `turn` from one). A head whose raw
            # gradient is an order of magnitude larger is clipped an order of
            # magnitude harder, so its effective learning rate is that much
            # lower. head_gnorm/head_scale record that, so the imbalance is
            # visible rather than inferred.
            # Only heads this chunk actually drives are stepped: grads[p] exists
            # for every primitive but is exactly zero for one no task in this
            # chunk uses, and Adam steps on a zero gradient anyway, moving the
            # weights on stale momentum and decaying v toward zero. Skipping
            # them keeps each head's step count equal to chunk_count[p].
            chunk_prims = set()
            for _n in cnames:
                chunk_prims.update(tasks[_n]['prim_seq'])
            for p in ACTIVE_PRIMS:
                if p not in chunk_prims:
                    continue
                g = grads[p]
                gn = optax.global_norm(g)
                scale = jnp.minimum(1.0, prim_clip[p] / (gn + 1e-6))
                if log_now:    # float() syncs; only pay it on logs
                    head_gnorm[p] = max(head_gnorm.get(p, 0.0), float(gn))
                    head_scale[p] = min(head_scale.get(p, 1.0), float(scale))
                g = jax.tree.map(lambda x: x * scale, g)
                upd, head_opt_states[p] = head_optimizers[p].update(
                    g, head_opt_states[p], policy_params[p])
                if _comp and chunk_count.get(p, 0) > 1:
                    _s = 1.0 / chunk_count[p]
                    upd = jax.tree.map(lambda x: x * _s, upd)
                policy_params[p] = optax.apply_updates(policy_params[p], upd)

            if log_now:
                chunk_gnorms.append(float(optax.global_norm(grads)))

        if consecutive_skips >= args.max_nan_skips:
            print(f"  {consecutive_skips} consecutive non-finite iterations "
                  f"— stopping. The last checkpoint is still good; something "
                  f"structural is wrong rather than one bad batch.")
            stop_reason = f"{consecutive_skips} consecutive non-finite iters"
            break
        if chunk_skipped:
            # Updates from the chunks that were fine still stand; only the log
            # line is dropped. A skipped chunk's auxes hold the non-finite
            # rewards that triggered the skip, and plotting those would put a
            # NaN into the reward history permanently.
            continue

        # ── ObsRMS warm-up ──────────────────────────────────────────────────
        # Iteration 0 would otherwise normalise with mean=0/std=1 (i.e. the raw
        # observation) and iteration 1 with real statistics — a distribution
        # shift under the policy right after its first update. Collect stats for
        # a few iterations first, applying no weight update.
        if iteration < args.warmup_iters:
            if iteration == args.warmup_iters - 1:
                _s = np.asarray(obs_rms.get_jnp()[1])
                _floored = obs_rms.floored_dims()
                print(f"  [warmup {args.warmup_iters} iters, "
                      f"{int(obs_rms.count)} samples]  "
                      f"std min={_s.min():.3e} max={_s.max():.3e}")
                # Expected: the padded slots, plus any slot no selected task
                # uses. Anything else means a coordinate the policy is being fed
                # never moves, which is worth knowing before a long run.
                print(f"  variance floor ({VAR_FLOOR:g}) binding on "
                      f"{len(_floored)}/{OBS_DIM} dims: {_floored.tolist()}")
            continue

        # ── metrics / success / curriculum ──
        if log_now:
            # Σrew is the sum of the RAW per-task rewards, so it stays
            # comparable across runs; under --normalise the objective actually
            # being minimised is a different, per-step-weighted number, shown
            # alongside so the two are never confused.
            raw_rew = -sum(float(auxes[n][0]) for n in TASK_NAMES)
            # grad= is the MEAN over this iteration's chunks, so it stays on the
            # same scale as a single-update run; the max is appended when they
            # differ, since one spiking chunk is what precedes a NaN.
            grad_norm = float(np.mean(chunk_gnorms)) if chunk_gnorms else 0.0
            _gtxt = (f"grad={grad_norm:.1f}" if len(chunk_gnorms) <= 1 else
                     f"grad={grad_norm:.1f}(max {max(chunk_gnorms):.0f} "
                     f"of {len(chunk_gnorms)})")
            print(f"\nIter {iteration:4d}:  Σrew={raw_rew:.3f}  "
                  + (f"obj={-loss_f:.4f}  " if args.normalise else "")
                  + _gtxt + "  "
                  f"[reset={t_reset:.1f}s bptt={t_bptt:.1f}s "
                  f"elapsed={time.time() - train_start:.0f}s]")

            # Same numbers the per-head table below prints, kept for the
            # gradient curves. Recorded before the table so a later formatting
            # change there cannot silently drop the history.
            ghist.append((iteration, grad_norm, shared_gnorm,
                          {p: head_gnorm[p] for p in ACTIVE_PRIMS
                           if p in head_gnorm},
                          {p: head_scale[p] for p in ACTIVE_PRIMS
                           if p in head_scale}))

            # ── per-head gradients, worst-clipped first ──
            # `clip` is the factor the head's update was scaled by: 1.00 means
            # it passed through untouched, 0.10 means nine tenths of its step
            # was thrown away. Heads sitting at a much smaller clip than their
            # neighbours are being trained at a correspondingly smaller
            # effective learning rate, whatever --lr-head says.
            order = sorted(ACTIVE_PRIMS, key=lambda p: -head_gnorm[p])
            n_clipped = sum(1 for p in ACTIVE_PRIMS if head_scale[p] < 1.0)
            _tscale = min(1.0, args.grad_clip / (shared_gnorm + 1e-6))
            print(f"  heads ({n_clipped}/{len(ACTIVE_PRIMS)} clipped)   "
                  f"trunk |g|={shared_gnorm:.0f} lim={args.grad_clip:.0f} "
                  f"x{_tscale:.2f}")
            for i in range(0, len(order), 2):
                print("   " + "".join(
                    f"{p:>12s} x{HEAD_SEGMENTS[p]:<2d} |g|={head_gnorm[p]:>8.0f}"
                    f" lim={prim_clip[p]:>6.0f} x{head_scale[p]:.2f}  "
                    for p in order[i:i + 2]))

            all_solved = True
            qpos_env0 = None
            for name in TASK_NAMES:
                t = tasks[name]
                states_b, qpos_b, fng_b = t['jit_batch_metrics'](
                    policy_params, states[name], obs_mean, obs_std,
                    active_steps[name])
                all_metrics, s_fng, extras = t['compute_metrics'](states_b, fng_b)
                metrics = all_metrics.mean(axis=0)
                frac_ok = pass_fractions(name, all_metrics)
                n_ok, rate = t['success_fn'](all_metrics, extras)

                # env 0's trajectory, for the viewer. Taken from the metrics
                # rollout rather than collected separately: it is the same
                # policy on the same start states, and it is already computed.
                if viewer is not None:
                    qpos_env0 = np.array(qpos_b[0])      # (steps, substeps, nq)

                # Success counts only once the WHOLE episode is being
                # simulated. Below that, every step past active_steps is frozen
                # by lax.cond, so the terminal metric is read off a truncated
                # rollout: drawer_close scored 90% at phase 0 because the arm
                # knocked the drawer shut during reach and grasp, and the push
                # segment — whose head had received exactly zero gradient —
                # never ran at all. Gating on active_steps rather than on a
                # phase index covers every task length uniformly: a one-segment
                # task is full-length from the start, container only at phase 13.
                full_episode = int(active_steps[name]) >= t['n_total']
                streak[name] = (streak[name] + 1
                                if rate >= SUCCESS_RATE and full_episode else 0)
                if streak[name] < SUCCESS_STREAK:
                    all_solved = False

                rew_n = -float(auxes[name][0])
                hist[name].append((iteration, rew_n))
                segs = "  ".join(f"{lbl}={frac_ok[i]:.0%}/{metrics[i]:.4f}"
                                 for i, lbl in enumerate(t['seg_labels']))
                mark = "✓" if streak[name] >= SUCCESS_STREAK else " "
                print(f"  {mark} {name:15s} rew={rew_n:8.3f}  ph={phase[name]}  "
                      f"success={n_ok:g}/{t['batch_size']} ({rate:.0%}) "
                      # "(partial)" marks a rate measured on a truncated
                      # rollout, which cannot count toward the streak.
                      + ("(partial) " if not full_episode else "")
                      + f"streak={streak[name]}  | {segs}")

                advance(name, frac_ok)

            # Replay env 0 into the viewer — single-task runs only.
            if viewer is not None and qpos_env0 is not None:
                viewer.replay([qpos_env0], skip=2)

            if all_solved:
                print(f"\n✓ All {len(TASK_NAMES)} tasks solved "
                      f"({SUCCESS_STREAK} consecutive ≥{SUCCESS_RATE:.0%}) "
                      f"at iter {iteration}.")
                stop_reason = "all tasks solved"
                break

        # Periodic checkpoint. Placed after the optimizer step so what lands on
        # disk is the policy as of the end of this iteration.
        if args.save_every > 0 and iteration > 0 \
                and iteration % args.save_every == 0:
            save_checkpoint(iteration)

        if iteration % 50 == 0:
            gc.collect()

except KeyboardInterrupt:
    # Ctrl+C must not throw away the run — fall through to the save below.
    stop_reason = "interrupted (Ctrl+C)"
    print(f"\n\n⚠ Interrupted at iter {iteration} — saving before exit...")


# ═══════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════
print(f"\nTraining stopped: {stop_reason} "
      f"after {iteration} iters, {time.time() - train_start:.1f}s"
      + (f"  ({n_skipped} non-finite iterations skipped)" if n_skipped else ""))
# Always safe to save: the non-finite guard means no bad update ever reached
# the weights, so whatever is in memory is the best state the run produced.
save_checkpoint(iteration, tag=" (final)")
if viewer is not None:
    viewer.close()
