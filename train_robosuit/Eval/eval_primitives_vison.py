"""
Common Eval — every robosuite task, one script.
================================================
Counterpart to train_primitives_vison.py.  It owns no task logic: every
per-task object comes from that task's environment in envs/, so eval and
training cannot drift apart — in particular the start states come from the
*same* `rebatch_jit` the policy was trained on.

Each task is scored by its own criterion (the `success_fn` on its env class):
|joint - target| < success_thresh for the articulated tasks, the xy-stacking
check for cube_stacking, partial credit per object for container, and so on.

Works with either checkpoint type, auto-detected:
  - shared-primitive (train_primitives_vison.py) → named heads (reach, pull, …)
  - per-segment      (a legacy per-task checkpoint) → positional seg0..seg4

Usage:
    python eval_primitives_vison.py                                  # default tasks
    python eval_primitives_vison.py --tasks all --n-trials 100
    python eval_primitives_vison.py --tasks drawer_open --params ../my_run.pkl
    python eval_primitives_vison.py --tasks drawer_open --run "Drawer open"   # opens the viewer

Viewer: a single-task eval opens a passive MuJoCo viewer and replays one env
per --render-every trials (default 20: env 0 of each 20-env batch, so 5 replays
over 100 trials). Suppress it with --no-viewer.

Recording: the same replays are rendered into one mp4 per task,
eval_<task>.mp4 next to the checkpoint, framed green/red by the terminal
segment's criterion — in a separate process, by Eval/record_replays.py. Works
headless (--no-viewer); suppress with --no-record.
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAIN = os.path.join(_HERE, '..')            # train_robosuit/  (checkpoints)
_ROOT = os.path.join(_HERE, '..', '..')       # repo root        (envs/, models/)
sys.path.insert(0, _ROOT)

# XLA memory/compile settings — MUST precede `import jax`, since the backend
# reads them once at initialisation. Anything already set in the shell wins.
from helpers.xla_env import apply_xla_defaults
apply_xla_defaults()

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")

import argparse, pickle, shutil, subprocess, tempfile, time
import numpy as np
import jax.numpy as jnp
import mujoco

from helpers.mjx_utils import patch_solver
from helpers.obs import ObsRMS
from models.vision_backbone import DINOv3Backbone
from models.vision_policy import vision_policy_forward, N_SEGMENTS
from models.primitive_policy import (make_primitive_forward, SHARED_KEYS,
                                    make_monolithic_forward)
import envs

patch_solver()


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG + ARGS
# ═══════════════════════════════════════════════════════════════════════════
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config", type=str, default=None,
                  help="task config YAML (default: configs/tasks.yaml)")
_cfg_args, _ = _pre.parse_known_args()

cfg = envs.load_config(_cfg_args.config)
TR = cfg['training']
POL = envs.policy_cfg(cfg)
PRIM_NAMES = envs.primitive_names(cfg)

parser = argparse.ArgumentParser(parents=[_pre])
parser.add_argument("--tasks", type=str, default=','.join(TR['default_tasks']),
                    help="comma-separated task names, or 'all'")
parser.add_argument("--run", type=str, default="all", metavar="DIR",
                    help="run folder under 'Plots and Checkpoints/Vision/' to "
                         "evaluate, e.g. 'all', 'Reach', 'Cube stacking'. The "
                         "_params.pkl and _obs_rms.pkl inside are found by "
                         "suffix, so the run's --out-prefix does not have to be "
                         "known here. Override either file with --params / "
                         "--obs-rms.")
parser.add_argument("--ckpt-root", type=str,
                    default=os.path.join(_TRAIN, "Plots and Checkpoints",
                                         "Vision"),
                    help="where the run folders live")
parser.add_argument("--params", type=str, default=None,
                    help="explicit params .pkl (overrides --run)")
parser.add_argument("--obs-rms", type=str, default=None,
                    help="explicit obs_rms .pkl (overrides --run)")

parser.add_argument("--n-trials", type=int, default=100,
                    help="per task; rounded up to a whole number of batches")
parser.add_argument("--batch", type=int, default=TR['batch'],
                    help="envs per rollout")
parser.add_argument("--task-batch", type=str, default="",
                    help="per-task override, e.g. 'cube_stacking=10'")
parser.add_argument("--img-size", type=int, default=TR['img_size'])
parser.add_argument("--substeps", type=int, default=TR['substeps'])
parser.add_argument("--chunk-size", type=int, default=TR['chunk_size'])
parser.add_argument("--proj-dim", type=int, default=POL['proj_dim'])
parser.add_argument("--temperature", type=float, default=POL['temperature'],
                    help="must match the value the checkpoint trained at")
parser.add_argument("--seed", type=int, default=10000)
parser.add_argument("--no-viewer", action="store_true",
                    help="suppress the MuJoCo viewer in single-task evals")
parser.add_argument("--render-every", type=int, default=20, metavar="N",
                    help="replay one env per N trials in the viewer (default "
                         "20: env 0 of each 20-env batch); 0 disables")
parser.add_argument("--no-record", action="store_true",
                    help="don't write the replays to eval_<task>.mp4")
args = parser.parse_args()

# Eval covers retired tasks too: they are excluded from a training sweep, not
# from measurement.
TASK_NAMES = (envs.task_names(cfg, include_retired=True)
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

PROPRIO_DIM = cfg['robot']['proprio_dim']

def _find_ckpt(run_dir, suffix, explicit):
    """Locate the one file ending in `suffix` inside `run_dir`.

    Runs are saved under a per-run folder with the run's own --out-prefix
    baked into every filename ('state_all_params.pkl', 'reach_params.pkl'),
    so the prefix cannot be reconstructed from the task list. Matching on the
    suffix keeps the eval independent of what the run was called.
    """
    if explicit:
        return explicit
    if not os.path.isdir(run_dir):
        avail = (sorted(os.listdir(os.path.dirname(run_dir)))
                 if os.path.isdir(os.path.dirname(run_dir)) else [])
        raise SystemExit(f"no run folder {run_dir!r}"
                         + (f"; available: {avail}" if avail else ""))
    hits = sorted(f for f in os.listdir(run_dir) if f.endswith(suffix))
    if not hits:
        raise SystemExit(f"no *{suffix} in {run_dir!r} "
                         f"(found: {sorted(os.listdir(run_dir))})")
    if len(hits) > 1:
        raise SystemExit(f"several *{suffix} in {run_dir!r}: {hits}. "
                         f"Pick one explicitly.")
    return os.path.join(run_dir, hits[0])


_RUN_DIR = os.path.join(args.ckpt_root, args.run)
PARAMS_PATH = _find_ckpt(_RUN_DIR, "_params.pkl", args.params)
OBS_RMS_PATH = _find_ckpt(_RUN_DIR, "_obs_rms.pkl", args.obs_rms)

# The viewer is built from one task's XML, so — as in the trainers — it only
# makes sense for a single-task eval, i.e. a separately trained checkpoint.
USE_VIEWER = (len(TASK_NAMES) == 1 and not args.no_viewer
              and args.render_every > 0)
FFMPEG = shutil.which("ffmpeg")
RECORD = (len(TASK_NAMES) == 1 and not args.no_record
          and args.render_every > 0 and FFMPEG is not None)
VIDEO_DIR = os.path.dirname(os.path.abspath(PARAMS_PATH))


print("=" * 74)
print("COMMON EVAL — robosuite vision primitives")
print(f"  Tasks:  {', '.join(TASK_NAMES)}")
print(f"  Run:    {args.run}")
print(f"  Params: {PARAMS_PATH}")
print(f"  ObsRMS: {OBS_RMS_PATH}")
print(f"  Proprio dim: {PROPRIO_DIM}")
print(f"  Viewer: "
      + (f"on (one replay per {args.render_every} trials)" if USE_VIEWER else
         "off  (multi-task eval)" if len(TASK_NAMES) > 1 else "off"))
print(f"  Record: "
      + (f"on → {VIDEO_DIR}/eval_<task>.mp4" if RECORD else
         "off  (ffmpeg not found)" if FFMPEG is None and not args.no_record
         and len(TASK_NAMES) == 1 else "off"))
print("=" * 74)


# ═══════════════════════════════════════════════════════════════════════════
# CHECKPOINT + BACKBONE
# ═══════════════════════════════════════════════════════════════════════════
print("\n[1] Loading checkpoint...")
with open(PARAMS_PATH, "rb") as f:
    policy_params = jax.tree.map(jnp.array, pickle.load(f))
with open(OBS_RMS_PATH, "rb") as f:
    rms_data = pickle.load(f)

obs_rms = ObsRMS(PROPRIO_DIM)
obs_rms.mean, obs_rms.var, obs_rms.count = (
    rms_data['mean'], rms_data['var'], rms_data['count'])
obs_mean, obs_std = obs_rms.get_jnp()
print(f"  OBS RMS: count={obs_rms.count}, "
      f"mean_range=[{obs_rms.mean.min():.3f}, {obs_rms.mean.max():.3f}]")

# The --monolithic ablation keeps its whole network under 'trunk', ending in
# w_out/b_out, and has no per-primitive heads at all. Detected from the
# checkpoint rather than from a flag, so a run is evaluated the way it was
# trained without the caller having to remember which ablation it was.
IS_MONOLITHIC = 'w_out' in policy_params.get('trunk', {})
IS_PER_SEGMENT = (not IS_MONOLITHIC) and 'seg0' in policy_params
if IS_MONOLITHIC:
    _n = sum(x.size for x in jax.tree.leaves(policy_params))
    _L = len(policy_params['trunk']['hidden'])
    print(f"  checkpoint: MONOLITHIC ablation — one MLP, {_L} hidden layers, "
          f"no per-primitive heads, {_n:,} params")
elif IS_PER_SEGMENT:
    print(f"  checkpoint: per-segment policy (seg0..seg{N_SEGMENTS - 1})")
    if len(TASK_NAMES) > 1:
        print("  ! a per-segment checkpoint is task-specific — evaluating it on "
              "several tasks is only meaningful for the task it was trained on")
else:
    n_heads = len([k for k in policy_params if k not in SHARED_KEYS])
    print(f"  checkpoint: shared-primitive policy ({n_heads} heads)")

print("\n[2] DINOv3 backbone (shared)...")
backbone = DINOv3Backbone(img_size=args.img_size, device="cuda")


def pick_forward(name):
    """Forward matching the checkpoint, for this task's primitive sequence."""
    if IS_PER_SEGMENT:
        return vision_policy_forward
    seq = envs.prim_seq(cfg, name)
    if IS_MONOLITHIC:
        # Same call signature as make_primitive_forward; the primitive becomes
        # a one-hot input instead of selecting a head.
        return make_monolithic_forward(seq, PRIM_NAMES, args.temperature)
    missing = [p for p in seq if p not in policy_params]
    if missing:
        raise SystemExit(
            f"checkpoint lacks primitives {missing} needed by '{name}'. "
            f"Keys present: {sorted(policy_params)}")
    # Must match the temperature the checkpoint was trained at, or evaluation
    # feeds the trunk a differently-scaled vision code than it ever saw.
    return make_primitive_forward(seq, PRIM_NAMES, args.temperature)


def rebatch(t, key):
    """→ states, hiding the 6-arg/7-arg task difference."""
    out = t['rebatch_jit'](key, t['batch_size'])
    return out[0] if isinstance(out, tuple) else out


# ═══════════════════════════════════════════════════════════════════════════
# EVAL
# ═══════════════════════════════════════════════════════════════════════════
def pass_fractions(criteria, all_metrics):
    """Fraction of trials meeting each segment's own criterion.

    The same rule the trainers advance the curriculum on, so an eval number is
    directly comparable with the `Reach=100%/0.0087` column in a training log.
    A mean alone hides the shape of the distribution: a segment can sit on a
    respectable mean while most trials fail and a few succeed wildly.
    """
    frac = np.ones(len(criteria))
    for si, (ctype, thr) in enumerate(criteria):
        col = all_metrics[:, si]
        if ctype == "always":
            frac[si] = 1.0
        elif ctype in ("lift", "angle"):
            frac[si] = float((col > thr).mean())
        else:
            frac[si] = float((col < thr).mean())
    return frac


# ═══════════════════════════════════════════════════════════════════════════
# VIEWER — single-task evals only
# ═══════════════════════════════════════════════════════════════════════════
class TrajectoryViewer:
    """Passive MuJoCo viewer replaying one env's qpos trajectory.

    The trainers' viewer, plus the mocap bodies: qpos alone would leave the
    reach / place / push target at its keyframe position rather than where
    this episode sampled it. Loads its own MjModel from the XML: the task's
    copy has collision geoms in group 3 and 64x64 cameras.
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
        print("✓ Viewer opened — closing it stops the replays, not the eval")

    @property
    def running(self):
        return self.handle.is_running()

    def replay(self, qpos, mocap_pos=None, skip=2, hold=1.0):
        """Play back (step, substep, nq) qpos in roughly real time."""
        if not self.running:
            return
        if mocap_pos is not None and self.model.nmocap:
            self.data.mocap_pos[:] = mocap_pos
        for step in range(qpos.shape[0]):
            for sub in range(0, qpos.shape[1], skip):
                if not self.running:
                    return
                self.data.qpos[:] = qpos[step, sub]
                mujoco.mj_forward(self.model, self.data)
                self.handle.sync()
                time.sleep(self.sim_dt * skip)
        time.sleep(hold)                          # linger on the final state

    def close(self):
        try:
            self.handle.close()
        except Exception:                                    # noqa: BLE001
            pass


class TrajectoryRecorder:
    """Collects the replayed envs, then renders them to one mp4 per task.

    The rendering runs in a process of its own (Eval/record_replays.py): an
    offscreen GL context next to the viewer's in THIS process renders textured
    geoms black on a two-GPU machine. See that file.
    """

    def __init__(self, task, path):
        self.task, self.path = task, path
        self.qpos, self.mocap, self.ok, self.cams = [], [], [], []

    def record(self, qpos, mocap_pos=None, ok=True, cam=None):
        """Keep one (step, substep, nq) trajectory, one frame per step."""
        self.qpos.append(qpos[:, -1])
        self.mocap.append(mocap_pos)
        self.ok.append(ok)
        # Copied now: the viewer's camera keeps moving after this replay.
        self.cams.append(np.full(9, np.nan) if cam is None else np.array(
            [cam.type, cam.fixedcamid, cam.trackbodyid, *cam.lookat,
             cam.distance, cam.azimuth, cam.elevation], dtype=float))

    def close(self):
        if not self.qpos:
            return
        fd, npz = tempfile.mkstemp(suffix=".npz")
        os.close(fd)
        try:
            np.savez(npz, xml=self.task['xml'], key_id=self.task['key_id'],
                     fps=1.0 / self.task['frame_dt'], qpos=np.stack(self.qpos),
                     mocap=np.stack(self.mocap), ok=np.array(self.ok),
                     cam=np.stack(self.cams))
            subprocess.run([sys.executable,
                            os.path.join(_HERE, "record_replays.py"),
                            npz, self.path])
        finally:
            os.remove(npz)


def replay_env(viewer, recorder, t, states_b, qpos_b, all_metrics, i, trial):
    """Replay env `i` of this batch, labelled with its terminal-segment metric."""
    ctype, thr = t['criteria'][-1]
    val = all_metrics[i, -1]
    ok = (True if ctype == "always" else
          val > thr if ctype in ("lift", "angle") else val < thr)
    print(f"      ▶ replaying trial {trial} (env {i}): "
          f"{t['seg_labels'][-1]}={val:.4f} "
          f"{'✓' if ok else '✗'} (criterion {ctype} {thr:.3f})")
    qpos = np.array(qpos_b[i])
    mocap = np.array(states_b.mocap_pos[i, 0])
    live = viewer is not None and viewer.running
    if live:
        viewer.replay(qpos, mocap_pos=mocap)
    if recorder is not None:
        # After the live replay, so a camera framed in the viewer is the one
        # recorded; the default free camera otherwise.
        recorder.record(qpos, mocap_pos=mocap, ok=ok,
                        cam=viewer.handle.cam if live else None)


results = {}
viewer = None

for name in TASK_NAMES:
    print(f"\n{'─' * 74}\n[3] {name} — building {cfg['tasks'][name]['env']} env")
    env = envs.make_env(
        name, cfg, backbone, pick_forward(name),
        batch_size=task_batch[name], img_size=args.img_size,
        substeps=args.substeps, chunk_size=args.chunk_size)
    t = env.build()

    if USE_VIEWER and viewer is None:
        try:
            viewer = TrajectoryViewer(t)
        except Exception as e:                               # noqa: BLE001
            # No display must not stop an eval that is otherwise fine.
            print(f"! could not open the viewer ({e!r}); evaluating headless. "
                  f"Pass --no-viewer to skip this attempt.")

    recorder = None
    if RECORD:
        try:
            recorder = TrajectoryRecorder(
                t, os.path.join(VIDEO_DIR, f"eval_{name}.mp4"))
        except Exception as e:                               # noqa: BLE001
            print(f"! could not start recording ({e!r}); continuing without. "
                  f"Pass --no-record to skip this attempt.")

    B = t['batch_size']
    active = jnp.int32(t['n_total'])          # full episode, no curriculum
    n_batches = max(1, (args.n_trials + B - 1) // B)
    print(f"    {t['n_total']} steps, {n_batches} batch(es) x {B} envs "
          f"= {n_batches * B} trials")

    succ, trials, errs = 0, 0, []
    metrics_all = []          # every batch, not just the last one
    t0 = time.time()
    for bi in range(n_batches):
        states = rebatch(t, jax.random.PRNGKey(args.seed + bi))
        jax.block_until_ready(states.qpos)

        vision, *_ = t['collect_features_batched'](
            policy_params, states, obs_mean, obs_std, active,
            collect_qpos=False)
        states_b, qpos_b, fng_b = t['jit_batch_metrics'](
            policy_params, vision, states, obs_mean, obs_std, active)

        all_metrics, _s_fng, extras = t['compute_metrics'](states_b, fng_b)
        n_ok, _rate = t['success_fn'](all_metrics, extras)   # task's own criterion

        # One env per --render-every trials, counted over the whole eval, so
        # the replays sample every seed block — env 0 of each batch at 5 x 20.
        if recorder is not None or (viewer is not None and viewer.running):
            for i in range(B):
                if (trials + i) % args.render_every == 0:
                    replay_env(viewer, recorder, t, states_b, qpos_b,
                               all_metrics, i, trials + i)

        succ += n_ok
        trials += B
        metrics_all.append(all_metrics)
        errs.extend(all_metrics[:, -1].tolist())
        print(f"      batch {bi + 1}/{n_batches}: "
              f"{succ}/{trials} ({succ / trials:.1%})  "
              f"final_err mean={np.mean(errs):.4f}")

    if recorder is not None:
        recorder.close()

    # Over every trial. This previously read `all_metrics`, which after the
    # loop holds only the LAST batch — 20 of 100 trials at the default sizes.
    metrics_cat = np.concatenate(metrics_all, axis=0)
    seg_means = metrics_cat.mean(axis=0)
    seg_frac = pass_fractions(t['criteria'], metrics_cat)
    results[name] = dict(
        succ=succ, trials=trials, rate=succ / trials,
        err_mean=float(np.mean(errs)), err_med=float(np.median(errs)),
        err_min=float(np.min(errs)), secs=time.time() - t0,
        seg_labels=t['seg_labels'], seg_means=seg_means,
        seg_frac=seg_frac, criteria=t['criteria'],
        prims=envs.prim_seq(cfg, name),
    )
    r = results[name]
    segs = "  ".join(f"{l}={seg_frac[i]:.0%}/{seg_means[i]:.4f}"
                     for i, l in enumerate(t['seg_labels']))
    print(f"    → {succ}/{trials} = {r['rate']:.1%}   [{segs}]")

if viewer is not None:
    viewer.close()


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 74)
print("EVAL SUMMARY".center(74))
print("=" * 74)
print(f"{'task':16s} {'success':>14s} {'rate':>8s} "
      f"{'err mean':>10s} {'err med':>9s} {'time':>7s}")
print("-" * 74)
for name in TASK_NAMES:
    r = results[name]
    frac = "{}/{}".format(r['succ'], r['trials'])
    print("{:16s} {:>14s} {:>7.1%} {:>10.4f} {:>9.4f} {:>6.0f}s".format(
        name, frac, r['rate'], r['err_mean'], r['err_med'], r['secs']))
print("-" * 74)
tot_s = sum(results[n]['succ'] for n in TASK_NAMES)
tot_t = sum(results[n]['trials'] for n in TASK_NAMES)
print("{:16s} {:>14s} {:>7.1%}".format(
    'OVERALL', "{}/{}".format(tot_s, tot_t), tot_s / tot_t))
print("=" * 74)
print("\nSEGMENT PASS FRACTIONS — trials meeting each segment's own criterion")
print("(a low fraction on an early segment is where the task actually breaks;")
print(" the terminal one usually just inherits that failure)")
print("-" * 74)
for name in TASK_NAMES:
    r = results[name]
    print(f"  {name}")
    for i, lbl in enumerate(r['seg_labels']):
        ctype, thr = r['criteria'][i]
        crit = ("always" if ctype == "always"
                else f"{'>' if ctype in ('lift', 'angle') else '<'}{thr:.3f}")
        bar = "█" * int(round(r['seg_frac'][i] * 20))
        print(f"     {lbl:10s} {r['seg_frac'][i]:>5.0%} {bar:<20s} "
              f"mean={r['seg_means'][i]:.4f}  ({crit})")
print("-" * 74)

print("\nPrimitives exercised per task:")
for name in TASK_NAMES:
    print(f"  {name:16s} {' → '.join(results[name]['prims'])}")
