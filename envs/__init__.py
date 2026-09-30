"""
Task registry — config in, built environment out.
==================================================
`configs/tasks.yaml` names an `env:` for each task; this module maps that name
onto the class in this package and hands the class its slice of the config.

    cfg = load_config()
    env = make_env('drawer_open', cfg, backbone, policy_forward, batch_size=20)
    task = env.build()          # the bundle the trainer consumes

Adding a task is a config edit plus, only if its logic is genuinely new, one
class here.  The six articulated tasks all reuse ArticulatedEnv; a seventh
hinge-or-slide task needs no Python at all.
"""
import os

import yaml

from envs.base import VisionWarpEnv
from envs.reach import ReachEnv
from envs.cube_stacking import CubeStackingEnv
from envs.pick_place import PickPlaceEnv
from envs.push_cube import PushCubeEnv
from envs.container import ContainerEnv
from envs.peg_insertion import PegInsertionEnv
from envs.articulated import ArticulatedEnv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(_ROOT, 'configs', 'tasks.yaml')
ASSET_DIR = os.path.join(_ROOT, 'assets')

#: `env:` value in the config → the class that implements it
ENV_CLASSES = {
    'reach':          ReachEnv,
    'cube_stacking':  CubeStackingEnv,
    'pick_place':     PickPlaceEnv,
    'push_cube':      PushCubeEnv,
    'container':      ContainerEnv,
    'peg_insertion':  PegInsertionEnv,
    'articulated':    ArticulatedEnv,
}

__all__ = ['load_config', 'make_env', 'task_names', 'primitive_names',
           'finger_biases', 'head_clips', 'head_widths', 'policy_cfg',
           'prim_seq', 'xml_path', 'ENV_CLASSES', 'VisionWarpEnv']


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
def load_config(path=None):
    """Read configs/tasks.yaml (override with $MJX_TASK_CONFIG or `path`)."""
    path = path or os.environ.get('MJX_TASK_CONFIG') or DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)

    unknown = {n: t['env'] for n, t in cfg['tasks'].items()
               if t['env'] not in ENV_CLASSES}
    if unknown:
        raise SystemExit(
            f"{path}: tasks name env classes that do not exist: {unknown}. "
            f"Known: {sorted(ENV_CLASSES)}")

    # Every primitive needs its own gradient-clip entry. Checked here rather
    # than defaulted, so adding a primitive to the vocabulary cannot silently
    # inherit someone else's clip.
    prims = set(cfg['primitives'])
    clips = cfg['training'].get('grad_clip_head') or {}
    missing = sorted(prims - set(clips))
    extra = sorted(set(clips) - prims)
    if missing or extra:
        raise SystemExit(
            f"{path}: training.grad_clip_head must list every primitive exactly "
            f"once."
            + (f" Missing: {missing}." if missing else "")
            + (f" Unknown: {extra}." if extra else ""))

    # An articulated task's start range must be a real interval. It is drawn
    # from directly (no clipping), so an inverted or empty range would silently
    # produce a constant start and undo the overlap it exists to create.
    for name, t in cfg['tasks'].items():
        if 'joint_init_range' in t:
            lo, hi = t['joint_init_range']
            if not hi > lo:
                raise SystemExit(
                    f"{path}: task '{name}' joint_init_range {[lo, hi]} is empty "
                    f"or inverted; it must satisfy hi > lo")

    # A criterion threshold of `success` means "the task's success_thresh".
    # The terminal criterion measures the same number success_fn scores, so
    # repeating the value invites the two drifting apart — which shows up as a
    # log line whose segment % and success % disagree. Tasks whose terminal
    # segment is scored on a DIFFERENT quantity (cube_stacking, container and
    # peg_insertion all end on a release that always passes) keep writing
    # `always` and are unaffected.
    for name, t in cfg['tasks'].items():
        for crit in (t.get('criteria', []) +
                     t.get('per_object', {}).get('criteria', [])):
            if crit[1] == 'success':
                if 'success_thresh' not in t:
                    raise SystemExit(
                        f"{path}: task '{name}' uses the `success` threshold "
                        f"sentinel but defines no success_thresh")
                crit[1] = float(t['success_thresh'])

    # Every prim_seq entry must name a head that actually gets created, and
    # must be as long as the task's segment list — a mismatch would otherwise
    # surface as a KeyError or a silently misrouted head deep inside a rollout.
    for name, t in cfg['tasks'].items():
        missing = [p for p in t['prim_seq'] if p not in prims]
        if missing:
            raise SystemExit(
                f"{path}: task '{name}' uses primitives absent from the "
                f"vocabulary: {missing}")
        n_seg = len(t['per_object']['seg_steps']) * len(t['per_object']['objects']) \
            if 'per_object' in t else len(t['seg_steps'])
        if len(t['prim_seq']) != n_seg:
            raise SystemExit(
                f"{path}: task '{name}' has {n_seg} segments but "
                f"{len(t['prim_seq'])} primitives in prim_seq")
    return cfg


def task_names(cfg, include_retired=False):
    """Every task in the config; retired ones only when asked for."""
    return [n for n, t in cfg['tasks'].items()
            if include_retired or not t.get('retired', False)]


def primitive_names(cfg):
    """The primitive vocabulary, in config order — this fixes the head order."""
    return list(cfg['primitives'])


def finger_biases(cfg):
    """primitive name → the head's initial finger bias."""
    return {n: float(p['finger_bias']) for n, p in cfg['primitives'].items()}


def head_clips(cfg):
    """primitive name -> that head's gradient-clip limit."""
    return {n: float(v) for n, v in cfg['training']['grad_clip_head'].items()}


def policy_cfg(cfg):
    """The `policy:` block — projection width, trunk width, head-width rule."""
    return cfg['policy']


def head_widths(cfg):
    """primitive name -> that head's hidden width.

    hidden = max(floor, per_segment x segments driving the head), counted over
    EVERY task in the config rather than the selected subset, so `--tasks`
    never changes the shape of the parameter tree.
    """
    hh = cfg['policy']['head_hidden']
    per, floor = int(hh['per_segment']), int(hh['floor'])
    segs = {p: 0 for p in cfg['primitives']}
    for t in cfg['tasks'].values():
        for p in t['prim_seq']:
            segs[p] += 1
    return {p: max(floor, per * n) for p, n in segs.items()}


def head_segments(cfg):
    """primitive name -> how many segments across all config tasks drive it."""
    segs = {p: 0 for p in cfg['primitives']}
    for t in cfg['tasks'].values():
        for p in t['prim_seq']:
            segs[p] += 1
    return segs


def prim_seq(cfg, name):
    """Which primitive head drives each segment of `name`."""
    return list(cfg['tasks'][name]['prim_seq'])


def xml_path(cfg, name):
    return os.path.join(ASSET_DIR, cfg['tasks'][name]['xml'])


# ═══════════════════════════════════════════════════════════════════════════
# CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════
def make_env(name, cfg, backbone, policy_forward, batch_size=None,
             img_size=None, substeps=None, chunk_size=None, gamma=None,
             full_chain=False):
    """Instantiate `name`'s environment. Unset arguments fall back to the config.

    `policy_forward` is called as
        (params, vis_list, proprio_norm, step_idx, seg_boundaries)
    so the same rollout serves the shared-primitive policy and any per-segment
    policy without the environment knowing which it has.
    """
    if name not in cfg['tasks']:
        raise SystemExit(
            f"unknown task {name!r}; choose from {sorted(cfg['tasks'])}")

    tcfg = cfg['tasks'][name]
    tr = cfg['training']
    cls = ENV_CLASSES[tcfg['env']]

    return cls(
        name=name, cfg=cfg, tcfg=tcfg, backbone=backbone,
        policy_forward=policy_forward, xml_path=xml_path(cfg, name),
        batch_size=tr['batch'] if batch_size is None else batch_size,
        img_size=tr['img_size'] if img_size is None else img_size,
        substeps=tr['substeps'] if substeps is None else substeps,
        chunk_size=tr['chunk_size'] if chunk_size is None else chunk_size,
        gamma=tr['gamma'] if gamma is None else gamma,
        full_chain=full_chain,
    )
