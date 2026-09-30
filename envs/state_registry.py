"""
State task registry — config in, built state environment out.
=============================================================
The counterpart to envs/__init__.py for the state-observation pipeline.  It is
a separate module rather than an addition to that one so the vision branch is
left exactly as it was:

    from envs.state_registry import load_state_config, make_state_env

    cfg = load_state_config()
    env = make_state_env('cube_stacking', cfg, forward, batch_size=20)
    task = env.build()          # the bundle the state trainer consumes

Everything that is common to both branches — reading the YAML, validating the
primitive vocabulary, resolving the `success` criterion sentinel, head widths,
asset paths — is reused from envs/__init__.py rather than copied.  What is added
here is the validation of the state-only `obs:` block and each task's
`state_slots:`.
"""
import os

import envs
from envs.state_base import SLOT_KINDS, parse_slot_specs
from envs.state_tasks import STATE_ENV_CLASSES

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_STATE_CONFIG = os.path.join(_ROOT, 'configs', 'tasks_state.yaml')

__all__ = ['load_state_config', 'make_state_env', 'STATE_ENV_CLASSES',
           'DEFAULT_STATE_CONFIG']


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
def load_state_config(path=None):
    """Read configs/tasks_state.yaml (override with $MJX_STATE_TASK_CONFIG).

    Runs every check the vision loader runs, then the state-specific ones.  All
    of them are hard failures: a wrong observation width is not something to
    discover from a shape error inside a jitted BPTT trace, or — worse — from a
    checkpoint that silently loads at the wrong size.
    """
    path = path or os.environ.get('MJX_STATE_TASK_CONFIG') or DEFAULT_STATE_CONFIG
    cfg = envs.load_config(path)          # shared validation + `success` sentinel

    if 'obs' not in cfg:
        raise SystemExit(
            f"{path}: no `obs:` block. A state config must declare n_slots and "
            f"slot_dim — see configs/tasks_state.yaml")
    obs = cfg['obs']
    for key in ('n_slots', 'slot_dim'):
        if key not in obs:
            raise SystemExit(f"{path}: obs.{key} is required")
    n_slots, slot_dim = int(obs['n_slots']), int(obs['slot_dim'])

    rb = cfg['robot']
    if 'proprio_prefix_dim' not in rb:
        raise SystemExit(
            f"{path}: robot.proprio_prefix_dim is required (25 for the standard "
            f"ee/vel/finger/q/qdot/quat prefix)")
    expect = int(rb['proprio_prefix_dim']) + n_slots * slot_dim
    if int(rb['proprio_dim']) != expect:
        raise SystemExit(
            f"{path}: robot.proprio_dim is {rb['proprio_dim']} but the layout "
            f"gives {expect} ({rb['proprio_prefix_dim']} prefix + "
            f"{n_slots} slots x {slot_dim}). One trunk and one ObsRMS consume "
            f"this for every task, so it has to be exact.")

    # Every task must declare what it observes. A missing `state_slots` would
    # otherwise train a policy on proprioception alone — which looks like a
    # task that simply never learns, rather than like a config mistake.
    for name, t in cfg['tasks'].items():
        if not t.get('state_slots'):
            raise SystemExit(
                f"{path}: task '{name}' declares no state_slots. Every state "
                f"task must say which exact coordinates it feeds the policy; "
                f"slot kinds are {list(SLOT_KINDS)}")
        specs = parse_slot_specs(t)                  # validates kinds and shape
        if len(specs) > n_slots:
            raise SystemExit(
                f"{path}: task '{name}' declares {len(specs)} state_slots but "
                f"obs.n_slots is {n_slots}. Raising n_slots changes the "
                f"observation width and invalidates existing checkpoints.")
        for kind, sname in specs:
            if kind == 'joint' and 'joint_target' not in t:
                raise SystemExit(
                    f"{path}: task '{name}' has a {{joint: {sname}}} slot but no "
                    f"joint_target for it to report progress against")

    return cfg


# ═══════════════════════════════════════════════════════════════════════════
# CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════
def make_state_env(name, cfg, policy_forward, batch_size=None, substeps=None,
                   gamma=None):
    """Instantiate `name`'s state environment. Unset arguments read the config.

    `policy_forward` is called as (params, obs_norm, step_idx, seg_boundaries) —
    one argument shorter than the vision pipeline's, which also passes a list of
    per-camera feature maps. The two signatures are deliberately different: a
    vision forward handed to a state env fails at the call rather than silently
    consuming the observation as if it were pixels.
    """
    if name not in cfg['tasks']:
        raise SystemExit(
            f"unknown task {name!r}; choose from {sorted(cfg['tasks'])}")

    tcfg = cfg['tasks'][name]
    tr = cfg['training']
    cls = STATE_ENV_CLASSES[tcfg['env']]

    return cls(
        name=name, cfg=cfg, tcfg=tcfg,
        # No backbone and nothing rendered; img_size and chunk_size are read by
        # VisionWarpEnv's constructor, which the state envs inherit, and are
        # unused from there on.
        backbone=None,
        policy_forward=policy_forward, xml_path=envs.xml_path(cfg, name),
        batch_size=tr['batch'] if batch_size is None else batch_size,
        img_size=tr['img_size'],
        substeps=tr['substeps'] if substeps is None else substeps,
        chunk_size=tr['chunk_size'],
        gamma=tr['gamma'] if gamma is None else gamma,
    )
