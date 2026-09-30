"""
State Policy — shared Trunk + one Head per primitive.
=====================================================
The state-observation counterpart of models/primitive_policy.py.  Same shared
trunk, same per-primitive heads, same parameter-tree layout — minus the two
layers that exist only to turn pixels into numbers:

    primitive_policy   DINOv3 patches → Projection(768→proj_dim)
                       → SpatialSoftmax → ‖ proprio → Trunk → Head
    state_policy                                obs → Trunk → Head

`init_trunk`, `trunk_forward`, `init_head` and `head_forward` are imported from
models/vision_policy.py rather than copied, so the two branches cannot drift in
their layer definitions, weight initialisation or action decoding (tanh-scaled
joint velocities and a sigmoid finger target).

Heads are keyed by *primitive name*, exactly as in the vision pipeline, so every
task's `reach` drives the same `reach` head and the heads accumulate gradient
across every task that uses them.  The vocabulary comes from the config:

    from envs.state_registry import load_state_config
    from envs import primitive_names, finger_biases, prim_seq
    cfg = load_state_config()
    names = primitive_names(cfg)
    params = init_state_policy(key, names, finger_biases(cfg), obs_dim=43)
    forward = make_state_forward(prim_seq(cfg, 'drawer_open'), names)

Order matters: it fixes the head order in the parameter tree, so appending to
the vocabulary is safe but reordering invalidates existing checkpoints.
"""
import jax
import jax.numpy as jnp

from models.vision_policy import (
    init_trunk, trunk_forward, init_head, head_forward,
    TRUNK_HIDDEN, HEAD_HIDDEN,
)

#: What the shared optimizer owns. One entry, where the vision policy has two —
#: there is no projection to train.
STATE_SHARED_KEYS = ('trunk',)


def init_state_policy(key, prim_names, finger_biases, obs_dim,
                      trunk_hidden=None, head_widths=None, head_layers=1):
    """Shared Trunk over the observation, one Head per primitive.

    `obs_dim` is the full state observation (proprio prefix + object slots), so
    it is the trunk's input width directly — there is no vision block to
    concatenate.

    `head_widths` maps each primitive to its own hidden width, so a head that
    drives many segments gets more capacity than one that drives a single
    segment. Omit it and every head falls back to the module default.

    `head_layers` sets how many hidden layers each head has, matching the
    vision policy's knob so the two frameworks can be configured the same way.
    """
    trunk_hidden = TRUNK_HIDDEN if trunk_hidden is None else trunk_hidden
    widths = {p: (head_widths or {}).get(p, HEAD_HIDDEN) for p in prim_names}

    keys = jax.random.split(key, len(prim_names) + 1)
    params = {'trunk': init_trunk(keys[0], obs_dim, hidden=trunk_hidden)}
    for i, name in enumerate(prim_names):
        params[name] = init_head(keys[i + 1], finger_bias=finger_biases[name],
                                 trunk_hidden=trunk_hidden, hidden=widths[name],
                                 n_layers=head_layers)

    n_params = sum(p.size for p in jax.tree.leaves(params))
    shared = sum(p.size for k in STATE_SHARED_KEYS
                 for p in jax.tree.leaves(params[k]))
    print(f"\n  State Policy:")
    print(f"    Observation: {obs_dim} (no projection, no spatial softmax)")
    print(f"    Trunk: {obs_dim} → {trunk_hidden} → {trunk_hidden} (shared)")
    print(f"    Heads ({len(prim_names)}), {head_layers} hidden layer(s), "
          f"width by usage:")
    for name in prim_names:
        n = sum(p.size for p in jax.tree.leaves(params[name]))
        chain = " → ".join([str(trunk_hidden)]
                           + [str(widths[name])] * head_layers + ["8"])
        print(f"      {name:14s} {chain}   {n:>8,} params")
    print(f"    Shared: {shared:,}   Heads: {n_params - shared:,}   "
          f"Total trainable: {n_params:,}")
    return params


def state_forward(params, prim_names, obs_norm, prim_idx):
    """Normalised observation → action, from the head named by prim_idx.

    Every head is evaluated and the result selected with `where`, because
    prim_idx is a traced value: it is not known at trace time which head the
    step will use, so all of them have to be in the graph.
    """
    trunk_features = trunk_forward(params['trunk'], obs_norm)
    results = [head_forward(params[name], trunk_features) for name in prim_names]

    qd, f = results[-1]
    for i in range(len(prim_names) - 2, -1, -1):
        qd = jnp.where(prim_idx == i, results[i][0], qd)
        f = jnp.where(prim_idx == i, results[i][1], f)
    return qd, f


def make_state_forward(prim_seq, prim_names):
    """Adapt state_forward to the state environment's policy_forward signature.

    Returns a callable (params, obs_norm, step_idx, seg_boundaries).  One
    argument shorter than the vision pipeline's forward, which also takes the
    per-camera feature list — deliberately, so the two cannot be swapped by
    accident.  `prim_seq` maps the task's segments onto `prim_names`, and the
    active segment is resolved from step_idx exactly as it is there.
    """
    prim_idx_of = {name: i for i, name in enumerate(prim_names)}
    prim_ids = [prim_idx_of[name] for name in prim_seq]

    def forward(params, obs_norm, step_idx, seg_boundaries):
        prim_idx = jnp.int32(prim_ids[-1])
        for i in range(len(seg_boundaries) - 1, -1, -1):
            prim_idx = jnp.where(step_idx < seg_boundaries[i],
                                 jnp.int32(prim_ids[i]), prim_idx)
        return state_forward(params, prim_names, obs_norm, prim_idx)

    return forward
