"""
Primitive Policy — shared Trunk + one Head per primitive.
=========================================================
Same architecture as models/vision_policy.py (Projection 768→proj_dim →
SpatialSoftmax → shared Trunk → Head) and built from its layers, but the heads
are keyed by *primitive name* instead of by positional segment index.  Every
task that performs a `reach` therefore drives the same `reach` head, and the
heads accumulate gradient across every task that uses them.

The vocabulary is not defined here.  Primitive names, their order and their
finger biases come from `configs/tasks.yaml` and are passed in, so adding a
primitive is a config edit:

    from envs import load_config, primitive_names, finger_biases
    cfg = load_config()
    names = primitive_names(cfg)
    params = init_primitive_policy(key, names, finger_biases(cfg))
    forward = make_primitive_forward(prim_seq(cfg, 'drawer_open'), names,
                                     policy_cfg(cfg)['temperature'])

Order matters: it fixes the head order in the parameter tree, so appending to
the vocabulary is safe but reordering invalidates existing checkpoints.
"""
import jax
import jax.numpy as jnp

from models.vision_policy import (
    spatial_softmax, project_features, init_projection,
    init_trunk, trunk_forward, norm_vision, init_head, head_forward,
    layer_norm, N_JOINTS, VEL_LIMITS, FINGER_OPEN,
    TRUNK_HIDDEN, HEAD_HIDDEN,
)

SHARED_KEYS = ('projection', 'trunk')


def init_primitive_policy(key, prim_names, finger_biases, proj_dim=64,
                          proprio_dim=31, n_cameras=1, trunk_hidden=None,
                          head_widths=None, head_layers=1):
    """Shared Projection + Trunk, one Head per primitive in `prim_names`.

    `head_widths` maps each primitive to its own hidden width, so a head that
    drives many segments gets more capacity than one that drives a single
    segment. Omit it and every head falls back to the module default.

    `head_layers` sets how many hidden layers each head has.  Measured head
    rank never exceeded 47 against widths up to 448, and width did not predict
    rank among heads of equal workload, so capacity added as depth is more
    likely to be used than capacity added as width.
    """
    trunk_hidden = TRUNK_HIDDEN if trunk_hidden is None else trunk_hidden
    widths = {p: (head_widths or {}).get(p, HEAD_HIDDEN) for p in prim_names}

    keys = jax.random.split(key, len(prim_names) + 2)
    vision_dim = n_cameras * (2 * proj_dim)   # spatial softmax → 2 coords/channel

    params = {
        'projection': init_projection(keys[0], in_dim=768, out_dim=proj_dim),
        'trunk': init_trunk(keys[1], vision_dim + proprio_dim,
                            hidden=trunk_hidden, vision_dim=vision_dim),
    }
    for i, name in enumerate(prim_names):
        params[name] = init_head(keys[i + 2], finger_bias=finger_biases[name],
                                 trunk_hidden=trunk_hidden, hidden=widths[name],
                                 n_layers=head_layers)

    n_params = sum(p.size for p in jax.tree.leaves(params))
    shared = sum(p.size for k in SHARED_KEYS for p in jax.tree.leaves(params[k]))
    print(f"\n  Primitive Policy:")
    print(f"    Projection: 768 → {proj_dim} ({n_cameras} camera) "
          f"→ spatial softmax → {vision_dim}")
    print(f"    Trunk: {vision_dim + proprio_dim} → {trunk_hidden} → "
          f"{trunk_hidden} (shared)")
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


def primitive_forward(params, prim_names, vision_features_list, proprio_norm,
                      prim_idx, temperature):
    """DINOv3 features + proprio → action, from the head named by prim_idx.

    Args:
        prim_names: the vocabulary, in parameter-tree order.
        vision_features_list: list of (H, W, 768) — one entry per camera.
        prim_idx: scalar int (traced) — index into prim_names.
        temperature: spatial-softmax sharpness, from policy.temperature.
            Required rather than defaulted: this used to carry a default that
            was passed on explicitly to spatial_softmax, so it silently
            overrode that function's own default and every run trained at 1.0
            regardless of what was configured.

    Every head is evaluated and the result selected with `where`, because
    prim_idx is a traced value: it is not known at trace time which head the
    step will use, so all of them have to be in the graph.
    """
    vision_parts = []
    for feat_map in vision_features_list:
        projected = project_features(params['projection'],
                                     feat_map.astype(jnp.float32))
        vision_parts.append(spatial_softmax(projected, temperature))
    vision_flat = norm_vision(params['trunk'], jnp.concatenate(vision_parts))

    inp = jnp.concatenate([vision_flat, proprio_norm])
    trunk_features = trunk_forward(params['trunk'], inp)

    results = [head_forward(params[name], trunk_features) for name in prim_names]

    qd, f = results[-1]
    for i in range(len(prim_names) - 2, -1, -1):
        qd = jnp.where(prim_idx == i, results[i][0], qd)
        f = jnp.where(prim_idx == i, results[i][1], f)
    return qd, f


def make_primitive_forward(prim_seq, prim_names, temperature):
    """Adapt primitive_forward to the environment's policy_forward signature.

    Returns a callable (params, vis_list, proprio_norm, step_idx,
    seg_boundaries) so a rollout can be driven by either this policy or the
    per-segment one without knowing which.  `prim_seq` maps the task's segments
    onto `prim_names`; the active segment is resolved from step_idx exactly as
    vision_policy_forward resolves its heads.

    `temperature` is bound here because the environment calls the returned
    function with exactly five positional arguments — there is no route for a
    caller to supply it per step, so a default here would be unoverridable.
    """
    prim_idx_of = {name: i for i, name in enumerate(prim_names)}
    prim_ids = [prim_idx_of[name] for name in prim_seq]

    def forward(params, vision_features_list, proprio_norm, step_idx,
                seg_boundaries):
        prim_idx = jnp.int32(prim_ids[-1])
        for i in range(len(seg_boundaries) - 1, -1, -1):
            prim_idx = jnp.where(step_idx < seg_boundaries[i],
                                 jnp.int32(prim_ids[i]), prim_idx)
        return primitive_forward(params, prim_names, vision_features_list,
                                 proprio_norm, prim_idx, temperature)

    return forward


# ═══════════════════════════════════════════════════════════════════
# ABLATION — one monolithic MLP instead of trunk + per-primitive heads
# ═══════════════════════════════════════════════════════════════════
# The thesis baseline factorises the policy into a shared trunk and one
# specialised head per primitive.  This variant removes that factorisation: a
# single MLP maps the observation straight to the action, and the primitive is
# supplied as a one-hot input instead of by selecting a head.
#
# The one-hot matters.  Without it the ablation would not be testing the
# architecture, it would be testing a network that cannot see which phase it is
# in — and `b4[7]` (the per-head finger bias, +2.0 open for reach, -2.0 closed
# for grasp) is a PER-HEAD parameter, so one output layer has exactly one finger
# bias and could not open for reach and close for grasp even in principle.
# Feeding the primitive keeps the information identical to the baseline's and
# leaves the architecture as the only difference.
#
# Everything upstream is shared with the baseline: the same projection, the same
# spatial softmax, the same vision LayerNorm, the same action parameterisation
# (tanh * VEL_LIMITS, sigmoid * FINGER_OPEN).  The MLP's width is the trunk's
# and its depth is the baseline's total (trunk layers + head layers), so the two
# are comparable in both.
def init_monolithic_policy(key, prim_names, proj_dim=64, proprio_dim=31,
                           n_cameras=1, trunk_hidden=None, n_layers=4):
    """One MLP: [vision | proprio | one-hot(primitive)] -> action.

    Parameters live under 'trunk' so the trainer's SHARED_KEYS, its optimizer
    and its checkpointing all work unchanged; there simply are no per-primitive
    head entries, so the per-head loops iterate over nothing.
    """
    trunk_hidden = TRUNK_HIDDEN if trunk_hidden is None else trunk_hidden
    n_prims = len(prim_names)
    vision_dim = n_cameras * (2 * proj_dim)
    in_dim = vision_dim + proprio_dim + n_prims

    keys = jax.random.split(key, n_layers + 2)
    layers = []
    d = in_dim
    for i in range(n_layers):
        layers.append({
            'w': (jax.random.normal(keys[i + 1], (d, trunk_hidden)) *
                  jnp.sqrt(2.0 / d)).astype(jnp.float32),
            'b': jnp.zeros(trunk_hidden, dtype=jnp.float32),
        })
        d = trunk_hidden

    mlp = {
        # LayerNorm on the input, as the trunk has. `vln_*` normalises the
        # vision block alone; the one-hot is left out of both, since a constant
        # 1.0 carries no scale to correct and normalising it would erase it.
        'ln_g': jnp.ones(in_dim, dtype=jnp.float32),
        'ln_b': jnp.zeros(in_dim, dtype=jnp.float32),
        'vln_g': jnp.full(vision_dim,
                          float(jnp.sqrt(proprio_dim / max(vision_dim, 1))),
                          dtype=jnp.float32),
        'vln_b': jnp.zeros(vision_dim, dtype=jnp.float32),
        'hidden': layers,
        'w_out': (jax.random.normal(keys[-1], (trunk_hidden, 8)) *
                  jnp.sqrt(2.0 / trunk_hidden)).astype(jnp.float32),
        # A single finger bias, left neutral: with the primitive one-hot on the
        # input the network can learn open-vs-closed itself, which is precisely
        # the capability the per-head biases hand the baseline for free.
        'b_out': jnp.zeros(8, dtype=jnp.float32),
    }
    params = {
        'projection': init_projection(keys[0], in_dim=768, out_dim=proj_dim),
        'trunk': mlp,
    }
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"\n  MONOLITHIC Policy (ablation — no trunk/head split):")
    print(f"    Projection: 768 → {proj_dim} ({n_cameras} camera) "
          f"→ spatial softmax → {vision_dim}")
    print(f"    MLP: {in_dim} ({vision_dim} vis + {proprio_dim} proprio + "
          f"{n_prims} one-hot) → "
          + " → ".join([str(trunk_hidden)] * n_layers) + " → 8")
    print(f"    No per-primitive heads; one output layer, one finger bias.")
    print(f"    Total trainable: {n_params:,}")
    return params


def monolithic_forward(params, prim_names, vision_features_list, proprio_norm,
                       prim_idx, temperature):
    """Same signature as primitive_forward, so the rollout cannot tell them
    apart. `prim_idx` becomes a one-hot input rather than a head selector."""
    vision_parts = []
    for feat_map in vision_features_list:
        projected = project_features(params['projection'],
                                     feat_map.astype(jnp.float32))
        vision_parts.append(spatial_softmax(projected, temperature))
    mlp = params['trunk']
    vision_flat = norm_vision(mlp, jnp.concatenate(vision_parts))

    onehot = jax.nn.one_hot(prim_idx, len(prim_names), dtype=jnp.float32)
    x = jnp.concatenate([vision_flat, proprio_norm, onehot]).astype(jnp.float32)
    x = layer_norm(x, mlp['ln_g'], mlp['ln_b'])
    for layer in mlp['hidden']:
        x = jax.nn.gelu(x @ layer['w'] + layer['b'])
    x = x @ mlp['w_out'] + mlp['b_out']
    qd = jnp.tanh(x[:N_JOINTS]) * VEL_LIMITS
    f = jax.nn.sigmoid(x[N_JOINTS]) * FINGER_OPEN
    return qd, f


def make_monolithic_forward(prim_seq, prim_names, temperature):
    """Monolithic twin of make_primitive_forward — identical call signature."""
    prim_ids = [prim_names.index(p) for p in prim_seq]

    def forward(params, vision_features_list, proprio_norm, step_idx,
                seg_boundaries):
        prim_idx = jnp.int32(prim_ids[-1])
        for i in range(len(seg_boundaries) - 1, -1, -1):
            prim_idx = jnp.where(step_idx < seg_boundaries[i],
                                 jnp.int32(prim_ids[i]), prim_idx)
        return monolithic_forward(params, prim_names, vision_features_list,
                                  proprio_norm, prim_idx, temperature)

    return forward
