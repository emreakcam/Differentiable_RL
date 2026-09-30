"""
Vision Policy — Projection + Spatial Softmax + Trunk + Heads (JAX)
===================================================================
DINOv3 feature'larını alıp action üretir.
Depth CNN pipeline ile aynı trunk+head yapısı.

Mimari:
    DINOv3 patches [H, W, 768] (frozen, numpy→jnp)
        ↓
    Projection: [H,W,768] → [H,W,proj_dim]
        ↓
    Depth concat: [H,W,proj_dim] + [H,W,1] → [H,W,proj_dim+1]
        ↓
    Spatial Softmax → [2 × (proj_dim+1)] per camera → concat → vision_flat
        ↓
    Trunk: [vision_flat ∥ proprio] → 128 → 128  (shared)
        ↓
    Head: 128 → 32 → 8  (per segment)
"""

import jax
import jax.numpy as jnp

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════
N_JOINTS = 7
FINGER_OPEN = 0.04
VEL_LIMITS = jnp.array([2.17, 2.17, 2.17, 2.17, 2.61, 2.61, 2.61])
ACT_DIM = N_JOINTS + 1

SEG_NAMES = ['reach', 'descend', 'grasp', 'move', 'release']
N_SEGMENTS = 5

# Holding biases are -3.0: f = sigmoid(bias)*FINGER_OPEN gives ~1.9 mm, well
# inside any object, while leaving the sigmoid in a region where the head can
# still learn to modulate the finger. -5.0 (~0.3 mm) grips no harder in
# practice but pushes the output ~7x further into saturation.
FINGER_BIASES = {
    'reach':    2.0,
    'descend':  2.0,
    'grasp':   -2.0,   # was 0.0 (~20 mm gap — barely gripped)
    'move':    -2.0,   # keep the same closure while carrying
    'release':  2.0,
}

TRUNK_HIDDEN = 256
HEAD_HIDDEN = 64


# ═══════════════════════════════════════════════════════════════════
# SPATIAL SOFTMAX
# ═══════════════════════════════════════════════════════════════════
def spatial_softmax(feature_map, temperature=0.1):
    """[H, W, C] → [2C] — per-channel expected (x, y)."""
    H, W, C = feature_map.shape
    ys = jnp.linspace(-1, 1, H)
    xs = jnp.linspace(-1, 1, W)
    grid_x, grid_y = jnp.meshgrid(xs, ys)

    flat = feature_map.reshape(-1, C)
    weights = jax.nn.softmax(flat / temperature, axis=0).reshape(H, W, C)

    exp_x = jnp.sum(grid_x[:, :, None] * weights, axis=(0, 1))
    exp_y = jnp.sum(grid_y[:, :, None] * weights, axis=(0, 1))

    return jnp.concatenate([exp_x, exp_y])


# ═══════════════════════════════════════════════════════════════════
# PROJECTION: 768 → proj_dim
# ═══════════════════════════════════════════════════════════════════
def init_projection(key, in_dim=768, out_dim=64):
    k1, k2 = jax.random.split(key)
    return {
        'w1': (jax.random.normal(k1, (in_dim, out_dim)) *
               jnp.sqrt(2.0 / in_dim)).astype(jnp.float32),
        'b1': jnp.zeros(out_dim, dtype=jnp.float32),
    }


def project_features(params, feature_map):
    """[H, W, 768] → [H, W, proj_dim]."""
    H, W, _ = feature_map.shape
    flat = feature_map.reshape(-1, feature_map.shape[-1])
    proj = jax.nn.gelu(flat @ params['w1'] + params['b1'])
    return proj.reshape(H, W, -1)


# ═══════════════════════════════════════════════════════════════════
# TRUNK (shared) — depth pipeline ile aynı yapı
# ═══════════════════════════════════════════════════════════════════
def init_trunk(key, input_dim, hidden=None, vision_dim=None,
               vision_share=0.5):
    """`hidden` defaults to TRUNK_HIDDEN; the primitive policy passes it from
    configs/tasks.yaml so the width is configurable without editing this file.

    `ln_g`/`ln_b` are a LayerNorm over the trunk's input.  The input is the
    vision code concatenated with ObsRMS-normalised proprio, and the two blocks
    do not arrive on the same scale: the spatial-softmax coordinates' magnitude
    is set by the softmax temperature, so at T=1.0 they averaged out to ~0.05
    against proprio's unit variance and contributed ~6% of the input's total
    variance -- the trunk was reading almost nothing but proprio.  Normalising
    the input makes the trunk indifferent to that magnitude.

    `vln_g`/`vln_b` normalise the vision block *alone*, before the concat.
    The joint `ln_g` above cannot fix a per-block imbalance: it divides both
    blocks by one shared std, which is dominated by whichever block carries
    more variance, so a vision code at ~0.05 against unit-variance proprio
    stays ~20x smaller afterwards.  Proprio already has its own normaliser
    (ObsRMS); this gives vision the equivalent, so both blocks reach the trunk
    at unit per-dimension variance whatever the softmax temperature is.
    Omitted when `vision_dim` is None, which is how the state-only policy --
    it has no vision block -- keeps the old parameter set.

    `vision_share` is the fraction of the trunk input's total variance the
    vision block starts with.  Unit per-dimension variance is not an even
    split: vision has `vision_dim` dimensions against proprio's, so at 128 vs
    31 it would carry 80% of the total simply by being wider.  `vln_g` is
    therefore initialised to the gain that hits the requested share rather than
    to ones.  It stays a trained parameter, so this only sets where training
    starts from -- the network moves the balance wherever it needs it.
    """
    hidden = TRUNK_HIDDEN if hidden is None else hidden
    k1, k2 = jax.random.split(key)
    if vision_dim is None:
        vln = {}
    else:
        proprio_dim = input_dim - vision_dim
        # per-dim var v for vision gives total v*vision_dim; proprio arrives at
        # ~unit variance for total proprio_dim.  Solve v*nv/(v*nv+np) = share.
        tot = vision_share * proprio_dim / max(1.0 - vision_share, 1e-6)
        gain = float(jnp.sqrt(tot / max(vision_dim, 1)))
        vln = {
            'vln_g': jnp.full(vision_dim, gain, dtype=jnp.float32),
            'vln_b': jnp.zeros(vision_dim, dtype=jnp.float32),
        }
    return {
        **vln,
        'ln_g': jnp.ones(input_dim, dtype=jnp.float32),
        'ln_b': jnp.zeros(input_dim, dtype=jnp.float32),
        'w1': (jax.random.normal(k1, (input_dim, hidden)) *
               jnp.sqrt(2.0 / input_dim)).astype(jnp.float32),
        'b1': jnp.zeros(hidden, dtype=jnp.float32),
        'w2': (jax.random.normal(k2, (hidden, hidden)) *
               jnp.sqrt(2.0 / hidden)).astype(jnp.float32),
        'b2': jnp.zeros(hidden, dtype=jnp.float32),
    }


def layer_norm(x, gain, bias, eps=1e-5):
    """Normalise over the feature axis, then rescale."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps) * gain + bias


def norm_vision(trunk_params, vision_flat):
    """LayerNorm the vision block on its own, before it meets proprio.

    A no-op for parameter sets built without `vision_dim`, so the state-only
    policy and any checkpoint predating this call through unchanged.
    """
    if 'vln_g' not in trunk_params:
        return vision_flat
    return layer_norm(vision_flat, trunk_params['vln_g'], trunk_params['vln_b'])


def trunk_forward(params, inp):
    """LayerNorm -> GELU -> GELU.

    GELU rather than tanh: measured on 760 rollout states, the input carried
    rank90 124 and the tanh trunk passed only 86 of those directions to the
    heads, against 112 for GELU.  tanh sheds rank by squashing, not by
    saturating -- under 4% of its units were past |z|>2 -- so widening the
    trunk would not have recovered it.

    There is no LayerNorm on the output.  One was tried: it measured +14 head
    rank at init, but over a 999-iteration run its gain moved 0.84% against the
    input norm's 1.91% and carried almost no spread, so the network found no
    use for it.  It also discards the per-sample magnitude of the trunk output,
    which for a control policy may itself be informative, and with GELU heads
    there is no saturation left for it to prevent.
    """
    x = inp.astype(jnp.float32)
    x = layer_norm(x, params['ln_g'], params['ln_b'])
    x = jax.nn.gelu(x @ params['w1'] + params['b1'])
    return jax.nn.gelu(x @ params['w2'] + params['b2'])


# ═══════════════════════════════════════════════════════════════════
# HEAD (per-segment) — depth pipeline ile aynı yapı
# ═══════════════════════════════════════════════════════════════════
def init_head(key, finger_bias=0.0, trunk_hidden=None, hidden=None,
              n_layers=1):
    """`n_layers` GELU hidden layers of width `hidden`, then the action readout.

    Hidden layers live in a list so depth is a config value rather than a code
    change; head_forward walks whatever is there, and heads may differ in both
    width and depth without special handling downstream.

    Width is deliberately modest.  Measured over 11 trained heads, rank90 ran
    13-47 against widths of 64-448, and among heads fed by exactly one segment
    -- identical workload, widths 64 to 160 -- width correlated with rank at
    +0.13.  Width buys nothing past a small multiple of the rank actually used,
    so extra capacity is better spent on depth.
    """
    trunk_hidden = TRUNK_HIDDEN if trunk_hidden is None else trunk_hidden
    hidden = HEAD_HIDDEN if hidden is None else hidden
    n_layers = max(1, int(n_layers))
    keys = jax.random.split(key, n_layers + 1)
    layers, n_in = [], trunk_hidden
    for i in range(n_layers):
        layers.append({
            'w': (jax.random.normal(keys[i], (n_in, hidden)) *
                  jnp.sqrt(2.0 / n_in)).astype(jnp.float32),
            'b': jnp.zeros(hidden, dtype=jnp.float32),
        })
        n_in = hidden
    return {
        'hidden': layers,
        'w4': (jax.random.normal(keys[-1], (hidden, ACT_DIM)) * 0.01
               ).astype(jnp.float32),
        'b4': jnp.array([0.0] * N_JOINTS + [finger_bias],
                        dtype=jnp.float32),
    }


def head_forward(params, trunk_features):
    """GELU hidden layer, then the bounded action readout.

    GELU for the same reason as the trunk: measured on 760 rollout states, a
    tanh hidden layer passed rank 119 against GELU's 136 off the same
    (LayerNormed) trunk output.  It is not a saturation fix -- pre-activations
    average 0.97 and no output logit came near saturating -- purely rank.

    The two output squashes below are NOT interchangeable with GELU: they are
    the action parameterisation, bounding joint velocity to VEL_LIMITS and the
    finger to its travel.
    """
    x = trunk_features.astype(jnp.float32)
    for layer in params['hidden']:
        x = jax.nn.gelu(x @ layer['w'] + layer['b'])
    x = x @ params['w4'] + params['b4']
    qd = jnp.tanh(x[:N_JOINTS]) * VEL_LIMITS
    f = jax.nn.sigmoid(x[N_JOINTS]) * FINGER_OPEN
    return qd, f


# ═══════════════════════════════════════════════════════════════════
# FULL POLICY
# ═══════════════════════════════════════════════════════════════════
def init_vision_policy(key, n_cameras=3, proj_dim=64, proprio_dim=31,
                       n_segments=None, finger_biases=None):
    """Projection + Trunk + one Head per segment (at least N_SEGMENTS).

    Tasks with more than N_SEGMENTS segments (peg_insertion=7, container=15)
    must pass n_segments, otherwise vision_policy_forward indexes past the end
    of the head list. Tasks with fewer keep the historical 5 heads so their
    existing seg4 overrides stay valid.

    finger_biases: optional per-head list; falls back to FINGER_BIASES[SEG_NAMES]
    for the first five and 0.0 beyond that.
    """
    n_heads = max(N_SEGMENTS, n_segments or 0)
    keys = jax.random.split(key, n_heads + 2)

    sm_channels = proj_dim  # proj + depth
    vision_dim = n_cameras * (2 * sm_channels)  # 3 × 130 = 390

    params = {
        'projection': init_projection(keys[0], in_dim=768, out_dim=proj_dim),
        'trunk': init_trunk(keys[1], vision_dim + proprio_dim,
                            vision_dim=vision_dim),
    }
    for i in range(n_heads):
        if finger_biases is not None:
            fb = finger_biases[i]
        elif i < len(SEG_NAMES):
            fb = FINGER_BIASES[SEG_NAMES[i]]
        else:
            fb = 0.0
        params[f'seg{i}'] = init_head(keys[i + 2], finger_bias=fb)

    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"\n  Vision Policy:")
    print(f"    Projection: 768 → {proj_dim}")
    print(f"    Spatial softmax: {sm_channels}ch → {2*sm_channels} per cam")
    print(f"    Vision input: {vision_dim} ({n_cameras} cameras)")
    print(f"    Trunk: {vision_dim + proprio_dim} → {TRUNK_HIDDEN} → {TRUNK_HIDDEN}")
    print(f"    Heads: {TRUNK_HIDDEN} → {HEAD_HIDDEN} → {ACT_DIM} × {n_heads}")
    print(f"    Total trainable: {n_params}")

    return params


def vision_policy_forward(params, vision_features_list,
                          proprio_norm, step_idx, seg_boundaries,
                          temperature=1.0):
    """
    DINOv3 features + depth + proprio → action.

    Args:
        params: policy params dict
        vision_features_list: list of (H, W, 768) — per camera
        depth_list: list of (H, W, 1) — per camera
        proprio_norm: (proprio_dim,) — normalized proprioception
        step_idx: scalar — current step
        seg_boundaries: list of ints — segment boundaries
        temperature: spatial softmax temperature

    Returns:
        qd: (7,) joint velocities
        f: scalar finger target
    """
    vision_parts = []
    for feat_map in vision_features_list:
        feat_map = feat_map.astype(jnp.float32)
        projected = project_features(params['projection'], feat_map)
        keypoints = spatial_softmax(projected, temperature)
        vision_parts.append(keypoints)

    vision_flat = norm_vision(params['trunk'], jnp.concatenate(vision_parts))

    # Trunk
    inp = jnp.concatenate([vision_flat, proprio_norm])
    trunk_features = trunk_forward(params['trunk'], inp)

    # Heads — nested where. The head count comes from params, not N_SEGMENTS,
    # so tasks with more than five segments work; min() keeps the historical
    # behaviour for tasks with fewer heads than boundaries.
    n_heads = sum(1 for k in params if k.startswith('seg'))
    results = [head_forward(params[f'seg{i}'], trunk_features)
               for i in range(n_heads)]

    qd = results[-1][0]
    f = results[-1][1]
    for i in range(len(seg_boundaries) - 1, -1, -1):
        j = min(i, n_heads - 1)
        qd = jnp.where(step_idx < seg_boundaries[i], results[j][0], qd)
        f = jnp.where(step_idx < seg_boundaries[i], results[j][1], f)
    return qd, f