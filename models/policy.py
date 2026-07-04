"""
Trunk + Head MLP politika ağı.

Mimari:
  Trunk:  obs → tanh(W1) → tanh(W2) → features (paylaşılan)
  Head:   features → tanh(W3) → W4 → [q_dot, finger]  (segment başına ayrı)

Trunk tüm segment'lerde paylaşılır, head'ler bağımsız eğitilir.
"""
import jax
import jax.numpy as jnp


def init_trunk(key, obs_dim, hidden_dim):
    """Paylaşılan trunk ağırlıklarını oluştur."""
    k1, k2 = jax.random.split(key)
    return {
        'w1': (jax.random.normal(k1, (obs_dim, hidden_dim)) *
               jnp.sqrt(2.0 / obs_dim)).astype(jnp.float32),
        'b1': jnp.zeros(hidden_dim, dtype=jnp.float32),
        'w2': (jax.random.normal(k2, (hidden_dim, hidden_dim)) *
               jnp.sqrt(2.0 / hidden_dim)).astype(jnp.float32),
        'b2': jnp.zeros(hidden_dim, dtype=jnp.float32),
    }


def init_head(key, hidden_trunk, hidden_head, act_dim, finger_bias=0.0):
    """Segment head ağırlıklarını oluştur."""
    k1, k2 = jax.random.split(key)
    return {
        'w3': (jax.random.normal(k1, (hidden_trunk, hidden_head)) *
               jnp.sqrt(2.0 / hidden_trunk)).astype(jnp.float32),
        'b3': jnp.zeros(hidden_head, dtype=jnp.float32),
        'w4': (jax.random.normal(k2, (hidden_head, act_dim)) * 0.01
               ).astype(jnp.float32),
        'b4': jnp.array([0.0] * (act_dim - 1) + [finger_bias],
                        dtype=jnp.float32),
    }


def init_all_mlps(key, obs_dim, hidden, act_dim,
                  n_segments=15, finger_biases=None):
    if finger_biases is None:
        finger_biases = [0.0] * n_segments
    keys = jax.random.split(key, n_segments)
    params = {}
    for i in range(n_segments):
        k1, k2, k3, k4 = jax.random.split(keys[i], 4)
        params[f'seg{i}'] = {
            'w1': (jax.random.normal(k1, (obs_dim, hidden)) * jnp.sqrt(2.0 / obs_dim)).astype(jnp.float32),
            'b1': jnp.zeros(hidden, dtype=jnp.float32),
            'w2': (jax.random.normal(k2, (hidden, hidden)) * jnp.sqrt(2.0 / hidden)).astype(jnp.float32),
            'b2': jnp.zeros(hidden, dtype=jnp.float32),
            'w3': (jax.random.normal(k3, (hidden, act_dim)) * 0.01).astype(jnp.float32),
            'b3': jnp.array([0.0] * (act_dim - 1) + [finger_biases[i]], dtype=jnp.float32),
        }
    return params


def trunk_forward(trunk_params, obs):
    """Trunk forward pass: obs → features."""
    obs32 = obs.astype(jnp.float32)
    x = jnp.tanh(obs32 @ trunk_params['w1'] + trunk_params['b1'])
    x = jnp.tanh(x @ trunk_params['w2'] + trunk_params['b2'])
    return x


def head_forward(trunk_features, head_params, n_joints, vel_limits,
                 finger_open):
    """Head forward pass: trunk_features → (q_dot_target, finger)."""
    tf32 = trunk_features.astype(jnp.float32)
    x = jnp.tanh(tf32 @ head_params['w3'] + head_params['b3'])
    x = x @ head_params['w4'] + head_params['b4']
    q_dot_target = jnp.tanh(x[:n_joints]) * vel_limits
    finger = jax.nn.sigmoid(x[n_joints]) * finger_open
    return q_dot_target, finger


def policy_forward(all_params, obs, step_idx, seg_boundaries, n_joints,
                   vel_limits, finger_open):
    def _single_mlp(p, obs):
        x = jnp.tanh(obs @ p['w1'] + p['b1'])
        x = jnp.tanh(x @ p['w2'] + p['b2'])
        x = x @ p['w3'] + p['b3']
        qd = jnp.tanh(x[:n_joints]) * vel_limits
        f = jax.nn.sigmoid(x[n_joints]) * finger_open
        return qd, f

    obs32 = obs.astype(jnp.float32)
    n_segs = len(seg_boundaries) + 1
    results = [_single_mlp(all_params[f'seg{i}'], obs32) for i in range(n_segs)]

    qd = results[-1][0]
    f = results[-1][1]
    for i in range(len(seg_boundaries) - 1, -1, -1):
        qd = jnp.where(step_idx < seg_boundaries[i], results[i][0], qd)
        f = jnp.where(step_idx < seg_boundaries[i], results[i][1], f)
    return qd, f