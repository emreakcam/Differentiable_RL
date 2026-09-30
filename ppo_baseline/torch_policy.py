"""
PyTorch port of the JAX trunk + per-primitive-head policy, for the PPO baseline.
===============================================================================
The DiffRL trainer's policy lives in models/state_policy.py as plain JAX param
dicts.  rsl_rl is PyTorch, so the network has to exist on both sides — and that
port is the one place a silent difference between the two arms could hide.  The
architecture is therefore mirrored exactly, and verify_policy_match.py asserts
the two agree numerically on identical weights.

Two framework defaults differ and both are set explicitly here:

  GELU   jax.nn.gelu defaults to approximate=True (the tanh approximation);
         torch.nn.GELU defaults to the exact erf form.  Straight across they
         disagree by ~4e-4 per activation; with approximate='tanh' it is ~6e-8.
  init   the JAX side uses normal * sqrt(2/fan_in) with a 0.01-scaled readout;
         torch's default is kaiming_uniform.

One deliberate difference: the action squash.  In the JAX policy head_forward
ends with tanh*VEL_LIMITS and sigmoid*FINGER_OPEN, so the network emits a
bounded action.  PPO needs a distribution over what it samples, and a Gaussian
over a bounded quantity is wrong without a change-of-variables correction.  The
squash therefore moves into the environment (rsl_vecenv.make_ctrl), so the
network emits the raw 8 numbers, PPO puts a Gaussian over those, and the
obs -> ctrl mapping is unchanged end to end.
"""
import torch
import torch.nn as nn


def gelu():
    """The tanh approximation, matching jax.nn.gelu's default."""
    return nn.GELU(approximate='tanh')


def _linear(n_in, n_out, scale=None):
    """Linear layer initialised the way the JAX side initialises its weights.

    `scale=None` reproduces normal * sqrt(2/fan_in); an explicit scale
    reproduces the readout's normal * 0.01.
    """
    lin = nn.Linear(n_in, n_out)
    std = (2.0 / n_in) ** 0.5 if scale is None else scale
    nn.init.normal_(lin.weight, mean=0.0, std=std)
    nn.init.zeros_(lin.bias)
    return lin


class TrunkHeadNet(nn.Module):
    """obs -> LayerNorm -> trunk -> the head named by the primitive one-hot.

    Input is [observation | one-hot(primitive)], concatenated by the caller,
    because rsl_rl hands a model one flat latent and this is the only route for
    a per-step quantity to reach the network.  The one-hot selects the head as a
    weighted sum rather than an index: every head is evaluated and the one-hot
    picks one, which is what the JAX side's chain of `jnp.where` does.
    """

    def __init__(self, obs_dim, n_prims, trunk_hidden, head_widths,
                 head_layers, finger_biases, act_dim=8, n_joints=7):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_prims = n_prims
        self.n_joints = n_joints

        # LayerNorm over the trunk input, as init_trunk's ln_g/ln_b provide.
        self.ln = nn.LayerNorm(obs_dim, eps=1e-5)

        self.trunk = nn.Sequential(
            _linear(obs_dim, trunk_hidden), gelu(),
            _linear(trunk_hidden, trunk_hidden), gelu(),
        )

        heads = []
        for w, fb in zip(head_widths, finger_biases):
            layers, n_in = [], trunk_hidden
            for _ in range(head_layers):
                layers += [_linear(n_in, w), gelu()]
                n_in = w
            readout = _linear(n_in, act_dim, scale=0.01)
            # b4 = [0]*n_joints + [finger_bias]: the per-head gripper bias is a
            # parameter of the head, which is exactly why one monolithic output
            # layer cannot express open-for-reach and closed-for-grasp.
            with torch.no_grad():
                readout.bias[-1] = float(fb)
            layers.append(readout)
            heads.append(nn.Sequential(*layers))
        self.heads = nn.ModuleList(heads)

    def forward(self, x):
        obs, onehot = x[..., :self.obs_dim], x[..., self.obs_dim:]
        feat = self.trunk(self.ln(obs))
        # (batch, n_prims, act_dim) -> select with the one-hot
        stacked = torch.stack([h(feat) for h in self.heads], dim=-2)
        return (stacked * onehot.unsqueeze(-1)).sum(dim=-2)
