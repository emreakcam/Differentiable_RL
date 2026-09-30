"""
Assert the PyTorch port matches the JAX policy numerically.
===========================================================
The PPO arm runs a PyTorch network and the DiffRL arm a JAX one.  If those two
differ, every result that follows is unattributable: a gap could be the learning
rule or it could be the port.  This script copies one policy's weights into the
other and compares outputs on identical inputs, so the claim "same model" is a
measurement.

It compares the network up to but NOT including the action squash: the JAX
head_forward ends with tanh*VEL_LIMITS / sigmoid*FINGER_OPEN, while the torch
side leaves that to the environment (see torch_policy's module docstring).  The
pre-squash logits are the whole network, so agreeing there means agreeing
everywhere the squash is applied identically.

    python ppo_baseline/verify_policy_match.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
import jax
import jax.numpy as jnp

import envs
from envs.state_registry import load_state_config
from models.state_policy import init_state_policy
from models.vision_policy import N_JOINTS
from ppo_baseline.torch_policy import TrunkHeadNet

TASK = 'cube_stacking'
TOL = 1e-5


def jax_logits(params, prim_names, obs_norm, prim_idx):
    """The JAX network up to the squash: trunk, then the selected head."""
    from models.vision_policy import layer_norm
    t = params['trunk']
    x = layer_norm(obs_norm.astype(jnp.float32), t['ln_g'], t['ln_b'])
    x = jax.nn.gelu(x @ t['w1'] + t['b1'])
    feat = jax.nn.gelu(x @ t['w2'] + t['b2'])
    outs = []
    for name in prim_names:
        h = params[name]
        y = feat
        for layer in h['hidden']:
            y = jax.nn.gelu(y @ layer['w'] + layer['b'])
        outs.append(y @ h['w4'] + h['b4'])
    return jnp.stack(outs)[prim_idx]


def main():
    cfg = load_state_config()
    prim_names = envs.primitive_names(cfg)
    fbs = envs.finger_biases(cfg)
    POL = envs.policy_cfg(cfg)
    obs_dim = cfg['robot']['proprio_dim']
    widths = envs.head_widths(cfg)
    head_layers = POL.get('head_layers', 1)
    trunk_hidden = POL['trunk_hidden']

    jp = init_state_policy(jax.random.PRNGKey(0), prim_names, fbs,
                           obs_dim=obs_dim, trunk_hidden=trunk_hidden,
                           head_widths=widths, head_layers=head_layers)

    net = TrunkHeadNet(
        obs_dim=obs_dim, n_prims=len(prim_names), trunk_hidden=trunk_hidden,
        head_widths=[widths[p] for p in prim_names], head_layers=head_layers,
        finger_biases=[fbs[p] for p in prim_names],
        act_dim=N_JOINTS + 1, n_joints=N_JOINTS).double()

    # ── copy JAX weights into torch (JAX is x @ W, torch is x @ W.T) ──
    with torch.no_grad():
        t = jp['trunk']
        net.ln.weight.copy_(torch.tensor(np.asarray(t['ln_g'], np.float64)))
        net.ln.bias.copy_(torch.tensor(np.asarray(t['ln_b'], np.float64)))
        for i, k in enumerate(('w1', 'w2')):
            lin = net.trunk[i * 2]
            lin.weight.copy_(torch.tensor(np.asarray(t[k], np.float64)).T)
            lin.bias.copy_(torch.tensor(
                np.asarray(t[f'b{i+1}'], np.float64)))
        for hi, name in enumerate(prim_names):
            h, head = jp[name], net.heads[hi]
            for li, layer in enumerate(h['hidden']):
                lin = head[li * 2]
                lin.weight.copy_(torch.tensor(
                    np.asarray(layer['w'], np.float64)).T)
                lin.bias.copy_(torch.tensor(
                    np.asarray(layer['b'], np.float64)))
            out = head[-1]
            out.weight.copy_(torch.tensor(np.asarray(h['w4'], np.float64)).T)
            out.bias.copy_(torch.tensor(np.asarray(h['b4'], np.float64)))

    # ── compare on random observations, every primitive ──
    rng = np.random.default_rng(0)
    worst, worst_p = 0.0, None
    for pi, pname in enumerate(prim_names):
        obs = rng.standard_normal((8, obs_dim))
        oh = np.zeros((8, len(prim_names))); oh[:, pi] = 1.0
        tj = np.stack([np.asarray(jax_logits(jp, prim_names,
                                             jnp.asarray(o, jnp.float32), pi),
                                  np.float64) for o in obs])
        tt = net(torch.tensor(np.concatenate([obs, oh], -1))).detach().numpy()
        d = np.abs(tj - tt).max()
        if d > worst:
            worst, worst_p = d, pname
        print(f"  {pname:14s} max|jax - torch| = {d:.3e}")

    print("-" * 56)
    print(f"  worst: {worst:.3e} on {worst_p}   (tolerance {TOL:.0e})")
    ok = worst < TOL
    print("  VERDICT:", "MATCH — the two arms share one network"
          if ok else "MISMATCH — the port differs, results not comparable")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
