"""
Convert a PPO (torch) checkpoint into the JAX parameter tree the eval expects.
=============================================================================
The PPO arm trains a PyTorch network; the DiffRL arm trains the JAX one; and the
comparison is only worth anything if both are scored by the SAME eval, with the
same criteria, the same success_fn and the same observation scaling. So rather
than writing a second evaluator, this converts the trained torch actor back into
the JAX layout and writes it as a normal `*_params.pkl` / `*_obs_rms.pkl` pair.

Eval/eval_primitives_state.py then reads it like any other run.

Only the ACTOR converts. The critic has no DiffRL counterpart and plays no part
in evaluation — a deterministic rollout needs the mean action and nothing else.
The Gaussian's log_std is likewise dropped: evaluation is deterministic in both
arms, so the mean is the policy.

    python ppo_baseline/ppo_to_jax.py ppo_runs/<run>/model_4000.pt \\
        --out "train_robosuit/Plots and Checkpoints/State/PPO cube stacking"
"""
import sys, os, argparse, pickle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch

import envs
from envs.state_registry import load_state_config


def torch_to_jax_params(sd, prim_names, head_layers):
    """actor state_dict -> the dict init_state_policy produces.

    torch Linear stores (out, in) and JAX stores (in, out), so every weight is
    transposed. Names map as:

        mlp.ln.{weight,bias}          -> trunk.{ln_g,ln_b}
        mlp.trunk.{0,2}.{weight,bias} -> trunk.{w1,b1},{w2,b2}
        mlp.heads.<i>.<2k>.{w,b}      -> <prim>.hidden[k].{w,b}
        mlp.heads.<i>.<last>.{w,b}    -> <prim>.{w4,b4}
    """
    def T(k):
        return np.asarray(sd[k].detach().cpu().numpy().T, dtype=np.float32)

    def V(k):
        return np.asarray(sd[k].detach().cpu().numpy(), dtype=np.float32)

    params = {'trunk': {
        'ln_g': V('mlp.ln.weight'), 'ln_b': V('mlp.ln.bias'),
        'w1': T('mlp.trunk.0.weight'), 'b1': V('mlp.trunk.0.bias'),
        'w2': T('mlp.trunk.2.weight'), 'b2': V('mlp.trunk.2.bias'),
    }}
    for i, name in enumerate(prim_names):
        hidden = [{'w': T(f'mlp.heads.{i}.{2*k}.weight'),
                   'b': V(f'mlp.heads.{i}.{2*k}.bias')}
                  for k in range(head_layers)]
        last = 2 * head_layers
        params[name] = {
            'hidden': hidden,
            'w4': T(f'mlp.heads.{i}.{last}.weight'),
            'b4': V(f'mlp.heads.{i}.{last}.bias'),
        }
    return params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="ppo_runs/<run>/model_N.pt")
    p.add_argument("--out", required=True,
                   help="run folder to write into; created if absent")
    p.add_argument("--prefix", default=None,
                   help="filename prefix (default: the checkpoint's stem)")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_state_config(args.config)
    prim_names = envs.primitive_names(cfg)
    head_layers = envs.policy_cfg(cfg).get('head_layers', 1)

    d = torch.load(args.ckpt, weights_only=False, map_location='cpu')
    params = torch_to_jax_params(d['actor_state_dict'], prim_names, head_layers)

    os.makedirs(args.out, exist_ok=True)
    stem = args.prefix or os.path.splitext(os.path.basename(args.ckpt))[0]
    pp = os.path.join(args.out, f"{stem}_params.pkl")
    with open(pp, "wb") as f:
        pickle.dump(params, f)

    # ObsRMS comes from the sidecar the resumable runner writes. Without it the
    # eval would normalise with fresh statistics while the policy was trained
    # against accumulated ones — the same silent mismatch --resume guards
    # against, and it would quietly understate the policy.
    side = args.ckpt.replace(".pt", "_env.pkl")
    op = os.path.join(args.out, f"{stem}_obs_rms.pkl")
    if os.path.exists(side):
        with open(side, "rb") as f:
            env_state = pickle.load(f)
        with open(op, "wb") as f:
            pickle.dump(env_state['obs_rms'], f)
        print(f"  obs_rms from {os.path.basename(side)} "
              f"(count={env_state['obs_rms']['count']:.0f}, "
              f"phase reached {env_state['phase']})")
    else:
        raise SystemExit(
            f"no sidecar at {side}. The observation statistics the policy "
            f"trained against are not in the .pt, and evaluating without them "
            f"scores the policy on inputs it never saw. Re-run training with "
            f"the current train_ppo.py, which writes the sidecar beside every "
            f"checkpoint.")

    n = sum(v.size for d_ in params.values()
            for v in (d_.values() if isinstance(d_, dict) else [])
            if hasattr(v, 'size'))
    print(f"  wrote {pp}")
    print(f"        {op}")
    print(f"  {len(params) - 1} primitive heads converted")
    print(f"\n  evaluate with:\n"
          f"    python Eval/eval_primitives_state.py \\\n"
          f"        --ckpt-root {os.path.dirname(args.out)!r} \\\n"
          f"        --run {os.path.basename(args.out)!r} --tasks cube_stacking")


if __name__ == "__main__":
    main()
