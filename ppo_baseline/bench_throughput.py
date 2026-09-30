"""Steady-state env throughput vs batch size, after compilation.

PPO's cost is environment samples, and each `step()` here is a separate jitted
call dispatched from Python — unlike the DiffRL arm, where a whole episode is
one fused scan.  If per-call overhead dominates, samples/sec is set by dispatch
rather than by physics and the fix is a bigger batch.

Forward-only rollouts have no gradient tape, so the batch sizes that fit here
are far larger than the 20 the BPTT trainer is limited to.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import jax, jax.numpy as jnp, numpy as np, torch
import envs
from envs.state_registry import load_state_config, make_state_env
from models.state_policy import make_state_forward
from models.vision_policy import N_JOINTS, VEL_LIMITS, FINGER_OPEN
from helpers.state_obs import StateObsRMS
from ppo_baseline.rsl_vecenv import MJXStateVecEnv

TASK = 'cube_stacking'
BATCHES = [1536, 2048]
WARM, MEASURE = 10, 60


def bench(nb, cfg, names, seq):
    env = make_state_env(TASK, cfg, make_state_forward(seq, names),
                         batch_size=nb, substeps=cfg['training']['substeps'])
    t = env.build()
    rms = StateObsRMS(cfg['robot']['proprio_dim'],
                      var_floor=float(cfg['obs']['var_floor']))
    TR = cfg['training']
    ve = MJXStateVecEnv(t, rms, N_JOINTS, VEL_LIMITS, FINGER_OPEN,
                        [names.index(p) for p in seq], len(names),
                        TR['curriculum']['patience'],
                        TR['curriculum']['pass_frac'], device='cuda:0')
    a = torch.zeros(nb, ve.num_actions, device='cuda:0')
    t0 = time.time()
    for _ in range(WARM):
        ve.step(a)
    compile_s = time.time() - t0
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(MEASURE):
        ve.step(a)
    torch.cuda.synchronize()
    el = time.time() - t0
    import subprocess; vram = float(subprocess.check_output('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits',shell=True).decode())/1024
    return compile_s, MEASURE / el, nb * MEASURE / el, vram


def main():
    cfg = load_state_config()
    names = envs.primitive_names(cfg)
    seq = envs.prim_seq(cfg, TASK)
    print(f"\n{'batch':>7s}{'compile':>10s}{'steps/s':>10s}"
          f"{'env-steps/s':>14s}{'GPU GB':>9s}")
    print("-" * 52)
    for nb in BATCHES:
        try:
            c, sps, esps, vram = bench(nb, cfg, names, seq)
            print(f"{nb:>7d}{c:>9.0f}s{sps:>10.1f}{esps:>14,.0f}{vram:>9.2f}")
        except Exception as e:
            print(f"{nb:>7d}   FAILED: {type(e).__name__}: {str(e)[:40]}")
            break
    print("-" * 52)
    print("  steps/s flat across batch => dispatch-bound, raise the batch")
    print("  steps/s falling           => physics-bound, batch is saturated")


if __name__ == "__main__":
    main()
