"""
XLA environment defaults — apply BEFORE `import jax`.
======================================================
These settings are read once, when the XLA backend initialises. Setting them
after JAX has started has no effect, which is why `apply_xla_defaults()` has to
be called above the `import jax` line rather than tucked in with the other
imports.

What they do, and why this project wants them:

  XLA_PYTHON_CLIENT_PREALLOCATE=false
      JAX otherwise claims ~75% of VRAM on its first allocation and keeps it.
      Three things share one GPU in these scripts — JAX/MJX physics, the MJWarp
      renderer, and PyTorch running DINOv3 — so preallocation starves the
      backbone and it OOMs before training starts.

  --xla_gpu_autotune_level=0
      Turns off compile-time benchmarking of GEMM/conv implementations. Saves
      substantial compile time, and avoids autotune scratch buffers competing
      for VRAM with the torch backbone. Costs some kernel throughput.

  --xla_gpu_force_compilation_parallelism=1
      Compiles single-threaded. Each compile thread holds its own LLVM state,
      so peak HOST RAM scales with thread count — and compiling the BPTT graphs
      for a full task sweep is what exhausts host memory, not VRAM.

Anything set in the shell wins: `XLA_PYTHON_CLIENT_PREALLOCATE` is only
defaulted if unset, and an individual XLA flag is only appended if that flag is
not already present in `XLA_FLAGS`. So the historical invocation

    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_autotune_level=0 \
        --xla_gpu_force_compilation_parallelism=1" python train_primitives_vison.py

still behaves identically — it is simply no longer necessary to type it.
To override just one, e.g. to re-enable autotuning:

    XLA_FLAGS="--xla_gpu_autotune_level=4" python train_primitives_vison.py
"""
import os

#: XLA_FLAGS entries, applied only when the flag is not already present
XLA_FLAG_DEFAULTS = {
    '--xla_gpu_autotune_level': '0',
    '--xla_gpu_force_compilation_parallelism': '1',
}

#: plain environment variables, applied only when unset
ENV_DEFAULTS = {
    'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
}


def _flag_names(xla_flags):
    """The flag names already present in an XLA_FLAGS string."""
    names = set()
    for token in xla_flags.split():
        if token.startswith('--'):
            names.add(token.split('=', 1)[0])
    return names


def apply_xla_defaults(verbose=True):
    """Set the XLA env defaults, leaving anything the shell already set alone.

    Returns the list of settings this call actually added, so a caller can see
    whether the environment or the defaults are in force.
    """
    added = []

    for key, value in ENV_DEFAULTS.items():
        if key not in os.environ:
            os.environ[key] = value
            added.append(f"{key}={value}")

    existing = os.environ.get('XLA_FLAGS', '')
    present = _flag_names(existing)
    new_flags = [f"{name}={value}"
                 for name, value in XLA_FLAG_DEFAULTS.items()
                 if name not in present]
    if new_flags:
        os.environ['XLA_FLAGS'] = ' '.join(
            ([existing] if existing else []) + new_flags)
        added.extend(new_flags)

    if verbose:
        inherited = [f"{k}={os.environ[k]}" for k in ENV_DEFAULTS
                     if f"{k}={os.environ[k]}" not in added]
        inherited += sorted(present & set(XLA_FLAG_DEFAULTS))
        if added:
            print(f"XLA defaults applied: {' '.join(added)}")
        if inherited:
            print(f"XLA settings from the environment (kept): "
                  f"{' '.join(inherited)}")

    return added
