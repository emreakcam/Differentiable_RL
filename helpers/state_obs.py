"""
Observation normalization for the state pipeline: RunningMeanStd + a variance
floor.
============================================================================
`helpers.obs.ObsRMS` normalises with `sqrt(var) + 1e-8`, which is fine when
every observation dimension actually varies.  The state observation has
dimensions that never do:

  - the padded object slots are constant zero for every task, and
  - a slot used by one task is constant zero for every other task in the run,
    because one ObsRMS is shared across all of them.

For those dims var → 0, so the divisor collapses to 1e-8 and anything in them —
solver noise, a single f32 rounding artefact — is amplified by 1e8 straight into
the trunk.  The vision trainer already warned about exactly this case ("A
variance floor in helpers/obs.py would be the actual fix"); this is that fix,
applied where it is needed rather than to the shared vision code.

The floor only ever *raises* a divisor, so a dimension that does vary normally
is untouched: at the default 1e-4 the floor binds below a standard deviation of
1 cm, and every real coordinate in these scenes moves further than that.
"""
import numpy as np
import jax.numpy as jnp

from helpers.obs import ObsRMS


class StateObsRMS(ObsRMS):
    """ObsRMS with a lower bound on the per-dimension variance."""

    def __init__(self, shape, var_floor=1e-4):
        super().__init__(shape)
        self.var_floor = float(var_floor)

    def get_jnp(self):
        """JAX-compatible (mean, std), with the floor applied to std."""
        mean = jnp.array(self.mean, dtype=jnp.float32)
        var = np.maximum(self.var, self.var_floor)
        std = jnp.sqrt(jnp.array(var, dtype=jnp.float32)) + 1e-8
        return mean, std

    def floored_dims(self):
        """Which dimensions the floor is currently binding on.

        Reported once after warm-up: on a normal run these are exactly the
        padded and cross-task-unused slots, and anything else in the list means
        a coordinate the policy is being fed is not actually moving.
        """
        return np.flatnonzero(self.var < self.var_floor)
