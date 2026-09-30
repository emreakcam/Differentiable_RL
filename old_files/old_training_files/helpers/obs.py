"""
Observation normalization: RunningMeanStd.
Should only be applied to proprioceptive dimensions (not to CNN latent variables or embeddings).
"""
import numpy as np
import jax.numpy as jnp


class ObsRMS:
    """Welford online mean/variance — batch update destekli."""

    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, batch):
        """batch: (N, obs_dim) numpy array."""
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        self.var = (self.var * self.count + batch_var * batch_count +
                    delta**2 * self.count * batch_count / total) / total
        self.count = total

    def get_jnp(self):
        """JAX-uyumlu mean, std döndür."""
        mean = jnp.array(self.mean, dtype=jnp.float32)
        std = jnp.sqrt(jnp.array(self.var, dtype=jnp.float32)) + 1e-8
        return mean, std