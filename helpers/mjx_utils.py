"""
MJX utility fonksiyonları:
  - Solver monkey-patch (while_loop → fori_loop)
  - shaped_distance: ödül şekillendirme fonksiyonu
  - safe_action: eklem limitlerinde güvenli aksiyon
"""
import jax
import jax.numpy as jnp
import mujoco.mjx._src.solver as _mjx_solver

# ---------------------------------------------------------------------------
# Solver monkey-patch: while_loop → fori_loop (reverse-mode AD uyumu)
# ---------------------------------------------------------------------------
_original_solve = _mjx_solver.solve

def _patched_solve(m, d):
    original_while_loop = jax.lax.while_loop
    def fixed_iter_while_loop(cond_fun, body_fun, init_val):
        def fori_body(_, carry):
            return body_fun(carry)
        return jax.lax.fori_loop(0, m.opt.iterations, fori_body, init_val)
    jax.lax.while_loop = fixed_iter_while_loop
    try:
        result = _original_solve(m, d)
    finally:
        jax.lax.while_loop = original_while_loop
    return result

def patch_solver():
    """Solver'ı patch'le, birden fazla çağrıda güvenli."""
    _mjx_solver.solve = _patched_solve
    print("✓ MJX solver patched (while_loop → fori_loop)")


# ---------------------------------------------------------------------------
# Shaped distance — ödül şekillendirme
# ---------------------------------------------------------------------------
def shaped_distance(a, b, s, t):
    """
    Mesafeye dayalı şekillendirilmiş ödül.
    s: scale parametresi, t: eşik (altında 1.0 döner).
    """
    dist = jnp.sqrt(jnp.dot(a - b, a - b) + 1e-6)
    scale = 1.8318 / jnp.maximum(s, 1e-6)
    shaped = (1.0 - jnp.tanh(dist * scale)) ** 2
    return jnp.where(dist < t, 1.0, shaped)


# ---------------------------------------------------------------------------
# Safe action — eklem limitlerinde güvenli velocity komutu
# ---------------------------------------------------------------------------
def safe_action(q_dot_target, current_q, q_lo, q_hi, margin=0.01):
    """Eklem limitlerine yakınken velocity'yi sınırla."""
    at_min = (current_q <= q_lo + margin)
    at_max = (current_q >= q_hi - margin)
    safe = jnp.where(at_min, jnp.maximum(0, q_dot_target), q_dot_target)
    safe = jnp.where(at_max, jnp.minimum(0, safe), safe)
    return safe