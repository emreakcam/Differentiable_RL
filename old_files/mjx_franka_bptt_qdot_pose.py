import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
import mujoco
from mujoco import mjx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# SOLVER MONKEY-PATCH
# ---------------------------------------------------------------------------
import mujoco.mjx._src.solver as _mjx_solver
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

_mjx_solver.solve = _patched_solve
print("✓ MJX solver patched")

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--substeps", type=int, default=8)
parser.add_argument("--iters", type=int, default=400)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--solver-iters", type=int, default=4)
parser.add_argument("--grad-clip", type=float, default=10.0)
parser.add_argument("--no-viewer", action="store_true")
parser.add_argument("--xml", type=str,
                    default="/home/emre/mjx_diffsim/franka_emika_panda/"
                            "mjx_single_cube.xml")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_STEPS     = args.steps
N_SUBSTEPS  = args.substeps
TRAIN_ITERS = args.iters
TRAIN_LR    = args.lr
GAMMA       = args.gamma
GRAD_CLIP   = args.grad_clip
N_JOINTS    = 7

EE_SITE_IDX   = 0
CUBE_BODY_IDX = 12
FINGER_OPEN   = 0.04
GRASP_Z_OFFSET = 0.0

Q_LO = jnp.array([-2.897, -1.763, -2.897, -3.072, -2.897, -0.018, -2.897])
Q_HI = jnp.array([ 2.897,  1.763,  2.897, -0.070,  2.897,  3.752,  2.897])
VEL_LIMITS = jnp.array([2.17, 2.17, 2.17, 2.17, 2.61, 2.61, 2.61])

TARGET_POS = jnp.array([0.4, 0.15, 0.55])
TARGET_QUAT = jnp.array([0.0, 0.7071068, 0.7071068, 0.0])

OBS_DIM = 36  # ee(3) + ee_to_target(3) + ee_vel(3) + finger(1) + joint_pos(7) + joint_vel(7) + quats(12)
ACT_DIM = N_JOINTS + 1
HIDDEN  = 64

APPROACH_S = 0.05
APPROACH_T = 0.005

# ---------------------------------------------------------------------------
# Shaped distance function
# ---------------------------------------------------------------------------
def shaped_distance(a, b, s, t):
    """D(a, b, s, t) ∈ [0, 1]. Returns 1.0 when ||a-b|| < t.
    Decays to 0.05 at distance s. Gradient everywhere.
    """
    dist = jnp.sqrt(jnp.dot(a - b, a - b) + 1e-8)
    scale = 1.8318 / jnp.maximum(s, 1e-6)
    shaped = (1.0 - jnp.tanh(dist * scale)) ** 2
    return jnp.where(dist < t, 1.0, shaped)

# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------
def init_mlp(key):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        'w1': jax.random.normal(k1, (OBS_DIM, HIDDEN)) * jnp.sqrt(2.0 / OBS_DIM),
        'b1': jnp.zeros(HIDDEN),
        'w2': jax.random.normal(k2, (HIDDEN, HIDDEN)) * jnp.sqrt(2.0 / HIDDEN),
        'b2': jnp.zeros(HIDDEN),
        'w3': jax.random.normal(k3, (HIDDEN, ACT_DIM)) * 0.01,
        'b3': jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0]),
    }


def mlp_forward(params, obs):
    x = jnp.tanh(obs @ params['w1'] + params['b1'])
    x = jnp.tanh(x @ params['w2'] + params['b2'])
    x = x @ params['w3'] + params['b3']
    q_dot_target = jnp.tanh(x[:N_JOINTS]) * VEL_LIMITS
    finger = jax.nn.sigmoid(x[N_JOINTS]) * FINGER_OPEN
    return q_dot_target, finger

# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
def compute_reward(state):
    ee_pos = state.site_xpos[EE_SITE_IDX]
    cube_pos = state.xpos[CUBE_BODY_IDX]
    finger_width = state.qpos[7] + state.qpos[8]
    grip_closedness = 1.0 - finger_width / (2.0 * FINGER_OPEN)
    grip_closedness = jnp.clip(grip_closedness, 0.0, 1.0)
    left_finger_z = state.xpos[10][2]
    right_finger_z = state.xpos[11][2]
    ee_pos_z = state.site_xpos[EE_SITE_IDX][2]

    #delta = ee_pos - TARGET_POS
    pre_grip_pos = cube_pos + jnp.array([0.0, 0.0, GRASP_Z_OFFSET])
    delta = ee_pos - pre_grip_pos
    r_approach = jnp.sqrt(jnp.dot(delta, delta) + 1e-6) * 1.0 - shaped_distance(ee_pos, pre_grip_pos, 0.3, 0.001) * 3.0  
    #r_approach = jnp.sqrt(jnp.dot(delta, delta) + 1e-6) 
    #r_approach = shaped_distance(ee_pos, cube_pos, 0.3, 0.003)

    ee_quat = state.xquat[9]
    quat_dot = jnp.dot(ee_quat, TARGET_QUAT)
    ori_loss = (1.0 - quat_dot * quat_dot)  # 0 = aligned, 1 = 180

    min_finger_z = jnp.minimum(left_finger_z, right_finger_z)
    ground_penalty1 = jnp.where(ee_pos_z < 0.1, 10.0 * (0.1 - ee_pos_z) / 0.1, 0.0)

    return  -r_approach * 2.0 - ori_loss * 3.0 - ground_penalty1 * 1.0

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("\nLoading MuJoCo model...")
mj_model = mujoco.MjModel.from_xml_path(args.xml)
mj_data = mujoco.MjData(mj_model)

key_id = mj_model.keyframe("home").id
mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
mujoco.mj_forward(mj_model, mj_data)

sim_dt = mj_model.opt.timestep
frame_dt = N_SUBSTEPS * sim_dt
total_time = N_STEPS * frame_dt
total_physics = N_STEPS * N_SUBSTEPS

print(f"  sim_dt={sim_dt}, frame_dt={frame_dt:.4f}s")
print(f"  {N_STEPS} steps × {N_SUBSTEPS} substeps = {total_physics} physics steps")
print(f"  Total trajectory: {total_time:.3f}s")
print(f"  EE home (site): {mj_data.site_xpos[EE_SITE_IDX]}")
print(f"  Target: {TARGET_POS}")

home_ctrl = jnp.array(mj_data.ctrl)

# ---------------------------------------------------------------------------
# MJX
# ---------------------------------------------------------------------------
print(f"\nSetting up MJX (solver_iters={args.solver_iters})...")
mj_model.opt.iterations = args.solver_iters
mj_model.opt.ls_iterations = 4
mjx_model = mjx.put_model(mj_model)

mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
mjx_data_init = mjx.put_data(mj_model, mj_data)
mjx_data_init = mjx.forward(mjx_model, mjx_data_init)

# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------
rng = jax.random.PRNGKey(42)
mlp_params = init_mlp(rng)
n_params = sum(p.size for p in jax.tree.leaves(mlp_params))

optimizer = optax.chain(
    optax.clip_by_global_norm(GRAD_CLIP),
    optax.adam(TRAIN_LR),
)
opt_state = optimizer.init(mlp_params)

print(f"\n{'='*60}")
print(f"BPTT CLOSED-LOOP — Joint Velocity + Nested lax.scan")
print(f"  MLP: {OBS_DIM}→{HIDDEN}→{HIDDEN}→{ACT_DIM} ({n_params} params)")
print(f"  Action: q_dot → ctrl = q + q_dot*dt")
print(f"  {N_STEPS}×{N_SUBSTEPS} = {total_physics} physics steps ({total_time:.3f}s)")
print(f"  Target: {TARGET_POS}")
print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# BPTT Loss — nested lax.scan (fast compile)
# ---------------------------------------------------------------------------
def bptt_loss_fn(mlp_params):
    """
    Nested lax.scan:
      Outer scan: N_STEPS frame steps (MLP called each)
      Inner scan: N_SUBSTEPS physics substeps (ctrl constant)
    
    JIT traces 1 frame step + 1 substep → compiles fast.
    """
    def frame_step_fn(carry, _):
        state, prev_ee_pos, total_reward, gamma_acc = carry

        # 1. Observation
        ee_pos = state.site_xpos[EE_SITE_IDX]
        ee_vel = (ee_pos - prev_ee_pos) / frame_dt
        cube_pos = state.xpos[CUBE_BODY_IDX]
        cube_quat = state.xquat[CUBE_BODY_IDX]
        ee_to_cube = cube_pos - ee_pos
        finger_width = state.qpos[7] + state.qpos[8]
        current_q = state.qpos[:N_JOINTS]
        current_q_dot = state.qvel[:N_JOINTS]
        ee_quat_obs = state.xquat[9]  # (4,)
        quat_err = TARGET_QUAT - ee_quat_obs  # (4,)
        obs = jnp.concatenate([ee_pos, ee_to_cube, ee_vel,
                                jnp.array([finger_width]), current_q, current_q_dot,
                                ee_quat_obs, quat_err, cube_quat])

        # 2. MLP → q_dot_target + finger
        q_dot_target, finger_target = mlp_forward(mlp_params, obs)

        at_min = (current_q <= Q_LO + 0.01)  # 0.01 radyanlık bir tampon bölge
        at_max = (current_q >= Q_HI - 0.01)
        
        safe_q_dot_target = jnp.where(at_min, jnp.maximum(0, q_dot_target), q_dot_target)
        safe_q_dot_target = jnp.where(at_max, jnp.minimum(0, safe_q_dot_target), safe_q_dot_target)

        ctrl = home_ctrl.at[:N_JOINTS].set(safe_q_dot_target)
        ctrl = ctrl.at[7].set(finger_target)

        # 4. Physics substeps (inner scan)
        def substep_fn(s, _):
            s = s.replace(ctrl=ctrl)
            s = mjx.step(mjx_model, s)
            return s, None

        state, _ = jax.lax.scan(substep_fn, state, None, length=N_SUBSTEPS)

        # 5. Reward        
        rew = compute_reward(state)
        total_reward = total_reward + gamma_acc * rew
        gamma_acc = gamma_acc * GAMMA

        return (state, ee_pos, total_reward, gamma_acc), None

    init_carry = (
        mjx_data_init,
        mjx_data_init.site_xpos[EE_SITE_IDX],  # prev_ee_pos
        jnp.float32(0.0),                        # total_reward
        jnp.float32(1.0),                        # gamma_acc
    )

    (final_state, _, total_reward, _), _ = jax.lax.scan(
        frame_step_fn, init_carry, None, length=N_STEPS)

    return -total_reward


def bptt_forward_fn(mlp_params):
    """Forward rollout — per-step info + qpos for rendering."""
    def frame_step_fn(carry, _):
        state, prev_ee_pos = carry

        ee_pos = state.site_xpos[EE_SITE_IDX]
        ee_vel = (ee_pos - prev_ee_pos) / frame_dt
        cube_pos = state.xpos[CUBE_BODY_IDX]
        cube_quat = state.xquat[CUBE_BODY_IDX]
        ee_to_cube = cube_pos - ee_pos
        finger_width = state.qpos[7] + state.qpos[8]
        current_q = state.qpos[:N_JOINTS]
        current_q_dot = state.qvel[:N_JOINTS]
        ee_quat_obs = state.xquat[9]  # (4,)
        quat_err = TARGET_QUAT - ee_quat_obs  # (4,)
        obs = jnp.concatenate([ee_pos, ee_to_cube, ee_vel,
                                jnp.array([finger_width]), current_q, current_q_dot,
                                ee_quat_obs, quat_err, cube_quat])

        q_dot_target, finger_target = mlp_forward(mlp_params, obs)

        at_min = (current_q <= Q_LO + 0.01)  # 0.01 radyanlık bir tampon bölge
        at_max = (current_q >= Q_HI - 0.01)
        
        safe_q_dot_target = jnp.where(at_min, jnp.maximum(0, q_dot_target), q_dot_target)
        safe_q_dot_target = jnp.where(at_max, jnp.minimum(0, safe_q_dot_target), safe_q_dot_target)

        ctrl = home_ctrl.at[:N_JOINTS].set(safe_q_dot_target)
        ctrl = ctrl.at[7].set(finger_target)

        def substep_fn(s, _):
            s = s.replace(ctrl=ctrl)
            s = mjx.step(mjx_model, s)
            return s, s.qpos

        state, step_qpos = jax.lax.scan(substep_fn, state, None, length=N_SUBSTEPS)

        ee_new = state.site_xpos[EE_SITE_IDX]
        ee_vel_new = (ee_new - ee_pos) / frame_dt
        delta = ee_new - TARGET_POS
        dist = jnp.sqrt(jnp.dot(delta, delta) + 1e-6)
        actual_qvel = state.qvel[:N_JOINTS]

        actual_rew = compute_reward(state)
        step_info = jnp.concatenate([ee_new, ee_vel_new, jnp.array([finger_target, actual_rew]), safe_q_dot_target, actual_qvel])
        # step_info: ee(3) + v(3) + finger(1) + rew(1) = 8

        return (state, ee_pos), (step_info, step_qpos)

    init_carry = (mjx_data_init, mjx_data_init.site_xpos[EE_SITE_IDX])

    (final_state, _), (all_step_info, all_qpos) = jax.lax.scan(
        frame_step_fn, init_carry, None, length=N_STEPS)

    return final_state, all_step_info, all_qpos

# ---------------------------------------------------------------------------
# JIT
# ---------------------------------------------------------------------------
print("JIT compiling BPTT loss + grad...")
t0 = time.time()
value_and_grad_fn = jax.jit(jax.value_and_grad(bptt_loss_fn))
loss_val, grad_val = value_and_grad_fn(mlp_params)
loss_val.block_until_ready()
print(f"  JIT compile: {time.time()-t0:.1f}s")
print(f"  Initial loss: {float(loss_val):.6f}")
print(f"  Initial grad norm: {float(optax.global_norm(grad_val)):.4f}")

print("\nJIT compiling forward rollout...")
t0 = time.time()
jit_forward = jax.jit(bptt_forward_fn)
_, info_test, qpos_test = jit_forward(mlp_params)
info_test.block_until_ready()
print(f"  JIT compile: {time.time()-t0:.1f}s")
print(f"  all_step_info shape: {info_test.shape}")   # (N_STEPS, 8)
print(f"  all_qpos shape: {qpos_test.shape}")         # (N_STEPS, N_SUBSTEPS, nq)

# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------
USE_VIEWER = not args.no_viewer

if USE_VIEWER:
    from mujoco import viewer as mj_viewer

    mj_model_render = mujoco.MjModel.from_xml_path(args.xml)
    mj_data_render = mujoco.MjData(mj_model_render)
    mj_data_render.qpos[:] = np.array(mjx_data_init.qpos)
    mj_data_render.qvel[:] = np.array(mjx_data_init.qvel)
    mujoco.mj_forward(mj_model_render, mj_data_render)

    mocap_id = mj_model_render.body("mocap_target").mocapid[0]
    mj_data_render.mocap_pos[mocap_id] = np.array(TARGET_POS)

    viewer_handle = mj_viewer.launch_passive(mj_model_render, mj_data_render)
    print("✓ Viewer açıldı\n")


def render_trajectory(all_qpos, skip=1):
    """all_qpos shape: (N_STEPS, N_SUBSTEPS, nq)"""
    if not USE_VIEWER or not viewer_handle.is_running():
        return
    qpos_np = np.array(all_qpos)
    for step in range(qpos_np.shape[0]):
        for sub in range(0, qpos_np.shape[1], skip):
            mj_data_render.qpos[:] = qpos_np[step, sub]
            mujoco.mj_forward(mj_model_render, mj_data_render)
            mj_data_render.mocap_pos[mocap_id] = np.array(TARGET_POS)
            viewer_handle.sync()
            time.sleep(sim_dt * skip)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
loss_history = []
dist_history = []

print("Training başlıyor...\n")
train_start = time.time()

for iteration in range(TRAIN_ITERS):
    if USE_VIEWER and not viewer_handle.is_running():
        print("\nViewer kapatıldı.")
        break

    loss_val, grad_val = value_and_grad_fn(mlp_params)
    loss_f = float(loss_val)

    if np.isnan(loss_f):
        print(f"\n⚠ NaN at iter {iteration}!")
        break

    loss_history.append(loss_f)

    grad_norm = float(optax.global_norm(grad_val))

    # Log + render every 5 iters
    if iteration % 5 == 0 or iteration < 3:
        _, all_info, all_qpos = jit_forward(mlp_params)
        info_np = np.array(all_info)  # (N_STEPS, 8)

        ee_final = info_np[-1, :3]
        dist = np.linalg.norm(ee_final - np.array(TARGET_POS))
        dist_history.append(dist)

        print(f"\nIter {iteration:4d}:  loss={loss_f:.4f}  dist={dist:.4f}  "
              f"grad_norm={grad_norm:.4f}")
        for s in range(N_STEPS):
            ee = info_np[s, :3]
            v = info_np[s, 3:6]
            fng = info_np[s, 6]
            rew = info_np[s, 7]
            print(f"  step {s}: EE=[{ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}]  "
                  f"v=[{v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f}]  "
                  f"fng={fng:.4f}  rew={rew:.4f}")
            
        fig, axes = plt.subplots(7, 1, figsize=(12, 20))
        for j in range(7):
            axes[j].plot(info_np[:, 8+j], label=f"target j{j}", linestyle="--")
            axes[j].plot(info_np[:, 15+j], label=f"actual j{j}")
            axes[j].set_ylabel("rad/s")
            axes[j].legend()
            axes[j].grid(True, alpha=0.3)
        axes[-1].set_xlabel("Step")
        fig.suptitle(f"Joint Vel: Target vs Actual — Iter {iteration}")
        fig.tight_layout()
        fig.savefig(f"joint_vel_iter_{iteration:04d}.png", dpi=100)
        plt.close(fig)

        render_trajectory(all_qpos, skip=1)
    else:
        dist_history.append(dist_history[-1] if dist_history else 0)

    updates, opt_state = optimizer.update(grad_val, opt_state, mlp_params)
    mlp_params = optax.apply_updates(mlp_params, updates)

train_time = time.time() - train_start
print(f"\nTraining done: {train_time:.1f}s ({len(loss_history)} iters, "
      f"{train_time/max(1,len(loss_history)):.3f}s/iter)")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"SONUÇLAR")
print(f"{'='*60}")
print(f"  Initial loss: {loss_history[0]:.4f}")
print(f"  Final loss:   {loss_history[-1]:.4f}")
print(f"  Final dist:   {dist_history[-1]:.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(loss_history, linewidth=1.5)
axes[0].set_ylabel("Loss (-reward)")
axes[0].set_title("Loss")
axes[0].grid(True, alpha=0.3)

axes[1].plot(dist_history, linewidth=1.5)
axes[1].set_ylabel("Distance (m)")
axes[1].set_xlabel("Iteration")
axes[1].set_title("EE → Target Distance")
axes[1].grid(True, alpha=0.3)

fig.suptitle(f"BPTT EE-Vel — {N_STEPS}×{N_SUBSTEPS}, {len(loss_history)} iters")
fig.tight_layout()
plot_path = "bptt_results.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {plot_path}")

if USE_VIEWER and viewer_handle.is_running():
    print("\nViewer açık — ESC ile kapat.")
    try:
        while viewer_handle.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nÇıkılıyor.")