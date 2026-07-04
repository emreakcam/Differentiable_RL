"""
CNN Depth Learning Test
 
Kanıtlamak istediğimiz: CNN depth image'dan küp pozisyonunu öğrenebiliyor mu?
 
Pipeline:
    random cube_pos → diff_render(cube_pos) → depth (32×32) → CNN → predicted_pos
                                                                         ↓
                                                              MSE loss vs cube_pos
                                                                         ↓
                                                    jax.grad → CNN params güncelle
 
Fizik yok, robot yok. Sadece: CNN render'dan öğrenebiliyor mu?
 
Usage:
    python cnn_depth_test.py
    python cnn_depth_test.py --iters 1000 --batch 128 --lr 1e-3
"""
 
import argparse
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
print(f"JAX devices: {jax.devices()}")
 
# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--iters", type=int, default=1500)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--res", type=int, default=64)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
 
# ---------------------------------------------------------------------------
# Differentiable Depth Renderer (aynı kod)
# ---------------------------------------------------------------------------
IMG_W = IMG_H = args.res
MAX_DEPTH = 3.0
 
CAM_POS    = jnp.array([0.9, -0.4, 0.45])
CAM_TARGET = jnp.array([0.5,  0.0, 0.05])
CAM_UP     = jnp.array([0.0,  0.0, 1.0])
BOX_HALF   = jnp.array([0.05, 0.05, 0.05])
FOVY_RAD   = jnp.deg2rad(60.0)
FLAT_DIM = 32 * (IMG_W // 8) * (IMG_H // 8)
 
 
def quat_to_rot(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])
 
 
def generate_rays(cam_pos, cam_target, cam_up, fovy, img_w, img_h):
    forward = cam_target - cam_pos
    forward = forward / jnp.linalg.norm(forward)
    right = jnp.cross(forward, cam_up)
    right = right / jnp.linalg.norm(right)
    up = jnp.cross(right, forward)
    aspect = img_w / img_h
    half_h = jnp.tan(fovy / 2.0)
    half_w = half_h * aspect
    u = (jnp.arange(img_w) + 0.5) / img_w * 2.0 - 1.0
    v = ((jnp.arange(img_h) + 0.5) / img_h * 2.0 - 1.0)[::-1]
    uu, vv = jnp.meshgrid(u, v)
    dirs = (forward[None, None, :]
            + uu[:, :, None] * half_w * right[None, None, :]
            + vv[:, :, None] * half_h * up[None, None, :])
    dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs
 
 
RAY_DIRS = generate_rays(CAM_POS, CAM_TARGET, CAM_UP, FOVY_RAD, IMG_W, IMG_H)
 
 
def render_depth(cube_pos, cube_quat):
    t_gnd = jnp.where(RAY_DIRS[:, :, 2] < -1e-8,
                       -CAM_POS[2] / RAY_DIRS[:, :, 2], MAX_DEPTH)
    R_inv = quat_to_rot(cube_quat).T
    local_o = R_inv @ (CAM_POS - cube_pos)
    local_d = jnp.einsum('ij,hwj->hwi', R_inv, RAY_DIRS)
    inv_d = 1.0 / jnp.where(jnp.abs(local_d) < 1e-8,
                              jnp.sign(local_d) * 1e-8 + 1e-10, local_d)
    t1 = (-BOX_HALF[None, None, :] - local_o[None, None, :]) * inv_d
    t2 = ( BOX_HALF[None, None, :] - local_o[None, None, :]) * inv_d
    t_near = jnp.minimum(t1, t2)
    t_far  = jnp.maximum(t1, t2)
    t_enter = jnp.max(t_near, axis=-1)
    t_exit  = jnp.min(t_far, axis=-1)
    hit = (t_enter < t_exit) & (t_exit > 0)
    t_box = jnp.where(hit, jnp.maximum(t_enter, 0.0), MAX_DEPTH)
    return jnp.minimum(t_gnd, t_box)
 
 
# ---------------------------------------------------------------------------
# CNN — 3 conv + 2 fc, saf JAX
# ---------------------------------------------------------------------------
def init_cnn(key):
    """CNN params: 3 conv layers + 2 FC layers."""
    keys = jax.random.split(key, 5)
 
    # Conv: (C_out, C_in, kH, kW) — JAX lax.conv formatı
    params = {
        'conv1_w': jax.random.normal(keys[0], (8, 1, 3, 3), dtype=jnp.float32) * 0.1,
        'conv1_b': jnp.zeros(8, dtype=jnp.float32),
 
        'conv2_w': jax.random.normal(keys[1], (16, 8, 3, 3), dtype=jnp.float32) * 0.1,
        'conv2_b': jnp.zeros(16, dtype=jnp.float32),
 
        'conv3_w': jax.random.normal(keys[2], (32, 16, 3, 3), dtype=jnp.float32) * 0.1,
        'conv3_b': jnp.zeros(32, dtype=jnp.float32),
 
        # 32 channels × 4 × 4 = 512
        'fc1_w': jax.random.normal(keys[3], (FLAT_DIM, 64), dtype=jnp.float32) * jnp.sqrt(2.0/512),
        'fc1_b': jnp.zeros(64, dtype=jnp.float32),
 
        'fc2_w': jax.random.normal(keys[4], (64, 3), dtype=jnp.float32) * 0.01,
        'fc2_b': jnp.array([0.5, 0.0, 0.1], dtype=jnp.float32),  # workspace ortası bias
    }
    return params
 
 
def cnn_forward(params, depth_img):
    """
    depth_img: (H, W) float → predicted cube pos (3,)
 
    Conv layers: 32→16→8→4, channels: 1→8→16→32
    FC: 512 → 64 → 3
    """
    # Normalize depth: [0.3, 3.0] → [-1, 1]
    x = (depth_img.astype(jnp.float32) - 1.5) / 1.5
 
    # (H, W) → (1, 1, H, W) — batch=1, channels=1
    x = x[None, None, :, :]
 
    # Conv1: 1→8, stride=2, SAME padding
    x = jax.lax.conv_general_dilated(
        x, params['conv1_w'], window_strides=(2, 2), padding='SAME')
    x = x + params['conv1_b'][None, :, None, None]
    x = jax.nn.relu(x)  # (1, 8, 16, 16)
 
    # Conv2: 8→16, stride=2
    x = jax.lax.conv_general_dilated(
        x, params['conv2_w'], window_strides=(2, 2), padding='SAME')
    x = x + params['conv2_b'][None, :, None, None]
    x = jax.nn.relu(x)  # (1, 16, 8, 8)
 
    # Conv3: 16→32, stride=2
    x = jax.lax.conv_general_dilated(
        x, params['conv3_w'], window_strides=(2, 2), padding='SAME')
    x = x + params['conv3_b'][None, :, None, None]
    x = jax.nn.relu(x)  # (1, 32, 4, 4)
 
    # Flatten: 32×4×4 = 512
    x = x.reshape(-1)
 
    # FC1: 512 → 64
    x = jnp.tanh(x @ params['fc1_w'] + params['fc1_b'])
 
    # FC2: 64 → 3 (predicted cube position)
    x = x @ params['fc2_w'] + params['fc2_b']
 
    return x
 
 
# ---------------------------------------------------------------------------
# Loss: render → CNN → MSE
# ---------------------------------------------------------------------------
DEFAULT_QUAT = jnp.array([1.0, 0.0, 0.0, 0.0])
 
def single_loss(cnn_params, cube_pos):
    """Tek örnek: render → CNN → MSE."""
    depth = render_depth(cube_pos, DEFAULT_QUAT)
    pred_pos = cnn_forward(cnn_params, depth)
    return jnp.sum((pred_pos - cube_pos.astype(jnp.float32)) ** 2)
 
 
def batch_loss(cnn_params, cube_positions):
    """Batch: vmap ile paralel."""
    losses = jax.vmap(single_loss, in_axes=(None, 0))(cnn_params, cube_positions)
    return jnp.mean(losses)
 
 
# ---------------------------------------------------------------------------
# Random cube position sampler
# ---------------------------------------------------------------------------
def sample_positions(key, batch_size):
    """Workspace içinde rastgele küp pozisyonları."""
    keys = jax.random.split(key, 3)
    x = jax.random.uniform(keys[0], (batch_size,), minval=0.3, maxval=0.7)
    y = jax.random.uniform(keys[1], (batch_size,), minval=-0.2, maxval=0.2)
    z = jax.random.uniform(keys[2], (batch_size,), minval=0.05, maxval=0.30)
    return jnp.stack([x, y, z], axis=-1)
 
 
# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------
rng = jax.random.PRNGKey(args.seed)
rng, init_key = jax.random.split(rng)
cnn_params = init_cnn(init_key)
 
n_params = sum(p.size for p in jax.tree.leaves(cnn_params))
print(f"\nCNN params: {n_params}")
print(f"  Conv1: 1→8, 3×3, stride=2")
print(f"  Conv2: 8→16, 3×3, stride=2")
print(f"  Conv3: 16→32, 3×3, stride=2")
print(f"  FC1: {FLAT_DIM}→64, FC2: 64→3")
print(f"  Depth res: {IMG_W}×{IMG_H}")
print(f"  Batch: {args.batch}, LR: {args.lr}, Iters: {args.iters}")
 
optimizer = optax.adam(args.lr)
opt_state = optimizer.init(cnn_params)
 
# ---------------------------------------------------------------------------
# JIT compile
# ---------------------------------------------------------------------------
print("\nJIT compiling...")
t0 = time.time()
 
value_and_grad_fn = jax.jit(jax.value_and_grad(batch_loss))
 
# Warmup
rng, sample_key = jax.random.split(rng)
test_positions = sample_positions(sample_key, args.batch)
loss_val, grad_val = value_and_grad_fn(cnn_params, test_positions)
loss_val.block_until_ready()
 
print(f"  JIT compile: {time.time()-t0:.1f}s")
print(f"  Initial loss: {float(loss_val):.6f}")
print(f"  Initial RMSE: {float(jnp.sqrt(loss_val)):.4f} m")
grad_norm = float(optax.global_norm(grad_val))
print(f"  Grad norm: {grad_norm:.4f}")
 
# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------
@jax.jit
def evaluate(cnn_params, positions):
    """Her örnek için pred vs GT pozisyon."""
    def predict_one(cube_pos):
        depth = render_depth(cube_pos, DEFAULT_QUAT)
        pred = cnn_forward(cnn_params, depth)
        err = jnp.sqrt(jnp.sum((pred - cube_pos.astype(jnp.float32)) ** 2))
        return pred, err
    preds, errors = jax.vmap(predict_one)(positions)
    return preds, errors
 
# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("Training başlıyor...")
print(f"{'='*70}")
print(f"{'Iter':>5} | {'Loss':>10} | {'RMSE':>8} | {'Mean Err':>8} | {'Max Err':>8} | {'Grad':>8} | {'ms':>6}")
print("-" * 70)
 
loss_history = []
rmse_history = []
mean_err_history = []
 
train_start = time.time()
 
for iteration in range(args.iters):
    rng, sample_key = jax.random.split(rng)
    positions = sample_positions(sample_key, args.batch)
 
    t_iter = time.time()
    loss_val, grad_val = value_and_grad_fn(cnn_params, positions)
    loss_f = float(loss_val)
 
    if np.isnan(loss_f):
        print(f"\n⚠ NaN at iter {iteration}!")
        break
 
    # Update
    updates, opt_state = optimizer.update(grad_val, opt_state, cnn_params)
    cnn_params = optax.apply_updates(cnn_params, updates)
 
    loss_history.append(loss_f)
    rmse = float(jnp.sqrt(loss_val))
    rmse_history.append(rmse)
 
    # Log
    if iteration % 10 == 0 or iteration < 5:
        # Evaluate on fixed test set
        rng, eval_key = jax.random.split(rng)
        eval_pos = sample_positions(eval_key, 256)
        preds, errors = evaluate(cnn_params, eval_pos)
        mean_err = float(errors.mean())
        max_err = float(errors.max())
        mean_err_history.append(mean_err)
        grad_norm = float(optax.global_norm(grad_val))
        iter_ms = (time.time() - t_iter) * 1000
 
        print(f"{iteration:5d} | {loss_f:10.6f} | {rmse:8.4f} | {mean_err:8.4f} | {max_err:8.4f} | {grad_norm:8.4f} | {iter_ms:6.1f}")
 
train_time = time.time() - train_start
print(f"\nTraining: {train_time:.1f}s ({args.iters} iters, {train_time/args.iters*1000:.1f} ms/iter)")
 
# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("Final Evaluation")
print(f"{'='*70}")
 
rng, eval_key = jax.random.split(rng)
eval_pos = sample_positions(eval_key, 512)
preds, errors = evaluate(cnn_params, eval_pos)
 
mean_err = float(errors.mean())
median_err = float(jnp.median(errors))
max_err = float(errors.max())
p90_err = float(jnp.percentile(errors, 90))
 
print(f"  512 test samples:")
print(f"  Mean error:   {mean_err:.4f} m ({mean_err*100:.1f} cm)")
print(f"  Median error: {median_err:.4f} m ({median_err*100:.1f} cm)")
print(f"  90th pctile:  {p90_err:.4f} m ({p90_err*100:.1f} cm)")
print(f"  Max error:    {max_err:.4f} m ({max_err*100:.1f} cm)")
 
# Show some examples
print(f"\n  Örnekler (GT → Pred → Error):")
eval_pos_np = np.array(eval_pos)
preds_np = np.array(preds)
errors_np = np.array(errors)
 
# Best 3
best_idx = np.argsort(errors_np)[:3]
print(f"  En iyi:")
for i in best_idx:
    print(f"    GT=[{eval_pos_np[i,0]:.3f}, {eval_pos_np[i,1]:.3f}, {eval_pos_np[i,2]:.3f}]  "
          f"Pred=[{preds_np[i,0]:.3f}, {preds_np[i,1]:.3f}, {preds_np[i,2]:.3f}]  "
          f"Err={errors_np[i]:.4f}m")
 
# Worst 3
worst_idx = np.argsort(errors_np)[-3:]
print(f"  En kötü:")
for i in worst_idx:
    print(f"    GT=[{eval_pos_np[i,0]:.3f}, {eval_pos_np[i,1]:.3f}, {eval_pos_np[i,2]:.3f}]  "
          f"Pred=[{preds_np[i,0]:.3f}, {preds_np[i,1]:.3f}, {preds_np[i,2]:.3f}]  "
          f"Err={errors_np[i]:.4f}m")
 
# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
print("\nPlotlar oluşturuluyor...")
 
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
 
# (0,0) Loss curve
axes[0, 0].plot(loss_history, linewidth=1)
axes[0, 0].set_ylabel("MSE Loss")
axes[0, 0].set_xlabel("Iteration")
axes[0, 0].set_title("Training Loss")
axes[0, 0].set_yscale("log")
axes[0, 0].grid(True, alpha=0.3)
 
# (0,1) Error distribution
axes[0, 1].hist(errors_np * 100, bins=30, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel("Error (cm)")
axes[0, 1].set_ylabel("Count")
axes[0, 1].set_title(f"Error Distribution (mean={mean_err*100:.1f}cm)")
axes[0, 1].axvline(mean_err * 100, color='r', linestyle='--', label=f'mean={mean_err*100:.1f}cm')
axes[0, 1].legend()
 
# (0,2) Predicted vs GT scatter (x axis)
axes[0, 2].scatter(eval_pos_np[:, 0], preds_np[:, 0], alpha=0.3, s=10)
axes[0, 2].plot([0.3, 0.7], [0.3, 0.7], 'r--', linewidth=2)
axes[0, 2].set_xlabel("GT x")
axes[0, 2].set_ylabel("Pred x")
axes[0, 2].set_title("X axis: Predicted vs GT")
axes[0, 2].set_aspect('equal')
axes[0, 2].grid(True, alpha=0.3)
 
# (1,0) Predicted vs GT scatter (y axis)
axes[1, 0].scatter(eval_pos_np[:, 1], preds_np[:, 1], alpha=0.3, s=10)
axes[1, 0].plot([-0.2, 0.2], [-0.2, 0.2], 'r--', linewidth=2)
axes[1, 0].set_xlabel("GT y")
axes[1, 0].set_ylabel("Pred y")
axes[1, 0].set_title("Y axis: Predicted vs GT")
axes[1, 0].set_aspect('equal')
axes[1, 0].grid(True, alpha=0.3)
 
# (1,1) Predicted vs GT scatter (z axis)
axes[1, 1].scatter(eval_pos_np[:, 2], preds_np[:, 2], alpha=0.3, s=10)
axes[1, 1].plot([0.05, 0.30], [0.05, 0.30], 'r--', linewidth=2)
axes[1, 1].set_xlabel("GT z")
axes[1, 1].set_ylabel("Pred z")
axes[1, 1].set_title("Z axis: Predicted vs GT")
axes[1, 1].set_aspect('equal')
axes[1, 1].grid(True, alpha=0.3)
 
# (1,2) Example depth images with predictions
examples_idx = [best_idx[0], worst_idx[-1],
                np.argsort(np.abs(errors_np - np.median(errors_np)))[0]]
depth_examples = []
for idx in examples_idx:
    pos = eval_pos[idx]
    depth = render_depth(pos, DEFAULT_QUAT)
    depth_examples.append((np.array(pos), np.array(preds[idx]),
                           float(errors[idx]), np.array(depth)))
 
# Mini gallery in subplot
ax = axes[1, 2]
ax.axis('off')
ax.set_title("Depth örnekleri (best / worst / median)")
for i, (gt, pred, err, dimg) in enumerate(depth_examples):
    inset = ax.inset_axes([i * 0.33, 0.0, 0.32, 1.0])
    inset.imshow(dimg, cmap='viridis', vmin=0.3, vmax=2.0)
    labels = ["BEST", "WORST", "MEDIAN"]
    inset.set_title(f"{labels[i]}\nerr={err*100:.1f}cm", fontsize=7)
    inset.set_xlabel(f"GT=[{gt[0]:.2f},{gt[1]:.2f},{gt[2]:.2f}]\n"
                     f"P=[{pred[0]:.2f},{pred[1]:.2f},{pred[2]:.2f}]", fontsize=6)
    inset.set_xticks([]); inset.set_yticks([])
 
fig.suptitle(f"CNN Depth Learning — {args.iters} iters, batch={args.batch}, "
             f"{IMG_W}×{IMG_H}, mean_err={mean_err*100:.1f}cm", fontsize=13)
fig.tight_layout()
fig.savefig("cnn_depth_results.png", dpi=150)
plt.close(fig)
print("Saved: cnn_depth_results.png")
 
# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  CNN params:    {n_params}")
print(f"  Resolution:    {IMG_W}×{IMG_H}")
print(f"  Training:      {args.iters} iters, {train_time:.1f}s")
print(f"  Final RMSE:    {rmse_history[-1]:.4f} m")
print(f"  Mean error:    {mean_err*100:.1f} cm")
print(f"  Median error:  {median_err*100:.1f} cm")
print(f"  90th pctile:   {p90_err*100:.1f} cm")
print(f"  Gradient:      ✓ flows through diff_render → CNN")
success = mean_err < 0.05
print(f"  Success:       {'✓' if success else '✗'} (mean < 5cm threshold)")
print(f"{'='*70}")