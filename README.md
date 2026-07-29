# Differentiable Reinforcement Learning in Aerial Manipulation

First-order analytic policy gradients (BPTT) through differentiable physics for Franka Emika Panda manipulation tasks, implemented in MuJoCo MJX (JAX).

This project trains neural network policies by backpropagating directly through the physics simulator — computing exact analytical gradients of the task reward with respect to policy parameters, rather than estimating them from sampled rollouts (as in PPO). Five manipulation tasks of increasing complexity demonstrate the approach.

---

## Tasks

### 1. Cube Stacking

Pick up a cube and stack it on a second cube.

<!-- VIDEO: cube stacking -->

https://github.com/user-attachments/assets/fecb58d7-15a2-46b2-8e5f-d0a4e4d7eb80

5 segments: pre-grasp → descend → grasp → move → release
Batch: 32 envs with randomized cube positions.

---

### 2. Drawer Opening + Cube Placement

Open a drawer, pick up a nearby cube, place it inside, close the drawer.

<!-- VIDEO: drawer -->

https://github.com/user-attachments/assets/6bead930-2bc3-419a-bdec-5c4d9ea2da66

10 segments: approach handle → grasp → pull → release+lift → above cube → descend → grasp cube → place → re-handle → push closed.

---

### 3. Peg Insertion

Pick up a peg from the table and insert it into a tight-clearance slot (5mm per side).

<!-- VIDEO: peg insertion -->

https://github.com/user-attachments/assets/b166cbb2-8637-4b3e-8f0f-1c2a5c02cc4c

7 segments: pre-grasp → descend → grasp → lift → align → insert → release.

---

### 4. Container Sorting

Pick up three different objects (ball, cube, prism) and place them into a container, one by one.

<!-- VIDEO: container sorting -->

https://github.com/user-attachments/assets/576896bb-3d5e-4fab-a838-8cd6919e983e

15 segments: 3 objects × (pre-grasp → descend → grasp → move → release).

---

### 5. Hinge Cabinet Door Opening

Approach a cabinet handle, grasp it, and swing the door open.

<!-- VIDEO: cabinet door -->

https://github.com/user-attachments/assets/15bb50c7-183b-4193-a118-ef046032110c

4 segments: approach → grasp → pull → release.

---

## Method

### Backpropagation Through Time (BPTT)

The policy network is a neural network that maps observations to joint velocity commands. The entire rollout — observation → policy → physics step → reward — forms a differentiable computation graph. `jax.grad` computes exact gradients of the cumulative reward with respect to all policy parameters in a single backward pass.

```
obs → MLP → joint velocities → mjx.step → next state → reward
 ↑                                  ↓
 └──────────── jax.lax.scan ────────┘
                    ↓
          jax.grad(total_reward) → parameter update
```

### Segmented Training

Each task is decomposed into temporal segments, each with its own MLP head and reward function. At segment boundaries, `jax.lax.stop_gradient` cuts the gradient chain and resets the discount factor. This addresses BPTT's core weakness — gradient vanishing/exploding over long horizons — while keeping exact gradients within each segment.

```
[Seg 0: approach]──stop_grad──[Seg 1: grasp]──stop_grad──[Seg 2: move]──...
    ↑ own MLP                     ↑ own MLP                  ↑ own MLP
    ↑ own reward                  ↑ own reward                ↑ own reward
```

### Progressive Training

Segments are activated incrementally based on convergence criteria. Training starts with only the first two segments active. When the current phase meets its success criterion for `PATIENCE` consecutive evaluations, the next segment is activated. This prevents early segments from receiving noisy gradients through unconverged later segments.

### Gradient Checkpointing

`jax.checkpoint` (rematerialization) is applied at two levels — frame steps and physics substeps. During the backward pass, intermediate activations are recomputed on the fly instead of being stored in memory, reducing VRAM usage by approximately 4× at a ~1.5× compute cost. This makes batch sizes of 32 feasible on a 16GB GPU.

### Observation Normalization

A running mean/variance estimator (`ObsRMS`) normalizes observations online. Statistics are updated each iteration using the full batch of rollout observations returned via `has_aux=True`, avoiding an extra forward pass.

---

## Key Technical Details

### MJX Solver Patch

MJX's constraint solver uses `jax.lax.while_loop`, which does not support reverse-mode automatic differentiation. A monkey-patch replaces it with `jax.lax.fori_loop` using a fixed iteration count, making the solver fully differentiable.

### Mixed Precision

Physics runs in float64 (`jax.config.update("jax_enable_x64", True)`) to avoid NaN gradients in contact resolution. The policy MLP runs in float32 for efficiency. Observations are cast to float32 before entering the network.

### Independent Per-Segment MLPs

Each segment has its own two-layer MLP (32 neurons per layer) rather than sharing a trunk. This was found to outperform shared-trunk architectures because segments require qualitatively different behaviors. `jax.lax.switch` or nested `jnp.where` selects the active head based on the current step index.

### Shaped Distance Reward

A custom reward shaping function provides smooth gradients at all distances:

```python
def shaped_distance(a, b, s, t):
    dist = sqrt(dot(a - b, a - b) + 1e-6)
    scale = 1.8318 / max(s, 1e-6)
    shaped = (1.0 - tanh(dist * scale)) ** 2
    return where(dist < t, 1.0, shaped)
```

The parameter `s` controls the gradient falloff width, `t` is the "close enough" threshold below which the reward saturates to 1.0.

### MJX Constraints

- Cylinder collision is not supported — use capsule or box
- Box-box SAT contact can produce NaN gradients — use capsule for peg-like objects, or use sphere geometry where possible
- `margin` and `gap` geom attributes cause vibration — remove them
- `max_contact_points` must be increased (via `<custom><numeric>`) for multi-object scenes
- Asset includes must be flat (no nested includes) to avoid MuJoCo's path doubling bug

---

## Running

### Prerequisites

```
Python 3.10+
JAX with GPU support
MuJoCo >= 3.0
mujoco-mjx
optax
```

### Training

```bash
# Cube stacking
python train.py --iters 900 --batch 32

# Drawer + cube (progressive, ~3000 iters)
python train_drawer.py --iters 3000 --no-viewer

# Peg insertion
python train_peg.py --iters 3000 --batch 32

# Container sorting (3 objects, 15 segments)
python train_container.py --iters 3000 --batch 32

# Cabinet door opening
python train_cabinet.py --iters 2000 --batch 32

# Cabinet door — single segment variant (no stop_gradient)
python train_cabinet_single.py --batch 16 --grad-clip 800
```

Each script saves trained parameters as a `.pkl` file and a training plot as `.png`.

---

## References

- SHAC: Xu et al., "Accelerated Policy Learning with Parallel Differentiable Simulation," ICLR 2022
- SAPO / Rewarped: entropy regularization for differentiable RL
- DiffSkill: Lin et al., "DiffSkill: Skill Abstraction from Differentiable Physics," ICLR 2022
- STAP: Agia et al., "STAP: Sequencing Task-Agnostic Policies," ICRA 2023
- Suh et al., "Do Differentiable Simulators Give Better Policy Gradients?", ICML 2022
- MuJoCo MJX: Freeman et al., JAX-based differentiable physics
