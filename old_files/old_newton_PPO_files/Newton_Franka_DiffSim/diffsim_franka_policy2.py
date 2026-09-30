# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Diffsim Franka — MLP POLICY VERSION (first step toward APG)
#
# Instead of directly optimizing v_target vectors, an MLP policy network
# maps observations to EE velocity commands:
#
#   obs = [ee_pos(3), target_pos(3)]  →  MLP(θ)  →  v_target(3)
#
# The gradient flows end-to-end:
#   loss ← physics ← q_dot ← J_pinv ← v_target ← MLP ← θ
#
# This is the same pipeline as diffsim_franka_ee_vel.py, but the learnable
# parameters are MLP weights instead of raw v_target vectors.
#
# PyTorch ↔ Warp bridge:
#   - MLP outputs v_target as a torch tensor (requires_grad=True)
#   - wp.from_torch converts it to a warp array (gradient flows through)
#   - Warp tape handles physics backward
#   - After tape.backward(), warp gradients propagate back to torch via
#     the from_torch bridge, and torch autograd updates MLP weights
#
# Command: python diffsim_franka_mlp.py
###########################################################################

import math
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import warp as wp

import newton
import newton.examples
import newton.utils

from helpers.kinematic_helpers import parse_urdf_kinematic_chain
from helpers.warp_kinematics import (
    N_JOINTS,
    fk_tcp_kernel,
    jacobian_p_tcp_kernel,
    damped_pinv_3xN_kernel,
    load_chain_to_warp,
)
from helpers.geom_diff_helpers import (
    rpy_to_quat_wxyz,
    transform_by_quat_diff,
    transform_quat_by_quat_diff,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

TRAJ_STEPS     = 1
SIM_SUBSTEPS   = 100
TRAIN_ITERS    = 500
TRAIN_LR       = 1e-3        # MLP learning rate (smaller than raw v_target)
PINV_DAMPING   = 0.01
DEVICE         = "cuda:0"
V_MAX          = 3.0         # clamp MLP output to physical limits

# Target EE position (world frame) — used as fixed target if RANDOMIZE_TARGET=False
TARGET_POS = (0.4, 0.1, 0.6)

# Random target sampling
RANDOMIZE_TARGET = True
TARGET_RADIUS    = 0.08      # distance from initial EE to target (meters)
TARGET_Z_MIN     = 0.35      # don't sample below this (table height safety)

# Franka rest pose
INITIAL_ARM_Q = [0.0, -0.785, 0.0, -2.356, 0.0, 2.571, 0.785]

# PD gains (velocity mode)
JOINT_KE = [0.0] * 7
JOINT_KD = [450, 450, 350, 350, 200, 200, 200]

# Loss weight
W_POS = 10.0


# ---------------------------------------------------------------------------
# MLP Policy
# ---------------------------------------------------------------------------

class ReachingPolicy(nn.Module):
    """Simple MLP: obs → v_target (EE velocity command).

    Input:  [ee_pos(3), target_pos(3)] = 6 dimensions
    Output: v_target(3) in m/s, clamped to [-V_MAX, V_MAX]
    """
    def __init__(self, obs_dim=6, hidden=64, v_max=V_MAX):
        super().__init__()
        self.v_max = v_max
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
        )
        # Initialize last layer small so initial v_target ≈ 0
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs):
        """obs: [6] → v_target: [3], clamped to [-v_max, v_max]."""
        raw = self.net(obs)
        return self.v_max * torch.tanh(raw)


# ---------------------------------------------------------------------------
# Warp Kernels (local to this training script)
# ---------------------------------------------------------------------------

@wp.kernel
def pose_loss_kernel(
    body_q: wp.array(dtype=wp.transform),
    target_pos: wp.vec3,
    ee_body_index: int,
    weight: float,
    loss: wp.array(dtype=float),
):
    ee_tf = body_q[ee_body_index]
    ee_pos = wp.transform_get_translation(ee_tf)
    delta = ee_pos - target_pos
    wp.atomic_add(loss, 0, weight * wp.dot(delta, delta))


@wp.kernel
def zero_joint_qd_kernel(joint_qd: wp.array(dtype=float)):
    tid = wp.tid()
    joint_qd[tid] = 0.0


@wp.kernel
def copy_first7_kernel(
    src: wp.array(dtype=float),
    dst: wp.array(dtype=float),
):
    tid = wp.tid()
    dst[tid] = src[tid]


@wp.kernel
def copy_fingers_from_rest_kernel(
    rest_fingers: wp.array(dtype=float),
    joint_target: wp.array(dtype=float),
):
    tid = wp.tid()
    joint_target[7 + tid] = rest_fingers[tid]


# ---------------------------------------------------------------------------
# Example class
# ---------------------------------------------------------------------------

class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.verbose = getattr(args, "verbose", True)
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 10.0
        self.sim_dt = self.frame_dt / SIM_SUBSTEPS
        self.train_iter = 0
        self.total_substeps = TRAJ_STEPS * SIM_SUBSTEPS
        self.torch_device = torch.device(DEVICE)
        self.loss_mean = 0.0
        self.loss_mean_alpha = 0.99  # exponential moving average

        # ------------------------------------------------------------------ #
        # 1. Build Newton model                                              #
        # ------------------------------------------------------------------ #
        self.urdf_path = str(
            newton.utils.download_asset("franka_emika_panda")
            / "urdf/fr3_franka_hand.urdf"
        )

        builder = newton.ModelBuilder()
        builder.add_urdf(
            self.urdf_path,
            xform=wp.transform_identity(),
            floating=False,
            enable_self_collisions=False,
        )

        builder.joint_q[:7] = INITIAL_ARM_Q
        builder.joint_q[7:9] = [0.04, 0.04]
        builder.joint_target_mode[:7] = [int(newton.JointTargetMode.VELOCITY)] * 7
        builder.joint_target_mode[7:9] = [int(newton.JointTargetMode.POSITION)] * 2
        builder.joint_target_ke[:7] = JOINT_KE
        builder.joint_target_ke[7:9] = [100.0, 100.0]
        builder.joint_target_kd[:7] = JOINT_KD
        builder.joint_target_kd[7:9] = [10.0, 10.0]
        builder.joint_armature[:9] = [0.3, 0.3, 0.3, 0.3, 0.11, 0.11, 0.11, 0.15, 0.15]
        builder.joint_target_pos[:7] = INITIAL_ARM_Q
        builder.joint_target_pos[7:9] = [0.04, 0.04]

        scene = newton.ModelBuilder()
        scene.replicate(builder, 1)
        scene.add_ground_plane()
        self.model = scene.finalize(requires_grad=True)

        # ------------------------------------------------------------------ #
        # 2. Solver, states, controls                                        #
        # ------------------------------------------------------------------ #
        self.solver = newton.solvers.SolverFeatherstone(self.model)

        self.states = []
        for _ in range(self.total_substeps + 1):
            self.states.append(self.model.state(requires_grad=True))
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.states[0]
        )
        self.n_joint_qd = len(self.model.joint_qd.numpy())

        self.controls = []
        for _ in range(TRAJ_STEPS):
            self.controls.append(self.model.control())

        # ------------------------------------------------------------------ #
        # 3. Find EE body                                                    #
        # ------------------------------------------------------------------ #
        self.ee_body_index = -1
        for i, label in enumerate(self.model.body_label):
            if "fr3_hand_tcp" in label:
                self.ee_body_index = i
                break
        if self.ee_body_index < 0:
            raise RuntimeError("Could not find fr3_hand_tcp body")
        print(f"EE body: {self.ee_body_index} ({self.model.body_label[self.ee_body_index]})")

        # ------------------------------------------------------------------ #
        # 4. Parse URDF → kinematic chain → Warp arrays                      #
        # ------------------------------------------------------------------ #
        chain = parse_urdf_kinematic_chain(
            urdf_path=self.urdf_path,
            root_link="fr3_link0",
            ee_link="fr3_link7",
            device=self.torch_device,
        )

        fix_p1 = torch.tensor([0.0, 0.0, 0.107], device=self.torch_device, dtype=torch.float32)
        fix_q1 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.torch_device, dtype=torch.float32)
        fix_p2 = torch.tensor([0.0, 0.0, 0.0], device=self.torch_device, dtype=torch.float32)
        fix_q2 = torch.tensor(rpy_to_quat_wxyz(0.0, 0.0, -0.7853981633974483), device=self.torch_device, dtype=torch.float32)

        composed_p = fix_p1 + transform_by_quat_diff(fix_p2, fix_q1)
        composed_q = transform_quat_by_quat_diff(fix_q1, fix_q2)

        grasp_offset_hand = torch.tensor([0.0, 0.0, 0.1034], device=self.torch_device, dtype=torch.float32)
        composed_p = composed_p + transform_by_quat_diff(grasp_offset_hand, composed_q)

        chain["ee_offset_p"] = composed_p
        chain["ee_offset_q"] = composed_q

        assert len(chain["joint_names"]) == N_JOINTS
        print(f"Parsed chain: {chain['joint_names']}")
        print(f"TCP offset: pos={chain['ee_offset_p'].cpu().numpy().tolist()}")

        self.chain_wp = load_chain_to_warp(chain, device=DEVICE)

        # Constant: finger rest values
        self.rest_fingers = wp.array([0.04, 0.04], dtype=float, device=DEVICE)

        # ------------------------------------------------------------------ #
        # 5. MLP Policy                                                      #
        # ------------------------------------------------------------------ #
        self.policy = ReachingPolicy(obs_dim=6, hidden=64, v_max=V_MAX).to(self.torch_device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=TRAIN_LR, betas=(0.0, 0.999))

        # Target as torch tensor (for building observations)
        self.target_pos_torch = torch.tensor(TARGET_POS, device=self.torch_device, dtype=torch.float32)
        self.target_pos_wp = wp.vec3(*TARGET_POS)

        # ------------------------------------------------------------------ #
        # 6. Get initial EE position (for first observation)                 #
        # ------------------------------------------------------------------ #
        self.initial_ee_pos = wp.to_torch(
            self.states[0].body_q
        )[self.ee_body_index, :3].detach().clone()

        # Compute initial target distance for reference
        initial_dist = torch.linalg.norm(self.target_pos_torch - self.initial_ee_pos).item()

        print(f"\n{'='*60}")
        print(f"MLP POLICY DIFFSIM (first step toward APG)")
        print(f"  Policy: ReachingPolicy(6 → 64 → 64 → 3)")
        print(f"  Policy params: {sum(p.numel() for p in self.policy.parameters())}")
        print(f"  Observation: [ee_pos(3), target_pos(3)]")
        print(f"  Action: v_target(3) in m/s")
        print(f"  Randomize target: {RANDOMIZE_TARGET}")
        if RANDOMIZE_TARGET:
            print(f"  Target radius: {TARGET_RADIUS} m (from initial EE)")
            print(f"  Target z_min: {TARGET_Z_MIN} m")
        else:
            print(f"  Fixed target: {TARGET_POS}")
        print(f"  Initial EE: {self.initial_ee_pos.cpu().numpy().tolist()}")
        print(f"  Initial target dist: {initial_dist:.3f} m")
        print(f"{'='*60}\n")

        # ------------------------------------------------------------------ #
        # 7. Loss + logging                                                  #
        # ------------------------------------------------------------------ #
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.loss_history = []
        self.joint_q_history = None

        # ------------------------------------------------------------------ #
        # 8. Viewer                                                          #
        # ------------------------------------------------------------------ #
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "renderer"):
            self.viewer.set_camera(
                wp.vec3(1.5, -1.0, 1.2), pitch=-20.0, yaw=130.0
            )

    # ---------------------------------------------------------------------- #
    # Forward                                                                #
    # ---------------------------------------------------------------------- #

    def forward(self):
        """Run the full trajectory forward.

        For each control step:
          1. Read current EE position from Newton state
          2. Build observation [ee_pos, target_pos]
          3. Run MLP → v_target (torch, differentiable)
          4. Convert to warp via wp.from_torch (gradient bridge)
          5. FK + Jacobian + pinv + physics (warp, under tape)
        """
        # Store v_targets for logging
        self._v_targets = []
        self._v_target_tensors = []

        for step in range(TRAJ_STEPS):
            start_substep = step * SIM_SUBSTEPS

            # --- (1) Read current EE position from Newton ---
            # At step 0 this is the initial pose; at step>0 it's the
            # result of previous physics substeps.
            ee_pos_torch = wp.to_torch(
                self.states[start_substep].body_q
            )[self.ee_body_index, :3]

            # --- (2) Build observation ---
            obs = torch.cat([ee_pos_torch, self.target_pos_torch])  # [6]

            # --- (3) MLP forward → v_target ---
            v_target_torch = self.policy(obs)  # [3], differentiable
            self._v_targets.append(v_target_torch.detach().cpu().numpy())

            # Store for debug (keep reference to gradient-bearing tensor)
            if not hasattr(self, '_v_target_tensors'):
                self._v_target_tensors = []
            self._v_target_tensors.append(v_target_torch)

            # --- (4) Convert to warp (gradient bridge) ---
            v_target_wp = wp.from_torch(v_target_torch, dtype=wp.float32)

            # --- (5) Warp pipeline: FK → Jacobian → pinv → physics ---
            # Scratch buffers (fresh each step to avoid stale tape refs)
            fk_ee_pos = wp.zeros(1, dtype=wp.vec3, device=DEVICE, requires_grad=True)
            fk_joint_pos = wp.zeros(N_JOINTS, dtype=wp.vec3, device=DEVICE, requires_grad=True)
            fk_joint_axis = wp.zeros(N_JOINTS, dtype=wp.vec3, device=DEVICE, requires_grad=True)
            J_p_flat = wp.zeros(3 * N_JOINTS, dtype=float, device=DEVICE, requires_grad=True)
            q_dot = wp.zeros(N_JOINTS, dtype=float, device=DEVICE, requires_grad=True)
            q_current_7 = wp.zeros(N_JOINTS, dtype=float, device=DEVICE, requires_grad=True)

            # (a) Snapshot current joint angles
            wp.launch(
                copy_first7_kernel,
                dim=N_JOINTS,
                inputs=[self.states[start_substep].joint_q],
                outputs=[q_current_7],
            )

            # (b) FK at current q
            wp.launch(
                fk_tcp_kernel,
                dim=1,
                inputs=[
                    q_current_7,
                    self.chain_wp["joint_axes"],
                    self.chain_wp["link_pos"],
                    self.chain_wp["link_quats"],
                    self.chain_wp["ee_offset_p"],
                    self.chain_wp["ee_offset_q"],
                    fk_ee_pos,
                    fk_joint_pos,
                    fk_joint_axis,
                ],
            )

            # (c) Position Jacobian
            wp.launch(
                jacobian_p_tcp_kernel,
                dim=N_JOINTS,
                inputs=[fk_joint_pos, fk_joint_axis, fk_ee_pos],
                outputs=[J_p_flat],
            )

            # (d) Damped pinv: q_dot from v_target
            wp.launch(
                damped_pinv_3xN_kernel,
                dim=1,
                inputs=[J_p_flat, v_target_wp, PINV_DAMPING],
                outputs=[q_dot],
            )

            # (e) Write q_dot as velocity target
            wp.launch(
                copy_first7_kernel,
                dim=N_JOINTS,
                inputs=[q_dot],
                outputs=[self.controls[step].joint_target_vel],
            )

            # (f) Finger position targets
            wp.launch(
                copy_fingers_from_rest_kernel,
                dim=2,
                inputs=[self.rest_fingers],
                outputs=[self.controls[step].joint_target_pos],
            )

            # (g) Physics substeps
            for sub in range(SIM_SUBSTEPS):
                t = start_substep + sub
                self.states[t].clear_forces()
                self.solver.step(
                    self.states[t],
                    self.states[t + 1],
                    self.controls[step],
                    None,
                    self.sim_dt,
                )

            # (h) Loss at end of this control step
            final_substep = (step + 1) * SIM_SUBSTEPS
            wp.launch(
                pose_loss_kernel,
                dim=1,
                inputs=[
                    self.states[final_substep].body_q,
                    self.target_pos_wp,
                    self.ee_body_index,
                    W_POS,
                    self.loss,
                ],
            )

    # ---------------------------------------------------------------------- #
    # Target sampling                                                        #
    # ---------------------------------------------------------------------- #

    def sample_target(self):
        """Sample a random target on a sphere of radius TARGET_RADIUS
        centered at the initial EE position. Rejects samples below TARGET_Z_MIN."""
        if not RANDOMIZE_TARGET:
            return  # keep fixed target

        for _ in range(100):  # rejection sampling with safety limit
            # Random direction (uniform on unit sphere)
            direction = torch.randn(3, device=self.torch_device, dtype=torch.float32)
            direction = direction / direction.norm().clamp_min(1e-8)

            target = self.initial_ee_pos + direction * TARGET_RADIUS

            if target[2].item() >= TARGET_Z_MIN:
                self.target_pos_torch = target
                self.target_pos_wp = wp.vec3(
                    float(target[0]),
                    float(target[1]),
                    float(target[2]),
                )
                return

        # Fallback: if all samples rejected, use fixed target
        self.target_pos_torch = torch.tensor(TARGET_POS, device=self.torch_device, dtype=torch.float32)
        self.target_pos_wp = wp.vec3(*TARGET_POS)

    # ---------------------------------------------------------------------- #
    # Training step                                                          #
    # ---------------------------------------------------------------------- #

    def step(self):
        # Sample new target each iteration
        self.sample_target()

        # Reset velocities + pose
        wp.launch(
            zero_joint_qd_kernel,
            dim=self.n_joint_qd,
            inputs=[self.states[0].joint_qd],
        )
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.states[0]
        )
        self.loss.zero_()

        # Zero PyTorch gradients
        self.policy_optimizer.zero_grad()

        # Forward + backward (Warp tape + PyTorch autograd)
        tape = wp.Tape()
        with tape:
            self.forward()
        tape.backward(self.loss)

        # Bridge: propagate warp gradients into PyTorch autograd
        for v_torch in self._v_target_tensors:
            if v_torch.grad is not None:
                v_torch.backward(v_torch.grad)

        # PyTorch optimizer step (updates MLP weights)
        self.policy_optimizer.step()

        # Log
        loss_val = self.loss.numpy()[0]
        self.loss_history.append(loss_val)

        if self.train_iter == 0:
            self.loss_mean = loss_val
        else:
            self.loss_mean = self.loss_mean_alpha * self.loss_mean + (1 - self.loss_mean_alpha) * loss_val

        ee_pos = wp.to_torch(
            self.states[self.total_substeps].body_q
        )[self.ee_body_index, :3].detach().cpu().numpy()

        if self.verbose:
            v_str = ""
            for i, v in enumerate(self._v_targets):
                v_str += f"\n    Step {i}: v=[{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}] m/s"

            # Print MLP gradient norm
            grad_norm = 0.0
            for p in self.policy.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            target_np = self.target_pos_torch.detach().cpu().numpy()
            print(
                f"Iter {self.train_iter:4d}  "
                f"loss={loss_val:.4f}  "
                f"loss_mean={self.loss_mean:.4f}  "
                f"EE=[{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]  "
                f"Target=[{target_np[0]:.3f}, {target_np[1]:.3f}, {target_np[2]:.3f}]  "
                f"grad_norm={grad_norm:.4f}"
                f"{v_str}"
            )

        if self.verbose and self.train_iter % 50 == 0:
            target_np = self.target_pos_torch.detach().cpu().numpy()
            ee_np = ee_pos  # zaten numpy
            err = target_np - ee_np
            dist = np.linalg.norm(err)
            print(f"    error: [{err[0]:+.3f}, {err[1]:+.3f}, {err[2]:+.3f}]  dist={dist:.3f}m")
            print(f"    v_target magnitude: {np.linalg.norm(self._v_targets[0]):.3f} m/s")

        # Record trajectory
        q_traj = np.zeros((self.total_substeps + 1, 9))
        for t in range(self.total_substeps + 1):
            q_traj[t] = self.states[t].joint_q.numpy()
        self.joint_q_history = q_traj

        self.viewer.log_scalar("/loss", loss_val)
        self.train_iter += 1

        if self.train_iter % 50 == 0:
            self.plot_joint_positions(f"joint_positions_iter{self.train_iter:04d}.png")

        tape.zero()

    # ---------------------------------------------------------------------- #
    # Render                                                                 #
    # ---------------------------------------------------------------------- #

    def render(self):
        if self.viewer.is_paused():
            self.viewer.begin_frame(self.sim_time)
            self.viewer.end_frame()
            return

        for t in range(self.total_substeps + 1):
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.states[t])
            self.viewer.log_shapes(
                "/target",
                newton.GeoType.SPHERE,
                (0.03,),
                wp.array(
                    [wp.transform(self.target_pos_wp, wp.quat_identity())],
                    dtype=wp.transform,
                ),
                wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3),
            )
            self.viewer.end_frame()
            self.sim_time += self.sim_dt

    # ---------------------------------------------------------------------- #
    # Plot                                                                   #
    # ---------------------------------------------------------------------- #

    def plot_joint_positions(self, save_path="joint_positions.png"):
        if self.joint_q_history is None:
            return
        q = self.joint_q_history
        t_axis = np.arange(q.shape[0]) * self.sim_dt
        fig, axes = plt.subplots(7, 1, figsize=(10, 12), sharex=True)
        for j in range(7):
            axes[j].plot(t_axis, q[:, j], linewidth=1.5, color="tab:blue")
            axes[j].set_ylabel(f"q{j} [rad]")
            axes[j].grid(True, alpha=0.3)
        axes[-1].set_xlabel("time [s]")
        fig.suptitle(f"Joint positions — iter {self.train_iter} (MLP policy)")
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f"Saved: {save_path}")

    # ---------------------------------------------------------------------- #
    # Test hook                                                              #
    # ---------------------------------------------------------------------- #

    def test_final(self):
        assert len(self.loss_history) > 10
        recent = self.loss_history[-max(1, len(self.loss_history) // 5):]
        assert recent[-1] < self.loss_history[0] * 0.5

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--verbose", action="store_true", default=True)
        return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)