# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Diffsim Franka — EE-VELOCITY VERSION (task-space control)
#
# Learnable parameter: linear EE velocity in the world frame, v_target ∈ ℝ³,
# one 3-vector per control step.
#
# Control pipeline (every control step):
#   1. Copy current joint angles q_current from states[step*SUBSTEPS].joint_q
#   2. Warp FK kernel       → EE pos, joint world positions, joint world axes
#   3. Warp Jacobian kernel → J_p ∈ ℝ^{3×7} (geometric position Jacobian at TCP)
#   4. Warp damped-pinv     → q_dot = Jᵀ (J Jᵀ + λI)⁻¹ v_target
#   5. Warp integrate       → q_target[:7] = q_current[:7] + q_dot * frame_dt
#   6. Copy rest fingers    → q_target[7:9] = [0.04, 0.04]
#   7. Physics substeps
#
# Everything runs under the Warp tape — gradients flow:
#   loss ← body_q ← physics ← q_target ← (integrate ← pinv ← J ← FK) ← v_target
#                                           ↑
#                             also ← q_current from previous physics state
#
# This gives the optimizer a task-space action space: "commanded EE velocity"
# instead of "per-joint position target". Expected to converge faster and
# produce more interpretable trajectories than the joint-space N-DOF version.
#
# Command: python diffsim_franka_ee_vel.py
###########################################################################

import math
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import torch
import warp as wp
import warp.optim

import newton
import newton.examples
import newton.utils

# Local imports — helpers are the torch-side ones used for URDF parsing only.
# The Warp-native kinematics live in warp_kinematics.py.
from helpers.kinematic_helpers import parse_urdf_kinematic_chain
from helpers.warp_kinematics import (
    N_JOINTS,
    fk_tcp_kernel,
    jacobian_p_tcp_kernel,
    damped_pinv_3xN_kernel,
    integrate_q_kernel,
    load_chain_to_warp,
)
from helpers.geom_diff_helpers import rpy_to_quat_wxyz, transform_by_quat_diff, transform_quat_by_quat_diff

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SIM_STEPS = 2
TRAJ_STEPS     = 1     #1
SIM_SUBSTEPS   = 100     #100
TRAIN_ITERS    = 200
TRAIN_LR       = 3e-2         # v_target is in m/s; large LR is appropriate
GRAD_CLIP      = 50.0
PINV_DAMPING   = 0.01        # λ in (JJᵀ + λI)⁻¹
DEVICE         = "cuda:0"

# Target EE position (world frame)
TARGET_POS = (0.4, 0.1, 0.6)

# Franka rest pose
INITIAL_ARM_Q = [0.0, -0.785, 0.0, -2.356, 0.0, 2.571, 0.785]

# PD gains
JOINT_KE = [0.0] * 7
JOINT_KD = [450, 450, 350, 350, 200, 200, 200]

# Loss weight
W_POS = 10.0

# Initial guess for v_target (m/s). Start at zero — optimizer finds direction.
V_TARGET_INIT = [0.0, 0.0, 0.0]
V_MAX = 3.0  # m/s — Franka'nın makul EE hızı


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
    """Squared EE position error → loss. Launched with dim=1."""
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
    src: wp.array(dtype=float),   # length >= 7
    dst: wp.array(dtype=float),   # length 7
):
    tid = wp.tid()
    dst[tid] = src[tid]


@wp.kernel
def copy_fingers_from_rest_kernel(
    rest_fingers: wp.array(dtype=float),   # [2]
    joint_target: wp.array(dtype=float),   # [9]
):
    """Keep finger targets at rest values. integrate_q_kernel only wrote
    the first 7 slots; this fills the remaining 2."""
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
        self.frame_dt = 1.0 / 10.0         #10
        self.sim_dt = self.frame_dt / SIM_SUBSTEPS
        self.train_iter = 0
        self.total_substeps = SIM_STEPS * SIM_SUBSTEPS

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
        for _ in range(SIM_STEPS):
            self.controls.append(self.model.control())

        # ------------------------------------------------------------------ #
        # 3. Find EE body — use fr3_hand_tcp (grasp center, matches RL env)  #
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
            device=torch.device(DEVICE),
        )

        # FIX: parser fr3_link7 → fr3_hand arasındaki fixed joint'ları
        # ee_offset'e katamıyor. URDF'ten elle ekliyoruz:
        #   fr3_link7 → fr3_link8: xyz=[0, 0, 0.107]  rpy=[0, 0, 0]
        #   fr3_link8 → fr3_hand:  xyz=[0, 0, 0]       rpy=[0, 0, -0.7854]
        fix_p1 = torch.tensor([0.0, 0.0, 0.107], device=torch.device(DEVICE), dtype=torch.float32)
        fix_q1 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=torch.device(DEVICE), dtype=torch.float32)
        fix_p2 = torch.tensor([0.0, 0.0, 0.0], device=torch.device(DEVICE), dtype=torch.float32)
        fix_q2 = torch.tensor(rpy_to_quat_wxyz(0.0, 0.0, -0.7853981633974483), device=torch.device(DEVICE), dtype=torch.float32)

        # Compose: (fix_p1, fix_q1) then (fix_p2, fix_q2)
        composed_p = fix_p1 + transform_by_quat_diff(fix_p2, fix_q1)
        composed_q = transform_quat_by_quat_diff(fix_q1, fix_q2)

        # Add grasp center offset (fingertip midpoint) — matches RL env
        grasp_offset_hand = torch.tensor([0.0, 0.0, 0.1034], device=torch.device(DEVICE), dtype=torch.float32)
        composed_p = composed_p + transform_by_quat_diff(grasp_offset_hand, composed_q)

        chain["ee_offset_p"] = composed_p
        chain["ee_offset_q"] = composed_q

        assert len(chain["joint_names"]) == N_JOINTS, (
            f"Expected {N_JOINTS} revolute joints, got {len(chain['joint_names'])}"
        )
        print(f"Parsed chain: {chain['joint_names']}")
        print(f"TCP offset: pos={chain['ee_offset_p'].cpu().numpy().tolist()}")
        print(f"TCP offset: quat={chain['ee_offset_q'].cpu().numpy().tolist()}")

        self.chain_wp = load_chain_to_warp(chain, device=DEVICE)
        # ------------------------------------------------------------------ #
        # 5. Per-control-step scratch buffers                                #
        #                                                                    #
        # Each control step needs its own set of intermediates so every      #
        # forward-pass node has a distinct tape slot. Reusing a single buffer#
        # across steps would overwrite values the tape still needs for       #
        # backprop.                                                          #
        # ------------------------------------------------------------------ #
        def _mk_vec3(n): return wp.zeros(n, dtype=wp.vec3, device=DEVICE, requires_grad=True)
        def _mk_float(n): return wp.zeros(n, dtype=float,   device=DEVICE, requires_grad=True)

        # Constant: finger rest values
        self.rest_fingers = wp.array([0.04, 0.04], dtype=float, device=DEVICE)

        # ------------------------------------------------------------------ #
        # 6. Learnable EE-velocity targets                                   #
        # ------------------------------------------------------------------ #
        self.ctrl_v = []
        for _ in range(TRAJ_STEPS):
            self.ctrl_v.append(
                wp.array(V_TARGET_INIT, dtype=float, device=DEVICE, requires_grad=True)
            )

        print(f"\n{'='*60}")
        print(f"EE-VELOCITY DIFFSIM (task-space control)")
        print(f"  Learnable params per control step: 3 (vx, vy, vz in world frame)")
        print(f"  Total learnable params: {3 * TRAJ_STEPS}")
        print(f"  Pinv damping λ: {PINV_DAMPING}")
        print(f"  Target: {TARGET_POS}")
        print(f"{'='*60}\n")

        # ------------------------------------------------------------------ #
        # 7. Target, loss, optimizer                                         #
        # ------------------------------------------------------------------ #
        self.target_pos = wp.vec3(*TARGET_POS)
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.loss_history = []
        self.joint_q_history = None
        self.optimizer = warp.optim.Adam(self.ctrl_v, lr=TRAIN_LR, betas=(0.0, 0.999))

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
        """Run the full trajectory forward under the Warp tape."""
        for step in range(SIM_STEPS):
            start_substep = step * SIM_SUBSTEPS

            # Hangi traj point'e karşılık geliyor?
            traj_idx = step * TRAJ_STEPS // SIM_STEPS  # TRAJ_STEPS=1 ise hep 0

            # Scratch buffers — her step için taze
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

            # (c) Position Jacobian at current q
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
                inputs=[J_p_flat, self.ctrl_v[traj_idx], PINV_DAMPING],
                outputs=[q_dot],
            )

            # (e) Velocity targets for arm joints
            wp.launch(
                copy_first7_kernel,
                dim=N_JOINTS,
                inputs=[q_dot],
                outputs=[self.controls[step].joint_target_vel],
            )

            # (f) Finger targets
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

            # (h) Loss only at the final step
            if step == SIM_STEPS - 1:
                final_substep = (step + 1) * SIM_SUBSTEPS
                wp.launch(
                    pose_loss_kernel,
                    dim=1,
                    inputs=[
                        self.states[final_substep].body_q,
                        self.target_pos,
                        self.ee_body_index,
                        W_POS,
                        self.loss,
                    ],
                )

    # ---------------------------------------------------------------------- #
    # Training step                                                          #
    # ---------------------------------------------------------------------- #

    def step(self):
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

        # Forward + backward
        tape = wp.Tape()
        with tape:
            self.forward()
        tape.backward(self.loss)

        # Print per-step v_target and gradient
        if self.verbose:
            print(f"\nIter {self.train_iter:4d}:")
            for i, cv in enumerate(self.ctrl_v):
                v = cv.numpy()
                g = cv.grad.numpy() if cv.grad is not None else np.zeros(3)
                print(
                    f"  Step {i}: "
                    f"v=[{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}] m/s  "
                    f"grad=[{g[0]:+.4f}, {g[1]:+.4f}, {g[2]:+.4f}]"
                )

        # Gradient clipping
        for cv in self.ctrl_v:
            if cv.grad is not None:
                g = cv.grad.numpy()
                n = np.linalg.norm(g)
                if n > GRAD_CLIP:
                    cv.grad = wp.array(
                        g * (GRAD_CLIP / n),
                        dtype=float, requires_grad=False, device=DEVICE,
                    )

        # Adam update
        self.optimizer.step([cv.grad for cv in self.ctrl_v])

        # 3. Clamp v_target to physical limits
        for cv in self.ctrl_v:
            v_np = cv.numpy()
            v_clamped = np.clip(v_np, -V_MAX, V_MAX)
            if not np.allclose(v_np, v_clamped):
                cv.assign(wp.array(v_clamped, dtype=float, requires_grad=True, device=DEVICE))

        # Log
        loss_val = self.loss.numpy()[0]
        self.loss_history.append(loss_val)

        ee_pos = wp.to_torch(self.states[self.total_substeps].body_q)[self.ee_body_index, :3]
        if self.verbose:
            print(
                f"  Loss={loss_val:.4f}  "
                f"EE=[{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]  "
                f"Target=[{TARGET_POS[0]:.3f}, {TARGET_POS[1]:.3f}, {TARGET_POS[2]:.3f}]"
            )

        # Record trajectory
        q_traj = np.zeros((self.total_substeps + 1, 9))
        for t in range(self.total_substeps + 1):
            q_traj[t] = self.states[t].joint_q.numpy()
        self.joint_q_history = q_traj

        self.viewer.log_scalar("/loss", loss_val)
        self.train_iter += 1

        if self.train_iter % 50 == 0:
            self.plot_joint_positions(f"joint_positions_iter{self.train_iter:04d}.png")

        if self.train_iter % 20 == 0:
            self.plot_ee_velocity(f"ee_velocity_iter{self.train_iter:04d}.png")

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
                    [wp.transform(self.target_pos, wp.quat_identity())],
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
            print("No trajectory recorded yet.")
            return

        q = self.joint_q_history
        t_axis = np.arange(q.shape[0]) * self.sim_dt
        fig, axes = plt.subplots(7, 1, figsize=(10, 12), sharex=True)
        for j in range(7):
            axes[j].plot(t_axis, q[:, j], linewidth=1.5, color="tab:blue")
            axes[j].set_ylabel(f"q{j} [rad]")
            axes[j].grid(True, alpha=0.3)
            for step in range(1, TRAJ_STEPS):
                axes[j].axvline(
                    step * SIM_SUBSTEPS * self.sim_dt,
                    color="red", linestyle="--", alpha=0.3, linewidth=0.8,
                )
        axes[-1].set_xlabel("time [s]")
        fig.suptitle(f"Joint positions — iter {self.train_iter} (EE-velocity control)")
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f"Saved joint position plot to {save_path}")

    def plot_ee_velocity(self, save_path="ee_velocity.png"):
        """Plot achieved EE velocity vs commanded (both using current Jacobian)."""
        from helpers.kinematic_helpers import forward_kinematics_jacobian_analytical

        n_substeps = self.total_substeps + 1
        ee_vel_achieved = np.zeros((n_substeps, 3))
        ee_vel_commanded = np.zeros((n_substeps, 3))

        cmd_qd = self.controls[0].joint_target_vel.numpy()[:7]
        v_target = self.ctrl_v[0].numpy()

        chain = parse_urdf_kinematic_chain(
            self.urdf_path, "fr3_link0", "fr3_link7", torch.device(DEVICE)
        )

        for t in range(n_substeps):
            q_cur = self.states[t].joint_q.numpy()[:7]
            qd_actual = self.states[t].joint_qd.numpy()[:7]

            q_torch = torch.tensor(q_cur, device=torch.device(DEVICE), dtype=torch.float32)
            J_p, _ = forward_kinematics_jacobian_analytical(
                q_torch, chain["joint_axes"], chain["p_rel"], chain["q_rel"]
            )
            J_np = J_p.detach().cpu().numpy()

            ee_vel_achieved[t] = J_np @ qd_actual
            ee_vel_commanded[t] = J_np @ cmd_qd

        t_axis = np.arange(n_substeps) * self.sim_dt

        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        labels = ['vx', 'vy', 'vz']
        colors = ['tab:blue', 'tab:orange', 'tab:green']

        for i in range(3):
            axes[i].plot(t_axis, ee_vel_achieved[:, i], linewidth=1.5, color=colors[i], label=f'achieved {labels[i]}')
            axes[i].plot(t_axis, ee_vel_commanded[:, i], linewidth=1.5, linestyle='--', color=colors[i], alpha=0.7, label=f'commanded {labels[i]}')
            axes[i].axhline(v_target[i], color='gray', linestyle=':', alpha=0.4, label=f'v_target {labels[i]}')
            axes[i].set_ylabel(f'{labels[i]} [m/s]')
            axes[i].legend(loc='upper right', fontsize=8)
            axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel('time [s]')
        fig.suptitle(f'EE velocity tracking — iter {self.train_iter}')
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f"Saved: {save_path}")

    # ---------------------------------------------------------------------- #
    # Test hook                                                               #
    # ---------------------------------------------------------------------- #

    def test_final(self):
        assert len(self.loss_history) > 10
        recent = self.loss_history[-max(1, len(self.loss_history) // 5):]
        assert recent[-1] < self.loss_history[0] * 0.5, (
            f"Loss did not decrease: {self.loss_history[0]:.4f} -> {recent[-1]:.4f}"
        )

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