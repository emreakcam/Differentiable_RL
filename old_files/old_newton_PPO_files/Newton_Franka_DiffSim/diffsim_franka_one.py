# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Diffsim Franka — 1-DOF VERSION (Joint 0 Only)
#
# Pedagogical version that optimizes ONLY the base joint (joint 0) while
# all other joints are frozen at their rest pose. This makes it easy to
# understand the full diffsim pipeline:
#
#   1. Forward: physics runs for T substeps with a position target on joint 0
#   2. Loss: squared distance from EE to a target position
#   3. Backward: tape.backward() gives ∂loss/∂target_q0 — how should we
#      change the base rotation to move the EE closer?
#   4. Adam: update target_q0
#
# Since only 1 scalar is optimized (per control step), you can print the
# gradient directly and verify it makes physical sense:
#   - Target to the right of EE → positive gradient (rotate CW)
#   - Target to the left of EE → negative gradient (rotate CCW)
#
# Command: python diffsim_franka_1dof.py
###########################################################################

import math
import sys
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import torch
import warp as wp
import warp.optim

import newton
import newton.examples
import newton.utils

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

TRAJ_STEPS     = 1           # fewer steps — simpler trajectory
SIM_SUBSTEPS   = 50          # physics substeps per control step
TRAIN_ITERS    = 200         # optimization iterations
TRAIN_LR       = 1e-2        # larger LR — single parameter, easy landscape
GRAD_CLIP      = 5.0         # max gradient magnitude
DEVICE         = "cuda:0"

# Target EE position — offset to the side so joint 0 must rotate
# (The rest pose has the EE roughly at x≈0.3, y≈0.0, z≈0.5)
# We place the target at y=0.3 so the base must rotate ~30° to reach it
TARGET_POS = (0.3, 0.3, 0.5)

# Franka rest pose — all joints frozen EXCEPT joint 0
INITIAL_ARM_Q = [0.0, -0.785, 0.0, -2.356, 0.0, 2.571, 0.785]

# PD gains
JOINT_KE = [1500.0] * 7
JOINT_KD = [50.0] * 7

# Loss weight
W_POS = 1.0


# ---------------------------------------------------------------------------
# Warp Kernels
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
def write_single_target(
    src: wp.array(dtype=float),      # shape (1,) — our learnable q0
    dst: wp.array(dtype=float),      # shape (9,) — full joint_target_pos
    joint_index: int,
):
    """Write a single learnable target into the control buffer.

    Only modifies dst[joint_index], leaving all other joints at their
    default (frozen) targets.
    """
    dst[joint_index] = src[0]


@wp.kernel
def zero_joint_qd_kernel(
    joint_qd: wp.array(dtype=float),
):
    """Zero out joint velocities."""
    tid = wp.tid()
    joint_qd[tid] = 0.0


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
        self.playback_step = 0
        self.total_substeps = TRAJ_STEPS * SIM_SUBSTEPS

        # ------------------------------------------------------------------ #
        # 1. Build Franka model                                              #
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

        # All joints in position mode
        builder.joint_target_mode[:9] = [int(newton.JointTargetMode.POSITION)] * 9

        builder.joint_target_ke[:7] = JOINT_KE
        builder.joint_target_ke[7:9] = [100.0, 100.0]
        builder.joint_target_kd[:7] = JOINT_KD
        builder.joint_target_kd[7:9] = [10.0, 10.0]
        builder.joint_armature[:9] = [0.3, 0.3, 0.3, 0.3, 0.11, 0.11, 0.11, 0.15, 0.15]

        # Set ALL targets to rest pose — joints 1-6 + fingers stay here
        builder.joint_target_pos[:7] = INITIAL_ARM_Q
        builder.joint_target_pos[7:9] = [0.04, 0.04]

        scene = newton.ModelBuilder()
        scene.replicate(builder, 1)
        scene.add_ground_plane()
        self.model = scene.finalize(requires_grad=True)

        # ------------------------------------------------------------------ #
        # 2. Solver                                                          #
        # ------------------------------------------------------------------ #
        self.solver = newton.solvers.SolverFeatherstone(self.model)

        # ------------------------------------------------------------------ #
        # 3. States                                                          #
        # ------------------------------------------------------------------ #
        self.states = []
        for _ in range(self.total_substeps + 1):
            self.states.append(self.model.state(requires_grad=True))
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.states[0]
        )

        self.n_joint_qd = len(self.model.joint_qd.numpy())

        # ------------------------------------------------------------------ #
        # 4. Controls                                                        #
        # ------------------------------------------------------------------ #
        self.controls = []
        for _ in range(TRAJ_STEPS):
            self.controls.append(self.model.control())

        # ------------------------------------------------------------------ #
        # 5. Find EE body                                                    #
        # ------------------------------------------------------------------ #
        self.ee_body_index = -1
        for i, label in enumerate(self.model.body_label):
            if "fr3_hand" in label and "tcp" not in label and "finger" not in label:
                self.ee_body_index = i
                break
        if self.ee_body_index < 0:
            raise RuntimeError("Could not find fr3_hand body")
        print(f"EE body: {self.ee_body_index} ({self.model.body_label[self.ee_body_index]})")

        # ------------------------------------------------------------------ #
        # 6. THE KEY: Only ONE learnable parameter per control step          #
        #                                                                    #
        # ctrl_q0[t] is the position target for joint 0 at control step t.   #
        # Shape: (1,) — a single scalar.                                     #
        # All other joints keep their rest-pose targets (frozen).            #
        # ------------------------------------------------------------------ #
        self.ctrl_q0 = []
        for _ in range(TRAJ_STEPS):
            self.ctrl_q0.append(
                wp.array([INITIAL_ARM_Q[0]], dtype=float, requires_grad=True)
            )
        self.joint_0_index = 0  # which joint we're optimizing

        print(f"\n{'='*60}")
        print(f"1-DOF DIFFSIM: Optimizing joint 0 (base rotation) only")
        print(f"  Learnable params: {TRAJ_STEPS} scalars (one per control step)")
        print(f"  Total params: {TRAJ_STEPS}")
        print(f"  Target: {TARGET_POS}")
        print(f"  Initial q0: {INITIAL_ARM_Q[0]:.3f} rad")
        print(f"{'='*60}\n")

        # ------------------------------------------------------------------ #
        # 7. Target                                                          #
        # ------------------------------------------------------------------ #
        self.target_pos = wp.vec3(*TARGET_POS)

        # ------------------------------------------------------------------ #
        # 8. Loss and optimizer                                              #
        # ------------------------------------------------------------------ #
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.loss_history = []

        # Joint position history for plotting
        # Will be filled each render() call with the latest trajectory
        self.joint_q_history = None

        self.optimizer = warp.optim.Adam(
            self.ctrl_q0,
            lr=TRAIN_LR,
        )

        # ------------------------------------------------------------------ #
        # 9. Viewer                                                          #
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
        """Run trajectory forward, writing only joint 0 target at each step."""
        for step in range(TRAJ_STEPS):
            # Write ONLY joint 0 target — all others stay at rest pose
            wp.launch(
                write_single_target,
                dim=1,
                inputs=[
                    self.ctrl_q0[step],
                    self.controls[step].joint_target_pos,
                    self.joint_0_index,
                ],
            )

            # Physics substeps
            for sub in range(SIM_SUBSTEPS):
                t = step * SIM_SUBSTEPS + sub
                self.states[t].clear_forces()
                self.solver.step(
                    self.states[t],
                    self.states[t + 1],
                    self.controls[step],
                    None,
                    self.sim_dt,
                )

            # Loss at end of this control step
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
        # Zero velocities before reset
        wp.launch(
            zero_joint_qd_kernel,
            dim=self.n_joint_qd,
            inputs=[self.states[0].joint_qd],
        )

        # Reset to initial pose
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.states[0]
        )
        self.loss.zero_()

        # Forward + backward
        tape = wp.Tape()
        with tape:
            self.forward()
        tape.backward(self.loss)

        # ------------------------------------------------------------------ #
        # Print the gradient — since it's a single scalar, you can reason    #
        # about it directly:                                                 #
        #   ∂loss/∂q0 > 0 → increasing q0 (rotating CCW) increases loss      #
        #                    → optimizer will decrease q0 (rotate CW)        #
        #   ∂loss/∂q0 < 0 → increasing q0 decreases loss                     #
        #                    → optimizer will increase q0 (rotate CCW)       #
        # ------------------------------------------------------------------ #
        if self.verbose:
            print(f"\nIter {self.train_iter:4d}:")
            for i, cq in enumerate(self.ctrl_q0):
                q0_val = cq.numpy()[0]
                grad_val = cq.grad.numpy()[0] if cq.grad is not None else 0.0
                print(f"  Step {i}: q0={q0_val:+.4f} rad ({math.degrees(q0_val):+.1f}°)  "
                      f"grad={grad_val:+.6f}")

        # Gradient clipping
        for cq in self.ctrl_q0:
            if cq.grad is not None:
                grad_np = cq.grad.numpy()
                grad_norm = np.linalg.norm(grad_np)
                if grad_norm > GRAD_CLIP:
                    cq.grad = wp.array(
                        grad_np * (GRAD_CLIP / grad_norm),
                        dtype=float, requires_grad=False,
                    )

        # Update
        self.optimizer.step([cq.grad for cq in self.ctrl_q0])

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

        self.viewer.log_scalar("/loss", loss_val)
        self.train_iter += 1
        if self.train_iter % 50 == 0:
            self.plot_joint_positions(f"joint_positions_iter{self.train_iter:04d}.png")
        tape.zero()

    # ---------------------------------------------------------------------- #
    # Render                                                                 #
    # ---------------------------------------------------------------------- #

    def render(self):
        """Play back the FULL latest trajectory each call.

        Also records joint positions at every substep for later plotting.
        """
        if self.viewer.is_paused():
            self.viewer.begin_frame(self.sim_time)
            self.viewer.end_frame()
            return

        # Record joint positions for the entire latest trajectory
        # Shape: (total_substeps + 1, n_joints)
        q_traj = np.zeros((self.total_substeps + 1, 9))
        for t in range(self.total_substeps + 1):
            q_traj[t] = self.states[t].joint_q.numpy()
        self.joint_q_history = q_traj

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

    def plot_joint_positions(self, save_path="joint_positions.png"):
        """Plot joint positions over time for the latest trajectory.

        Also marks control-step boundaries with vertical dashed lines so you
        can see where the position targets change discretely.
        """
        if self.joint_q_history is None:
            print("No trajectory recorded yet — call render() first.")
            return

        q = self.joint_q_history                    # (total_substeps+1, 9)
        t_axis = np.arange(q.shape[0]) * self.sim_dt  # seconds

        # 7 arm joints + 2 finger joints = 9 total; plot only the 7 arm joints
        n_joints = 7
        fig, axes = plt.subplots(n_joints, 1, figsize=(10, 12), sharex=True)

        for j in range(n_joints):
            ax = axes[j]
            ax.plot(t_axis, q[:, j], linewidth=1.5)
            ax.set_ylabel(f"q{j} [rad]")
            ax.grid(True, alpha=0.3)
            # Vertical lines at control-step boundaries
            for step in range(1, TRAJ_STEPS):
                ax.axvline(
                    step * SIM_SUBSTEPS * self.sim_dt,
                    color="red", linestyle="--", alpha=0.3, linewidth=0.8,
                )

        axes[-1].set_xlabel("time [s]")
        fig.suptitle(f"Joint positions — iter {self.train_iter}")
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f"Saved joint position plot to {save_path}")

    # ---------------------------------------------------------------------- #
    # Test hook                                                              #
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
        parser.add_argument(
            "--verbose", action="store_true", default=True,
            help="Print loss and gradient each iteration.",
        )
        return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)


# ---------------------------------------------------------------------------
# WHY 1-DOF?
# ---------------------------------------------------------------------------
#
# With 7 joints × 10 control steps = 70 parameters, it's hard to build
# intuition for what the gradients mean. With 1 DOF:
#
#   - There's 1 parameter per control step (5 total)
#   - Each gradient is a single number you can reason about:
#     "The EE is to the LEFT of the target, so ∂loss/∂q0 < 0,
#      meaning increasing q0 (rotating base CCW) reduces loss"
#   - You can verify the optimizer does what you'd expect by hand
#   - You can see that the gradient flows through the full chain:
#     loss ← EE position ← body_q ← physics substeps ← joint target ← q0
#
# Once this makes sense, the 7-DOF version is the same thing but with
# a 7-element gradient vector instead of a scalar.
# ---------------------------------------------------------------------------