# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Diffsim Franka — N-DOF VERSION (parametrized joint selection)
#
# Generalization of the 1-DOF demo. Pick any subset of joints to optimize
# via ACTIVE_JOINTS; the rest stay frozen at their rest-pose targets.
#
# Examples:
#   ACTIVE_JOINTS = [0]          → 1-DOF, only base rotates
#   ACTIVE_JOINTS = [0, 2]       → 2-DOF, base + upper-arm twist
#   ACTIVE_JOINTS = [0, 1, 3]    → 3-DOF, base + shoulder + elbow
#   ACTIVE_JOINTS = [0,1,2,3,4,5,6] → full 7-DOF
#
# Everything else (loss, optimizer, plotting, gradient printout) works
# identically regardless of how many joints are active.
#
# Command: python diffsim_franka_ndof.py
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

# ★ CHANGE THIS LINE to pick which joints are optimized. ★
# Indices are into the 7 arm joints (0 = base rotation, 6 = wrist roll).
# All other arm joints + 2 finger joints stay frozen at their rest-pose targets.
ACTIVE_JOINTS  = [0, 3]       # e.g. base rotation + upper-arm twist

TRAJ_STEPS     = 1
SIM_SUBSTEPS   = 50
TRAIN_ITERS    = 200
TRAIN_LR       = 1e-2
GRAD_CLIP      = 5.0
DEVICE         = "cuda:0"

# Target EE position (world frame)
TARGET_POS = (0.3, 0.3, 0.5)

# Franka rest pose
INITIAL_ARM_Q = [0.0, -0.785, 0.0, -2.356, 0.0, 2.571, 0.785]

# PD gains
JOINT_KE = [1500.0] * 7
JOINT_KD = [50.0] * 7

# Loss weight
W_POS = 10.0


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
def scatter_active_targets(
    src: wp.array(dtype=float),          # shape (n_active,) — learnable targets
    active_indices: wp.array(dtype=int), # shape (n_active,) — which joint slots
    dst: wp.array(dtype=float),          # shape (9,) — full joint_target_pos
):
    """Write each active-joint target into the correct slot of the control buffer.

    Each thread handles one active joint. Inactive joints are untouched, so
    they keep the rest-pose defaults set in the builder.
    """
    tid = wp.tid()
    joint_idx = active_indices[tid]
    dst[joint_idx] = src[tid]


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
        self.total_substeps = TRAJ_STEPS * SIM_SUBSTEPS

        # ------------------------------------------------------------------ #
        # 0. Validate ACTIVE_JOINTS                                          #
        # ------------------------------------------------------------------ #
        assert len(ACTIVE_JOINTS) > 0, "Need at least one active joint"
        assert all(0 <= j < 7 for j in ACTIVE_JOINTS), (
            f"ACTIVE_JOINTS must be in [0, 6], got {ACTIVE_JOINTS}"
        )
        assert len(set(ACTIVE_JOINTS)) == len(ACTIVE_JOINTS), (
            f"ACTIVE_JOINTS must have unique entries, got {ACTIVE_JOINTS}"
        )
        self.n_active = len(ACTIVE_JOINTS)

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

        builder.joint_target_mode[:9] = [int(newton.JointTargetMode.POSITION)] * 9
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
        # 6. Learnable params: one array of shape (n_active,) per control   #
        #    step. Initialized to the rest pose for the active joints.     #
        # ------------------------------------------------------------------ #
        initial_active = np.array(
            [INITIAL_ARM_Q[j] for j in ACTIVE_JOINTS], dtype=np.float32
        )
        self.ctrl_q = []
        for _ in range(TRAJ_STEPS):
            self.ctrl_q.append(
                wp.array(initial_active, dtype=float, requires_grad=True)
            )

        # Index array — which joint slots the active params map to.
        # Created once, reused every step. Must be a wp.array of int32.
        self.active_idx = wp.array(
            np.array(ACTIVE_JOINTS, dtype=np.int32), dtype=int
        )

        print(f"\n{'='*60}")
        print(f"N-DOF DIFFSIM: Optimizing {self.n_active} joint(s): {ACTIVE_JOINTS}")
        print(f"  Learnable params per control step: {self.n_active}")
        print(f"  Total learnable params: {self.n_active * TRAJ_STEPS}")
        print(f"  Target: {TARGET_POS}")
        print(f"  Initial q[active]: {initial_active.tolist()}")
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
        self.joint_q_history = None
        self.optimizer = warp.optim.Adam(self.ctrl_q, lr=TRAIN_LR, betas=(0.0, 0.999))

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
        """Run trajectory forward, scattering active targets at each step."""
        for step in range(TRAJ_STEPS):
            # Scatter the n_active learnable values into the full target buffer
            wp.launch(
                scatter_active_targets,
                dim=self.n_active,
                inputs=[
                    self.ctrl_q[step],
                    self.active_idx,
                    self.controls[step].joint_target_pos,
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
        # Zero velocities + reset pose
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

        # Print per-joint values and gradients
        if self.verbose:
            print(f"\nIter {self.train_iter:4d}:")
            for step_i, cq in enumerate(self.ctrl_q):
                q_vals = cq.numpy()
                grad_vals = cq.grad.numpy() if cq.grad is not None else np.zeros_like(q_vals)
                parts = []
                for k, j in enumerate(ACTIVE_JOINTS):
                    parts.append(
                        f"q{j}={q_vals[k]:+.3f}({math.degrees(q_vals[k]):+.1f}°) "
                        f"g={grad_vals[k]:+.4f}"
                    )
                print(f"  Step {step_i}: " + " | ".join(parts))

        # Gradient clipping (per control step)
        for cq in self.ctrl_q:
            if cq.grad is not None:
                grad_np = cq.grad.numpy()
                grad_norm = np.linalg.norm(grad_np)
                if grad_norm > GRAD_CLIP:
                    cq.grad = wp.array(
                        grad_np * (GRAD_CLIP / grad_norm),
                        dtype=float, requires_grad=False,
                    )

        # Update
        self.optimizer.step([cq.grad for cq in self.ctrl_q])

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

        # Record trajectory for plotting (works with or without viewer)
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
    # Render — plays back full latest trajectory each call                   #
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
        """Plot all 7 arm joints; highlight the active ones."""
        if self.joint_q_history is None:
            print("No trajectory recorded yet.")
            return

        q = self.joint_q_history
        t_axis = np.arange(q.shape[0]) * self.sim_dt
        n_joints = 7

        fig, axes = plt.subplots(n_joints, 1, figsize=(10, 12), sharex=True)
        for j in range(n_joints):
            ax = axes[j]
            is_active = j in ACTIVE_JOINTS
            color = "tab:blue" if is_active else "tab:gray"
            lw = 2.0 if is_active else 1.0
            label_suffix = "  (ACTIVE)" if is_active else "  (frozen)"
            ax.plot(t_axis, q[:, j], linewidth=lw, color=color)
            ax.set_ylabel(f"q{j} [rad]{label_suffix}")
            ax.grid(True, alpha=0.3)
            for step in range(1, TRAJ_STEPS):
                ax.axvline(
                    step * SIM_SUBSTEPS * self.sim_dt,
                    color="red", linestyle="--", alpha=0.3, linewidth=0.8,
                )

        axes[-1].set_xlabel("time [s]")
        fig.suptitle(
            f"Joint positions — iter {self.train_iter}  "
            f"(active: {ACTIVE_JOINTS})"
        )
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