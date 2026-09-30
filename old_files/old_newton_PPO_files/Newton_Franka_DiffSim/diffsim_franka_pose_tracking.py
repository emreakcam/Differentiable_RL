# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Diffsim Franka Pose Tracking
#
# Optimises a joint-angle trajectory so the Franka Panda end-effector
# tracks a sequence of target poses.  Gradients flow through the
# differentiable PyTorch FK helpers (forward_kinematics_batch, etc.)
# already used by the PPO environment.
#
# Training loop mirrors example_diffsim_bear.py:
#   forward()  – roll out trajectory, accumulate FK pose-error loss
#   backward() – torch.autograd through FK
#   step()     – Adam update, log, visualise one playback frame
#
# Extend to full Newton-physics diffsim by replacing the FK-only forward
# pass with a warp.Tape recording of newton.eval_fk + solver.step (see
# comment block at bottom of this file).
#
# Command: python diffsim_franka_pose_tracking.py
###########################################################################

import math
import sys
from pathlib import Path

import torch
import warp as wp
import newton
import newton.examples
import newton.utils

sys.path.insert(0, str(Path(__file__).parent))

from helpers.geom_diff_helpers import (
    quat_error_to_rotvec,
    transform_by_quat_diff,
    transform_quat_by_quat_diff,
)
from helpers.kinematic_helpers import (
    forward_kinematics_batch,
    parse_urdf_kinematic_chain,
    solve_ik_batch,
)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

TRAJ_STEPS   = 150        # trajectory length (1 s at 60 fps)
SIM_SUBSTEPS = 10        # Newton substeps per trajectory step (visualisation)
TRAIN_ITERS  = 500
TRAIN_LR     = 5e-3
POS_WEIGHT   = 100.0
ROT_WEIGHT   = 1.0       # weight on orientation loss term
W_ACC        = 0.001
W_VEL        = 0.0
DEVICE       = "cuda:0"

# Target EE poses: list of (pos [3], quat_wxyz [4]) tuples.
# You can add as many as you like; the trajectory is shared across all targets.
TARGET_EE_POSES = [
    # In front of the robot, slightly above table
    (torch.tensor([0.5,  0.0, 0.5]), torch.tensor([1.0, 0.0, 0.0, 0.0])),
    (torch.tensor([0.4,  0.2, 0.4]), torch.tensor([1.0, 0.0, 0.0, 0.0])),
    (torch.tensor([0.45,-0.2, 0.45]), torch.tensor([1.0, 0.0, 0.0, 0.0])),
]

INITIAL_ARM_Q = [0.0, -0.785, 0.0, -2.356, 0.0, 2.571, 0.785]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def fk_ee(
    q_batch:     torch.Tensor,   # [B, 7]
    joint_axes:  torch.Tensor,   # [7, 3]
    link_pos:    torch.Tensor,   # [7, 3]
    link_quats:  torch.Tensor,   # [7, 4]
    ee_offset_p: torch.Tensor,   # [3]
    ee_offset_q: torch.Tensor,   # [4]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch FK with fixed EE offset. Returns (pos [B,3], quat [B,4])."""
    pos, quat = forward_kinematics_batch(q_batch, joint_axes, link_pos, link_quats)
    pos  = pos  + transform_by_quat_diff(ee_offset_p.expand(q_batch.shape[0], -1), quat)
    quat = transform_quat_by_quat_diff(quat, ee_offset_q.expand(q_batch.shape[0], -1))
    return pos, quat


def pose_loss(
    pos:         torch.Tensor,   # [T, 3] or [3]
    quat:        torch.Tensor,   # [T, 4] or [4]
    tgt_pos:     torch.Tensor,   # [T, 3] or [3]
    tgt_quat:    torch.Tensor,   # [T, 4] or [4]
    pos_w:       float = POS_WEIGHT,
    rot_w:       float = ROT_WEIGHT,
) -> torch.Tensor:
    """Scalar pose-tracking loss: L2 position + rotation-vector orientation."""
    pos_err  = (pos - tgt_pos).pow(2).sum(dim=-1).mean()
    rot_err  = quat_error_to_rotvec(tgt_quat, quat).pow(2).sum(dim=-1).mean()
    return pos_w * pos_err + rot_w * rot_err


# ---------------------------------------------------------------------------
# Main Example class  (mirrors bear example API)
# ---------------------------------------------------------------------------

class Example:
    def __init__(self, viewer, args):
        self.viewer     = viewer
        self.device     = torch.device(DEVICE)
        self.sim_time   = 0.0
        self.frame_dt   = 1.0 / 60.0
        self.sim_dt     = self.frame_dt / SIM_SUBSTEPS
        self.train_iter = 0
        self.verbose    = getattr(args, "verbose", True)
        self.playback_step = 0

        # ------------------------------------------------------------------ #
        # 1. Load URDF & build Newton model (for visualisation)              #
        # ------------------------------------------------------------------ #
        self.urdf_path = str(
            newton.utils.download_asset("franka_emika_panda")
            / "urdf/fr3_franka_hand.urdf"
        )

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_urdf(
            self.urdf_path,
            xform=wp.transform_identity(),
            floating=False,
            enable_self_collisions=False,
        )
        # Start from the canonical rest pose
        builder.joint_q[:7] = INITIAL_ARM_Q
        builder.joint_q[7:9] = [0.04, 0.04]

        scene = newton.ModelBuilder()
        scene.replicate(builder, 1)
        scene.add_ground_plane()
        self.model = scene.finalize()

        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "renderer"):
            self.viewer.set_camera(wp.vec3(1.5, -1.0, 1.2), pitch=-20.0, yaw=130.0)

        # ------------------------------------------------------------------ #
        # 2. Parse kinematic chain (same as PPO env)                         #
        # ------------------------------------------------------------------ #
        chain = parse_urdf_kinematic_chain(
            urdf_path = self.urdf_path,
            root_link = "fr3_link0",
            ee_link   = "fr3_hand",
            device    = self.device,
        )
        self.joint_axes  = chain["joint_axes"]   # [7, 3]
        self.link_pos    = chain["p_rel"]         # [7, 3]
        self.link_quats  = chain["q_rel"]         # [7, 4]
        self.joint_limits= chain["joint_ranges"]  # [7, 2]
        self.ee_offset_p = chain["ee_offset_p"]   # [3]
        self.ee_offset_q = chain["ee_offset_q"]   # [4]
        n_joints = self.joint_axes.shape[0]       # 7

        # Grasp-centre offset (same value as the PPO env)
        grasp_offset = torch.tensor([0.0, 0.0, 0.1034], device=self.device)
        self.ee_offset_p = self.ee_offset_p + transform_by_quat_diff(
            grasp_offset, self.ee_offset_q
        )

        # ------------------------------------------------------------------ #
        # 3. Solve IK for each target pose → warm-start trajectory           #
        # ------------------------------------------------------------------ #
        q_rest = torch.tensor(INITIAL_ARM_Q, device=self.device, dtype=torch.float32)

        targets = TARGET_EE_POSES
        n_targets = len(targets)
        tgt_pos_list, tgt_quat_list = zip(*targets)
        tgt_pos  = torch.stack(tgt_pos_list).to(self.device)   # [K, 3]
        tgt_quat = torch.stack(tgt_quat_list).to(self.device)  # [K, 4]
        tgt_quat = _normalize_quat(tgt_quat)

        q_ik = solve_ik_batch(
            target_pos  = tgt_pos,
            target_quat = tgt_quat,
            joint_axes  = self.joint_axes,
            link_pos    = self.link_pos,
            link_quats  = self.link_quats,
            ee_offset_p = self.ee_offset_p,
            ee_offset_q = self.ee_offset_q,
            joint_limits= self.joint_limits,
            q_init      = q_rest.unsqueeze(0).expand(n_targets, -1),
            n_iters     = 100,
            rot_weight  = ROT_WEIGHT,
        )  # [K, 7]

        # ------------------------------------------------------------------ #
        # 4. Learnable trajectory  q_traj : [T, 7]                           #
        # ------------------------------------------------------------------ #
        # Warm-start: interpolate from rest → first IK solution over T steps.
        # Build waypoints: rest → IK1 → IK2 → IK3
        waypoints = torch.stack([q_rest, *[q_ik[i] for i in range(n_targets)]])  # [K+1, 7]
        n_segments = len(waypoints) - 1  # 3 segments
        steps_per_segment = TRAJ_STEPS // n_segments  # 300 // 3 = 100

        q_warm_parts = []
        for seg in range(n_segments):
            n_steps = steps_per_segment if seg < n_segments - 1 else TRAJ_STEPS - seg * steps_per_segment
            alpha = torch.linspace(0.0, 1.0, n_steps, device=self.device).unsqueeze(1)
            part = (1 - alpha) * waypoints[seg].unsqueeze(0) + alpha * waypoints[seg + 1].unsqueeze(0)
            q_warm_parts.append(part)

        q_warm = torch.cat(q_warm_parts, dim=0)  # [T, 7]
        self.q_traj = torch.nn.Parameter(q_warm)   # [T, 7]  – differentiable

        # Build per-step targets by repeating IK solutions across trajectory
        # (evenly spaced; final fraction of steps tracks the last target)
        steps_per_target = math.ceil(TRAJ_STEPS / n_targets)
        tgt_pos_full  = tgt_pos.repeat_interleave(steps_per_target, dim=0)[:TRAJ_STEPS]
        tgt_quat_full = tgt_quat.repeat_interleave(steps_per_target, dim=0)[:TRAJ_STEPS]
        self.tgt_pos_full  = tgt_pos_full   # [T, 3]  constant
        self.tgt_quat_full = tgt_quat_full  # [T, 4]  constant

        # --- init içine eklenecekler ---
        n_targets = len(TARGET_EE_POSES)
        # Hedefleri yörünge boyunca eşit aralıklarla dağıtır (0, 99, 199 gibi)
        self.target_indices = torch.linspace(0, TRAJ_STEPS - 1, steps=n_targets).long().to(self.device)

        # Hedefleri tensor'e dönüştür
        tgt_pos_list, tgt_quat_list = zip(*TARGET_EE_POSES)
        self.sparse_tgt_pos = torch.stack(tgt_pos_list).to(self.device)
        self.sparse_tgt_quat = _normalize_quat(torch.stack(tgt_quat_list).to(self.device))

        # ------------------------------------------------------------------ #
        # 5. Optimiser                                                        #
        # ------------------------------------------------------------------ #
        self.optimizer    = torch.optim.Adam([self.q_traj], lr=TRAIN_LR)
        self.loss_history: list[float] = []

    # ---------------------------------------------------------------------- #
    # Forward pass: roll out trajectory via FK, compute loss                 #
    # ---------------------------------------------------------------------- #

    def forward(self) -> torch.Tensor:
        """Compute FK for all T steps, return scalar pose-tracking loss."""
        # Clamp to joint limits throughout (soft, differentiable)
        q_clamped = torch.clamp(
            self.q_traj,
            self.joint_limits[:, 0],
            self.joint_limits[:, 1],
        )  # [T, 7]

        ee_pos, ee_quat = fk_ee(
            q_clamped,
            self.joint_axes,
            self.link_pos,
            self.link_quats,
            self.ee_offset_p,
            self.ee_offset_q,
        )  # [T, 3], [T, 4]

        # 3. Select only the positions in the target indices
        selected_pos = ee_pos[self.target_indices]
        selected_quat = ee_quat[self.target_indices]

        # 4. Calculate the error using SUM (total) for accuracy
        # We calculate the error for each target separately and sum them (we do not use the mean)
        pos_err = (selected_pos - self.sparse_tgt_pos).pow(2).sum() 
        rot_err = quat_error_to_rotvec(self.sparse_tgt_quat, selected_quat).pow(2).sum()
        
        total_loss = (POS_WEIGHT * pos_err) + (ROT_WEIGHT * rot_err)

        # 5. Smoothness - Essential to prevent interpolated points from becoming nonsensical
        velocity = q_clamped[1:] - q_clamped[:-1]
        acceleration = velocity[1:] - velocity[:-1]
        
        # If you keep the W_ACC value low, the quats will align more “stiffly” with the targets
        total_loss += W_ACC * acceleration.pow(2).sum() 
        total_loss += W_VEL * velocity.pow(2).sum() 
        
        return total_loss

    # ---------------------------------------------------------------------- #
    # One optimisation step                                                   #
    # ---------------------------------------------------------------------- #

    def step(self):
        self.optimizer.zero_grad()
        loss = self.forward()
        loss.backward()

        # Gradient clipping (keeps early updates stable)
        torch.nn.utils.clip_grad_norm_([self.q_traj], max_norm=1.0)
        self.optimizer.step()

        loss_val = float(loss.detach())
        self.loss_history.append(loss_val)
        if self.verbose:
            print(f"Iter {self.train_iter:4d}  loss={loss_val:.6f}")
        self.viewer.log_scalar("/loss", loss_val)
        self.train_iter += 1

        # Advance playback pointer for render()
        self.playback_step = (self.playback_step + 1) % TRAJ_STEPS

    # ---------------------------------------------------------------------- #
    # Render: apply current trajectory step to Newton model                  #
    # ---------------------------------------------------------------------- #

    def render(self):
        if self.viewer.is_paused():
            self.viewer.begin_frame(self.sim_time)
            self.viewer.end_frame()
            return

        t = self.playback_step

        # Write q_traj[t] into the Newton model and run eval_fk so the
        # viewer shows the robot at the trajectory pose.
        with torch.no_grad():
            joint_q_torch = wp.to_torch(self.model.joint_q)
            joint_q_torch[:7] = self.q_traj[t].detach().to(joint_q_torch.device)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

        self.sim_time += self.frame_dt

    # ---------------------------------------------------------------------- #
    # Test hook (used by the Newton test harness)                            #
    # ---------------------------------------------------------------------- #

    def test_final(self):
        import numpy as np
        recent = self.loss_history[-max(1, len(self.loss_history) // 5):]
        assert recent[-1] < self.loss_history[0] * 0.5, (
            f"Loss did not decrease sufficiently: {self.loss_history[0]:.4f} → {recent[-1]:.4f}"
        )

    # ---------------------------------------------------------------------- #
    # Argument parser (Newton examples convention)                           #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--verbose", action="store_true", default=True,
            help="Print loss each iteration."
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
# NOTE: Extending to full Newton-physics diffsim (warp tape)
# ---------------------------------------------------------------------------
# To differentiate *through the physics dynamics* (not just FK), replace the
# PyTorch forward pass above with a warp.Tape recording:
#
#   self.model = scene.finalize(requires_grad=True)
#   self.states = [self.model.state(requires_grad=True) for _ in range(T+1)]
#   self.q_ctrl = [wp.zeros(7, dtype=float, requires_grad=True) for _ in range(T)]
#
#   def forward_backward_physics(self):
#       self.tape = wp.Tape()
#       with self.tape:
#           for t in range(T):
#               # write learnable control into warp control buffer
#               wp.copy(self.control.joint_target_pos, self.q_ctrl[t], count=7)
#               self.solver.step(self.states[t], self.states[t+1],
#                                self.control, self.contacts, self.sim_dt)
#               # accumulate FK-based loss via warp kernel
#               wp.launch(joint_error_kernel,
#                         dim=7,
#                         inputs=[self.states[t+1].joint_q, self.q_target, self.loss])
#       self.tape.backward(self.loss)
#       self.optimizer.step([ctrl.grad.flatten() for ctrl in self.q_ctrl])
#
# Use SolverSemiImplicit (requires_grad-compatible) instead of SolverMuJoCo.
# ---------------------------------------------------------------------------
