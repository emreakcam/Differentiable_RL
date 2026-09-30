# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Diffsim Franka — MLP POLICY + TRUNCATED BPTT
#
# Combines:
#   - MLP policy: obs → v_target (instead of raw learnable vectors)
#   - TBPTT: long horizons without memory explosion
#
# Per segment:
#   1. Read EE pose from (detached) state
#   2. Build obs = [ee_pos, target_pos]
#   3. MLP(obs) → v_target (torch, differentiable)
#   4. wp.from_torch → warp pipeline (FK → J → pinv → physics)
#   5. Loss at segment end
#   6. tape.backward() → warp grads → from_torch bridge → torch autograd
#   7. Detach state for next segment
#   8. After all segments: torch optimizer step
#
# Gradient flow within each segment:
#   loss ← body_q ← physics ← q_dot ← J_pinv ← v_target ← MLP(θ) ← obs
#
# Command: python diffsim_franka_mlp_tbptt.py
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
# Hyperparameters — USER-FACING
# ---------------------------------------------------------------------------

# 1. Total simulation time (seconds)
TOTAL_SIM_TIME = 1.0

# 2. Number of TBPTT segments (gradient window count)
N_SEGMENTS     = 10

# 3. Physics substeps per control step (sim_step)
SIM_SUBSTEPS   = 100

# 4. Number of MLP decision points per trajectory.
#    MLP is called once per traj_step; the v_target is held constant for
#    (SIM_STEPS / TRAJ_STEPS) consecutive sim_steps.
#    Set to 0 to call MLP every sim_step.
TRAJ_STEPS     = 10

SHAPED_SCALE = 1.8318 / 0.20   

# ---------------------------------------------------------------------------
# Derived parameters — DO NOT EDIT
# ---------------------------------------------------------------------------
FPS            = 10
FRAME_DT       = 1.0 / FPS
SIM_DT         = FRAME_DT / SIM_SUBSTEPS
SIM_STEPS      = round(TOTAL_SIM_TIME / FRAME_DT)
SEGMENT_LEN    = max(1, SIM_STEPS // N_SEGMENTS)
N_SEGMENTS     = math.ceil(SIM_STEPS / SEGMENT_LEN)

# Resolve TRAJ_STEPS: 0 means every sim_step
if TRAJ_STEPS <= 0:
    TRAJ_STEPS = SIM_STEPS
STEPS_PER_TRAJ = max(1, SIM_STEPS // TRAJ_STEPS)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
TRAIN_ITERS    = 500
TRAIN_LR       = 1e-3
GRAD_CLIP_NORM = 10.0
PINV_DAMPING   = 0.01
DEVICE         = "cuda:0"

# Target EE position (world frame)
TARGET_POS = (0.5, 0.3, 0.8)

# Franka rest pose
INITIAL_ARM_Q = [0.0, -0.785, 0.0, -2.356, 0.0, 2.571, 0.785]

# PD gains (velocity mode for arm)
JOINT_KE = [0.0] * 7
JOINT_KD = [450, 450, 350, 350, 200, 200, 200]

# Loss weight
W_POS = 5.0

# EE velocity limits
V_MAX = 3.0


# ---------------------------------------------------------------------------
# MLP Policy
# ---------------------------------------------------------------------------

class ReachingPolicy(nn.Module):
    """obs → v_target (EE velocity command).

    Input:  [ee_pos(3), target_pos(3)] = 6D
    Output: v_target(3) in m/s, scaled by tanh
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
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs):
        raw = self.net(obs)
        return self.v_max * torch.tanh(raw)


# ---------------------------------------------------------------------------
# Warp Kernels
# ---------------------------------------------------------------------------

@wp.kernel
def shaped_loss_kernel(
    body_q: wp.array(dtype=wp.transform),
    target_pos: wp.vec3,
    ee_body_index: int,
    weight: float,
    scale: float,
    loss: wp.array(dtype=float),
):
    ee_tf = body_q[ee_body_index]
    ee_pos = wp.transform_get_translation(ee_tf)
    delta = ee_pos - target_pos
    dist = wp.length(delta)
    shaped = 1.0 - wp.tanh(dist * scale)
    wp.atomic_add(loss, 0, weight * (1.0 - shaped * shaped))

@wp.kernel
def linear_loss_kernel(
    body_q: wp.array(dtype=wp.transform),
    target_pos: wp.vec3,
    ee_body_index: int,
    weight: float,
    loss: wp.array(dtype=float),
):
    ee_tf = body_q[ee_body_index]
    ee_pos = wp.transform_get_translation(ee_tf)
    delta = ee_pos - target_pos
    dist = wp.length(delta)
    wp.atomic_add(loss, 0, weight * dist)


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
        self.frame_dt = FRAME_DT
        self.sim_dt = SIM_DT
        self.train_iter = 0
        self.total_substeps = SIM_STEPS * SIM_SUBSTEPS
        self.segment_substeps = SEGMENT_LEN * SIM_SUBSTEPS
        self.torch_device = torch.device(DEVICE)

        # ------------------------------------------------------------------ #
        # 1. Build Newton model
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
        # 2. Solver
        # ------------------------------------------------------------------ #
        self.solver = newton.solvers.SolverFeatherstone(self.model)
        self.n_joint_q = len(self.model.joint_q.numpy())
        self.n_joint_qd = len(self.model.joint_qd.numpy())

        # ------------------------------------------------------------------ #
        # 3. Find EE body
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
        # 4. Parse URDF → kinematic chain → Warp arrays
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

        self.chain_wp = load_chain_to_warp(chain, device=DEVICE)

        # ------------------------------------------------------------------ #
        # 5. Constants
        # ------------------------------------------------------------------ #
        self.rest_fingers = wp.array([0.04, 0.04], dtype=float, device=DEVICE)
        self.target_pos_torch = torch.tensor(TARGET_POS, device=self.torch_device, dtype=torch.float32)
        self.target_pos_wp = wp.vec3(*TARGET_POS)

        # Controls — one per sim_step
        self.controls = []
        for _ in range(SIM_STEPS):
            self.controls.append(self.model.control())

        # ------------------------------------------------------------------ #
        # 6. MLP Policy + optimizer
        # ------------------------------------------------------------------ #
        self.policy = ReachingPolicy(obs_dim=6, hidden=64, v_max=V_MAX).to(self.torch_device)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=TRAIN_LR, betas=(0.0, 0.999)
        )

        # ------------------------------------------------------------------ #
        # 7. Logging
        # ------------------------------------------------------------------ #
        self.loss_history = []
        self.render_states = []

        # ------------------------------------------------------------------ #
        # 8. Print config
        # ------------------------------------------------------------------ #
        grad_window_s = SEGMENT_LEN * FRAME_DT
        n_policy_params = sum(p.numel() for p in self.policy.parameters())

        print(f"\n{'='*60}")
        print(f"MLP POLICY + TRUNCATED BPTT")
        print(f"  --- User params ---")
        print(f"  TOTAL_SIM_TIME:  {TOTAL_SIM_TIME:.2f} s")
        print(f"  N_SEGMENTS:      {N_SEGMENTS}")
        print(f"  SIM_SUBSTEPS:    {SIM_SUBSTEPS}")
        print(f"  TRAJ_STEPS:      {TRAJ_STEPS} (MLP called {TRAJ_STEPS}x, held for {STEPS_PER_TRAJ} sim_steps each)")
        print(f"  --- Derived ---")
        print(f"  FPS (control):   {FPS} Hz")
        print(f"  FRAME_DT:        {FRAME_DT*1000:.1f} ms")
        print(f"  SIM_DT:          {SIM_DT*1000:.3f} ms")
        print(f"  SIM_STEPS:       {SIM_STEPS}")
        print(f"  SEGMENT_LEN:     {SEGMENT_LEN} sim_steps ({grad_window_s*1000:.0f} ms gradient window)")
        print(f"  --- Policy ---")
        print(f"  Architecture:    6 → 64 → 64 → 3")
        print(f"  Parameters:      {n_policy_params}")
        print(f"  Observation:     [ee_pos(3), target_pos(3)]")
        print(f"  Action:          v_target(3) m/s (tanh scaled)")
        print(f"  Target:          {TARGET_POS}")
        print(f"{'='*60}\n")

        # ------------------------------------------------------------------ #
        # 9. Viewer
        # ------------------------------------------------------------------ #
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "renderer"):
            self.viewer.set_camera(
                wp.vec3(1.5, -1.0, 1.2), pitch=-20.0, yaw=130.0
            )

    # ---------------------------------------------------------------------- #
    # Detach state
    # ---------------------------------------------------------------------- #

    def _detach_state(self, src_state):
        """Copy joint state via numpy — fully breaks gradient chain."""
        new_state = self.model.state(requires_grad=True)
        jq_np = src_state.joint_q.numpy().copy()
        jqd_np = src_state.joint_qd.numpy().copy()
        new_state.joint_q.assign(wp.array(jq_np, dtype=float, device=DEVICE))
        new_state.joint_qd.assign(wp.array(jqd_np, dtype=float, device=DEVICE))
        return new_state

    # ---------------------------------------------------------------------- #
    # Forward one segment
    # ---------------------------------------------------------------------- #

    def _forward_segment(self, seg_idx, states, loss, v_target_tensors):
        """
        Run one TBPTT segment.

        For each sim_step in the segment:
          1. Read EE pos from current state (warp → torch)
          2. Build obs, run MLP → v_target (torch)
          3. Bridge to warp, run FK → J → pinv → physics

        Args:
            seg_idx: segment index
            states: pre-allocated states for this segment
            loss: warp loss array
            v_target_tensors: list to append torch tensors for gradient bridge
        """
        step_start = seg_idx * SEGMENT_LEN
        step_end = min(step_start + SEGMENT_LEN, SIM_STEPS)

        # Rebuild body transforms inside tape
        newton.eval_fk(
            self.model, states[0].joint_q, states[0].joint_qd, states[0]
        )

        for local_step, global_step in enumerate(range(step_start, step_end)):
            sub_start = local_step * SIM_SUBSTEPS

            # Check if this sim_step is a traj decision point
            traj_idx = global_step * TRAJ_STEPS // SIM_STEPS
            prev_traj_idx = (global_step - 1) * TRAJ_STEPS // SIM_STEPS if global_step > step_start else -1
            is_new_traj_point = (traj_idx != prev_traj_idx) or (local_step == 0)

            if is_new_traj_point:
                # --- (1) Read EE position from warp state → torch ---
                ee_pos_torch = wp.to_torch(
                    states[sub_start].body_q
                )[self.ee_body_index, :3]

                # --- (2) Build observation and run MLP ---
                obs = torch.cat([ee_pos_torch, self.target_pos_torch])  # [6]
                v_target_torch = self.policy(obs)  # [3]
                v_target_tensors.append(v_target_torch)

                # --- (3) Bridge torch → warp ---
                v_target_wp = wp.from_torch(v_target_torch, dtype=wp.float32)

            # --- (4) Warp FK → Jacobian → pinv → physics ---
            fk_ee_pos = wp.zeros(1, dtype=wp.vec3, device=DEVICE, requires_grad=True)
            fk_joint_pos = wp.zeros(N_JOINTS, dtype=wp.vec3, device=DEVICE, requires_grad=True)
            fk_joint_axis = wp.zeros(N_JOINTS, dtype=wp.vec3, device=DEVICE, requires_grad=True)
            J_p_flat = wp.zeros(3 * N_JOINTS, dtype=float, device=DEVICE, requires_grad=True)
            q_dot = wp.zeros(N_JOINTS, dtype=float, device=DEVICE, requires_grad=True)
            q_current_7 = wp.zeros(N_JOINTS, dtype=float, device=DEVICE, requires_grad=True)

            wp.launch(
                copy_first7_kernel, dim=N_JOINTS,
                inputs=[states[sub_start].joint_q], outputs=[q_current_7],
            )

            wp.launch(
                fk_tcp_kernel, dim=1,
                inputs=[
                    q_current_7,
                    self.chain_wp["joint_axes"],
                    self.chain_wp["link_pos"],
                    self.chain_wp["link_quats"],
                    self.chain_wp["ee_offset_p"],
                    self.chain_wp["ee_offset_q"],
                    fk_ee_pos, fk_joint_pos, fk_joint_axis,
                ],
            )

            wp.launch(
                jacobian_p_tcp_kernel, dim=N_JOINTS,
                inputs=[fk_joint_pos, fk_joint_axis, fk_ee_pos],
                outputs=[J_p_flat],
            )

            wp.launch(
                damped_pinv_3xN_kernel, dim=1,
                inputs=[J_p_flat, v_target_wp, PINV_DAMPING],
                outputs=[q_dot],
            )

            wp.launch(
                copy_first7_kernel, dim=N_JOINTS,
                inputs=[q_dot],
                outputs=[self.controls[global_step].joint_target_vel],
            )

            wp.launch(
                copy_fingers_from_rest_kernel, dim=2,
                inputs=[self.rest_fingers],
                outputs=[self.controls[global_step].joint_target_pos],
            )

            for sub in range(SIM_SUBSTEPS):
                t = sub_start + sub
                states[t].clear_forces()
                self.solver.step(
                    states[t], states[t + 1],
                    self.controls[global_step], None, self.sim_dt,
                )

        # Loss at segment end
        seg_weight = W_POS * math.exp(-20.0 * (N_SEGMENTS - 1 - seg_idx) / N_SEGMENTS) 
        final_sub = (step_end - step_start) * SIM_SUBSTEPS
        
        # wp.launch(
        #     shaped_loss_kernel, dim=1,
        #     inputs=[states[final_sub].body_q, self.target_pos_wp,
        #             self.ee_body_index, seg_weight, SHAPED_SCALE, loss],
        # )

        wp.launch(
            linear_loss_kernel, dim=1,
            inputs=[states[final_sub].body_q, self.target_pos_wp,
                    self.ee_body_index, seg_weight, loss],
        )

    # ---------------------------------------------------------------------- #
    # Training step
    # ---------------------------------------------------------------------- #

    def step(self):
        # Zero MLP gradients
        self.policy_optimizer.zero_grad()

        # Reset to initial pose
        init_state = self.model.state(requires_grad=True)
        wp.launch(
            zero_joint_qd_kernel, dim=self.n_joint_qd,
            inputs=[init_state.joint_qd],
        )
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, init_state
        )

        total_loss = 0.0
        self.render_states = []
        current_state = init_state

        for seg_idx in range(N_SEGMENTS):
            step_start = seg_idx * SEGMENT_LEN
            step_end = min(step_start + SEGMENT_LEN, SIM_STEPS)
            n_sub = (step_end - step_start) * SIM_SUBSTEPS

            # Allocate segment states
            seg_states = [current_state]
            for _ in range(n_sub):
                seg_states.append(self.model.state(requires_grad=True))

            seg_loss = wp.zeros(1, dtype=float, requires_grad=True)
            v_target_tensors = []  # collect torch tensors for gradient bridge

            # Forward under warp tape
            tape = wp.Tape()
            with tape:
                self._forward_segment(seg_idx, seg_states, seg_loss, v_target_tensors)

            # Warp backward
            tape.backward(seg_loss)

            # Bridge: warp grads → torch autograd → MLP weights
            for v_torch in v_target_tensors:
                if v_torch.grad is not None:
                    v_torch.backward(v_torch.grad, retain_graph=True)

            seg_loss_val = seg_loss.numpy()[0]
            total_loss += seg_loss_val

            # Debug print
            if self.verbose and seg_idx % max(1, N_SEGMENTS // 5) == 0:
                ee_pos = wp.to_torch(seg_states[n_sub].body_q)[
                    self.ee_body_index, :3
                ].detach().cpu().numpy()
                print(
                    f"  Seg {seg_idx}/{N_SEGMENTS}: "
                    f"loss={seg_loss_val:.4f}  "
                    f"EE=[{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]"
                )

            # Save for rendering
            self.render_states.append(seg_states[n_sub])

            # Detach for next segment
            if seg_idx < N_SEGMENTS - 1:
                current_state = self._detach_state(seg_states[n_sub])

            tape.zero()

        # Gradient clipping on MLP
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), GRAD_CLIP_NORM
        )

        # Optimizer step
        self.policy_optimizer.step()

        # Log
        self.loss_history.append(total_loss)

        if self.verbose:
            print(
                f"Iter {self.train_iter:4d}: "
                f"total_loss={total_loss:.4f}  "
                f"grad_norm={grad_norm:.4f}  "
                f"Target=[{TARGET_POS[0]:.3f}, {TARGET_POS[1]:.3f}, {TARGET_POS[2]:.3f}]"
            )

        self.viewer.log_scalar("/loss", total_loss)
        self.train_iter += 1

        if self.train_iter % 50 == 0:
            self.plot_loss(f"loss_iter{self.train_iter:04d}.png")

    # ---------------------------------------------------------------------- #
    # Render
    # ---------------------------------------------------------------------- #

    def render(self):
        if self.viewer.is_paused():
            self.viewer.begin_frame(self.sim_time)
            self.viewer.end_frame()
            return

        for state in self.render_states:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(state)
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
            self.sim_time += self.frame_dt * SEGMENT_LEN

    # ---------------------------------------------------------------------- #
    # Plot
    # ---------------------------------------------------------------------- #

    def plot_loss(self, save_path="loss.png"):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.loss_history, linewidth=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Total loss")
        ax.set_title("MLP Policy + TBPTT Loss")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f"Saved: {save_path}")

    # ---------------------------------------------------------------------- #
    # Test hook
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