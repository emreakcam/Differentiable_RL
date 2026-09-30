# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Diffsim Franka — EE-VELOCITY with TRUNCATED BPTT
#
# Same as diffsim_franka_ee_vel.py but uses Truncated Backpropagation
# Through Time (TBPTT) to handle long horizons without running out of
# memory or slowing down.
#
# Instead of taping the entire trajectory and doing one backward pass,
# we split the trajectory into segments of SEGMENT_LEN sim_steps.
# Each segment gets its own tape → forward → loss → backward → grad accumulate.
# Between segments, we detach the state (break the gradient chain).
#
# This means gradients only flow within a segment window, not across
# the full trajectory. This is biased but practical for long horizons.
#
# Command: python diffsim_franka_tbptt.py
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
# Hyperparameters — USER-FACING
# ---------------------------------------------------------------------------

# 1. Total simulation time (seconds)
TOTAL_SIM_TIME = 0.1

# 2. Number of TBPTT segments (gradient window count)
#    More segments = longer horizon but each segment is shorter gradient window
N_SEGMENTS     = 1

# 3. Physics substeps per control step (sim_step)
#    Higher = more accurate physics, but slower
SIM_SUBSTEPS   = 100

# 4. Learnable trajectory points (independent of sim structure)
TRAJ_STEPS     = 1

# ---------------------------------------------------------------------------
# Derived parameters — DO NOT EDIT
# ---------------------------------------------------------------------------
FPS            = 10                                    # control frequency (Hz)
FRAME_DT       = 1.0 / FPS                            # time per sim_step (s)
SIM_DT         = FRAME_DT / SIM_SUBSTEPS              # physics dt (s)
SIM_STEPS      = round(TOTAL_SIM_TIME / FRAME_DT)     # total sim_steps
SEGMENT_LEN    = max(1, SIM_STEPS // N_SEGMENTS)       # sim_steps per segment
N_SEGMENTS     = math.ceil(SIM_STEPS / SEGMENT_LEN)    # recalculate (rounding)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
TRAIN_ITERS    = 200
TRAIN_LR       = 3e-2
GRAD_CLIP      = 50.0
PINV_DAMPING   = 0.01
DEVICE         = "cuda:0"

# Target EE position (world frame)
TARGET_POS = (0.4, 0.1, 0.6)

# Franka rest pose
INITIAL_ARM_Q = [0.0, -0.785, 0.0, -2.356, 0.0, 2.571, 0.785]

# PD gains
JOINT_KE = [0.0] * 7
JOINT_KD = [450, 450, 350, 350, 200, 200, 200]

# Loss weight
W_POS = 100.0

# EE velocity control
V_TARGET_INIT = [0.0, 0.0, 0.0]
V_MAX = 3.0


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
        self.frame_dt = FRAME_DT
        self.sim_dt = SIM_DT
        self.train_iter = 0

        # Total substeps for the FULL trajectory
        self.total_substeps = SIM_STEPS * SIM_SUBSTEPS

        # Substeps per segment
        self.segment_substeps = SEGMENT_LEN * SIM_SUBSTEPS

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

        # We need n_joint_qd for zero_joint_qd_kernel
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
            device=torch.device(DEVICE),
        )

        fix_p1 = torch.tensor([0.0, 0.0, 0.107], device=torch.device(DEVICE), dtype=torch.float32)
        fix_q1 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=torch.device(DEVICE), dtype=torch.float32)
        fix_p2 = torch.tensor([0.0, 0.0, 0.0], device=torch.device(DEVICE), dtype=torch.float32)
        fix_q2 = torch.tensor(rpy_to_quat_wxyz(0.0, 0.0, -0.7853981633974483), device=torch.device(DEVICE), dtype=torch.float32)

        composed_p = fix_p1 + transform_by_quat_diff(fix_p2, fix_q1)
        composed_q = transform_quat_by_quat_diff(fix_q1, fix_q2)

        grasp_offset_hand = torch.tensor([0.0, 0.0, 0.1034], device=torch.device(DEVICE), dtype=torch.float32)
        composed_p = composed_p + transform_by_quat_diff(grasp_offset_hand, composed_q)

        chain["ee_offset_p"] = composed_p
        chain["ee_offset_q"] = composed_q

        assert len(chain["joint_names"]) == N_JOINTS
        print(f"Parsed chain: {chain['joint_names']}")

        self.chain_wp = load_chain_to_warp(chain, device=DEVICE)

        # ------------------------------------------------------------------ #
        # 5. Constant arrays
        # ------------------------------------------------------------------ #
        self.rest_fingers = wp.array([0.04, 0.04], dtype=float, device=DEVICE)

        # ------------------------------------------------------------------ #
        # 6. Learnable EE-velocity targets (one per sim_step)
        # ------------------------------------------------------------------ #
        self.ctrl_v = []
        for _ in range(TRAJ_STEPS):
            self.ctrl_v.append(
                wp.array(V_TARGET_INIT, dtype=float, device=DEVICE, requires_grad=True)
            )

        # ------------------------------------------------------------------ #
        # 7. Controls (one per sim_step)
        # ------------------------------------------------------------------ #
        self.controls = []
        for _ in range(SIM_STEPS):
            self.controls.append(self.model.control())

        # ------------------------------------------------------------------ #
        # 8. State storage for rendering (full trajectory)
        #    We store the final state of each sim_step for visualization.
        # ------------------------------------------------------------------ #
        self.render_states = []  # will be populated during training

        # ------------------------------------------------------------------ #
        # 9. Target, loss, optimizer
        # ------------------------------------------------------------------ #
        self.target_pos = wp.vec3(*TARGET_POS)
        self.loss_history = []
        self.joint_q_history = None

        # Optimizer over all ctrl_v
        self.optimizer = warp.optim.Adam(self.ctrl_v, lr=TRAIN_LR, betas=(0.0, 0.999))

        steps_per_traj = SIM_STEPS // TRAJ_STEPS
        grad_window_s = SEGMENT_LEN * FRAME_DT

        print(f"\n{'='*60}")
        print(f"EE-VELOCITY DIFFSIM with TRUNCATED BPTT")
        print(f"  --- User params ---")
        print(f"  TOTAL_SIM_TIME:  {TOTAL_SIM_TIME:.2f} s")
        print(f"  N_SEGMENTS:      {N_SEGMENTS}")
        print(f"  SIM_SUBSTEPS:    {SIM_SUBSTEPS}")
        print(f"  TRAJ_STEPS:      {TRAJ_STEPS}")
        print(f"  --- Derived ---")
        print(f"  FPS (control):   {FPS} Hz")
        print(f"  FRAME_DT:        {FRAME_DT*1000:.1f} ms")
        print(f"  SIM_DT:          {SIM_DT*1000:.3f} ms")
        print(f"  SIM_STEPS:       {SIM_STEPS}")
        print(f"  SEGMENT_LEN:     {SEGMENT_LEN} sim_steps ({grad_window_s*1000:.0f} ms gradient window)")
        print(f"  Substeps/seg:    {self.segment_substeps}")
        print(f"  Total substeps:  {self.total_substeps}")
        print(f"  Steps/v_target:  {steps_per_traj}")
        print(f"  Learnable:       {3 * TRAJ_STEPS} params ({TRAJ_STEPS} velocity vectors)")
        print(f"  Target:          {TARGET_POS}")
        print(f"{'='*60}\n")

        # ------------------------------------------------------------------ #
        # 10. Viewer
        # ------------------------------------------------------------------ #
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "renderer"):
            self.viewer.set_camera(
                wp.vec3(1.5, -1.0, 1.2), pitch=-20.0, yaw=130.0
            )

    # ---------------------------------------------------------------------- #
    # Detach state: copy values to a fresh state, breaking gradient chain
    # ---------------------------------------------------------------------- #

    def _detach_state(self, src_state):
        """Create a fresh state with same values but no gradient history.
        
        Only copies joint_q and joint_qd via numpy (fully breaks the graph).
        Body transforms (body_q) are NOT set here — they must be recomputed
        via eval_fk INSIDE the next segment's tape so that the gradient
        chain loss → body_q → joint_q → ctrl_v is intact.
        """
        new_state = self.model.state(requires_grad=True)

        # Copy via numpy — completely breaks gradient chain
        jq_np = src_state.joint_q.numpy().copy()
        jqd_np = src_state.joint_qd.numpy().copy()
        new_state.joint_q.assign(wp.array(jq_np, dtype=float, device=DEVICE))
        new_state.joint_qd.assign(wp.array(jqd_np, dtype=float, device=DEVICE))

        return new_state

    # ---------------------------------------------------------------------- #
    # Forward one segment
    # ---------------------------------------------------------------------- #

    def _forward_segment(self, seg_idx, initial_state, states, loss):
        """
        Run forward for one segment of SEGMENT_LEN sim_steps.
        
        Args:
            seg_idx: which segment (0, 1, 2, ...)
            initial_state: starting state (already in states[0])
            states: pre-allocated list of states for this segment
            loss: loss array to accumulate into
        """
        step_start = seg_idx * SEGMENT_LEN
        step_end = min(step_start + SEGMENT_LEN, SIM_STEPS)

        # Rebuild body transforms from joint state INSIDE the tape.
        # This is critical: loss reads body_q, so body_q must be connected
        # to joint_q within the tape for gradients to flow.
        newton.eval_fk(
            self.model, states[0].joint_q, states[0].joint_qd, states[0]
        )

        for local_step, global_step in enumerate(range(step_start, step_end)):
            sub_start = local_step * SIM_SUBSTEPS

            # Scratch buffers — fresh per step for tape safety
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
                inputs=[states[sub_start].joint_q],
                outputs=[q_current_7],
            )

            # (b) FK
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
            traj_idx = global_step * TRAJ_STEPS // SIM_STEPS
            wp.launch(
                damped_pinv_3xN_kernel,
                dim=1,
                inputs=[J_p_flat, self.ctrl_v[traj_idx], PINV_DAMPING],
                outputs=[q_dot],
            )

            # (e) Velocity targets
            wp.launch(
                copy_first7_kernel,
                dim=N_JOINTS,
                inputs=[q_dot],
                outputs=[self.controls[global_step].joint_target_vel],
            )

            # (f) Finger targets
            wp.launch(
                copy_fingers_from_rest_kernel,
                dim=2,
                inputs=[self.rest_fingers],
                outputs=[self.controls[global_step].joint_target_pos],
            )

            # (g) Physics substeps
            for sub in range(SIM_SUBSTEPS):
                t = sub_start + sub
                states[t].clear_forces()
                self.solver.step(
                    states[t],
                    states[t + 1],
                    self.controls[global_step],
                    None,
                    self.sim_dt,
                )

        # (h) Loss at the end of this segment
        final_sub = (step_end - step_start) * SIM_SUBSTEPS
        wp.launch(
            pose_loss_kernel,
            dim=1,
            inputs=[
                states[final_sub].body_q,
                self.target_pos,
                self.ee_body_index,
                W_POS,
                loss,
            ],
        )

    # ---------------------------------------------------------------------- #
    # Training step (TBPTT)
    # ---------------------------------------------------------------------- #

    def step(self):
        # Zero ctrl_v grads from previous iteration
        for cv in self.ctrl_v:
            if cv.grad is not None:
                cv.grad.zero_()

        # Reset to initial pose
        init_state = self.model.state(requires_grad=True)
        wp.launch(
            zero_joint_qd_kernel,
            dim=self.n_joint_qd,
            inputs=[init_state.joint_qd],
        )
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, init_state
        )

        total_loss = 0.0
        self.render_states = []  # collect states for rendering

        # Current state — will be detached between segments
        current_state = init_state

        for seg_idx in range(N_SEGMENTS):
            step_start = seg_idx * SEGMENT_LEN
            step_end = min(step_start + SEGMENT_LEN, SIM_STEPS)
            actual_segment_len = step_end - step_start
            n_sub_this_segment = actual_segment_len * SIM_SUBSTEPS

            # Allocate states for this segment
            seg_states = [current_state]  # states[0] = current_state
            for _ in range(n_sub_this_segment):
                seg_states.append(self.model.state(requires_grad=True))

            # Loss for this segment
            seg_loss = wp.zeros(1, dtype=float, requires_grad=True)

            # Forward + tape
            tape = wp.Tape()
            with tape:
                self._forward_segment(seg_idx, current_state, seg_states, seg_loss)

            # Backward
            tape.backward(seg_loss)

            # Accumulate gradients into optimizer
            seg_loss_val = seg_loss.numpy()[0]
            total_loss += seg_loss_val

            # DEBUG: Check gradients RIGHT AFTER backward, before tape.zero()
            if self.verbose and seg_idx < 3:
                for i, cv in enumerate(self.ctrl_v):
                    if cv.grad is not None:
                        g = cv.grad.numpy()
                        gnorm = np.linalg.norm(g)
                        if gnorm > 0:
                            print(f"    [seg {seg_idx}] ctrl_v[{i}] grad norm={gnorm:.2e}")
                    else:
                        print(f"    [seg {seg_idx}] ctrl_v[{i}] grad=None")

            # Print per-segment info
            if self.verbose and seg_idx % max(1, N_SEGMENTS // 5) == 0:
                ee_pos = wp.to_torch(seg_states[n_sub_this_segment].body_q)[
                    self.ee_body_index, :3
                ]
                print(
                    f"  Seg {seg_idx}/{N_SEGMENTS}: "
                    f"loss={seg_loss_val:.4f}  "
                    f"EE=[{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]"
                )

            # Save last state for rendering
            self.render_states.append(seg_states[n_sub_this_segment])

            # Detach: prepare next segment's initial state
            if seg_idx < N_SEGMENTS - 1:
                current_state = self._detach_state(seg_states[n_sub_this_segment])

            # Zero the tape — but NOT ctrl_v grads!
            # tape.zero() zeros ALL grads including ctrl_v.grad
            # We need to save them first.
            saved_grads = {}
            for i, cv in enumerate(self.ctrl_v):
                if cv.grad is not None:
                    saved_grads[i] = cv.grad.numpy().copy()

            tape.zero()

            # Restore accumulated ctrl_v grads
            for i, cv in enumerate(self.ctrl_v):
                if i in saved_grads:
                    if cv.grad is None:
                        cv.grad = wp.array(saved_grads[i], dtype=float, device=DEVICE)
                    else:
                        # Add back the saved grad
                        cv.grad.assign(wp.array(saved_grads[i], dtype=float, device=DEVICE))

        # --- After all segments: apply gradients ---

        # Debug: print gradient info
        if self.verbose:
            grad_info = []
            for i, cv in enumerate(self.ctrl_v):
                if cv.grad is not None:
                    g = cv.grad.numpy()
                    grad_info.append(f"  ctrl_v[{i}]: grad=[{g[0]:+.2e}, {g[1]:+.2e}, {g[2]:+.2e}] norm={np.linalg.norm(g):.2e}")
                else:
                    grad_info.append(f"  ctrl_v[{i}]: grad=None")
            print("\n".join(grad_info))

        # Gradient clipping (per ctrl_v)
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

        # Clamp v_target
        for cv in self.ctrl_v:
            v_np = cv.numpy()
            v_clamped = np.clip(v_np, -V_MAX, V_MAX)
            if not np.allclose(v_np, v_clamped):
                cv.assign(wp.array(v_clamped, dtype=float, requires_grad=True, device=DEVICE))

        # Log
        self.loss_history.append(total_loss)

        if self.verbose:
            print(
                f"Iter {self.train_iter:4d}: "
                f"total_loss={total_loss:.4f}  "
                f"Target=[{TARGET_POS[0]:.3f}, {TARGET_POS[1]:.3f}, {TARGET_POS[2]:.3f}]"
            )

        self.viewer.log_scalar("/loss", total_loss)
        self.train_iter += 1

    # ---------------------------------------------------------------------- #
    # Render
    # ---------------------------------------------------------------------- #

    def render(self):
        if self.viewer.is_paused():
            self.viewer.begin_frame(self.sim_time)
            self.viewer.end_frame()
            return

        # Render the last state of each segment
        for i, state in enumerate(self.render_states):
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(state)
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
            self.sim_time += self.frame_dt * SEGMENT_LEN

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