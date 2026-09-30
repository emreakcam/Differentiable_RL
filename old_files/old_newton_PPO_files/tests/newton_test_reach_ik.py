# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# IK Test Environment
#
# Newton simulation environment for IK testing using the Franka Panda arm.
# The kinematic chain is parsed directly from the robot URDF and IK is
# solved via a PyTorch differentiable FK function with an analytical
# Jacobian (Levenberg-Marquardt, position + orientation).
#
# Command: python newton_test_reach_ik.py
#
###########################################################################

import copy
import math
import sys
from pathlib import Path

import torch
import warp as wp

import newton
import newton.examples
import newton.utils

sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.geom_diff_helpers import (    
    quat_conjugate,
    quat_error_to_rotvec,
    quat_mul,
    transform_by_quat_diff,
    transform_quat_by_quat_diff,
)
from helpers.kinematic_helpers import forward_kinematics, solve_ik, parse_urdf_kinematic_chain


# ---------------------------------------------------------------------------
# FK with rigid EE offset
# ---------------------------------------------------------------------------

def fk_with_ee_offset(
    q:           torch.Tensor,   # [N]
    joint_axes:  torch.Tensor,   # [N, 3]
    link_pos:    torch.Tensor,   # [N, 3]
    link_quats:  torch.Tensor,   # [N, 4]
    ee_offset_p: torch.Tensor,   # [3]  fixed translation in last-joint frame
    ee_offset_q: torch.Tensor,   # [4]  fixed rotation   in last-joint frame
):
    """
    Run FK and append the fixed EE offset (flange / TCP mount).

    Returns:
        pos  : [3]  world-frame EE position
        quat : [4]  world-frame EE orientation [w, x, y, z]
    """
    pos, quat = forward_kinematics(q, joint_axes, link_pos, link_quats)
    pos  = pos  + transform_by_quat_diff(ee_offset_p, quat)
    quat = transform_quat_by_quat_diff(quat, ee_offset_q)
    return pos, quat



# ---------------------------------------------------------------------------
# Warp broadcast kernel
# ---------------------------------------------------------------------------

@wp.kernel
def broadcast_joint_targets_kernel(
    ik_solution:   wp.array(dtype=wp.float32),    # pyright: ignore[reportInvalidTypeForm]  # [n_arm_joints]
    joint_targets: wp.array2d(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]  # [world_count, joints_per_world]
    n_arm_joints:  int,
    gripper_value: float,
):
    """Copy IK solution into every simulated world and set gripper joints."""
    world_idx = wp.tid()
    for j in range(n_arm_joints):
        joint_targets[world_idx, j] = ik_solution[j]
    joint_targets[world_idx, n_arm_joints]     = gripper_value
    joint_targets[world_idx, n_arm_joints + 1] = gripper_value


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class IKTestEnvironment:
    def __init__(self, viewer, args):
        self.fps              = 60
        self.frame_dt         = 1.0 / self.fps
        self.sim_time         = 0.0
        self.sim_substeps     = 5        # reduced from 10 for speed
        self.collide_substeps = 2
        self.sim_dt           = self.frame_dt / self.sim_substeps
        self.world_count      = args.world_count
        self.viewer           = viewer
        self.ik_update_freq   = 1
        self.ik_blend         = 0.2
        self.ik_frame_count   = 0
        self.ik_solve_count   = 0        # count actual solves for diagnostics

        # Tiny 7-DOF IK solves are lower-latency and more stable on CPU.
        self.torch_device = torch.device("cpu")

        # ------------------------------------------------------------------ #
        # Newton simulation                                                   #
        # ------------------------------------------------------------------ #
        shape_cfg                  = newton.ModelBuilder.ShapeConfig(margin=0.0, gap=0.005)
        shape_cfg.ke               = 5.0e3
        shape_cfg.kd               = 5.0e6
        shape_cfg.kf               = 1.0e3
        shape_cfg.mu               = 0.75
        shape_cfg.mu_torsional     = 0.0
        shape_cfg.mu_rolling       = 0.0        

        builder = newton.ModelBuilder()
        self._build_scene(builder, shape_cfg)
        
        # Set PD control gains on the builder BEFORE finalization
        # These keep the arm from collapsing under gravity
        builder.joint_target_ke[:7] = [5000.0] * 7    # arm position/stiffness gains
        builder.joint_target_kd[:7] = [100.0] * 7     # arm velocity/damping gains
        builder.joint_target_ke[7:9] = [1000.0] * 2   # gripper gains
        builder.joint_target_kd[7:9] = [10.0] * 2
        builder.joint_effort_limit[:7] = [87.0] * 7   # arm effort limits
        builder.joint_effort_limit[7:9] = [20.0] * 2  # gripper effort limits

        scene = newton.ModelBuilder()
        scene.replicate(builder, self.world_count)
        scene.add_ground_plane()
        self.model = scene.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.collision_pipeline = newton.CollisionPipeline(
            self.model, reduce_contacts=True, broad_phase="explicit"
        )
        self.contacts = self.collision_pipeline.contacts()

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=500,
            nconmax=500,
            iterations=15,
            ls_iterations=100,
            impratio=1000.0,
        )

        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = False
        if hasattr(self.viewer, "renderer"):
            self.viewer.set_camera(wp.vec3(0.5, 0.0, 0.5), -15, -140)
            self.viewer.set_world_offsets(wp.vec3(1.0, 1.0, 0.0))

        self.control = self.model.control()
        
        joint_target_2d_shape = self.control.joint_target_pos.reshape(
            (self.world_count, -1)
        ).shape
        self.joint_targets_2d = wp.zeros(joint_target_2d_shape, dtype=wp.float32)
        wp.copy(self.control.joint_target_pos[:9], self.model.joint_q[:9])

        # ------------------------------------------------------------------ #
        # PyTorch differentiable FK + IK                                     #
        # ------------------------------------------------------------------ #
        self._setup_pytorch_ik()

        # ------------------------------------------------------------------ #
        # CUDA-graph capture for the simulation step                         #
        # ------------------------------------------------------------------ #
        self.graph = None
        self.capture()

    # ---------------------------------------------------------------------- #

    def _build_scene(self, builder: newton.ModelBuilder, shape_cfg):
        """Add the Franka Panda robot to the scene."""
        builder.default_shape_cfg = copy.deepcopy(shape_cfg)

        self.urdf_path = str(
            newton.utils.download_asset("franka_emika_panda")
            / "urdf/fr3_franka_hand.urdf"
        )
        builder.add_urdf(
            self.urdf_path,
            xform=wp.transform((-0.5, -0.5, 0.05), wp.quat_identity()),
            enable_self_collisions=False,
        )
        builder.approximate_meshes("convex_hull")

    # ---------------------------------------------------------------------- #

    def _setup_pytorch_ik(self):
        """Parse the URDF kinematic chain and initialise PyTorch IK tensors."""
        print("[IK] Parsing URDF kinematic chain …")
        chain = parse_urdf_kinematic_chain(
            urdf_path  = self.urdf_path,
            root_link  = "fr3_link0",
            ee_link    = "fr3_link7",   # 7-DOF arm; fixed flange/hand become ee_offset
            device     = self.torch_device,
        )

        self.joint_axes   = chain["joint_axes"]    # [7, 3]
        self.link_pos     = chain["p_rel"]         # [7, 3]
        self.link_quats   = chain["q_rel"]         # [7, 4]
        self.joint_limits = chain["joint_ranges"]  # [7, 2]
        self.ee_offset_p  = chain["ee_offset_p"]   # [3]
        self.ee_offset_q  = chain["ee_offset_q"]   # [4]
        self.n_arm_joints = self.joint_axes.shape[0]

        print(f"[IK]   {self.n_arm_joints} revolute joints: {chain['joint_names']}")
        print(f"[IK]   EE offset (flange→TCP): {self.ee_offset_p.tolist()}")

        # Warm-start: use the robot's initial joint configuration.
        q_np      = self.model.joint_q.numpy()[:self.n_arm_joints]
        self.q_ik = torch.tensor(q_np, dtype=torch.float32, device=self.torch_device)

        # Compute the initial EE pose and use it as the first IK target.
        pos0, quat0 = fk_with_ee_offset(
            self.q_ik, self.joint_axes, self.link_pos, self.link_quats,
            self.ee_offset_p, self.ee_offset_q,
        )
        self.target_pos   = pos0.clone()
        self.target_quat  = quat0.clone()
        self._ik_center   = pos0.clone()   # centre for the animated trajectory
        print(f"[IK]   Initial EE position: {self.target_pos.tolist()}")

    # ---------------------------------------------------------------------- #

    def _animate_target(self):
        """Move the IK target along a small circular arc in the XZ plane."""
        radius = 0.1
        omega  = 1.5
        angle  = omega * self.sim_time
        dy = radius * math.cos(angle)
        dz = radius * math.sin(angle) * 0.5
        self.target_pos = self._ik_center + torch.tensor(
            [0.0, dy, dz], dtype=torch.float32, device=self.torch_device
        )

    # ---------------------------------------------------------------------- #

    def _update_joint_targets(self):
        """Solve IK for the current target and broadcast result to all worlds."""
        self._animate_target()
        
        # Solve a small damped IK problem every frame for smooth target updates.
        if self.ik_frame_count % self.ik_update_freq == 0:
            q_solved = solve_ik(
                target_pos   = self.target_pos,
                target_quat  = self.target_quat,
                joint_axes   = self.joint_axes,
                link_pos     = self.link_pos,
                link_quats   = self.link_quats,
                ee_offset_p  = self.ee_offset_p,
                ee_offset_q  = self.ee_offset_q,
                joint_limits = self.joint_limits,
                q_init       = self.q_ik,
                n_iters      = 8,
                rot_weight   = 0.0,
                max_step_norm= 0.08,
            )
            self.q_ik = torch.lerp(self.q_ik, q_solved, self.ik_blend)
            self.ik_solve_count += 1

        ik_warp = wp.array(self.q_ik.detach().cpu().numpy(), dtype=wp.float32)
        wp.launch(
            broadcast_joint_targets_kernel,
            dim=self.world_count,
            inputs=[ik_warp, self.joint_targets_2d, self.n_arm_joints, 0.0],
        )
        self.control.joint_target_pos.assign(self.joint_targets_2d.reshape((-1,)))
        self.ik_frame_count += 1

    # ---------------------------------------------------------------------- #

    def capture(self):
        """Optionally capture the physics step as a CUDA graph."""
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        for i in range(self.sim_substeps):
            if i % self.collide_substeps == 0:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """One frame: update IK targets, then advance the physics."""
        self._update_joint_targets()

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    # ---------------------------------------------------------------------- #

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(num_frames=1800)
        parser.set_defaults(world_count=1)
        return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = IKTestEnvironment.create_parser()
    viewer, args = newton.examples.init(parser)

    env = IKTestEnvironment(viewer, args)

    for frame in range(args.num_frames):
        env.step()
        env.render()
        if frame % 30 == 0:  # print more frequently (every 0.5 sec)
            pos_ee, _ = fk_with_ee_offset(
                env.q_ik, env.joint_axes, env.link_pos, env.link_quats,
                env.ee_offset_p, env.ee_offset_q,
            )
            target_np = env.target_pos.cpu().numpy()
            ee_np = pos_ee.cpu().numpy()
            err = (pos_ee - env.target_pos).norm().item()
            print(
                f"Frame {frame:4d}/{args.num_frames}  solves={env.ik_solve_count}  "
                f"target=({target_np[0]:7.4f}, {target_np[1]:7.4f}, {target_np[2]:7.4f})  "
                f"ee     =({ee_np[0]:7.4f}, {ee_np[1]:7.4f}, {ee_np[2]:7.4f})  "
                f"|err|={err:.4f} m"
            )

    print("Simulation complete!")
