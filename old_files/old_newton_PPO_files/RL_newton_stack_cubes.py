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
from datetime import datetime
from pathlib import Path

import torch
import warp as wp
import yaml

import newton
import newton.examples
import newton.utils
from tensordict import TensorDict
from newton.sensors import SensorContact

try:
    from rsl_rl.env import VecEnv  # pyright: ignore[reportMissingImports]
    from rsl_rl.runners import OnPolicyRunner  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rsl_rl"))
    from rsl_rl.env import VecEnv  # pyright: ignore[reportMissingImports]
    from rsl_rl.runners import OnPolicyRunner  # pyright: ignore[reportMissingImports]

# Ensure helpers/ is importable regardless of working directory.
sys.path.insert(0, str(Path(__file__).parent))

from helpers.geom_diff_helpers import (
    analytic_to_geometric_jacobian,
    quat_mul,
    quat_error_to_rotvec,
    transform_by_quat_diff,
    transform_quat_by_quat_diff,
)
from helpers.kinematic_helpers import (
    forward_kinematics,
    forward_kinematics_batch,
    forward_kinematics_jacobian_analytical,
    solve_ik_batch,
    parse_urdf_kinematic_chain,
)


def _load_stack_cubes_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise TypeError(f"Config at {config_path} must be a dictionary")

    if "franka" not in config:
        raise KeyError("Missing 'franka' section in stack-cubes config")
    if "cubes" not in config:
        raise KeyError("Missing 'cubes' section in stack-cubes config")
    if "cluster_center" not in config:
        raise KeyError("Missing 'cluster_center' section in stack-cubes config")    
    if "rl" not in config:
        raise KeyError("Missing 'rl' section in stack-cubes config")
    if "reward" not in config:
        raise KeyError("Missing 'reward' section in stack-cubes config")
    if "training" not in config:
        raise KeyError("Missing 'training' section in stack-cubes config")
    if "run" not in config:
        raise KeyError("Missing 'run' section in stack-cubes config")

    reward_cfg = config["reward"]
    reward_cfg.setdefault("stack_pair_bonus", 5.0)
    reward_cfg.setdefault("stack_xy_threshold_m", reward_cfg.get("success_threshold_m", 0.03))
    reward_cfg.setdefault("stack_z_threshold_m", reward_cfg.get("success_threshold_m", 0.03))
    reward_cfg.setdefault("lift_bonus_weight", 0.5)
    reward_cfg.setdefault("ee_proximity_weight", 0.5)
    reward_cfg.setdefault("ee_proximity_scale_m", 0.08)
    reward_cfg.setdefault("ground_contact_penalty", 1.0)

    cubes = config["cubes"]
    if not isinstance(cubes, list) or len(cubes) != 3:
        raise ValueError("Config must define exactly 3 cubes in 'cubes'")

    franka_cfg = config["franka"]
    if "initial_arm_q" not in franka_cfg:
        raise KeyError("Missing 'franka.initial_arm_q' in stack-cubes config")
    if "initial_gripper_q" not in franka_cfg:
        raise KeyError("Missing 'franka.initial_gripper_q' in stack-cubes config")
    initial_arm_q = franka_cfg["initial_arm_q"]
    if not isinstance(initial_arm_q, (list, tuple)) or len(initial_arm_q) != 7:
        raise ValueError("'franka.initial_arm_q' must be a 7-element list")
    initial_gripper_q = franka_cfg["initial_gripper_q"]
    if not isinstance(initial_gripper_q, (int, float)):
        raise ValueError("'franka.initial_gripper_q' must be a scalar")

    action_scales_cfg = config["rl"]["action_scales"]
    if "gripper" not in action_scales_cfg:
        raise KeyError("Missing 'rl.action_scales.gripper' in stack-cubes config")
    if "gripper_open_limit" not in config["rl"]:
        raise KeyError("Missing 'rl.gripper_open_limit' in stack-cubes config")
    if "ee_velocity_tracking" not in config["rl"]:
        raise KeyError("Missing 'rl.ee_velocity_tracking' in stack-cubes config")
    velocity_tracking_cfg = config["rl"]["ee_velocity_tracking"]
    if "jacobian_damping" not in velocity_tracking_cfg:
        raise KeyError("Missing 'rl.ee_velocity_tracking.jacobian_damping' in stack-cubes config")

    cube_cluster_center_local = config["cluster_center"]
    if (
        not isinstance(cube_cluster_center_local, (list, tuple))
        or len(cube_cluster_center_local) != 3
    ):
        raise ValueError("'cluster_center' must be a [x, y, z] list")

    return config


def _parse_quat_wxyz(quat_cfg):
    if isinstance(quat_cfg, str):
        if quat_cfg != "identity":
            raise ValueError("franka.orientation must be 'identity' or a [w,x,y,z] list")
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    if not isinstance(quat_cfg, (list, tuple)) or len(quat_cfg) != 4:
        raise ValueError("franka.orientation must be 'identity' or a [w,x,y,z] list")

    return torch.tensor([float(v) for v in quat_cfg], dtype=torch.float32)


def _wxyz_to_wp_quat(q_wxyz: torch.Tensor):
    return wp.quat(float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]), float(q_wxyz[0]))


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


def fk_with_ee_offset_batch(
    q_batch: torch.Tensor,
    joint_axes: torch.Tensor,
    link_pos: torch.Tensor,
    link_quats: torch.Tensor,
    ee_offset_p: torch.Tensor,
    ee_offset_q: torch.Tensor,
):
    pos, quat = forward_kinematics_batch(q_batch, joint_axes, link_pos, link_quats)
    pos = pos + transform_by_quat_diff(ee_offset_p.expand(q_batch.shape[0], -1), quat)
    quat = transform_quat_by_quat_diff(quat, ee_offset_q.expand(q_batch.shape[0], -1))
    return pos, quat


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)


def _rotvec_to_quat_batch(rotvec: torch.Tensor) -> torch.Tensor:
    angle = rotvec.norm(dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    small = angle < 1.0e-8
    axis = rotvec / angle.clamp_min(1.0e-8)
    quat = torch.cat([torch.cos(half_angle), axis * torch.sin(half_angle)], dim=-1)
    quat_small = torch.cat([torch.ones_like(half_angle), 0.5 * rotvec], dim=-1)
    return _normalize_quat(torch.where(small.expand_as(quat), quat_small, quat))


class IKTestEnvironment(VecEnv):
    _ARM_REACH_RADIUS_M = 0.6
    _ROBOT_GROUND_CONTACT_BODY_LOCAL_IDS = tuple(range(2, 14))

    def __init__(self, viewer, config: dict, config_path: str):
        self.cfg = config
        self.rl_cfg = self.cfg["rl"]
        self.reward_cfg = self.cfg["reward"]
        self.run_cfg = self.cfg["run"]

        self.fps              = 60
        self.frame_dt         = 1.0 / self.fps
        self.sim_time         = 0.0
        self.sim_substeps     = 10
        self.collide_substeps = False
        self.sim_dt           = self.frame_dt / self.sim_substeps
        self.num_envs = int(self.rl_cfg["num_envs"])
        self.viewer           = viewer                
        self.config_path      = config_path
        self.init_orient_rand_rad = float(self.rl_cfg["init_orientation_rand_rad"])
        self.max_episode_length = int(self.rl_cfg["max_episode_length"])
        self.num_actions = 7
        self.device = torch.device(self.run_cfg["rl_device"])
        self.mode = str(self.run_cfg["mode"])
        self.render_train = bool(self.run_cfg["render_train"])
        self.max_rendered_worlds = int(self.rl_cfg["num_rendered_envs"])
        self.num_debug_render_envs = min(self.max_rendered_worlds, self.num_envs)
        self.debug_render_env_ids = torch.arange(self.num_debug_render_envs, device=self.device, dtype=torch.long)
        self.viewer_cfg = self.run_cfg["viewer"]
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        franka_cfg = self.cfg["franka"]
        self.franka_base_pos = torch.tensor(
            [float(v) for v in franka_cfg["position"]], dtype=torch.float32
        )
        self.franka_base_quat = _parse_quat_wxyz(franka_cfg["orientation"])
        self.franka_initial_arm_q = torch.tensor(
            [float(v) for v in franka_cfg["initial_arm_q"]],
            device=self.device,
            dtype=torch.float32,
        )
        self.franka_initial_gripper_q = float(franka_cfg["initial_gripper_q"])
        self.cube_cfgs = self.cfg["cubes"]
        self.cube_sizes = torch.tensor(
            [float(cube["size"]) for cube in self.cube_cfgs],
            device=self.device,
            dtype=torch.float32,
        )
        self.stack_pair_i = torch.tensor([0, 0, 1], device=self.device, dtype=torch.long)
        self.stack_pair_j = torch.tensor([1, 2, 2], device=self.device, dtype=torch.long)
        self.cluster_center = torch.tensor(
            [float(v) for v in self.cfg["cluster_center"]], dtype=torch.float32
        )
        self.cube_targets_world = []
        self.cube_body_indices_local = []

        self.action_scales_pos = float(self.rl_cfg["action_scales"]["delta_pos"])
        self.action_scales_theta = float(self.rl_cfg["action_scales"]["delta_theta"])
        self.action_scales_gripper = float(self.rl_cfg["action_scales"]["gripper"])
        self.jacobian_damping = float(self.rl_cfg["ee_velocity_tracking"]["jacobian_damping"])
        self.gripper_open_limit = float(self.rl_cfg["gripper_open_limit"])
        ws = self.rl_cfg["workspace_limits"]
        self.workspace_min = torch.tensor(
            [float(ws["x"][0]), float(ws["y"][0]), float(ws["z"][0])],
            device=self.device,
            dtype=torch.float32,
        )
        self.workspace_max = torch.tensor(
            [float(ws["x"][1]), float(ws["y"][1]), float(ws["z"][1])],
            device=self.device,
            dtype=torch.float32,
        )

        self.torch_device = self.device
        self.use_mujoco_contacts = True

        # ------------------------------------------------------------------ #
        # Newton
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
        
        # Arm DOFs track velocity targets; gripper DOFs keep simple position control.
        builder.joint_target_mode[:7] = [int(newton.JointTargetMode.VELOCITY)] * 7
        builder.joint_target_mode[7:9] = [int(newton.JointTargetMode.POSITION)] * 2
        builder.joint_target_ke[:7] = [0.0] * 7
        builder.joint_target_ke[7:9] = [100.0, 100.0]
        builder.joint_target_kd[:9] = [450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0]
        builder.joint_effort_limit[:9] = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 100.0, 100.0]
        builder.joint_armature[:9] = [0.3, 0.3, 0.3, 0.3, 0.11, 0.11, 0.11, 0.15, 0.15]

        builder.joint_q[:7] = self.franka_initial_arm_q.tolist()
        builder.joint_q[7:9] = [self.franka_initial_gripper_q] * 2
        builder.joint_target_pos[:7] = self.franka_initial_arm_q.tolist()
        builder.joint_target_pos[7:9] = [self.franka_initial_gripper_q] * 2
        builder.joint_target_vel[:9] = [0.0] * 9

        scene = newton.ModelBuilder()
        scene.replicate(builder, self.num_envs)
        scene.add_ground_plane()
        self.model = scene.finalize()
        self.num_bodies_per_world = self.model.body_count // self.num_envs
        self._setup_ground_contact_sensor()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=self.use_mujoco_contacts,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=1000,
            nconmax=2000,
            iterations=20,
            ls_iterations=100,
            impratio=1000.0,
        )
        self.contacts = newton.Contacts(
            self.solver.get_max_contact_count(),
            0,
            device=self.model.device,
            requested_attributes=self.model.get_requested_contact_attributes(),
        )

        self.viewer.set_model(self.model, max_worlds=self.max_rendered_worlds)
        self.viewer.picking_enabled = False
        self._apply_cube_colors()
        if hasattr(self.viewer, "renderer"):
            self.viewer.set_camera(wp.vec3(1.0, 0.0, 1.0), -15, -140)
            self.viewer.set_world_offsets(wp.vec3(1.0, 1.0, 0.0))

        self.control = self.model.control()
        
        joint_target_2d_shape = self.control.joint_target_pos.reshape(
            (self.num_envs, -1)
        ).shape
        self.joint_targets_torch = torch.zeros(joint_target_2d_shape, device=self.torch_device, dtype=torch.float32)
        self.control.joint_target_pos = wp.from_torch(
            self.joint_targets_torch.reshape(-1), dtype=wp.float32, requires_grad=False
        )
        joint_target_vel_2d_shape = self.control.joint_target_vel.reshape(
            (self.num_envs, -1)
        ).shape
        self.joint_target_vel_torch = torch.zeros(
            joint_target_vel_2d_shape,
            device=self.torch_device,
            dtype=torch.float32,
        )
        self.control.joint_target_vel = wp.from_torch(
            self.joint_target_vel_torch.reshape(-1), dtype=wp.float32, requires_grad=False
        )
        model_joint_q_torch = wp.to_torch(self.model.joint_q).reshape(self.num_envs, -1)
        model_joint_qd_torch = wp.to_torch(self.model.joint_qd).reshape(self.num_envs, -1)
        self.joint_targets_torch.copy_(model_joint_q_torch[:, : joint_target_2d_shape[1]])
        self.joint_target_vel_torch.copy_(model_joint_qd_torch[:, : joint_target_vel_2d_shape[1]])

        state0_joint_q_torch = wp.to_torch(self.state_0.joint_q)
        state0_joint_qd_torch = wp.to_torch(self.state_0.joint_qd)
        state0_body_q_torch = wp.to_torch(self.state_0.body_q)
        state0_body_qd_torch = wp.to_torch(self.state_0.body_qd)
        self.num_joint_coords_per_world = state0_joint_q_torch.numel() // self.num_envs
        self.num_joint_dofs_per_world = state0_joint_qd_torch.numel() // self.num_envs

        self.cube_body_ids = torch.zeros((self.num_envs, len(self.cube_body_indices_local)), device=self.torch_device, dtype=torch.long)
        for env_id in range(self.num_envs):
            body_start = env_id * self.num_bodies_per_world
            for cube_idx, body_local in enumerate(self.cube_body_indices_local):
                self.cube_body_ids[env_id, cube_idx] = body_start + body_local

        # ------------------------------------------------------------------ #
        # PyTorch differentiable FK + IK                                     #
        # ------------------------------------------------------------------ #
        self._setup_pytorch_ik()
        self.joint_targets_torch[:, : self.n_arm_joints] = self.q_ik
        self.gripper_targets = torch.full(
            (self.num_envs,),
            self.franka_initial_gripper_q,
            device=self.torch_device,
            dtype=torch.float32,
        )
        self.joint_targets_torch[:, self.n_arm_joints] = self.gripper_targets
        self.joint_targets_torch[:, self.n_arm_joints + 1] = self.gripper_targets
        self.joint_target_vel_torch.zero_()
        self._set_gripper_joint_configuration(self.gripper_targets)
        self.arm_joint_velocity_limits = wp.to_torch(self.model.joint_velocity_limit).reshape(self.num_envs, -1)[
            :, : self.n_arm_joints
        ].to(self.torch_device).clone()

        self.initial_joint_q = state0_joint_q_torch.clone().reshape(self.num_envs, self.num_joint_coords_per_world)
        self.initial_joint_qd = state0_joint_qd_torch.clone().reshape(self.num_envs, self.num_joint_dofs_per_world)
        self.initial_body_q = state0_body_q_torch.clone().reshape(self.num_envs, self.num_bodies_per_world, 7)
        self.initial_body_qd = state0_body_qd_torch.clone().reshape(self.num_envs, self.num_bodies_per_world, -1)
        self.initial_joint_targets = self.joint_targets_torch.clone()
        self.initial_joint_target_vel = self.joint_target_vel_torch.clone()

        self._episode_reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.prev_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float32)
        self.prev_stack_err = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._latest_obs = None

        cube_pos0 = self._get_cube_positions_world()
        self.prev_stack_err = self._stack_completion_metric(cube_pos0)

        self.graph = None
        self.capture()

    def _build_scene(self, builder: newton.ModelBuilder, shape_cfg):
        """Add the Franka Panda robot and stack-cube assets to the scene."""
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg = copy.deepcopy(shape_cfg)

        self.urdf_path = str(
            newton.utils.download_asset("franka_emika_panda")
            / "urdf/fr3_franka_hand.urdf"
        )
        franka_base_xform = wp.transform(
            wp.vec3(*self.franka_base_pos.tolist()),
            _wxyz_to_wp_quat(self.franka_base_quat),
        )
        builder.add_urdf(
            self.urdf_path,
            xform=franka_base_xform,
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )

        gravcomp_attr = builder.custom_attributes["mujoco:jnt_actgravcomp"]
        if gravcomp_attr.values is None:
            gravcomp_attr.values = {}
        for dof_idx in range(7):
            gravcomp_attr.values[dof_idx] = True

        gravcomp_body = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp_body.values is None:
            gravcomp_body.values = {}
        for body_idx in range(2, 14):
            gravcomp_body.values[body_idx] = 1.0

        if self.use_mujoco_contacts:
            condim_attr = builder.custom_attributes["mujoco:condim"]
            if condim_attr.values is None:
                condim_attr.values = {}
            for shape_idx in range(builder.shape_count):
                if builder.shape_body[shape_idx] in (12, 13):
                    condim_attr.values[shape_idx] = 4

        cube_shape_cfg = newton.ModelBuilder.ShapeConfig(
            margin=0.0,
            gap=0.005,
        )
        cube_shape_cfg.ke = 5.0e4
        cube_shape_cfg.kd = 5.0e2
        cube_shape_cfg.kf = 1.0e3
        cube_shape_cfg.mu = 0.75

        raw_offsets_local = [
            torch.tensor([float(v) for v in cube["position"]], dtype=torch.float32)
            for cube in self.cube_cfgs
        ]

        xy_offsets = torch.stack([offset[:2] for offset in raw_offsets_local], dim=0)
        xy_offsets = xy_offsets - xy_offsets.mean(dim=0, keepdim=True)
        max_xy_radius = torch.linalg.norm(xy_offsets, dim=1).max().item()
        target_cluster_radius = 0.05
        if max_xy_radius > 1.0e-6:
            xy_offsets = xy_offsets * (target_cluster_radius / max_xy_radius)

        for cube_idx, cube in enumerate(self.cube_cfgs):
            cube_name = cube["name"]
            cube_size = float(cube["size"])
            cube_mass = float(cube["mass"])
            cube_pos_world = torch.tensor(
                [
                    float(self.cluster_center[0] + xy_offsets[cube_idx, 0]),
                    float(self.cluster_center[1] + xy_offsets[cube_idx, 1]),
                    float(self.cluster_center[2]),
                ],
                dtype=torch.float32,
            )

            distance_from_base = torch.linalg.norm(cube_pos_world - self.franka_base_pos).item()
            if distance_from_base > self._ARM_REACH_RADIUS_M:
                raise ValueError(
                    f"Cube '{cube_name}' is out of reach ({distance_from_base:.3f} m > "
                    f"{self._ARM_REACH_RADIUS_M:.3f} m). Update config positions."
                )

            self.cube_targets_world.append(cube_pos_world)

            cube_density = cube_mass / (cube_size**3)
            if cube_density <= 0.0:
                raise ValueError(f"Cube '{cube_name}' has non-positive density")

            cube_shape_cfg.density = cube_density
            cube_body = builder.add_body(
                xform=wp.transform(wp.vec3(*cube_pos_world.tolist()), wp.quat_identity())
            )
            self.cube_body_indices_local.append(cube_body)
            half_size = 0.5 * cube_size
            cube_shape_idx = builder.shape_count
            builder.add_shape_box(
                body=cube_body,
                hx=half_size,
                hy=half_size,
                hz=half_size,
                cfg=cube_shape_cfg,
                label=f"cube/{cube_name}",
            )
            if self.use_mujoco_contacts:
                condim_attr = builder.custom_attributes["mujoco:condim"]
                if condim_attr.values is None:
                    condim_attr.values = {}
                condim_attr.values[cube_shape_idx] = 4

        builder.approximate_meshes("convex_hull")

    # this can probably be simplified: self.viewer.update_shape_colors({self.shape_map[s]: v for s, v in self.cube_colors.items()})
    def _apply_cube_colors(self):
        shape_map = {key: idx for idx, key in enumerate(self.model.shape_label)}
        color_updates = {}

        for cube in self.cube_cfgs:
            cube_name = cube["name"]
            color_rgba = cube["color"]
            if len(color_rgba) < 3:
                raise ValueError(f"Cube '{cube_name}' color must have at least RGB values")
            rgb = [float(color_rgba[0]), float(color_rgba[1]), float(color_rgba[2])]

            key_suffix = f"cube/{cube_name}"
            for shape_label, shape_idx in shape_map.items():
                if key_suffix in shape_label:
                    color_updates[shape_idx] = rgb

        if color_updates:
            self.viewer.update_shape_colors(color_updates)

    def _setup_ground_contact_sensor(self):
        robot_ground_sensor_body_ids = []
        for env_id in range(self.num_envs):
            body_start = env_id * self.num_bodies_per_world
            robot_ground_sensor_body_ids.extend(
                body_start + body_local for body_local in self._ROBOT_GROUND_CONTACT_BODY_LOCAL_IDS
            )

        self.robot_ground_sensor_bodies_per_env = len(self._ROBOT_GROUND_CONTACT_BODY_LOCAL_IDS)
        self.robot_ground_contact_sensor = SensorContact(
            self.model,
            sensing_obj_bodies=robot_ground_sensor_body_ids,
            counterpart_shapes=["ground*"],
            include_total=False,
        )

    def _check_robot_ground_contact(self) -> torch.Tensor:
        self.robot_ground_contact_sensor.update(self.state_0, self.contacts)
        net_force = wp.to_torch(self.robot_ground_contact_sensor.net_force).to(self.torch_device)
        if net_force.ndim < 3 or net_force.shape[1] == 0:
            return torch.zeros(self.num_envs, device=self.torch_device, dtype=torch.bool)
        net_force = net_force.reshape(self.num_envs, self.robot_ground_sensor_bodies_per_env, -1, 3)
        contact_force_norm = torch.linalg.norm(net_force, dim=-1)
        return contact_force_norm.amax(dim=(1, 2)) > 1.0e-6



    def _set_arm_joint_configuration(self, q_batch: torch.Tensor):
        model_joint_q = wp.to_torch(self.model.joint_q).reshape(self.num_envs, -1)
        model_joint_qd = wp.to_torch(self.model.joint_qd).reshape(self.num_envs, -1)
        state0_joint_q = wp.to_torch(self.state_0.joint_q).reshape(self.num_envs, -1)
        state0_joint_qd = wp.to_torch(self.state_0.joint_qd).reshape(self.num_envs, -1)
        state1_joint_q = wp.to_torch(self.state_1.joint_q).reshape(self.num_envs, -1)
        state1_joint_qd = wp.to_torch(self.state_1.joint_qd).reshape(self.num_envs, -1)

        model_joint_q[:, : self.n_arm_joints] = q_batch
        state0_joint_q[:, : self.n_arm_joints] = q_batch
        state1_joint_q[:, : self.n_arm_joints] = q_batch

        model_joint_qd[:, : self.n_arm_joints] = 0.0
        state0_joint_qd[:, : self.n_arm_joints] = 0.0
        state1_joint_qd[:, : self.n_arm_joints] = 0.0

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

    def _set_gripper_joint_configuration(self, gripper_q_batch: torch.Tensor):
        model_joint_q = wp.to_torch(self.model.joint_q).reshape(self.num_envs, -1)
        model_joint_qd = wp.to_torch(self.model.joint_qd).reshape(self.num_envs, -1)
        state0_joint_q = wp.to_torch(self.state_0.joint_q).reshape(self.num_envs, -1)
        state0_joint_qd = wp.to_torch(self.state_0.joint_qd).reshape(self.num_envs, -1)
        state1_joint_q = wp.to_torch(self.state_1.joint_q).reshape(self.num_envs, -1)
        state1_joint_qd = wp.to_torch(self.state_1.joint_qd).reshape(self.num_envs, -1)

        # The Franka hand has two prismatic finger joints. We drive them with one
        # shared aperture scalar and write the same coordinate to both joints.
        model_joint_q[:, self.n_arm_joints] = gripper_q_batch
        model_joint_q[:, self.n_arm_joints + 1] = gripper_q_batch
        state0_joint_q[:, self.n_arm_joints] = gripper_q_batch
        state0_joint_q[:, self.n_arm_joints + 1] = gripper_q_batch
        state1_joint_q[:, self.n_arm_joints] = gripper_q_batch
        state1_joint_q[:, self.n_arm_joints + 1] = gripper_q_batch

        model_joint_qd[:, self.n_arm_joints] = 0.0
        model_joint_qd[:, self.n_arm_joints + 1] = 0.0
        state0_joint_qd[:, self.n_arm_joints] = 0.0
        state0_joint_qd[:, self.n_arm_joints + 1] = 0.0
        state1_joint_qd[:, self.n_arm_joints] = 0.0
        state1_joint_qd[:, self.n_arm_joints + 1] = 0.0

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

    def _refresh_joint_state_cache(self):
        joint_q = wp.to_torch(self.state_0.joint_q).reshape(self.num_envs, self.num_joint_coords_per_world)
        joint_qd = wp.to_torch(self.state_0.joint_qd).reshape(self.num_envs, self.num_joint_dofs_per_world)
        self.arm_q = joint_q[:, : self.n_arm_joints].to(self.torch_device)
        self.arm_qd = joint_qd[:, : self.n_arm_joints].to(self.torch_device)

    def _compute_ee_jacobian_batch(self, q_batch: torch.Tensor) -> torch.Tensor:
        J_p, J_q = forward_kinematics_jacobian_analytical(
            q_batch,
            self.joint_axes,
            self.link_pos,
            self.link_quats,
        )
        _, quat_fk = forward_kinematics_batch(
            q_batch,
            self.joint_axes,
            self.link_pos,
            self.link_quats,
        )
        J_omega = analytic_to_geometric_jacobian(J_q, quat_fk)
        rot_offset_w = transform_by_quat_diff(self.ee_offset_p.expand(q_batch.shape[0], -1), quat_fk)

        rx, ry, rz = rot_offset_w.unbind(-1)
        zero = torch.zeros_like(rx)
        skew_r = torch.stack([
            torch.stack([zero, -rz, ry], dim=-1),
            torch.stack([rz, zero, -rx], dim=-1),
            torch.stack([-ry, rx, zero], dim=-1),
        ], dim=-2)
        J_p_ee = J_p - torch.matmul(skew_r, J_omega)
        return torch.cat([J_p_ee, J_omega], dim=1)

    def _compute_joint_velocity_targets(
        self,
        ee_linear_vel: torch.Tensor,
        ee_angular_vel: torch.Tensor,
    ) -> torch.Tensor:
        J = self._compute_ee_jacobian_batch(self.arm_q)
        twist = torch.cat([ee_linear_vel, ee_angular_vel], dim=-1)
        eye6 = torch.eye(6, device=self.torch_device, dtype=torch.float32).unsqueeze(0).expand(self.num_envs, -1, -1)
        damping = self.jacobian_damping * self.jacobian_damping
        JJt = torch.matmul(J, J.transpose(1, 2))
        solve_rhs = torch.linalg.solve(JJt + damping * eye6, twist.unsqueeze(-1))
        joint_vel = torch.matmul(J.transpose(1, 2), solve_rhs).squeeze(-1)

        velocity_limits = torch.where(
            self.arm_joint_velocity_limits > 0.0,
            self.arm_joint_velocity_limits,
            torch.full_like(self.arm_joint_velocity_limits, torch.inf),
        )
        return torch.clamp(joint_vel, min=-velocity_limits, max=velocity_limits)



    def _setup_pytorch_ik(self):
        """Parse the URDF kinematic chain and initialise PyTorch IK tensors."""
        chain = parse_urdf_kinematic_chain(
            urdf_path  = self.urdf_path,
            root_link  = "fr3_link0",
            ee_link    = "fr3_hand",    # stop before finger joints; control a fixed grasp point via ee_offset
            device     = self.torch_device,
        )

        self.joint_axes   = chain["joint_axes"]    # [7, 3]
        self.link_pos     = chain["p_rel"]         # [7, 3]
        self.link_quats   = chain["q_rel"]         # [7, 4]
        self.joint_limits = chain["joint_ranges"]  # [7, 2]
        self.ee_offset_p  = chain["ee_offset_p"]   # [3]
        self.ee_offset_q  = chain["ee_offset_q"]   # [4]
        self.n_arm_joints = self.joint_axes.shape[0]

        grasp_center_offset_hand = torch.tensor(
            [0.0, 0.0, 0.1034],
            device=self.torch_device,
            dtype=torch.float32,
        )
        self.ee_offset_p = self.ee_offset_p + transform_by_quat_diff(
            grasp_center_offset_hand,
            self.ee_offset_q,
        )

        q_seed = self.franka_initial_arm_q.clone()

        stack_center_local = torch.tensor([0.46, 0.0], device=self.torch_device, dtype=torch.float32)
        size0 = float(self.cube_cfgs[0]["size"])
        stack_base_local = torch.tensor(
            [stack_center_local[0], stack_center_local[1], 0.5 * size0],
            device=self.torch_device,
            dtype=torch.float32,
        )
        stack_base_world = self.franka_base_pos.to(self.torch_device) + transform_by_quat_diff(
            stack_base_local,
            self.franka_base_quat.to(self.torch_device),
        )
        self.stack_targets = torch.zeros((len(self.cube_cfgs), 3), device=self.torch_device, dtype=torch.float32)
        height_accum = 0.0
        for cube_idx, cube in enumerate(self.cube_cfgs):
            cube_size = float(cube["size"])
            half = 0.5 * cube_size
            z = float(stack_base_world[2] + height_accum + half)
            self.stack_targets[cube_idx] = torch.tensor(
                [float(stack_base_world[0]), float(stack_base_world[1]), z],
                device=self.torch_device,
                dtype=torch.float32,
            )
            height_accum += cube_size

        resting_target_world = self.cluster_center.to(
            device=self.torch_device,
            dtype=torch.float32,
        ) + torch.tensor(
            [0.0, 0.0, 0.10],
            device=self.torch_device,
            dtype=torch.float32,
        )
        resting_target_quat = torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            device=self.torch_device,
            dtype=torch.float32,
        )

        target_pos = resting_target_world.unsqueeze(0).repeat(self.num_envs, 1)
        target_quat = resting_target_quat.unsqueeze(0).repeat(self.num_envs, 1)
        q_seed_batch = q_seed.unsqueeze(0).repeat(self.num_envs, 1)

        self.q_ik = solve_ik_batch(
            target_pos=target_pos,
            target_quat=target_quat,
            joint_axes=self.joint_axes,
            link_pos=self.link_pos,
            link_quats=self.link_quats,
            ee_offset_p=self.ee_offset_p,
            ee_offset_q=self.ee_offset_q,
            joint_limits=self.joint_limits,
            q_init=q_seed_batch,
            n_iters=int(100),
            rot_weight=float(0.0),
            max_step_norm=float(0.1),
        )

        self._set_arm_joint_configuration(self.q_ik)
        self._refresh_ee_pose_cache()
        self.resting_q_ik = self.q_ik[0].clone()
        self.resting_pos = target_pos[0].clone()
        self.resting_quat = target_quat[0].clone()
        self.target_pos = target_pos.clone()
        self.target_quat = target_quat.clone()

    def _sample_random_ee_quat(self, n: int) -> torch.Tensor:
        """Sample *n* orientations: resting_quat perturbed by a random rotation.

        The perturbation axis is uniform on S2 and the angle is uniform in
        [-init_orient_rand_rad, +init_orient_rand_rad].  When the config
        value is 0 this returns the resting quat unchanged.
        """
        if self.init_orient_rand_rad <= 0.0:
            return self.resting_quat.unsqueeze(0).repeat(n, 1)

        # Random axis (unit vectors)
        axes = torch.randn(n, 3, device=self.torch_device, dtype=torch.float32)
        axes = axes / axes.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        # Random angle
        angles = (2.0 * torch.rand(n, 1, device=self.torch_device, dtype=torch.float32) - 1.0) * self.init_orient_rand_rad
        rotvec = axes * angles
        delta_quat = _rotvec_to_quat_batch(rotvec)
        return _normalize_quat(quat_mul(delta_quat, self.resting_quat.unsqueeze(0).expand(n, -1)))



    def _refresh_ee_pose_cache(self):
        self._refresh_joint_state_cache()
        self.ee_pos, self.ee_quat = fk_with_ee_offset_batch(
            self.arm_q,
            self.joint_axes,
            self.link_pos,
            self.link_quats,
            self.ee_offset_p,
            self.ee_offset_q,
        )

    def _get_cube_positions_world(self) -> torch.Tensor:
        body_q_torch = wp.to_torch(self.state_0.body_q).reshape(self.num_bodies_per_world * self.num_envs, 7)
        flat_ids = self.cube_body_ids.reshape(-1).to(device=body_q_torch.device)
        cube_pos = body_q_torch.index_select(0, flat_ids)[:, :3]
        return cube_pos.reshape(self.num_envs, len(self.cube_cfgs), 3).to(self.torch_device)

    def _get_pairwise_stack_geometry(self, cube_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xy_thresh = float(self.reward_cfg["stack_xy_threshold_m"])
        z_thresh = float(self.reward_cfg["stack_z_threshold_m"])

        pos_i = cube_pos[:, self.stack_pair_i]
        pos_j = cube_pos[:, self.stack_pair_j]
        xy_dist = torch.linalg.norm(pos_i[:, :, :2] - pos_j[:, :, :2], dim=-1)
        expected_dz = 0.5 * (self.cube_sizes[self.stack_pair_i] + self.cube_sizes[self.stack_pair_j])
        z_sep = torch.abs(pos_i[:, :, 2] - pos_j[:, :, 2])
        z_err = torch.abs(z_sep - expected_dz.unsqueeze(0))
        stacked_pairs = (xy_dist < xy_thresh) & (z_err < z_thresh)
        return xy_dist, z_err, stacked_pairs

    def _stack_completion_metric(self, cube_pos: torch.Tensor) -> torch.Tensor:
        _, _, stacked_pairs = self._get_pairwise_stack_geometry(cube_pos)
        return stacked_pairs.sum(dim=-1, dtype=torch.float32)

    def _stack_is_valid(self, cube_pos: torch.Tensor) -> torch.Tensor:
        return self._stack_completion_metric(cube_pos) >= 2.0

    def _compute_cost(self, actions: torch.Tensor, cube_pos: torch.Tensor, ee_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, stacked_pairs = self._get_pairwise_stack_geometry(cube_pos)
        stacked_pair_count = stacked_pairs.sum(dim=-1, dtype=torch.float32)

        # Dense lift bonus: encourages at least one cube to be lifted.
        cube_heights = cube_pos[:, :, 2]
        cube_base_heights = 0.5 * self.cube_sizes.unsqueeze(0)
        lift_amount = torch.clamp(cube_heights - cube_base_heights, min=0.0)
        lift_bonus = lift_amount.max(dim=-1).values

        # Dense EE proximity bonus.
        ee_to_cubes = torch.linalg.norm(cube_pos - ee_pos.unsqueeze(1), dim=-1)
        ee_scale = max(float(self.reward_cfg["ee_proximity_scale_m"]), 1.0e-6)
        ee_proximity = torch.exp(-ee_to_cubes.min(dim=-1).values / ee_scale)

        # Encourage stacking while the gripper is not holding a cube.
        no_grip_dist = max(float(self.reward_cfg["stacked_no_grip_distance_m"]), 1.0e-6)
        no_grip_mask = (ee_to_cubes.min(dim=-1).values > no_grip_dist).float()
        stacked_no_grip = stacked_pair_count * no_grip_mask

        reward = (
            float(self.reward_cfg["stack_pair_bonus"]) * stacked_pair_count
            + float(self.reward_cfg["lift_bonus_weight"]) * lift_bonus
            + float(self.reward_cfg["ee_proximity_weight"]) * ee_proximity
            + float(self.reward_cfg["stacked_no_grip_bonus_weight"]) * stacked_no_grip
        )

        success_mask = self._stack_is_valid(cube_pos)
        reward = reward + success_mask.float() * float(self.reward_cfg["success_bonus"])

        total_cost = -reward
        stack_err = 3.0 - stacked_pair_count
        return total_cost, reward, stack_err

    def _compute_observations(self, cube_pos: torch.Tensor) -> TensorDict:
        pos_err = self.target_pos - self.ee_pos
        rot_err = quat_error_to_rotvec(self.target_quat, self.ee_quat)
        obs = torch.cat(
            [
                self.ee_pos,           # 3
                self.ee_quat,          # 4
                self.gripper_targets.unsqueeze(-1),  # 1
                self.prev_actions,     # 7
                pos_err,               # 3
                rot_err,               # 3
                cube_pos.reshape(self.num_envs, -1),  # 9
            ],
            dim=-1,
        )
        return TensorDict({"policy": obs.to(self.device)}, batch_size=[self.num_envs], device=self.device)



    def _update_joint_targets(self, actions: torch.Tensor):
        """Track commanded end-effector velocities via Jacobian-based joint speeds.

        Actions layout: [ee_angular_vel (3), ee_linear_vel (3), delta_gripper (1)].
        The arm uses differential IK with the analytical FK Jacobian to map
        commanded task-space twist to joint velocity targets.
        """
        ee_angular_vel = actions[:, 0:3] * self.action_scales_theta
        ee_linear_vel = actions[:, 3:6] * self.action_scales_pos
        delta_gripper = actions[:, 6] * self.action_scales_gripper

        self.target_pos = torch.clamp(self.target_pos + ee_linear_vel * self.frame_dt, self.workspace_min, self.workspace_max)
        delta_quat = _rotvec_to_quat_batch(ee_angular_vel * self.frame_dt)
        self.target_quat = _normalize_quat(quat_mul(delta_quat, self.target_quat))
        self.gripper_targets = torch.clamp(
            self.gripper_targets + delta_gripper,
            min=0.0,
            max=self.gripper_open_limit,
        )

        self._refresh_joint_state_cache()
        joint_vel_targets = self._compute_joint_velocity_targets(ee_linear_vel, ee_angular_vel)

        self.joint_targets_torch[:, : self.n_arm_joints] = self.arm_q
        self.joint_targets_torch[:, self.n_arm_joints] = self.gripper_targets
        self.joint_targets_torch[:, self.n_arm_joints + 1] = self.gripper_targets
        self.joint_target_vel_torch[:, : self.n_arm_joints] = joint_vel_targets
        self.joint_target_vel_torch[:, self.n_arm_joints] = 0.0
        self.joint_target_vel_torch[:, self.n_arm_joints + 1] = 0.0

    def capture(self):
        """Optionally capture the physics step as a CUDA graph."""
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        use_newton_collision_pipeline = not self.use_mujoco_contacts
        if use_newton_collision_pipeline and not self.collide_substeps:
            self.model.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            if use_newton_collision_pipeline and self.collide_substeps:
                self.model.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
        if self.use_mujoco_contacts:
            self.solver.update_contacts(self.contacts, self.state_0)

    def _step_sim(self, actions: torch.Tensor):
        self._update_joint_targets(actions)

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

    def _reset_done_envs(self, done_ids: torch.Tensor):
        if done_ids.numel() == 0:
            return

        joint_q_state0 = wp.to_torch(self.state_0.joint_q).reshape(self.num_envs, self.num_joint_coords_per_world)
        joint_qd_state0 = wp.to_torch(self.state_0.joint_qd).reshape(self.num_envs, self.num_joint_dofs_per_world)
        joint_q_state1 = wp.to_torch(self.state_1.joint_q).reshape(self.num_envs, self.num_joint_coords_per_world)
        joint_qd_state1 = wp.to_torch(self.state_1.joint_qd).reshape(self.num_envs, self.num_joint_dofs_per_world)
        body_q_state0 = wp.to_torch(self.state_0.body_q).reshape(self.num_envs, self.num_bodies_per_world, 7)
        body_qd_state0 = wp.to_torch(self.state_0.body_qd).reshape(self.num_envs, self.num_bodies_per_world, -1)
        body_q_state1 = wp.to_torch(self.state_1.body_q).reshape(self.num_envs, self.num_bodies_per_world, 7)
        body_qd_state1 = wp.to_torch(self.state_1.body_qd).reshape(self.num_envs, self.num_bodies_per_world, -1)

        for env_id in done_ids.tolist():
            self.episode_length_buf[env_id] = 0
            self._episode_reward[env_id] = 0.0
            self.prev_actions[env_id] = 0.0

            joint_q_state0[env_id] = self.initial_joint_q[env_id]
            joint_qd_state0[env_id] = self.initial_joint_qd[env_id]
            joint_q_state1[env_id] = self.initial_joint_q[env_id]
            joint_qd_state1[env_id] = self.initial_joint_qd[env_id]
            body_q_state0[env_id] = self.initial_body_q[env_id]
            body_qd_state0[env_id] = self.initial_body_qd[env_id]
            body_q_state1[env_id] = self.initial_body_q[env_id]
            body_qd_state1[env_id] = self.initial_body_qd[env_id]
            self.joint_targets_torch[env_id] = self.initial_joint_targets[env_id]
            self.joint_target_vel_torch[env_id] = self.initial_joint_target_vel[env_id]

            self.q_ik[env_id] = self.resting_q_ik
            self.target_pos[env_id] = self.resting_pos
            self.target_quat[env_id] = self.resting_quat
            self.gripper_targets[env_id] = self.franka_initial_gripper_q
            cube_pos_env = self.initial_body_q[env_id, self.cube_body_indices_local, :3]
            self.prev_stack_err[env_id] = self._stack_completion_metric(cube_pos_env.unsqueeze(0))[0]

        self._refresh_ee_pose_cache()

    def get_observations(self) -> TensorDict:
        cube_pos = self._get_cube_positions_world()
        self._refresh_ee_pose_cache()
        self._latest_obs = self._compute_observations(cube_pos)
        return self._latest_obs

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions_cpu = torch.clamp(actions, -1.0, 1.0).to(self.torch_device)
        ee_pos_before = self.ee_pos.clone()
        self._step_sim(actions_cpu)

        self.episode_length_buf += 1
        cube_pos = self._get_cube_positions_world()
        self._refresh_ee_pose_cache()
        robot_ground_contact = self._check_robot_ground_contact()

        _, rewards, pos_err = self._compute_cost(actions_cpu, cube_pos, self.ee_pos)
        rewards = rewards - robot_ground_contact.float() * float(self.reward_cfg["ground_contact_penalty"])
        self._episode_reward += rewards
        self.prev_actions.copy_(actions_cpu)
        self.prev_stack_err.copy_(pos_err)

        timeout = self.episode_length_buf >= self.max_episode_length
        success = self._stack_is_valid(cube_pos)
        done_mask = timeout | success | robot_ground_contact

        done_ids = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
        self._reset_done_envs(done_ids)

        ee_step_move = torch.linalg.norm(self.ee_pos - ee_pos_before, dim=-1)

        obs = self._compute_observations(cube_pos)
        dones = done_mask.float().to(self.device)
        rew = rewards.to(self.device)
        extras = {
            "time_outs": timeout.float().to(self.device),
            "log": {
                "/task/ground_contact_rate": robot_ground_contact.float().mean().item(),
                "/task/success_rate": success.float().mean().item(),
                "/task/mean_reward": rewards.mean().item(),
                "/task/mean_action_abs": actions_cpu.abs().mean().item(),
                "/task/mean_ee_step_m": ee_step_move.mean().item(),
            },
        }
        if self.mode == "train" and self.render_train:
            self.render()
        self._latest_obs = obs
        return obs, rew, dones, extras



def _create_viewer_from_config(run_cfg: dict):
    import newton.viewer

    viewer_cfg = run_cfg["viewer"]
    viewer_type = str(viewer_cfg["type"])

    if viewer_type == "gl":
        return newton.viewer.ViewerGL(
            width=int(viewer_cfg["width"]),
            height=int(viewer_cfg["height"]),
            vsync=bool(viewer_cfg["vsync"]),
            headless=bool(viewer_cfg["headless"]),
        )

    if viewer_type == "null":
        return newton.viewer.ViewerNull(num_frames=int(run_cfg["num_frames"]))

    raise ValueError(f"Unsupported run.viewer.type: {viewer_type}")

if __name__ == "__main__":
    config_path = str(Path(__file__).parent / "configs/config_stack_cubes.yaml")
    config = _load_stack_cubes_config(config_path)
    run_cfg = config["run"]

    if bool(run_cfg["quiet"]):
        wp.config.quiet = True
    wp.set_device(str(run_cfg["sim_device"]))

    viewer = _create_viewer_from_config(run_cfg)
    env = IKTestEnvironment(viewer, config, config_path)

    rl_device = str(run_cfg["rl_device"])
    ckpt_path = run_cfg["ckpt"]
    mode = str(run_cfg["mode"])

    if mode == "train":
        train_cfg = env.cfg["training"]
        max_iterations = int(train_cfg["max_iterations"])
        log_dir = Path(str(run_cfg["log_dir"])) / datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        logger_type = str(train_cfg["logger"])
        if logger_type == "tensorboard":
            print(f"TensorBoard log dir: {log_dir}")
            print(f"Start TensorBoard: tensorboard --logdir {log_dir.parent} --port 6006")
            print("Open TensorBoard: http://localhost:6006")
        runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=rl_device)
        if ckpt_path is not None:
            runner.load(str(ckpt_path), map_location=rl_device)
        runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)
    elif mode == "eval":
        if ckpt_path is None:
            raise ValueError("--mode eval requires --ckpt")

        train_cfg = env.cfg["training"]
        runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=rl_device)
        runner.load(str(ckpt_path), map_location=rl_device)
        policy = runner.get_inference_policy(device=rl_device)

        for _ in range(int(run_cfg["num_frames"])):
            obs = env.get_observations().to(rl_device)
            with torch.inference_mode():
                actions = policy(obs)
            env.step(actions)
            env.render()
    else:
        raise ValueError(f"Unsupported run.mode: {mode}")