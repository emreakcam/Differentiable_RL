import torch

from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat, quat_to_R

def quat_log(q, eps=1e-6):
    # q = [w, x, y, z]
    v = q[:, 1:]
    w = torch.clamp(q[:, 0], -1.0, 1.0)

    norm_v = torch.norm(v, dim=1, keepdim=True)
    theta = 2.0 * torch.atan2(norm_v, w.unsqueeze(-1))

    scale = torch.where(
        norm_v > eps,
        theta / norm_v,
        torch.zeros_like(norm_v),
    )

    return scale * v   # R^3

def behavior_shaping_reward(env):            

    q_WB = env.drone.get_quat()
    p_WB_W = env.drone.get_pos()
    p_WE_W = (env.link_finger_left.get_pos() + env.link_finger_right.get_pos()) / 2.0
    p_BE_B = transform_by_quat(p_WE_W - p_WB_W, inv_quat(q_WB))
    
    penalty_ee_keepout = -0.5 * (p_BE_B[:, 2] < 0.2)
    
    return penalty_ee_keepout


def ee_velocity_tracking_rewards(env):
    """Reward the end-effector following the desired velocity and attitude."""

    v_WE_W = env.link_ee.get_vel()
    v_WG_W = env.v_WG_W

    # Velocity tracking reward - exponential based on squared error
    vel_err_sq = torch.sum((v_WG_W - v_WE_W)**2, dim=1)
    velocity_reward = torch.exp(-0.5 * vel_err_sq)  # ranges from 0 (bad) to 1.0 (perfect)

    task_reward = torch.nan_to_num(
        velocity_reward,
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )
    return task_reward


def ee_pose_tracking_rewards(env, return_components: bool = False):
    """Reward the end-effector matching the target pose and moving towards it."""
            
    p_WE_W = (env.link_finger_left.get_pos() + env.link_finger_right.get_pos()) / 2 # position vector from world to end-effector in world frame  
    pos_err_norm = torch.norm(env.p_WG_W - p_WE_W, dim=1)
    
    # Distance-based weight: de-emphasize attitude/vel terms when far from goal
    weight_near = 1.0 / (1.0 + 10 * pos_err_norm**2)

    # goal position reward
    reward_pos = 1.0 * weight_near - 0.5 * pos_err_norm    
    
    # orientation residual
    q_WE = env.link_ee.get_quat()
    q_GE = transform_quat_by_quat(q_WE, inv_quat(env.q_WG)) # transform_quat_by_quat(v,u): This is equivalent to quatmul(quat_u, quat_v) or R_u @ R_v    
    e_R = quat_log(q_GE)
    ori_err = torch.norm(e_R, dim=1) 
    reward_orientation = 2.0 * weight_near * torch.exp(-1.0 * ori_err)

    # ee velocity residual at the goal        
    vel_err_goal = torch.norm(env.v_WG_W - env.link_ee.get_vel(), dim=1)
    reward_goal_vel = 1.0 * weight_near * torch.exp(-4.0 * vel_err_goal)

    # ee angular velocity residual at the goal
    omega_WG_W = torch.zeros((env.num_envs, 3), device=env.device)
    ang_err_goal = torch.norm(omega_WG_W - env.link_ee.get_ang(), dim=1)
    reward_goal_ang = 1.0 * weight_near * torch.exp(-2.0 * ang_err_goal)

    # base velocity residual at the goal        
    vel_err_goal = torch.norm(env.v_WG_W - env.drone.get_vel(), dim=1)
    reward_goal_vel_base = 1.0 * weight_near * torch.exp(-4.0 * vel_err_goal)

    # base angular velocity residual at the goal
    omega_WG_W = torch.zeros((env.num_envs, 3), device=env.device)
    ang_err_goal = torch.norm(omega_WG_W - env.drone.get_ang(), dim=1)
    reward_goal_ang_base = 1.0 * weight_near * torch.exp(-4.0 * ang_err_goal)

    # penalty for deviating from arm home pos -> encourages the arm to stay in the nominal / resting configuration if not required otherwise 
    arm_pos_err = torch.norm(env.drone.get_dofs_position(dofs_idx_local = env._arm_dof_idx_local) - env.arm_home_q_pos, dim=1)
    # reward_arm_near_resting_pose = 1e0 * torch.exp(-1.0 * arm_pos_err)
    reward_arm_near_resting_pose = -0.3 * arm_pos_err

    # action smoothness penalty
    if getattr(env, "last_action", None) is None:
        env.last_action = env.curr_action.clone()

    norm_diff_action_ee = torch.norm(env.curr_action[:,6::] - env.last_action[:,6::], dim=1)   
    norm_diff_action_base = torch.norm(env.curr_action[:,0:6] - env.last_action[:,0:6], dim=1)   
    reward_action_smoothness_ee = 1.0 * torch.exp(-4.0 * norm_diff_action_ee)    
    reward_action_smoothness_base = 1.0 * torch.exp(-4.0 * norm_diff_action_base)    
    env.last_action = env.curr_action.clone()

    # jerk penalty for ee
    ee_link_acc = env.drone.get_links_acc(links_idx_local = env.link_ee.idx_local).squeeze(dim=1)
    if getattr(env, "last_ee_acc", None) is None:
        env.last_ee_acc = ee_link_acc
    norm_diff_ee_acc = torch.norm(ee_link_acc - env.last_ee_acc, dim=1)
    reward_ee_jerk_smoothness = 0.1 * torch.exp(-1.0 * norm_diff_ee_acc / env.ctrl_dt)
    env.last_ee_acc = ee_link_acc

    # jerk penalty for base
    base_link_acc = env.drone.get_links_acc(links_idx_local = env.drone.base_link_idx).squeeze(dim=1)
    if getattr(env, "last_base_acc", None) is None:
        env.last_base_acc = base_link_acc
    norm_diff_base_acc = torch.norm(base_link_acc - env.last_base_acc, dim=1)
    reward_base_jerk_smoothness = 0.2 * torch.exp(-1.0 * norm_diff_base_acc / env.ctrl_dt)
    env.last_base_acc = base_link_acc
    
    # high velocity penalty
    penalty_high_velocity = torch.zeros(env.num_envs, device=env.device)
    is_high_velocity = torch.norm(env.drone.get_vel(), dim=1) > 4.0
    penalty_high_velocity[is_high_velocity] = -1.0

    # high turn rate penalty
    penalty_high_body_rates = torch.zeros(env.num_envs, device=env.device)
    is_high_body_rate = torch.norm(env.drone.get_ang(), dim=1) > 6.2 # roughly 1 rotation/s
    penalty_high_body_rates[is_high_body_rate] = -1.0

    # encourage the system to stay level
    z_W = torch.tensor([0.0, 0.0, 1.0], device=env.device).unsqueeze(0).repeat(env.num_envs,1)
    zB_W = transform_by_quat(z_W, env.drone.get_quat())
    reward_upright = 1 * (z_W[:,None,:] @ -zB_W[:,:,None]).squeeze()
    
    total = (
        # rewards for matching the goal state
        reward_pos        
        + reward_orientation        
        + reward_upright
        + reward_goal_vel
        + reward_goal_ang
        # + reward_goal_vel_base
        # + reward_goal_ang_base
        # rewards for smoothing the motion
        + reward_action_smoothness_base
        + reward_action_smoothness_ee
        + reward_ee_jerk_smoothness
        + reward_base_jerk_smoothness
        + reward_arm_near_resting_pose
        + penalty_high_velocity
        + penalty_high_body_rates        
    )
    
    total = torch.nan_to_num(total, nan=0.0, posinf=1e6, neginf=-1e6)
    return total

def pick_object_rewards(env, return_components: bool = False):
    """Reward the agent for picking up an object and moving it to a goal pose."""
    
    p_WO_W = env.targetObject.get_pos()
    p_WFL_W = env.link_finger_left.get_pos()
    p_WFR_W = env.link_finger_right.get_pos()
    
    # Distance-based metrics
    finger_left_to_obj = torch.norm(p_WO_W - p_WFL_W, dim=1)
    finger_right_to_obj = torch.norm(p_WO_W - p_WFR_W, dim=1)
    avg_finger_dist = (finger_left_to_obj + finger_right_to_obj) / 2.0
    
    # Approach: exponential reward for getting fingers close
    approach_reward = 2.0 * torch.exp(-5.0 * avg_finger_dist)
    
    # Pinch reward: both fingers close to object simultaneously
    both_close = (finger_left_to_obj < 0.05).float() * (finger_right_to_obj < 0.05).float()
    pinch_reward = 1.5 * both_close
    
    # Contact-based grasp detection
    finger_actor_indices = torch.tensor(env._finger_link_idx_global, device=env.device)
    object_reactor_indices = torch.tensor(env._target_link_idx_global, device=env.device)
    
    contact_info = env.drone.get_contacts()
    finger_contact_mask = torch.isin(contact_info['link_a'], finger_actor_indices) | torch.isin(contact_info['link_b'], finger_actor_indices)
    object_contact_mask = torch.isin(contact_info['link_a'], object_reactor_indices) | torch.isin(contact_info['link_b'], object_reactor_indices)
    finger_object_contact = finger_contact_mask & object_contact_mask & contact_info['valid_mask']
    has_contact = torch.any(finger_object_contact, dim=1).float()
    
    contact_bonus = 2.0 * has_contact
    
    # Goal reward: move object to goal
    is_grasped = (both_close + has_contact).clamp(0, 1)
    obj_to_goal_dist = torch.norm(env.p_WG_W - p_WO_W, dim=1)
    goal_reward = is_grasped * 3.0 * torch.exp(-5.0 * obj_to_goal_dist)

    # Lift reward (only once we're plausibly grasping): encourages breaking contact with ground
    lift_height = torch.clamp(p_WO_W[:, 2] - 0.04, min=0.0, max=0.20)
    lift_reward = is_grasped * 4.0 * lift_height
    
    total = approach_reward + pinch_reward + contact_bonus + goal_reward + lift_reward
    total = torch.nan_to_num(total, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if return_components:
        return approach_reward, goal_reward, lift_reward, total
    return total

def base_pose_tracking_cost(env):
    
    # position residual
    p_WB_W = env.drone.get_pos()
    pos_err = env.p_WG_W - p_WB_W
    # cost_pos = 1.0 * torch.sum(pos_err ** 2, axis=1)
    cost_pos = -1 * torch.exp(-1 *torch.sum(pos_err ** 2, dim=1))

    # Distance-based weight: de-emphasize attitude/vel terms when far from goal
    dist = torch.norm(pos_err, dim=1)
    weight_near = 1.0 / (1.0 + dist)  # smoothly decays with distance

    # velocity residual
    v_WB_W = env.drone.get_vel()
    v_WG_W = torch.zeros((env.num_envs, 3), device=env.device)
    cost_vel = -weight_near * torch.exp(-1 * torch.sum((v_WG_W - v_WB_W) ** 2, axis=1))

    # orientation residual
    q_WB = env.drone.get_quat()
    q_GB = transform_quat_by_quat(q_WB, inv_quat(env.q_WG)) # transform_quat_by_quat(v,u): This is equivalent to quatmul(quat_u, quat_v) or R_u @ R_v
    cost_orientation = weight_near * (1 - torch.abs(q_GB[:, 0]))

    # ang velocity residual
    omega_WB_W = env.drone.get_ang()
    omega_WG_W = torch.zeros((env.num_envs, 3), device=env.device)
    cost_ang = -weight_near * torch.exp(-1 * torch.sum((omega_WG_W - omega_WB_W) ** 2, axis=1))

    task_cost = (
        cost_pos
        + cost_vel
        + cost_orientation        
        + cost_ang
    )

    return task_cost