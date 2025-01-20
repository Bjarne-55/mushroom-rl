from mushroom_rl.environments import IsaacSim
from mushroom_rl.environments.isaacsim_envs.isaac_a1_legged_gym import IsaacA1Description
from mushroom_rl.utils.isaac_sim import ActionType, ObservationType
from mushroom_rl.rl_utils.spaces import Box

import numpy as np
import torch

def quaternion_to_euler(quaternion):
    """
    Convert quaternion to Euler angles.
    Args:
        quaternion (torch.Tensor): A tensor of shape (..., 4) representing quaternions in (w, x, y, z) format.
    Returns:
        torch.Tensor: A tensor of shape (..., 3) representing Euler angles in radians.
    """
    q = quaternion
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Pre-compute repeated terms
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.copysign(torch.tensor(torch.pi / 2), sinp),  # Use 90 degrees if out of range
        torch.asin(sinp)
    )

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)

def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower


class HoneyBatcher(IsaacA1Description):
    def __init__(self, num_envs, horizon, headless):
        self.NUM_DOFS = 12

        backend="torch"
        device="cuda:0"

        usd_path = "/home/bjarne/GitWorkspace/BachelorThesis/mushroom-rl/isaac_assets/honey_badger_2/honey_badger/honey_badger.usd"

        self._action_spec = [
            "fl_j0", "fl_j1", "fl_j2",
            "fr_j0", "fr_j1", "fr_j2",   
            "rl_j0", "rl_j1", "rl_j2", 
            "rr_j0", "rr_j1", "rr_j2"
        ]
        self._default_joint_angles = torch.tensor([
            0.1, -0.8, 1.5,
            -0.1, 0.8, -1.5,
            0.1, -1., 1.5,
            -0.1, 1., -1.5
        ], device=device)
        
        observation_spec = [
            ("base_lin_vel", "", ObservationType.BODY_LIN_VEL),
            ("base_ang_vel", "", ObservationType.BODY_ANG_VEL),
            ("base_pos", "", ObservationType.BODY_POS),

            ("FL_hip_joint_pos", "/body/fl_j0", ObservationType.JOINT_POS),#FL            
            ("FL_thigh_joint_pos", "/fl_l0/fl_j1", ObservationType.JOINT_POS),            
            ("FL_calf_joint_pos", "/fl_l1/fl_j2", ObservationType.JOINT_POS),            

            ("FR_hip_joint_pos", "/body/fr_j0", ObservationType.JOINT_POS),#FR            
            ("FR_thigh_joint_pos", "/fr_l0/fr_j1", ObservationType.JOINT_POS),
            ("FR_calf_joint_pos", "/fr_l1/fr_j2", ObservationType.JOINT_POS),           

            ("RL_hip_joint_pos", "/body/rl_j0", ObservationType.JOINT_POS),#RL
            ("RL_thigh_joint_pos", "/rl_l0/rl_j1", ObservationType.JOINT_POS),
            ("RL_calf_joint_pos", "/rl_l1/rl_j2", ObservationType.JOINT_POS),

            ("RR_hip_joint_pos", "/body/rr_j0", ObservationType.JOINT_POS),#RR
            ("RR_thigh_joint_pos", "/rl_l0/rr_j1", ObservationType.JOINT_POS),
            ("RR_calf_joint_pos", "/rl_l1/rr_j2", ObservationType.JOINT_POS),
            

            ("FL_hip_joint_vel", "/body/fl_j0", ObservationType.JOINT_VEL),
            ("FL_thigh_joint_vel", "/fl_l0/fl_j1", ObservationType.JOINT_VEL),
            ("FL_calf_joint_vel", "/fl_l1/fl_j2", ObservationType.JOINT_VEL),

            ("FR_hip_joint_vel", "/body/fr_j0", ObservationType.JOINT_VEL),
            ("FR_thigh_joint_vel", "/fr_l0/fr_j1", ObservationType.JOINT_VEL),
            ("FR_calf_joint_vel", "/fr_l1/fr_j2", ObservationType.JOINT_VEL),

            ("RL_hip_joint_vel", "/body/rl_j0", ObservationType.JOINT_VEL),
            ("RL_thigh_joint_vel", "/rl_l0/rl_j1", ObservationType.JOINT_VEL),
            ("RL_calf_joint_vel", "/rl_l1/rl_j2", ObservationType.JOINT_VEL),
            
            ("RR_hip_joint_vel", "/body/rr_j0", ObservationType.JOINT_VEL),
            ("RR_thigh_joint_vel", "/rl_l0/rr_j1", ObservationType.JOINT_VEL),
            ("RR_calf_joint_vel", "/rl_l1/rr_j2", ObservationType.JOINT_VEL),
        ]
        additional_data_spec = [("body_rot", "", ObservationType.BODY_ROT), ("body_vel", "", ObservationType.BODY_VEL)]
        collision_groups = [
            ("groundplane", ["/World/groundPlane/collisionPlane"]), 
            ("FL_foot", ["/fl_foot"]), 
            ("FR_foot", ["/fr_foot"]), 
            ("RL_foot", ["/rl_foot"]), 
            ("RR_foot", ["/rr_foot"]),
            ("body", ["/body"]),
            ("lower_body", ["/fl_l2", "/fr_l2", "/rl_l2", "/rr_l2"])
        ]
        collision_between_envs = False
        env_spacing = 3.
        physics_material_spec = self._get_values_for_physics_materials(num_envs)
        IsaacSim.__init__(self, usd_path, self._action_spec, observation_spec, backend, device, collision_between_envs, num_envs, 
                         env_spacing, 0.99, horizon, additional_data_spec=additional_data_spec, collision_groups=collision_groups, 
                         action_type=ActionType.EFFORT, headless=headless, n_intermediate_steps=4, timestep=0.005, 
                         physics_material_spec=physics_material_spec) 
        self._mdp_info.action_space = Box(*((self._task.get_joint_pos_limits() - self._default_joint_angles) / 0.25))
        
        self.observation_helper.add_obs("projected_gravity", 3, -1, 1)
        self.observation_helper.add_obs("commands", 3, -1, 1)
        self.observation_helper.add_obs("actions", self.NUM_DOFS, self.info.action_space.low, self.info.action_space.high)
        self._mdp_info.observation_space = Box(*self.observation_helper.obs_limits)

        self.commands = torch.zeros(num_envs, 4, dtype=torch.float, device=device)
        
        self.normalization_obs_vec = self._get_obs_normilization_vec()
        self.noise_scale_vec = self._get_noise_scale_vec()

        self._soft_dof_pos_limits = self._get_soft_dof_pos_limit()
        
        self._actions = torch.zeros((num_envs, self.NUM_DOFS), device=device)

        self.feet_air_time = torch.zeros((num_envs, 4), device=device)
        self.last_actions =  torch.zeros((num_envs, self.NUM_DOFS), device=device)
        self.last_dof_vel = torch.zeros((num_envs, self.NUM_DOFS), device=device)
        self.last_contacts = torch.zeros((num_envs, 4), device=device, dtype=torch.bool)

        self.forward_vec = torch.tensor([1., 0., 0.], device=device).repeat((num_envs, 1))

        self._effort_limit = self._task.get_joint_max_efforts()
        self.episode_length = torch.zeros((num_envs, ), dtype=int, device=device)
        self.step_counter = 0

    def _get_noise_scale_vec(self):
        v = torch.zeros((self.observation_helper.obs_length), device=self._device)

        lin_vel = self.observation_helper.obs_idx_map["base_lin_vel"]
        ang_vel = self.observation_helper.obs_idx_map["base_ang_vel"]
        joint_positions = self.observation_helper.obs_types_idx_map[ObservationType.JOINT_POS]
        joint_velocities = self.observation_helper.obs_types_idx_map[ObservationType.JOINT_VEL]
        gravity = self.observation_helper.obs_idx_map["projected_gravity"]
        commands = self.observation_helper.obs_idx_map["commands"]
        actions = self.observation_helper.obs_idx_map["actions"]
        
        v[lin_vel] = 0
        v[ang_vel] = 0.2
        v[joint_positions] = 0.01
        v[joint_velocities] = 1.5
        v[gravity] = 0.05
        v[commands[:3]] = 0
        v[actions] = 0

        return v
    
    def _get_obs_normilization_vec(self):
        v = torch.ones((self.observation_helper.obs_length), device=self._device)

        lin_vel = self.observation_helper.obs_idx_map["base_lin_vel"]
        ang_vel = self.observation_helper.obs_idx_map["base_ang_vel"]
        pos = self.observation_helper.obs_idx_map["base_pos"]
        joint_positions = self.observation_helper.obs_types_idx_map[ObservationType.JOINT_POS]
        joint_velocities = self.observation_helper.obs_types_idx_map[ObservationType.JOINT_VEL]
        gravity = self.observation_helper.obs_idx_map["projected_gravity"]
        commands = self.observation_helper.obs_idx_map["commands"]
        actions = self.observation_helper.obs_idx_map["actions"]
        
        v[lin_vel] = 1./10.
        v[ang_vel] = 1./50.
        v[joint_positions] = 1./4.6
        v[joint_velocities] = 1./35.0
        v[commands[2]] = 0.25
        v[actions] = 1./10.
        v[pos] = 1./0.4 #0.4 is roughly the robot height probably a little more than robot height

        return v

    def _modify_observation(self, obs):
        base_pos_indices = self.observation_helper.obs_idx_map["base_pos"]
        obs[:, base_pos_indices[:2]] = 0.

        dof_pos_indices = self.observation_helper.obs_types_idx_map[ObservationType.JOINT_POS]
        obs[:, dof_pos_indices] -= self._default_joint_angles

        obs += (2 * torch.rand_like(obs) - 1) * self.noise_scale_vec
        #missing dropout
        obs *= self.normalization_obs_vec

        lin_vel = self.observation_helper.obs_idx_map["base_lin_vel"]
        ang_vel = self.observation_helper.obs_idx_map["base_ang_vel"]
        pos = self.observation_helper.obs_idx_map["base_pos"]
        obs[lin_vel] = torch.clip(obs[lin_vel], -1.0, 1.0)
        obs[ang_vel] = torch.clip(obs[ang_vel], -1.0, 1.0)
        obs[pos] = torch.clip(obs[pos]- 1.0, -1.0, 1.0)

        obs = torch.clamp(obs, max=100., min=-100.)
        
        return obs
    
    #rewards --------------------------------------------------------------------

    def reward(self, obs, action, next_obs, absorbing):
        base_lin_vel = self.observation_helper.get_from_obs(next_obs, "base_lin_vel")
        base_lin_vel_xy = base_lin_vel[:, 0:2]
        base_lin_vel_z = base_lin_vel[:, 2]

        base_ang_vel = self.observation_helper.get_from_obs(next_obs, "base_ang_vel")
        base_ang_vel_xy = base_ang_vel[:, 0:2]
        base_ang_vel_z = base_ang_vel[:, 2]

        base_rot = self._read_data("body_rot")
        base_rot_euler = quaternion_to_euler(base_rot)
        base_rot_xy = base_rot_euler[:, :2]

        base_pos = self.observation_helper.get_from_obs(next_obs, "base_pos")
        base_pos_z = base_pos[:, 2]

        dof_vel = self.observation_helper.get_by_type_from_obs(next_obs, ObservationType.JOINT_VEL)
        dof_pos = self.observation_helper.get_by_type_from_obs(next_obs, ObservationType.JOINT_POS)

        #--------------------------------------------------------------------------------
        #curriculum_coeff is missing
        r_tracking_lin_vel = self._reward_tracking_lin_vel(base_lin_vel_xy) * 2. * self.dt
        r_tracking_yaw_vel = self._reward_tracking_ang_vel(base_ang_vel_z) * 1. * self.dt
        r_lin_vel = self._reward_lin_vel_z(base_lin_vel_z) * -2. * self.dt
        r_ang_vel = self._reward_ang_vel_xy(base_ang_vel_xy) * -5e-2 * self.dt
        r_ang_pos = self._reward_ang_pos_xy(base_rot_xy) * -2e-1 * self.dt
        r_dof_pos_limits = self._reward_dof_pos_limits(dof_pos) * -1e1 * self.dt
        r_dof_acc = self._reward_dof_acc(dof_vel) * -2.5e-7 * self.dt
        r_torque = self._reward_torques(self._torques) * -2e-4 * self.dt
        r_action_rate = self._reward_action_rate(action) * -1e-2 * self.dt
        r_collision = (self._reward_collision() + absorbing) * -1 * self.dt
        r_height = self._reward_height(base_pos_z) * -3e1 * self.dt
        r_feet_air_time = self._reward_feet_air_time() * 1e-1 * self.dt
        r_symmetry = self._reward_symmetry() * -0.5 * self.dt

        self._extra_info_rewards = {
            "r_tracking_lin_vel": r_tracking_lin_vel, "r_tracking_yaw_vel": r_tracking_yaw_vel, "r_lin_vel_z": r_lin_vel,
            "r_ang_vel_xy": r_ang_vel, "r_torque": r_torque, "r_dof_acc": r_dof_acc, "r_feet_air_time": r_feet_air_time,
            "r_collision": r_collision, "r_action_rate": r_action_rate, "r_dof_pos_limits": r_dof_pos_limits, "r_ang_pos": r_ang_pos,
            "r_height": r_height, "r_symmetry": r_symmetry
        }

        reward = r_tracking_lin_vel + r_tracking_yaw_vel + r_lin_vel + r_ang_vel + r_ang_pos + r_dof_pos_limits \
                + r_dof_acc + r_torque + r_action_rate + r_collision + r_height + r_feet_air_time + r_symmetry

        reward = torch.clamp(reward, min=0.)

        self.last_actions[:] = action[:]
        self.last_dof_vel[:] = dof_vel[:]

        return reward
    
    def _reward_ang_pos_xy(self, ang_pos_xy):
        return torch.sum(torch.square(ang_pos_xy), dim=1)
    
    def _reward_height(self, base_z):
        nominal_base_z = 0.316
        return torch.square(base_z - nominal_base_z)
    
    def _reward_feet_air_time(self):
        contact = torch.zeros((self.number, 4), device=self._device, dtype=bool)
        for i, foot in enumerate(["FL_foot", "FR_foot", "RL_foot", "RR_foot"]):
            contact[:, i] = self._check_collision(foot, "groundplane")
        first_contact = (self.feet_air_time > 0.) * contact
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact
        return rew_airTime
    
    def _reward_symmetry(self):
        contact = torch.zeros((self.number, 4), device=self._device, dtype=bool)
        for i, foot in enumerate(["FL_foot", "FR_foot", "RL_foot", "RR_foot"]):
            contact[:, i] = self._check_collision(foot, "groundplane")
        symmetry_violations = torch.logical_and(torch.logical_not(contact[:, 0]), torch.logical_not(contact[:, 1])) + \
                                torch.logical_and(torch.logical_not(contact[:, 2]), torch.logical_not(contact[:, 3]))
        return symmetry_violations
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return self._get_collision_count("lower_body", "groundplane")

