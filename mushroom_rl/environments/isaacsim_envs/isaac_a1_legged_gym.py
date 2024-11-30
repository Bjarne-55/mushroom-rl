from mushroom_rl.environments import IsaacSim
from mushroom_rl.environments.isaac_sim_env import ActionType #TODO
from mushroom_rl.utils.isaac_sim import ObservationType
import numpy as np
import torch
import random
from mushroom_rl.rl_utils.spaces import Box

class IsaacA1Description(IsaacSim):
    def __init__(self, num_envs, horizon, headless):
        backend="torch"
        device="cuda:0"

        usd_path = "/home/bjarne/GitWorkspace/BachelorThesis/mushroom-rl/isaac_assets/a1_legged_gym_2/a1/a1.usd"

        self._action_spec = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", 
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",   
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", 
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
        ]
        self._default_joint_angles = torch.tensor([
            0.1, 0.8, -1.5,
            -0.1, 0.8, -1.5,
            0.1, 1., -1.5,
            -0.1, 1., -1.5
        ], device=device)
        #self._action_spec = ["FL_thigh_joint", "FR_thigh_joint"]
        observation_spec = [#Can't Add something like gravity or something like commands
            ("base_lin_vel", "/base", ObservationType.BODY_LIN_VEL),
            ("base_ang_vel", "/base", ObservationType.BODY_ANG_VEL),

            ("FL_hip_joint_pos", "/trunk/FL_hip_joint", ObservationType.JOINT_POS),#FL
            ("FL_hip_joint_vel", "/trunk/FL_hip_joint", ObservationType.JOINT_VEL),
            ("FL_thigh_joint_pos", "/FL_hip/FL_thigh_joint", ObservationType.JOINT_POS),
            ("FL_thigh_joint_vel", "/FL_hip/FL_thigh_joint", ObservationType.JOINT_VEL),
            ("FL_calf_joint_pos", "/FL_thigh/FL_calf_joint", ObservationType.JOINT_POS),
            ("FL_calf_joint_vel", "/FL_thigh/FL_calf_joint", ObservationType.JOINT_VEL),

            ("FR_hip_joint_pos", "/trunk/FR_hip_joint", ObservationType.JOINT_POS),#FR
            ("FR_hip_joint_vel", "/trunk/FR_hip_joint", ObservationType.JOINT_VEL),
            ("FR_thigh_joint_pos", "/FR_hip/FR_thigh_joint", ObservationType.JOINT_POS),
            ("FR_thigh_joint_vel", "/FR_hip/FR_thigh_joint", ObservationType.JOINT_VEL),
            ("FR_calf_joint_pos", "/FR_thigh/FR_calf_joint", ObservationType.JOINT_POS),
            ("FR_calf_joint_vel", "/FR_thigh/FR_calf_joint", ObservationType.JOINT_VEL),

            ("RL_hip_joint_pos", "/trunk/RL_hip_joint", ObservationType.JOINT_POS),#RL
            ("RL_hip_joint_vel", "/trunk/RL_hip_joint", ObservationType.JOINT_VEL),
            ("RL_thigh_joint_pos", "/RL_hip/RL_thigh_joint", ObservationType.JOINT_POS),
            ("RL_thigh_joint_vel", "/RL_hip/RL_thigh_joint", ObservationType.JOINT_VEL),
            ("RL_calf_joint_pos", "/RL_thigh/RL_calf_joint", ObservationType.JOINT_POS),
            ("RL_calf_joint_vel", "/RL_thigh/RL_calf_joint", ObservationType.JOINT_VEL),

            ("RR_hip_joint_pos", "/trunk/RR_hip_joint", ObservationType.JOINT_POS),#RR
            ("RR_hip_joint_vel", "/trunk/RR_hip_joint", ObservationType.JOINT_VEL),
            ("RR_thigh_joint_pos", "/RR_hip/RR_thigh_joint", ObservationType.JOINT_POS),
            ("RR_thigh_joint_vel", "/RR_hip/RR_thigh_joint", ObservationType.JOINT_VEL),
            ("RR_calf_joint_pos", "/RR_thigh/RR_calf_joint", ObservationType.JOINT_POS),
            ("RR_calf_joint_vel", "/RR_thigh/RR_calf_joint", ObservationType.JOINT_VEL),
        ]
        additional_data_spec = [("body_rot", "/base", ObservationType.BODY_ROT)]
        collision_groups = [
            ("groundplane", ["/World/defaultGroundPlane/GroundPlane/CollisionPlane"]), 
            ("FL_foot", ["/FL_foot"]), 
            ("FR_foot", ["/FR_foot"]), 
            ("RL_foot", ["/RL_foot"]), 
            ("RR_foot", ["/RR_foot"]),
            ("body", ["/trunk", "/FL_thigh_shoulder", "/FR_thigh_shoulder", "/RL_thigh_shoulder", "/RR_thigh_shoulder"]),
            ("lower_body", ["/FL_thigh", "/FR_thigh", "/RL_thigh", "/RR_thigh", "/FL_calf", "/FR_calf", "/RL_calf", "/RR_calf"])
        ]
        collision_between_envs = False
        env_spacing = 2.
        super().__init__(usd_path, self._action_spec, observation_spec, backend, device, collision_between_envs, num_envs, 
                         env_spacing, 0.99, horizon, additional_data_spec=additional_data_spec, collision_groups=collision_groups, 
                         action_type=ActionType.POSITION, headless=headless, n_substeps=1)
        
        self.observation_helper.add_obs("projected_gravity", 3, -1, 1)
        self.observation_helper.add_obs("commands", 3, -1, 1)
        self.observation_helper.add_obs("actions", 12, self.info.action_space.low, self.info.action_space.high)
        self._mdp_info.observation_space = Box(*self.observation_helper.obs_limits)

        self.goal_linear_velocity_xy = torch.tensor([0.5, 0], device=device).repeat(self.number, 1) #simplified in legged_gym every env has random goal
        self.goal_angular_velocity_z = torch.tensor([0], device=device).repeat(self.number)
        
        self.normalization_obs_vec = self._get_obs_normilization_vec()
        self.noise_scale_vec = self._get_noise_scale_vec()
        
        self._last_joint_vels = torch.zeros((num_envs, 12), device=device)
        self._last_action = torch.zeros((num_envs, 12), device=device)
        self.foot_air_time = torch.zeros((num_envs, 4), device=device)
        self._actions = torch.zeros((num_envs, 12), device=device)
    
    def _get_obs_normilization_vec(self):
        v = torch.zeros((self.observation_helper.obs_length), device=self._device)

        lin_vel = self.observation_helper.obs_idx_map["base_lin_vel"]
        ang_vel = self.observation_helper.obs_idx_map["base_ang_vel"]
        joint_positions = self.observation_helper.obs_types_idx_map[ObservationType.JOINT_POS]
        joint_velocities = self.observation_helper.obs_types_idx_map[ObservationType.JOINT_VEL]
        gravity = self.observation_helper.obs_idx_map["projected_gravity"]
        commands = self.observation_helper.obs_idx_map["commands"]
        actions = self.observation_helper.obs_idx_map["actions"]
        
        v[lin_vel[0]:lin_vel[1]] = 2.0
        v[ang_vel[0]:ang_vel[1]] = 0.25
        v[joint_positions] = 1.00
        v[joint_velocities] = 0.05
        v[gravity] = 1.
        v[commands[0:2]] = 2.0
        v[commands[2]] = 0.25
        v[actions] = 1.

        return v

    def _get_noise_scale_vec(self):
        v = torch.zeros((self.observation_helper.obs_length), device=self._device)

        lin_vel = self.observation_helper.obs_idx_map["base_lin_vel"]
        ang_vel = self.observation_helper.obs_idx_map["base_ang_vel"]
        joint_positions = self.observation_helper.obs_types_idx_map[ObservationType.JOINT_POS]
        joint_velocities = self.observation_helper.obs_types_idx_map[ObservationType.JOINT_VEL]
        gravity = self.observation_helper.obs_idx_map["projected_gravity"]
        commands = self.observation_helper.obs_idx_map["commands"]
        actions = self.observation_helper.obs_idx_map["actions"]
        
        v[lin_vel[0]:lin_vel[1]] = 0.1 * 2.0
        v[ang_vel[0]:ang_vel[1]] = 0.2 * 0.25
        v[joint_positions] = 0.01 * 1.00
        v[joint_velocities] = 1.5 * 0.05
        v[gravity] = 0.05
        v[commands] = 0
        v[actions] = 0

        return v

    def is_absorbing(self, obs):
        fallen = self._check_collision("body", "groundplane")
        return fallen
    
    def setup(self, env_indices, obs):
        self.foot_air_time[env_indices] = 0.
        self._last_joint_vels[env_indices] = torch.nan
        self._last_action[env_indices] = torch.nan
        self._actions[env_indices] = 0
        return

    def _modify_observation(self, obs):
        obs *= self.normalization_obs_vec
        obs += (2 * torch.rand_like(obs) - 1) * self.noise_scale_vec

        obs = torch.clamp(obs, max=100., min=-100.)

        return obs
    
    def _create_observation(self, obs):
        rot = self._read_data("body_rot")
        gravity_indices = self.observation_helper.obs_idx_map["projected_gravity"]
        gravity = torch.tensor([0., 0., 1.], device=self._device).repeat((self.number, 1))
        obs[:, gravity_indices] = self._quat_rotate_inverse(rot, gravity)

        command_indices = self.observation_helper.obs_idx_map["commands"]
        obs[:, command_indices[0:2]] = self.goal_linear_velocity_xy
        obs[:, command_indices[2]] = self.goal_angular_velocity_z

        action_indices = self.observation_helper.obs_idx_map["actions"]
        obs[:, action_indices] = self._actions

        return obs
    
    @staticmethod
    def _quat_rotate_inverse(q, v):
        shape = q.shape
        q_w = q[:, -1]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
        return a - b + c

    def _preprocess_action(self, action):
        action = 0.25 * action + self._default_joint_angles
        self._actions = action
        return action
    
    def _create_info_dictionary(self, obs):
        return {}
    
    def _get_reward_air_time(self):
        self.foot_air_time += self.dt
        contact = torch.zeros((self.number, 4), device=self._device, dtype=bool)
        for i, foot in enumerate(["FL_foot", "FR_foot", "RL_foot", "RR_foot"]):
            collision_mask = self._check_collision(foot, "groundplane")
            contact[collision_mask, i] = True

        rew = torch.sum((self.foot_air_time - 0.5) * contact, dim=1)
        rew *= torch.norm(self.goal_linear_velocity_xy, dim=1) > 0.1
        self.foot_air_time *= ~ contact
        return rew
    
    def _get_torque(self, action, joint_vels, last_joint_vels):
        #actions_scaled = action * 0.25
        torques = 20.0 * (action - joint_vels) - 0.5*(joint_vels - last_joint_vels) / self.dt
        return torch.clip(torques, -1.0, 1.0)
    
    #Taken from https://proceedings.mlr.press/v164/rudin22a.html
    #Ripped from https://github.com/leggedrobotics/legged_gym/blob/17847702f90d8227cd31cce9c920aa53a739a09a/legged_gym/envs/base/legged_robot.py#L815C3-L816C12
    def reward(self, obs, action, next_obs, absorbing):
        base_lin_vel = self.observation_helper.get_from_obs(next_obs, "base_lin_vel")
        base_lin_vel_xy = base_lin_vel[:, 0:2]
        base_lin_vel_z = base_lin_vel[:, 2]
        base_ang_vel = self.observation_helper.get_from_obs(next_obs, "base_ang_vel")
        base_ang_vel_xy = base_ang_vel[:, 0:2]
        base_ang_vel_z = base_ang_vel[:, 2]

        joint_vels = self.observation_helper.get_by_type_from_obs(next_obs, ObservationType.JOINT_VEL)

        self._last_joint_vels = torch.where(torch.isnan(self._last_joint_vels), joint_vels, self._last_joint_vels)
        self._last_action = torch.where(torch.isnan(self._last_action), action, self._last_action)

        dof_positions = torch.deg2rad(self.observation_helper.get_by_type_from_obs(next_obs, ObservationType.JOINT_POS))

        torques = self._get_torque(action, joint_vels, self._last_joint_vels)

        #calculations of rewards -------------------------------------------------------

        x = self.goal_linear_velocity_xy - base_lin_vel_xy
        reward_lin_vel_tracking = torch.exp(-(torch.sum(torch.square(x), dim=1) / 0.25))

        x = self.goal_angular_velocity_z - base_ang_vel_z
        reward_ang_vel_tracking = torch.exp(-(torch.square(x) / 0.25))

        penalty_lin_vel_z = - torch.square(base_lin_vel_z)
        penalty_ang_vel_xy = - torch.sum(torch.square(base_ang_vel_xy), dim=1)

        penalty_joint_accs = - torch.sum(torch.square((self._last_joint_vels - joint_vels) / self.dt), dim=1)

        penalty_joint_torques = - torch.sum(torch.square(torques), dim=1)#TODO

        penalty_action_rate = - torch.sum(torch.square(self._last_action - action), dim=1)#TODO rework with actual values

        penalty_collisions = - self._get_collision_count("lower_body", "groundplane")

        reward_feet_air_time = self._get_reward_air_time()

        x =  -(dof_positions - self.info.action_space.low).clip(max=0.)
        x += (dof_positions - self.info.action_space.high).clip(min=0.)
        penalty_dof_pos_limits = - torch.sum(x, dim=1)

        #---------------------------------------------------------------------------

        reward = reward_lin_vel_tracking * 1.0 \
                    + reward_ang_vel_tracking * 0.5  \
                    + penalty_lin_vel_z * 2.0  \
                    + penalty_ang_vel_xy * 0.05 \
                    + penalty_joint_torques * 0.0002 \
                    + penalty_joint_accs * 2.5e-7 \
                    + reward_feet_air_time * 1.0 \
                    + penalty_collisions * 1.0 \
                    + penalty_action_rate * 0.01 \
                    + penalty_dof_pos_limits * 0.0
        
        self._last_joint_vels = joint_vels
        self._last_action = action

        reward = torch.clamp(reward, min=0.)

        #reward -= 1.0 * absorbing
        
        return reward