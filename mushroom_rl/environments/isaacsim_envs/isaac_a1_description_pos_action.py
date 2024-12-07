from mushroom_rl.environments import IsaacSim
from mushroom_rl.environments.isaac_sim_env import ObservationType, ActionType #TODO
import numpy as np
import torch

class IsaacA1Description(IsaacSim):
    def __init__(self, num_envs, horizon, headless):
        backend="torch"
        device="cuda:0"

        self.goal_linear_velocity_xy = torch.tensor([1.0, 0], device=device)
        self.goal_angular_velocity_z = 0.0

        usd_path = "/home/bjarne/GitWorkspace/BachelorThesis/mushroom-rl/isaac_assets/a1_position/a1/a1.usd"

        self._action_spec = ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", 
                       "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",   
                       "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", 
                       "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"]
        #self._action_spec = ["FL_thigh_joint", "FR_thigh_joint"]
        observation_spec = [
            ("body_pos", "/base", ObservationType.BODY_POS),
            ("body_rot", "/base", ObservationType.BODY_ROT),
            ("base_lin_vel", "/base", ObservationType.BODY_LIN_VEL),
            ("base_ang_vel", "/base", ObservationType.BODY_ANG_VEL),
            ("FL_hip_joint_pos", "/trunk/FL_hip_joint", ObservationType.JOINT_POS),
            ("FL_hip_joint_vel", "/trunk/FL_hip_joint", ObservationType.JOINT_VEL),
            ("FR_hip_joint_pos", "/trunk/FR_hip_joint", ObservationType.JOINT_POS),
            ("FR_hip_joint_vel", "/trunk/FR_hip_joint", ObservationType.JOINT_VEL),
            ("RL_hip_joint_pos", "/trunk/RL_hip_joint", ObservationType.JOINT_POS),
            ("RL_hip_joint_vel", "/trunk/RL_hip_joint", ObservationType.JOINT_VEL),
            ("RR_hip_joint_pos", "/trunk/RR_hip_joint", ObservationType.JOINT_POS),
            ("RR_hip_joint_vel", "/trunk/RR_hip_joint", ObservationType.JOINT_VEL),
            ("FL_thigh_joint_pos", "/FL_hip/FL_thigh_joint", ObservationType.JOINT_POS),
            ("FL_thigh_joint_vel", "/FL_hip/FL_thigh_joint", ObservationType.JOINT_VEL),
            ("FL_calf_joint_pos", "/FL_thigh/FL_calf_joint", ObservationType.JOINT_POS),
            ("FL_calf_joint_vel", "/FL_thigh/FL_calf_joint", ObservationType.JOINT_VEL),
            ("FR_thigh_joint_pos", "/FR_hip/FR_thigh_joint", ObservationType.JOINT_POS),
            ("FR_thigh_joint_vel", "/FR_hip/FR_thigh_joint", ObservationType.JOINT_VEL),
            ("FR_calf_joint_pos", "/FR_thigh/FR_calf_joint", ObservationType.JOINT_POS),
            ("FR_calf_joint_vel", "/FR_thigh/FR_calf_joint", ObservationType.JOINT_VEL),
            ("RL_thigh_joint_pos", "/RL_hip/RL_thigh_joint", ObservationType.JOINT_POS),
            ("RL_thigh_joint_vel", "/RL_hip/RL_thigh_joint", ObservationType.JOINT_VEL),
            ("RL_calf_joint_pos", "/RL_thigh/RL_calf_joint", ObservationType.JOINT_POS),
            ("RL_calf_joint_vel", "/RL_thigh/RL_calf_joint", ObservationType.JOINT_VEL),
            ("RR_thigh_joint_pos", "/RR_hip/RR_thigh_joint", ObservationType.JOINT_POS),
            ("RR_thigh_joint_vel", "/RR_hip/RR_thigh_joint", ObservationType.JOINT_VEL),
            ("RR_calf_joint_pos", "/RR_thigh/RR_calf_joint", ObservationType.JOINT_POS),
            ("RR_calf_joint_vel", "/RR_thigh/RR_calf_joint", ObservationType.JOINT_VEL),
        ]
        additional_data_spec = []
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
        env_spacing = 1.
        super().__init__(usd_path, self._action_spec, observation_spec, backend, device, collision_between_envs, num_envs, 
                         env_spacing, 0.99, horizon, additional_data_spec=additional_data_spec, collision_groups=collision_groups, 
                         action_type=ActionType.POSITION, headless=headless)
        
        self._last_joint_vels = None
        self._last_action = None
        self.foot_air_time = torch.zeros((num_envs, 4), device=device)

    def is_absorbing(self, obs):
        #return torch.zeros(64, device=self._device)
        rot = obs[:, 3:7]
        roll, pitch, yaw = self._quaternion_to_euler(rot)
        ones = torch.ones_like(obs[:, 0], dtype=bool)
        zeros = torch.zeros_like(obs[:, 0], dtype=bool)
        lost_balance = torch.where((torch.abs(roll) > 30.0) | (torch.abs(pitch) > 30.0) | (torch.abs(yaw) > 30.0), ones, zeros)
        fallen = self._check_collision("body", "groundplane")
        return fallen
        return torch.logical_or(fallen, lost_balance)
    
    def _quaternion_to_euler(self, quaternion):
        # Quaternion should be in (w, x, y, z) format
        w = quaternion[:, 0]
        x = quaternion[:, 1]
        y = quaternion[:, 2]
        z = quaternion[:, 3]
        # Compute roll, pitch, and yaw
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = torch.arctan2(t0, t1) * 180 / np.pi

        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clip(t2, -1.0, +1.0)
        pitch = torch.arcsin(t2) * 180 / np.pi

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = torch.arctan2(t3, t4) * 180 / np.pi

        return roll, pitch, yaw

    def setup(self, env_indices, obs):
        joints_defaults = self._task.robots.get_joints_default_state()
        dof_pos = joints_defaults.positions[env_indices]
        dof_pos += (torch.rand(dof_pos.shape, device=self._device)) - 0.5
        self._task.robots.set_joint_positions(dof_pos, indices=env_indices)

    def _preprocess_action(self, action):
        #return torch.zeros((self.number, 1), device=self._device)
        #import random
        #random_value = random.choice([-1, 1])
        #return torch.full((self.number, 2), 30.0 * random_value, device=self._device)
        return action
    
    def _create_info_dictionary(self, obs):
        return {}
    
    def _get_joint_velocities(self, obs):
        index = self._obs_idx_map["FL_hip_joint_vel"][0]
        return obs[:, index::2]
    
    def _update_air_time(self):
        self.foot_air_time = self.foot_air_time + self.dt
        for i, foot in enumerate(["FL_foot", "FR_foot", "RL_foot", "RR_foot"]):
            collision_mask = self._check_collision(foot, "groundplane")
            self.foot_air_time[collision_mask, i] = 0.
    
    def _count_collisions(self):
        count = torch.zeros(self.number, device=self._device)
        
        for body in ["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh", "FL_calf", "FR_calf", "RL_calf", "RR_calf"]:
            collision_mask = self._check_collision(body, "groundplane")
            count += collision_mask
        
        return count
    
    #Taken from https://proceedings.mlr.press/v164/rudin22a.html
    #Ripped from https://github.com/leggedrobotics/legged_gym/blob/17847702f90d8227cd31cce9c920aa53a739a09a/legged_gym/envs/base/legged_robot.py#L815C3-L816C12
    def reward(self, obs, action, next_obs, absorbing):
        base_lin_vel = self._get_from_obs(next_obs, "base_lin_vel")
        base_lin_vel_xy = base_lin_vel[:, 0:2]
        base_lin_vel_z = base_lin_vel[:, 2]
        base_ang_vel = self._get_from_obs(next_obs, "base_ang_vel")
        base_ang_vel_xy = base_ang_vel[:, 0:2]
        base_ang_vel_z = base_ang_vel[:, 2]

        base_pos = self._get_from_obs(next_obs, "body_pos")
        base_height = base_pos[:, 2]

        joint_vels = self._get_joint_velocities(next_obs)

        torques = action #I think this is correct, paper uses positions for actions we use effort

        if self._last_joint_vels is None:
            self._last_joint_vels = joint_vels
            self._last_action = action

        x = self.goal_linear_velocity_xy - base_lin_vel_xy
        reward_lin_vel_tracking = torch.exp(-(torch.sum(torch.square(x), dim=1) / 0.25))

        x = self.goal_angular_velocity_z - base_ang_vel_z
        reward_ang_vel_tracking = torch.exp(-(torch.square(x) / 0.25))

        penalty_lin_vel_z = - torch.square(base_lin_vel_z)
        penalty_ang_vel_xy = - torch.sum(torch.square(base_ang_vel_xy), dim=1)

        penalty_joint_accs = torch.sum(torch.square((self._last_joint_vels - joint_vels) / self.dt), dim=1)
        penalty_joint_vels = torch.sum(torch.square(joint_vels), dim=1)
        penalty_joint_motion = - penalty_joint_accs - penalty_joint_vels

        penalty_joint_torques = - torch.sum(torch.square(torques), dim=1)

        penalty_action_rate = - torch.sum(torch.square(self._last_action - action), dim=1)
        penalty_collisions = - self._get_collision_count("lower_body", "groundplane")

        penalty_base_height = - torch.square(base_height - 0.4)

        self._update_air_time()
        reward_air_time = torch.sum(self.foot_air_time - 0.5, dim=1)

        reward = reward_lin_vel_tracking * self.dt \
                    + reward_ang_vel_tracking * 0.5 * self.dt  \
                    + penalty_lin_vel_z * 4 * self.dt \
                    + penalty_ang_vel_xy * 0.05 * self.dt \
                    + penalty_joint_motion * 0.001 * self.dt \
                    + penalty_joint_torques * 0.00002 * self.dt \
                    + penalty_collisions * 0.001 * self.dt \
                    + reward_air_time * 2.0 * self.dt \
                    + penalty_action_rate * 0.25 * self.dt \
                    + 250.0
        #
        #
        #
        
        self._last_joint_vels = joint_vels
        self._last_action = action
        
        return reward