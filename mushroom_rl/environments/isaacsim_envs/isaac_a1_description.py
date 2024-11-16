from mushroom_rl.environments import IsaacSim
from mushroom_rl.environments.isaac_sim_env import ObservationType #TODO
import numpy as np
import torch

class IsaacA1Description(IsaacSim):
    def __init__(self, num_envs):
        backend="torch"
        device="cuda:0"

        usd_path = "/home/bjarne/GitWorkspace/BachelorThesis/mushroom-rl/isaac_assets/a1/a1.usd"

        action_spec = ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", 
                       "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",   
                       "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", 
                       "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"]
        observation_spec = [
            ("body_pos", "/base", ObservationType.BODY_POS),
            ("body_rot", "/base", ObservationType.BODY_ROT),
            ("trunk_FL_hip_joint_pos", "/trunk/FL_hip_joint", ObservationType.JOINT_POS),
            ("trunk_FL_hip_joint_vel", "/trunk/FL_hip_joint", ObservationType.JOINT_VEL),
            ("trunk_FR_hip_joint_pos", "/trunk/FR_hip_joint", ObservationType.JOINT_POS),
            ("trunk_FR_hip_joint_vel", "/trunk/FR_hip_joint", ObservationType.JOINT_VEL),
            ("trunk_RL_hip_joint_pos", "/trunk/RL_hip_joint", ObservationType.JOINT_POS),
            ("trunk_RL_hip_joint_vel", "/trunk/RL_hip_joint", ObservationType.JOINT_VEL),
            ("trunk_RR_hip_joint_pos", "/trunk/RR_hip_joint", ObservationType.JOINT_POS),
            ("trunk_RR_hip_joint_vel", "/trunk/RR_hip_joint", ObservationType.JOINT_VEL),
            ("FL_hip_FL_thigh_joint_pos", "/FL_hip/FL_thigh_joint", ObservationType.JOINT_POS),
            ("FL_hip_FL_thigh_joint_vel", "/FL_hip/FL_thigh_joint", ObservationType.JOINT_VEL),
            ("FL_thigh_FL_calf_joint_pos", "/FL_thigh/FL_calf_joint", ObservationType.JOINT_POS),
            ("FL_thigh_FL_calf_joint_vel", "/FL_thigh/FL_calf_joint", ObservationType.JOINT_VEL),
            ("FR_hip_FR_thigh_joint_pos", "/FR_hip/FR_thigh_joint", ObservationType.JOINT_POS),
            ("FR_hip_FR_thigh_joint_vel", "/FR_hip/FR_thigh_joint", ObservationType.JOINT_VEL),
            ("FR_thigh_FR_calf_joint_pos", "/FR_thigh/FR_calf_joint", ObservationType.JOINT_POS),
            ("FR_thigh_FR_calf_joint_vel", "/FR_thigh/FR_calf_joint", ObservationType.JOINT_VEL),
            ("RL_hip_RL_thigh_joint_pos", "/RL_hip/RL_thigh_joint", ObservationType.JOINT_POS),
            ("RL_hip_RL_thigh_joint_vel", "/RL_hip/RL_thigh_joint", ObservationType.JOINT_VEL),
            ("RL_thigh_RL_calf_joint_pos", "/RL_thigh/RL_calf_joint", ObservationType.JOINT_POS),
            ("RL_thigh_RL_calf_joint_vel", "/RL_thigh/RL_calf_joint", ObservationType.JOINT_VEL),
            ("RR_hip_RR_thigh_joint_pos", "/RR_hip/RR_thigh_joint", ObservationType.JOINT_POS),
            ("RR_hip_RR_thigh_joint_vel", "/RR_hip/RR_thigh_joint", ObservationType.JOINT_VEL),
            ("RR_thigh_RR_calf_joint_pos", "/RR_thigh/RR_calf_joint", ObservationType.JOINT_POS),
            ("RR_thigh_RR_calf_joint_vel", "/RR_thigh/RR_calf_joint", ObservationType.JOINT_VEL),
        ]
        additional_data_spec = []
        collision_between_envs = False
        env_spacing = 2.5
        super().__init__(usd_path, action_spec, observation_spec, backend, device, collision_between_envs, num_envs, 
                         env_spacing, 0.99, 200, additional_data_spec=additional_data_spec)
        
    def reward(self, obs, action, next_obs, absorbing):
        x_pos = obs[:, 0]
        y_pos = obs[:, 1]
        reward = torch.sqrt(torch.square(x_pos) + torch.square(y_pos))
        reward = torch.where(absorbing, torch.ones_like(reward) * (-1000.0), reward)
        return reward

    def is_absorbing(self, obs):
        rot = obs[:, 3:7]
        roll, pitch, yaw = self._quaternion_to_euler(rot)
        ones = torch.ones_like(obs[:, 0], dtype=bool)
        zeros = torch.zeros_like(obs[:, 0], dtype=bool)
        fallen = torch.where((torch.abs(roll) > 30.0) | (torch.abs(pitch) > 30.0) | (torch.abs(yaw) > 30.0), ones, zeros)
        return fallen
    
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
        pass

    def _preprocess_action(self, action):
        return action * 10
    
    def _create_info_dictionary(self, obs):
        return {}