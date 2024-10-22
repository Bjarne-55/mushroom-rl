from mushroom_rl.environments import IsaacSim
from mushroom_rl.environments.isaac_sim_env import ObservationType #TODO
import numpy as np
import torch

class IsaacCartPole(IsaacSim):
    def __init__(self, num_envs, backend="torch", device="cuda:0"):
        usd_path = "/home/bjarne/GitWorkspace/BachelorThesis/mushroom-rl/isaac_assets/cartpole.usd"
        action_spec = ["cartJoint"]
        observation_spec = [
            ("poleJointPos", "/poleJoint", ObservationType.JOINT_POS),
            ("poleJointVel", "/poleJoint", ObservationType.JOINT_VEL),
            ("cartJointPos", "/cartJoint", ObservationType.JOINT_POS),
            ("cartJointVel", "/cartJoint", ObservationType.JOINT_VEL)
        ]
        additional_data_spec = [
            ("poleJointPos", "/poleJoint", ObservationType.JOINT_POS),
            ("poleJointVel", "/poleJoint", ObservationType.JOINT_VEL),
            ("cartJointPos", "/cartJoint", ObservationType.JOINT_POS),
            ("cartJointVel", "/cartJoint", ObservationType.JOINT_VEL),
            ("cartPos", "/cart", ObservationType.BODY_POS),
            ("polePos", "/pole", ObservationType.BODY_POS),
            ("poleAngVel", "/pole", ObservationType.BODY_ANG_VEL)
        ]
        collision_between_envs = False
        env_spacing = 7
        super().__init__(usd_path, action_spec, observation_spec, backend, device, collision_between_envs, num_envs, 
                         env_spacing, 0.99, 200, additional_data_spec=additional_data_spec)
        
    def reward(self, obs, action, next_obs, absorbing):
        pole_joint_pos = obs[:, 0]
        reward = torch.where(torch.abs(pole_joint_pos) > np.pi / 2, -torch.ones_like(pole_joint_pos), torch.zeros_like(pole_joint_pos))
        return reward

    def is_absorbing(self, obs):
        pole_joint_pos = obs[:, 0]
        return torch.where(torch.abs(pole_joint_pos) > np.pi / 2, torch.ones_like(pole_joint_pos, dtype=bool), torch.zeros_like(pole_joint_pos, dtype=bool))

    def setup(self, env_indices, obs):
        num_environments = len(env_indices)

        cart_dof_pos = 1.0 * (1.0 - 2.0 * torch.rand(num_environments, 1, device=self._device))
        pole_dof_pos = 0.125 * np.pi * (1.0 - 2.0 * torch.rand(num_environments, 1, device=self._device))

        cart_dof_vel = 0.5 * (1.0 - 2.0 * torch.rand(num_environments, 1, device=self._device))
        pole_dof_vel = 0.25 * np.pi * (1.0 - 2.0 * torch.rand(num_environments, 1, device=self._device))

        self._write_data("cartJointPos", cart_dof_pos, env_indices)
        self._write_data("poleJointPos", pole_dof_pos, env_indices)
        self._write_data("cartJointVel", cart_dof_vel, env_indices)
        self._write_data("poleJointVel", pole_dof_vel, env_indices)

    def _preprocess_action(self, action):
        return action * 400
    
    def _create_info_dictionary(self, obs):
        info = {}
        info["cartPosition"] = self._read_data("cartPos")
        info["polePosition"] = self._read_data("polePos")
        info["poleAngularVelocity"] = self._read_data("poleAngVel")
        return info

if __name__ == "__main__":
    print("hello")
    num_envs = 6
    env = IsaacCartPole(num_envs)
    while True:
        r = torch.tensor([1, 1, 1, 0, 0, 0]).to("cuda:0")
        env.reset_all(r)
        for _ in range(180):
            action = torch.rand(num_envs, 1) * 10 - 5
            r = torch.ones((num_envs, )).to("cuda:0")
            obs = env.step_all(r, action)
            print(obs)