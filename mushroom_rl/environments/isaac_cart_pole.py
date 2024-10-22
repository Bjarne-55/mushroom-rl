from mushroom_rl.environments import IsaacSim
from mushroom_rl.environments.isaac_sim_env import ObservationType #TODO
import numpy as np
import torch

class IsaacCartPole(IsaacSim):
    def __init__(self, num_envs, backend="torch", device="cuda:0"):
        usd_path = "/home/bjarne/GitWorkspace/BachelorThesis/mushroom-rl/isaac_assets/cartpole.usd"
        action_spec = ["cartJoint"]
        observation_spec = [
            ("cart", "/cart", ObservationType.BODY_POS),
            ("poleJointVel", "/poleJoint", ObservationType.JOINT_POS),
            ("cartJointPos", "/cartJoint", ObservationType.JOINT_POS)
        ]
        collision_between_envs = False
        env_spacing = 7
        super().__init__(usd_path, action_spec, observation_spec, backend, device, collision_between_envs, num_envs, 
                         env_spacing, 0.99, 200)
        
    def reward(self, cur_obs, action, obs, absorbing):
        pole_joint_pos = cur_obs[:, 3]
        reward = torch.where(torch.abs(pole_joint_pos) > np.pi / 2, -torch.ones_like(pole_joint_pos), torch.zeros_like(pole_joint_pos))
        return reward

    def is_absorbing(self, obs):
        pole_joint_pos = obs[:, 3]
        return torch.where(torch.abs(pole_joint_pos) > np.pi / 2, torch.ones_like(pole_joint_pos, dtype=bool), torch.zeros_like(pole_joint_pos, dtype=bool))


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