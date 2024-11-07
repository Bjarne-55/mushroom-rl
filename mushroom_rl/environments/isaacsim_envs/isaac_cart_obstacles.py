from mushroom_rl.environments import IsaacSim
from mushroom_rl.environments.isaac_sim_env import ObservationType #TODO
import numpy as np
import torch

class IsaacCartObstacles(IsaacSim):
    def __init__(self, num_envs):
        backend="torch"
        device="cuda:0"

        usd_path = "/home/bjarne/GitWorkspace/BachelorThesis/mushroom-rl/isaac_assets/cart_with_obstacles_2.usd"
        action_spec = ["rail_cart_joint"]
        observation_spec = [
            ("cartJointPos", "/rail_cart_joint", ObservationType.JOINT_POS),
            ("cartJointVel", "/rail_cart_joint", ObservationType.JOINT_VEL)
        ]
        additional_data_spec = [
            ("cartJointPos", "/rail_cart_joint", ObservationType.JOINT_POS),
            ("cartJointVel", "/rail_cart_joint", ObservationType.JOINT_VEL),
            ("cartPos", "/cart", ObservationType.BODY_POS)
        ]
        collision_groups = [("cart", ["/cart"]), ("obstacles", ["/obstacle1/o1", "/obstacle1/o2", "/obstacle2/o1"])]
        collision_between_envs = False
        env_spacing = 2.5
        super().__init__(usd_path, action_spec, observation_spec, backend, device, collision_between_envs, num_envs, 
                         env_spacing, 0.99, 200, additional_data_spec=additional_data_spec, collision_groups=collision_groups)
        
    def setup(self, env_indices, obs):
        num_environments = len(env_indices)

        cart_dof_vel = 10 * (2.0 * torch.rand(num_environments, 1, device=self._device) - 1)

        self._write_data("cartJointVel", cart_dof_vel, env_indices)

    def is_absorbing(self, obs):
        return torch.zeros(2).to("cuda:0")

    def reward(self, obs, action, next_obs, absorbing):
        return torch.zeros(2).to("cuda:0")

if __name__ == "__main__":
    num_envs = 2
    env = IsaacCartObstacles(num_envs)
    while True:
        r = torch.tensor([1, 1]).to("cuda:0")
        env.reset_all(r)
        for _ in range(180):
            action = torch.rand(num_envs, 1) * 20 - 10
            action = action.to("cuda:0")
            r = torch.ones((num_envs, )).to("cuda:0")
            obs = env.step_all(r, action)
            env.render()
            #print(obs)
        