from isaacgym.torch_utils import quat_apply, quat_rotate_inverse
from mushroom_rl.environments.isaacsim_envs.isaac_a1_legged_gym import IsaacA1Description
import torch

TOTAL_TESTED_STEPS = 100
TOTAL_TESTED_SIM_STEPS = TOTAL_TESTED_STEPS * 4

sim_counter = 0
step_counter = 0

class MockWorld:
    def step(self, render=False):
        global sim_counter, step_counter
        sim_counter += 1
        if sim_counter % 4 == 0:
            step_counter += 1

    def reset(self, soft=False):
        pass

class MockCollisionHelper:
    def __init__(self):
        forces = torch.load("tests/environments/isaac_sim_envs/data/contact_forces_legged_gym.pth")
        feet_idx = torch.tensor([ 4,  8, 12, 16])
        lower_body_idx = torch.tensor([ 2,  6, 10, 14,  3,  7, 11, 15])
        body_idx = torch.tensor([0])
        self.force_buffer = {
            "FL_foot": forces[:, :, [feet_idx[0]]],
            "FR_foot": forces[:, :, [feet_idx[1]]],
            "RL_foot": forces[:, :, [feet_idx[2]]],
            "RR_foot": forces[:, :, [feet_idx[3]]],
            "body": forces[:, :, body_idx],
            "lower_body": forces[:, :, lower_body_idx]
        }

    def get_net_contact_forces(self, group, dt=1.0):
        return self.force_buffer[group][step_counter].to("cuda:0")

class MockTask:
    def __init__(self, num_envs, device):
        self.device = device
        self.num_envs = num_envs
        self.collision_helper = MockCollisionHelper()
        self.counter = 0

        self.read_body_rot = torch.load("tests/environments/isaac_sim_envs/data/body_rot_legged_gym.pth")
        self.read_body_vel = torch.load("tests/environments/isaac_sim_envs/data/body_vel_legged_gym.pth")
        self.read_joint_pos = torch.load("tests/environments/isaac_sim_envs/data/joint_pos_legged_gym.pth")
        self.read_joint_vel = torch.load("tests/environments/isaac_sim_envs/data/joint_vel_legged_gym.pth")

        self.write_body_vel = []
        self.write_joint_pos = []
        self.write_joint_vel = []

        self.applied_actions = []

    def get_observation_limits(self):
        obs_high = torch.tensor([torch.inf, torch.inf, torch.inf, torch.inf, torch.inf, torch.inf,  0.8029,  4.1888,
        -0.9163,  0.8029,  4.1888, -0.9163,  0.8029,  4.1888, -0.9163,  0.8029,
         4.1888, -0.9163, 52.4000, 28.6000, 28.6000, 52.4000, 28.6000, 28.6000,
        52.4000, 28.6000, 28.6000, 52.4000, 28.6000, 28.6000], device=self.device)
        obs_low = torch.tensor([-torch.inf, -torch.inf, -torch.inf, -torch.inf, -torch.inf, -torch.inf,  -0.8029,
         -1.0472,  -2.6965,  -0.8029,  -1.0472,  -2.6965,  -0.8029,  -1.0472,
         -2.6965,  -0.8029,  -1.0472,  -2.6965, -52.4000, -28.6000, -28.6000,
        -52.4000, -28.6000, -28.6000, -52.4000, -28.6000, -28.6000, -52.4000,
        -28.6000, -28.6000], device=self.device)
        return obs_low, obs_high
    
    def get_action_limits(self):
        return self.get_joint_max_efforts()
    
    def get_joint_pos_limits(self):
        return torch.tensor([[-0.8029, -1.0472, -2.6965, -0.8029, -1.0472, -2.6965, -0.8029, -1.0472,
         -2.6965, -0.8029, -1.0472, -2.6965],
        [ 0.8029,  4.1888, -0.9163,  0.8029,  4.1888, -0.9163,  0.8029,  4.1888,
         -0.9163,  0.8029,  4.1888, -0.9163]], device=self.device)
    
    def get_joint_max_efforts(self):
        return torch.tensor([20., 55., 55., 20., 55., 55., 20., 55., 55., 20., 55., 55.], device=self.device)

    def write_data(self, name, value, env_indices=None):
        if env_indices is None:
            env_indices = torch.arange(0, self.num_envs, 1, dtype=int, device=self.device)
        if name == "body_vel":
            self.write_body_vel.append((value, env_indices))
        elif name == "joint_pos":
            self.write_joint_pos.append((value, env_indices))
        elif name == "joint_vel":
            self.write_joint_vel.append((value, env_indices))
        else:
            raise NotImplementedError() 
        
    def read_data(self, name, env_indices=None):
        if env_indices is None:
            env_indices = torch.arange(0, self.num_envs, 1, dtype=int, device=self.device)
        if name == "body_rot":
            return self.read_body_rot[sim_counter, env_indices].to("cuda:0")
        elif name == "body_vel":
            return self.read_body_vel[sim_counter, env_indices].to("cuda:0")
        elif name == "base_lin_vel":
            return self.read_body_vel[sim_counter, env_indices, :3].to("cuda:0")
        elif name == "base_ang_vel":
            return self.read_body_vel[sim_counter, env_indices, 3:7].to("cuda:0")
        elif name == "joint_pos":
            return self.read_joint_pos[sim_counter, env_indices].to("cuda:0")
        elif name == "joint_vel":
            return self.read_joint_vel[sim_counter, env_indices].to("cuda:0")
        raise NotImplementedError()
    
    def apply_action(self, action, env_indices=None):
        self.applied_actions.append((action, env_indices))

    def get_observations(self, clone=True):
        obs = {}
        for name in ["base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel"]:
            obs[name] = self.read_data(name)
        return obs
    
    def reset_env(self, env_indices):
        pass

    def teleport_away(self, env_indices):
        pass

import inspect

legged_gym_rand_dof = torch.load("tests/environments/isaac_sim_envs/data/rand_dof_legged_gym.pth")
legged_gym_rand_push = torch.load("tests/environments/isaac_sim_envs/data/rand_push_legged_gym.pth")
legged_gym_rand_root = torch.load("tests/environments/isaac_sim_envs/data/rand_root_legged_gym.pth")
legged_gym_rand_command = torch.load("tests/environments/isaac_sim_envs/data/rand_command_legged_gym.pth")

counter_rand_dof = 0
counter_rand_push = 0
counter_rand_root = 0
counter_rand_command = 0

def torch_rand_float(lower, upper, shape, device):
    global counter_rand_dof, counter_rand_push, counter_rand_root, counter_rand_command
    stack = inspect.stack()
    caller_function = stack[1].function

    if caller_function == "setup":
        if upper == 1.5:#dof
            counter_rand_dof += 1
            return legged_gym_rand_dof[counter_rand_dof - 1]
        else:#root
            counter_rand_root += 1
            return legged_gym_rand_root[counter_rand_root - 1]
    elif caller_function == "_push_robots":
        counter_rand_push += 1
        return legged_gym_rand_push[counter_rand_push - 1]
    elif caller_function == "_resample_commands":
        counter_rand_command += 1
        return legged_gym_rand_command[counter_rand_command - 1]
    raise NotImplementedError()


class MockIsaacA1(IsaacA1Description):
    def _create_simulation_app(self, headless):
        pass

    def _apply_carb_settings(self):
        pass

    def _create_world(self, timestep, custom_sim_params=None):
        self._timestep = timestep
        self._world = MockWorld()

    def _set_task(self, usd_path, num_envs, env_spacing, collision_between_envs, observation_spec, actuation_spec, additional_data_spec, collision_groups, physics_material_spec, camera_position, camera_target):
        self._task = MockTask(num_envs, "cuda:0")

    def render_all(self, env_mask, record=False):
        pass

    def _modify_observation(self, obs):
        obs = super()._modify_observation(obs)
        new_obs = obs.clone().detach()
        new_obs[:, 6:9] = self.observation_helper.get_from_obs(obs, "projected_gravity")
        new_obs[:, 9:12] = self.observation_helper.get_from_obs(obs, "commands")
        new_obs[:, 12:24] = self.observation_helper.get_from_obs(obs, "joint_pos")
        new_obs[:, 24:36] = self.observation_helper.get_from_obs(obs, "joint_vel")
        new_obs[:, 36:48] = self.observation_helper.get_from_obs(obs, "actions")

        return new_obs
    
    def _import_helper_functions(self):
        self.torch_rand_float = torch_rand_float
        self.quat_apply = quat_apply
        self.quat_rotate_inverse = quat_rotate_inverse

NUM_ENV = 4096
def test_run():
    expected_obs = torch.load("tests/environments/isaac_sim_envs/data/obs_legged_gym.pth")
    expected_reward = torch.load("tests/environments/isaac_sim_envs/data/reward_legged_gym.pth")
    expected_absorbing = torch.load("tests/environments/isaac_sim_envs/data/reset_legged_gym.pth")
    used_actions = torch.load("tests/environments/isaac_sim_envs/data/actions_legged_gym.pth")

    expected_torques = torch.load("tests/environments/isaac_sim_envs/data/torques_legged_gym.pth")

    expected_written_body_vel = torch.load("tests/environments/isaac_sim_envs/data/wr_body_vel_legged_gym.pth")
    expected_written_joint_pos = torch.load("tests/environments/isaac_sim_envs/data/joint_pos_legged_gym.pth")
    expected_written_joint_vel = torch.load("tests/environments/isaac_sim_envs/data/joint_vel_legged_gym.pth")


    a1 = MockIsaacA1(NUM_ENV, 1000, True, True)

    env_mask = torch.ones((NUM_ENV, ), dtype=bool, device="cuda:0")
    
    for i in range(TOTAL_TESTED_STEPS):
        obs, info = a1.reset_all(env_mask)
        #TODO test obs correct
        env_mask = torch.zeros((NUM_ENV, ), dtype=bool, device="cuda:0")

        action = used_actions[i].to("cuda:0")

        obs, reward, absorbing, info = a1.step_all(env_mask, action)

        assert torch.allclose(expected_obs[i].to("cuda:0"), obs, atol=1e-4)#TODO check index
        assert torch.allclose(expected_reward[i].to("cuda:0"), reward, atol=1e-4)#TODO check index
        assert torch.allclose(expected_absorbing[i].to("cuda:0"), absorbing, atol=1e-4)#TODO check index

        for j in range(4):
            applied_torque, env_ids = a1._task.applied_actions[j]
            expected_env_ids = torch.arange(0, NUM_ENV, 1, dtype=int)
            assert torch.allclose(expected_torques[i * 4 + j].to("cuda:0"), applied_torque)
            assert torch.equal(expected_env_ids.to("cuda:0"), env_ids)
            
        #TODO test written
        for expected, actual, name in [
            (expected_written_body_vel, a1._task.write_body_vel, "body_vel"), 
            (expected_written_joint_pos, a1._task.write_joint_pos, "joint_pos"), 
            (expected_written_joint_vel, a1._task.write_joint_vel, "joint_vel")
        ]:
            if i in expected:
                assert len(expected[i]) == len(actual)
                for (expected_value, expected_env_ids), (actual_vel, actual_env_ids) in zip(expected[i], actual):
                    assert torch.allclose(expected_value.to("cuda:0"), actual_vel), f"Written {name} values are not the same as expected values"
                    assert torch.equal(expected_env_ids.to("cuda:0"), actual_env_ids), f"Written {name} env_ids are not the same as expected env_ids"


        a1._task.write_body_vel = []
        a1._task.write_joint_pos = []
        a1._task.write_joint_vel = []


        env_mask = absorbing
test_run()

