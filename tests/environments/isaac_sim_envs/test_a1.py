#import isaacgym
from mushroom_rl.environments.isaacsim_envs.a1_walking import A1Walking
import torch

NUM_ENV = 64
TOTAL_TESTED_STEPS = 100
TOTAL_TESTED_SIM_STEPS = TOTAL_TESTED_STEPS * 4

sim_counter = 0
step_counter = 0

#legged_gym data
def load_data(idx):
    global expected_obs, expected_reward, expected_absorbing, used_actions, expected_torques, expected_written_body_vel
    global expected_written_joint_pos, expected_written_joint_vel, expected_detail_reward, forces, read_body_rot
    global read_body_vel, read_joint_pos, read_joint_vel, legged_gym_rand_dof, legged_gym_rand_push
    global legged_gym_rand_root, legged_gym_rand_command, legged_gym_rand_noise

    expected_obs = torch.load(f"tests/environments/isaac_sim_envs/data/obs_legged_gym_{idx}.pth")
    expected_reward = torch.load(f"tests/environments/isaac_sim_envs/data/reward_legged_gym_{idx}.pth")
    expected_absorbing = torch.load(f"tests/environments/isaac_sim_envs/data/reset_legged_gym_{idx}.pth")
    used_actions = torch.load(f"tests/environments/isaac_sim_envs/data/actions_legged_gym_{idx}.pth")

    expected_torques = torch.load(f"tests/environments/isaac_sim_envs/data/torques_legged_gym_{idx}.pth")

    expected_written_body_vel = torch.load(f"tests/environments/isaac_sim_envs/data/wr_body_vel_legged_gym_{idx}.pth")
    expected_written_joint_pos = torch.load(f"tests/environments/isaac_sim_envs/data/wr_joint_pos_legged_gym_{idx}.pth")
    expected_written_joint_vel = torch.load(f"tests/environments/isaac_sim_envs/data/wr_joint_vel_legged_gym_{idx}.pth")

    expected_detail_reward = torch.load(f"tests/environments/isaac_sim_envs/data/detail_reward_legged_gym_{idx}.pth")

    forces = torch.load(f"tests/environments/isaac_sim_envs/data/contact_forces_legged_gym_{idx}.pth")

    read_body_rot = torch.load(f"tests/environments/isaac_sim_envs/data/body_rot_legged_gym_{idx}.pth")
    read_body_vel = torch.load(f"tests/environments/isaac_sim_envs/data/body_vel_legged_gym_{idx}.pth")
    read_joint_pos = torch.load(f"tests/environments/isaac_sim_envs/data/joint_pos_legged_gym_{idx}.pth")
    read_joint_vel = torch.load(f"tests/environments/isaac_sim_envs/data/joint_vel_legged_gym_{idx}.pth")

    legged_gym_rand_dof = torch.load(f"tests/environments/isaac_sim_envs/data/rand_dof_legged_gym_{idx}.pth")
    legged_gym_rand_push = torch.load(f"tests/environments/isaac_sim_envs/data/rand_push_legged_gym_{idx}.pth")
    legged_gym_rand_root = torch.load(f"tests/environments/isaac_sim_envs/data/rand_root_legged_gym_{idx}.pth")
    legged_gym_rand_command = torch.load(f"tests/environments/isaac_sim_envs/data/rand_command_legged_gym_{idx}.pth")
    legged_gym_rand_noise = torch.load(f"tests/environments/isaac_sim_envs/data/rand_noise_legged_gym_{idx}.pth")

def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)

def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

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
        self.calc_forces_buffer()

    def calc_forces_buffer(self):
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
        return -self.get_joint_max_efforts(), self.get_joint_max_efforts()
    
    def get_joint_pos_limits(self):
        return torch.tensor([[-0.80285138, -1.04719758, -2.69653344, -0.80285138, -1.04719758,
         -2.69653344, -0.80285138, -1.04719758, -2.69653344, -0.80285138,
         -1.04719758, -2.69653344],
        [ 0.80285138,  4.18879032, -0.91629779,  0.80285138,  4.18879032,
         -0.91629779,  0.80285138,  4.18879032, -0.91629779,  0.80285138,
          4.18879032, -0.91629779]], device='cuda:0')
    
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
        env_indices = env_indices.to("cpu")
        if name == "body_rot":
            return read_body_rot[sim_counter, env_indices].to("cuda:0")
        elif name == "body_vel":
            return read_body_vel[sim_counter, env_indices].to("cuda:0")
        elif name == "base_lin_vel":
            return read_body_vel[sim_counter, env_indices, :3].to("cuda:0")
        elif name == "base_ang_vel":
            return read_body_vel[sim_counter, env_indices, 3:6].to("cuda:0")
        elif name == "joint_pos":
            return read_joint_pos[sim_counter, env_indices].to("cuda:0")
        elif name == "joint_vel":
            return read_joint_vel[sim_counter, env_indices].to("cuda:0")
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
            return legged_gym_rand_dof[counter_rand_dof - 1].to("cuda:0")
        else:#root
            counter_rand_root += 1
            return legged_gym_rand_root[counter_rand_root - 1].to("cuda:0")
    elif caller_function == "_push_robots":
        counter_rand_push += 1
        return legged_gym_rand_push[counter_rand_push - 1].to("cuda:0")
    elif caller_function == "_resample_commands":
        counter_rand_command += 1
        return legged_gym_rand_command[counter_rand_command - 1].to("cuda:0")
    raise NotImplementedError()

from unittest.mock import patch
counter_rand_noise = 0
inverted_new_obs_order = None
def mock_rand_like(tensor):
    global counter_rand_noise
    counter_rand_noise += 1
    if counter_rand_noise == 0:
        return 0
    return legged_gym_rand_noise[counter_rand_noise - 1][:, inverted_new_obs_order].to("cuda:0")

class MockIsaacA1(A1Walking):
    def __init__(self, num_envs, horizon, headless, domain_randomization=True):
        super().__init__(num_envs, horizon, headless, domain_randomization)

        self.new_obs_order = torch.cat((
            self.observation_helper.obs_idx_map["base_lin_vel"], 
            self.observation_helper.obs_idx_map["base_ang_vel"], 
            self.observation_helper.obs_idx_map["projected_gravity"], 
            self.observation_helper.obs_idx_map["commands"], 
            self.observation_helper.obs_idx_map["joint_pos"], 
            self.observation_helper.obs_idx_map["joint_vel"], 
            self.observation_helper.obs_idx_map["actions"]),
            dim=0
        )
        global inverted_new_obs_order
        inverted_new_obs_order = torch.argsort(self.new_obs_order).to("cpu")

    
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
        with patch('torch.rand_like', mock_rand_like):
            obs = super()._modify_observation(obs)
        obs = obs[:, self.new_obs_order]
        return obs
    
    def _import_helper_functions(self):
        self.torch_rand_float = torch_rand_float
        self.quat_apply = quat_apply
        self.quat_rotate_inverse = quat_rotate_inverse

    def stop(self):
        pass

    def __del__(self):
        pass

    def setup(self, env_indices, obs):
        super().setup(env_indices, obs)
        global counter_rand_noise
        counter_rand_noise -= 1 #in legged gym reset doesn't have extra observation calculation

def test_run():
    load_data(0)
    torch.set_printoptions(precision=8)
    a1 = MockIsaacA1(NUM_ENV, 1000, True, True)

    env_mask = torch.ones((NUM_ENV, ), dtype=bool, device="cuda:0")

    obs, info = a1.reset_all(env_mask)

    current_episode_length = torch.zeros(NUM_ENV, dtype=int, device="cuda:0")
    
    for h in range(1):
        if h != 0:
            print("finished dataset, go to next 2000 steps")
            load_data(h)
            a1._task.collision_helper.calc_forces_buffer()
            i = 0
            global sim_counter, step_counter, counter_rand_dof, counter_rand_push, counter_rand_root, counter_rand_command, counter_rand_noise
            sim_counter = 0
            step_counter = 0
            counter_rand_dof = 0
            counter_rand_push = 0
            counter_rand_root = 0
            counter_rand_command = 0
            counter_rand_noise = 0
        for i in range(TOTAL_TESTED_STEPS):
            #TODO test obs correct
            env_mask = torch.ones((NUM_ENV, ), dtype=bool, device="cuda:0")

            action = used_actions[i].to("cuda:0")

            obs, reward, absorbing, info = a1.step_all(env_mask, action)

            current_episode_length += 1

            absorbing = torch.logical_or(absorbing, current_episode_length > 1001)

            absorbing = absorbing.to("cuda:0")
            if torch.any(absorbing):
                obs_2, info_2 = a1.reset_all(absorbing)
                obs[absorbing] = obs_2[absorbing]
                current_episode_length[absorbing] = 0

            assert torch.allclose(expected_obs[i].to("cuda:0"), obs, atol=5e-07)
            assert torch.equal(expected_absorbing[i].to("cuda:0"), absorbing)

            assert len(expected_detail_reward[i]) == len(info)

            for key in expected_detail_reward[i]:#TODO ignore terminated rewards
                assert torch.allclose(expected_detail_reward[i][key].to("cuda:0"), info[key], atol=1e-07)
            assert torch.allclose(expected_reward[i].to("cuda:0"), reward, atol=1e-07)

            for j in range(4):
                applied_torque, env_ids = a1._task.applied_actions[j]
                expected_env_ids = torch.arange(0, NUM_ENV, 1, dtype=int)
                assert torch.allclose(expected_torques[i * 4 + j].to("cuda:0"), applied_torque, atol=1e-08)
                assert torch.equal(expected_env_ids.to("cuda:0"), env_ids)
                
            #TODO test written
            for expected, actual, name in [
                (expected_written_body_vel, a1._task.write_body_vel, "body_vel"), 
                (expected_written_joint_pos, a1._task.write_joint_pos, "joint_pos"), 
                (expected_written_joint_vel, a1._task.write_joint_vel, "joint_vel")
            ]:
                if i + 1 in expected:
                    assert len(expected[i + 1]) == len(actual)
                    for (expected_value, expected_env_ids), (actual_value, actual_env_ids) in zip(expected[i + 1], actual):
                        assert torch.equal(expected_env_ids.to("cuda:0"), actual_env_ids), f"Written {name} env_ids are not the same as expected env_ids"
                        assert torch.allclose(expected_value.to("cuda:0"), actual_value, atol=1e-08), f"Written {name} values are not the same as expected values"
                    

            a1._task.write_body_vel = []
            a1._task.write_joint_pos = []
            a1._task.write_joint_vel = []
            a1._task.applied_actions = []


            env_mask = absorbing

test_run()