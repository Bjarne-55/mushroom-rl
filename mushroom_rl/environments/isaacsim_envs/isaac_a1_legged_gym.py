from mushroom_rl.environments import IsaacSim
from mushroom_rl.environments.isaac_sim_env import ActionType #TODO
from mushroom_rl.utils.isaac_sim import ObservationType
import numpy as np
import torch
import random
from mushroom_rl.rl_utils.spaces import Box
from pathlib import Path

class IsaacA1Description(IsaacSim):
    def __init__(self, num_envs, horizon, headless, domain_randomization=True, camera_position=(105, 0, 4), camera_target=(95, 0, 0)):
        usd_path = str(Path(__file__).resolve().parent / "robots_usds/a1/a1.usd")
        self.NUM_DOFS = 12

        backend="torch"
        device="cuda:0"

        self.domain_randomization = domain_randomization

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
        observation_spec = [
            ("base_lin_vel", "", ObservationType.BODY_LIN_VEL, None),
            ("base_ang_vel", "", ObservationType.BODY_ANG_VEL, None),

            ("joint_pos", "", ObservationType.JOINT_POS, self._action_spec),
            ("joint_vel", "", ObservationType.JOINT_VEL, self._action_spec)
        ]
        additional_data_spec = [
            ("body_rot", "", ObservationType.BODY_ROT, None), 
            ("body_vel", "", ObservationType.BODY_VEL, None)
        ]
        """
        collision_groups = [
            ("foots", ["/FL_foot", "/FR_foot", "/RL_foot", "/RR_foot"]), 
            ("body", ["/trunk"]),
            ("lower_body", ["/FL_thigh", "/FR_thigh", "/RL_thigh", "/RR_thigh", "/FL_calf", "/FR_calf", "/RL_calf", "/RR_calf"])
        ]
        """
        collision_groups = [
            ("body", ["/trunk", "/FL_foot", "/FR_foot", "/RL_foot", "/RR_foot", "/FL_thigh", "/FR_thigh", "/RL_thigh", "/RR_thigh", "/FL_calf", "/FR_calf", "/RL_calf", "/RR_calf"]),
        ]
        collision_between_envs = False
        env_spacing = 3.
        physics_material_spec = self._get_values_for_physics_materials(num_envs) if domain_randomization else None
        sim_params = {
            "gpu_found_lost_aggregate_pairs_capacity": 128*1024, 
            "gpu_total_aggregate_pairs_capacity": 128*1024, 
            "gpu_temp_buffer_capacity": 16777216,
            "gpu_max_rigid_patch_count": 2 * 81920,
        }

        solver_pos = torch.full((num_envs, ), 4)
        solver_vel = torch.full((num_envs, ), 0)
        super().__init__(usd_path, self._action_spec, observation_spec, backend, device, collision_between_envs, num_envs, 
                         env_spacing, 0.99, horizon, additional_data_spec=additional_data_spec, collision_groups=collision_groups, 
                         action_type=ActionType.EFFORT, headless=headless, n_intermediate_steps=4, timestep=0.005, 
                         physics_material_spec=physics_material_spec, sim_params=sim_params, camera_position=camera_position, 
                         camera_target=camera_target, solver_pos_it_count=solver_pos, solver_vel_it_count=solver_vel) 
        self._import_helper_functions()
        action_limits = (self._task.get_joint_pos_limits() - self._default_joint_angles) / 0.25
        self._mdp_info.action_space = Box(*action_limits, data_type=action_limits[0].dtype)
        
        self.observation_helper.add_obs("projected_gravity", 3, -1, 1)
        commands_upper = torch.tensor([1., 1., np.pi], device=device)
        self.observation_helper.add_obs("commands", 3, -commands_upper, commands_upper)
        self.observation_helper.add_obs("actions", self.NUM_DOFS, self.info.action_space.low, self.info.action_space.high)

        self.normalization_obs_vec = self._get_obs_normilization_vec()
        self.noise_scale_vec = self._get_noise_scale_vec()

        obs_low, obs_high = self.observation_helper.obs_limits
        dof_pos_indices = self.observation_helper.obs_idx_map["joint_pos"]
        obs_low[dof_pos_indices] -= self._default_joint_angles
        obs_high[dof_pos_indices] -= self._default_joint_angles
        new_obs_low = obs_low * self.normalization_obs_vec - self.noise_scale_vec
        new_obs_high = obs_high * self.normalization_obs_vec + self.noise_scale_vec
        self._mdp_info.observation_space = Box(new_obs_low, new_obs_high, data_type=new_obs_high.dtype)

        self.commands = torch.zeros(num_envs, 4, dtype=torch.float, device=device)

        self._soft_dof_pos_limits = self._get_soft_dof_pos_limit()
        
        self._actions = torch.zeros((num_envs, self.NUM_DOFS), device=device)

        self.feet_air_time = torch.zeros((num_envs, 4), device=device)
        self.last_actions =  torch.zeros((num_envs, self.NUM_DOFS), device=device)
        self.last_dof_vel = torch.zeros((num_envs, self.NUM_DOFS), device=device)
        self.last_contacts = torch.zeros((num_envs, 4), device=device, dtype=torch.bool)
        self.episode_length = torch.zeros((num_envs, ), dtype=int, device=device)

        self.forward_vec = torch.tensor([1., 0., 0.], device=device).repeat((num_envs, 1))

        self._effort_limit = self._task.get_joint_max_efforts()

        self.step_counter = 0

        self._gravity = torch.tensor([0., 0., -1.], device=self._device).repeat((self.number, 1))

        self._obs = None
    
    def _import_helper_functions(self):
        from omni.isaac.core.utils.torch.rotations import quat_apply, quat_rotate_inverse
        from omni.isaac.core.utils.torch.maths import torch_rand_float
        self.quat_apply = quat_apply
        self.quat_rotate_inverse = quat_rotate_inverse
        self.torch_rand_float = torch_rand_float

    def _get_values_for_physics_materials(self, num_envs):
        friction_range = [0.5, 1.25]
        num_buckets = 64
        bucket_ids = torch.randint(0, num_buckets, (num_envs, ))
        friction_buckets = (friction_range[1] - friction_range[0]) * torch.rand((num_buckets, ), device='cpu') + friction_range[0]
        
        names = [f"custom_material_{i}" for i in bucket_ids.tolist()]
        dynamic_friction = [0.5] * num_envs
        static_friction = friction_buckets[bucket_ids].tolist()
        restitution = [0.0] * num_envs
        
        return list(zip(names, dynamic_friction, static_friction, restitution))
    
    def _step_finalize(self, env_indices):
        self.episode_length += 1
        self.step_counter += 1

        #resample commands and calculate yaw command
        #env_ids = (self.episode_length % int(10. / self.dt)==0).nonzero(as_tuple=False).flatten()
        do_resample = self.torch_rand_float(0., 1., (len(env_indices), 1), device=self._device).squeeze() < (1./500.)
        do_resample *= self.episode_length[env_indices] > 50
        env_ids = env_indices[do_resample]
        self._resample_commands(env_ids)

        base_quat = self._read_data("body_rot")
        forward = self.quat_apply(base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2] = torch.clip(0.5*self.wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        #domain randomization: push Robot
        do_push = self.torch_rand_float(0., 1., (len(env_indices), 1), device=self._device).squeeze() < (1./750.)
        do_push_ids = env_indices[do_push]
        do_push_ids = do_push_ids[self.episode_length[do_push_ids] > 50]
        if self.domain_randomization: #not self.eval
            self._push_robots(do_push_ids)
        #push_interval = np.ceil(15 / self.dt) + 1
        #if self.domain_randomization and (self.step_counter % push_interval == 0):
        #    self._push_robots(env_indices)
    
    def _push_robots(self, env_indices):
        max_vel= 1.
        vels = self.torch_rand_float(-max_vel, max_vel, (env_indices.shape[0], 2), device=self._device)
        extended_vels = self._read_data("body_vel", env_indices)
        extended_vels[:, :2] = vels
        self._write_data("body_vel", extended_vels, env_indices)
    
    def _get_obs_normilization_vec(self):
        v = torch.zeros((self.observation_helper.obs_length), device=self._device)

        lin_vel = self.observation_helper.obs_idx_map["base_lin_vel"]
        ang_vel = self.observation_helper.obs_idx_map["base_ang_vel"]
        joint_positions = self.observation_helper.obs_idx_map["joint_pos"]
        joint_velocities = self.observation_helper.obs_idx_map["joint_vel"]
        gravity = self.observation_helper.obs_idx_map["projected_gravity"]
        commands = self.observation_helper.obs_idx_map["commands"]
        actions = self.observation_helper.obs_idx_map["actions"]
        
        v[lin_vel] = 2.0
        v[ang_vel] = 0.25
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
        joint_positions = self.observation_helper.obs_idx_map["joint_pos"]
        joint_velocities = self.observation_helper.obs_idx_map["joint_vel"]
        gravity = self.observation_helper.obs_idx_map["projected_gravity"]
        commands = self.observation_helper.obs_idx_map["commands"]
        actions = self.observation_helper.obs_idx_map["actions"]
        
        v[lin_vel] = 0.1 * 2.0
        v[ang_vel] = 0.2 * 0.25
        v[joint_positions] = 0.01 * 1.00
        v[joint_velocities] = 1.5 * 0.05
        v[gravity] = 0.05
        v[commands[:3]] = 0
        v[actions] = 0

        return v
    
    def _get_soft_dof_pos_limit(self):
        soft_dof_pos_limits = torch.zeros(self.NUM_DOFS, 2, device=self._device, requires_grad=False)
        pos_limit = self._task.get_joint_pos_limits()
        low = pos_limit[0]
        high = pos_limit[1]
        
        middle = (low + high) / 2
        r = high - low
        soft_dof_pos_limits[:, 0] = middle - 0.5 * r * 0.9
        soft_dof_pos_limits[:, 1] = middle + 0.5 * r * 0.9
        return soft_dof_pos_limits

    def is_absorbing(self, obs):
        #fallen = self._check_collision("body", "groundplane", 0.)
        fallen = torch.norm(self._get_net_collision_forces("body", dt=self._timestep)[:, 0, :], dim=-1) > 1.
        
        return fallen
    
    def setup(self, env_indices, obs):
        #new
        self.feet_air_time[env_indices] = 0.
        #self.last_actions[env_indices] = 0.
        #self.last_dof_vel[env_indices] = 0.
        self.episode_length[env_indices] = 0

        dof_pos = self._default_joint_angles * self.torch_rand_float(0.5, 1.5, (len(env_indices), self.NUM_DOFS), device=self._device)
        dof_vel = torch.zeros((len(env_indices), len(self._action_spec)), device=self._device)

        self._write_data("joint_pos", dof_pos, env_indices)
        self._write_data("joint_vel", dof_vel, env_indices)

        body_vel = self.torch_rand_float(-0.5, 0.5, (len(env_indices), 6), device=self._device)
        self._write_data("body_vel", body_vel, env_indices)

        self._setup_dof_pos = dof_pos
        self._setup_dof_vel = dof_vel
        self._setup_env_indices = env_indices

        #update last_dof_vel
        self.last_dof_vel[env_indices] = dof_vel

        self._resample_commands(env_indices)

        zero = torch.zeros(self._n_envs, device=self._device)
        self._extra_info_rewards = self._extra_info_rewards = {
            "r_tracking_lin_vel": zero, "r_tracking_ang_vel": zero, "r_lin_vel_z": zero,
            "r_ang_vel_xy": zero, "r_torques": zero, "r_dof_acc": zero, "r_feet_air_time": zero,
            "r_collision": zero, "r_action_rate": zero, "r_dof_pos_limits": zero
        }

    def _modify_observation(self, obs):
        dof_pos_indices = self.observation_helper.obs_idx_map["joint_pos"]
        obs[:, dof_pos_indices] -= self._default_joint_angles

        command_indices = self.observation_helper.obs_idx_map["commands"]
        obs[:, command_indices] = self.commands[:, :3]

        obs *= self.normalization_obs_vec
        obs += (2 * torch.rand_like(obs) - 1) * self.noise_scale_vec

        obs = torch.clamp(obs, max=100., min=-100.)

        return obs
    
    def _create_observation(self, obs):
        #update observation with values set in setup
        if self._setup_env_indices is not None:
            dof_pos_indices = self.observation_helper.obs_idx_map["joint_pos"]
            obs[self._setup_env_indices.unsqueeze(1), dof_pos_indices] = self._setup_dof_pos

            dof_vel_indices = self.observation_helper.obs_idx_map["joint_vel"]
            obs[self._setup_env_indices.unsqueeze(1), dof_vel_indices] = self._setup_dof_vel

            self._setup_env_indices = None

        #set missing observations
        rot = self._read_data("body_rot")
        gravity_indices = self.observation_helper.obs_idx_map["projected_gravity"]
        obs[:, gravity_indices] = self.quat_rotate_inverse(rot, self._gravity)

        command_indices = self.observation_helper.obs_idx_map["commands"]
        obs[:, command_indices] = self.commands[:, :3]

        action_indices = self.observation_helper.obs_idx_map["actions"]
        obs[:, action_indices] = self._actions

        lin_vel_indices = self.observation_helper.obs_idx_map["base_lin_vel"] 
        lin_vel = self.observation_helper.get_from_obs(obs, "base_lin_vel")
        obs[:, lin_vel_indices] = self.quat_rotate_inverse(rot, lin_vel)

        ang_vel_indices = self.observation_helper.obs_idx_map["base_ang_vel"]
        ang_vel = self.observation_helper.get_from_obs(obs, "base_ang_vel")
        obs[:, ang_vel_indices] = self.quat_rotate_inverse(rot, ang_vel)

        return obs
    
    @staticmethod
    def wrap_to_pi(angles):
        angles %= 2*np.pi
        angles -= 2*np.pi * (angles > np.pi)
        return angles
    
    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = self.torch_rand_float(-1., 1., (len(env_ids), 1), device=self._device).squeeze(1)
        self.commands[env_ids, 1] = self.torch_rand_float(-1., 1., (len(env_ids), 1), device=self._device).squeeze(1)
        self.commands[env_ids, 3] = self.torch_rand_float(-3.14, 3.14, (len(env_ids), 1), device=self._device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _preprocess_action(self, action):
        action = torch.clip(action, min=-100., max=100.)
        self._actions[:] = action[:]
        return action
    
    def _compute_action(self, action):
        #joint_vels = self.observation_helper.get_from_obs(obs, "joint_vel")
        #dof_positions = self.observation_helper.get_from_obs(obs, "joint_pos")
        joint_vels = self._read_data("joint_vel")
        dof_positions = self._read_data("joint_pos")
        torque = self._compute_torque(action, joint_vels, dof_positions)
        return torque
    
    def _create_info_dictionary(self, obs):
        return self._extra_info_rewards
    
    def _compute_torque(self, action, joint_vels, joint_pos):
        actions_scaled = action * 0.25
        self._torques = 20.0 * (actions_scaled + self._default_joint_angles - joint_pos) - 0.5*joint_vels
        self._torques = torch.clip(self._torques, -self._effort_limit, self._effort_limit)
        
        return self._torques
    
    #Taken from https://proceedings.mlr.press/v164/rudin22a.html
    #Taken from https://github.com/leggedrobotics/legged_gym/blob/17847702f90d8227cd31cce9c920aa53a739a09a/legged_gym/envs/base/legged_robot.py#L815C3-L816C12
    def reward(self, obs, action, next_obs, absorbing):
        base_lin_vel = self.observation_helper.get_from_obs(next_obs, "base_lin_vel")
        base_lin_vel_xy = base_lin_vel[:, 0:2]
        base_lin_vel_z = base_lin_vel[:, 2]
        base_ang_vel = self.observation_helper.get_from_obs(next_obs, "base_ang_vel")
        base_ang_vel_xy = base_ang_vel[:, 0:2]
        base_ang_vel_z = base_ang_vel[:, 2]

        dof_vel = self.observation_helper.get_from_obs(next_obs, "joint_vel")
        dof_pos = self.observation_helper.get_from_obs(next_obs, "joint_pos")

        #---------------------------------------------------------------------------

        r_tracking_lin_vel = self._reward_tracking_lin_vel(base_lin_vel_xy) * 1.0 * self.dt
        r_tracking_ang_vel = self._reward_tracking_ang_vel(base_ang_vel_z) * 0.5 * self.dt
        r_lin_vel_z = self._reward_lin_vel_z(base_lin_vel_z) * -2.0 * self.dt
        r_ang_vel_xy = self._reward_ang_vel_xy(base_ang_vel_xy) * -0.05 * self.dt
        r_torques = self._reward_torques(self._torques) * -0.0002 * self.dt
        r_dof_acc = self._reward_dof_acc(dof_vel) * -2.5e-7 * self.dt
        r_feet_air_time = self._reward_feet_air_time() * 1.0 * self.dt
        r_collision = self._reward_collision() * -1. * self.dt
        r_action_rate = self._reward_action_rate(action) * -0.01 * self.dt
        r_dof_pos_limits = self._reward_dof_pos_limits(dof_pos) * -10.0 * self.dt

        self._extra_info_rewards = {
            "tracking_lin_vel": r_tracking_lin_vel, "tracking_ang_vel": r_tracking_ang_vel,
            "lin_vel_z": r_lin_vel_z, "ang_vel_xy": r_ang_vel_xy, 
            "torques": r_torques, "dof_acc": r_dof_acc, 
            "feet_air_time": r_feet_air_time, "collision": r_collision, 
            "action_rate": r_action_rate, "dof_pos_limits": r_dof_pos_limits
        }

        reward = r_tracking_lin_vel + r_tracking_ang_vel + r_lin_vel_z + r_ang_vel_xy + r_torques + r_dof_acc + r_feet_air_time \
                + r_collision + r_action_rate + r_dof_pos_limits
        
        reward = torch.clamp(reward, min=0.)

        self.last_actions = action.clone().detach()
        self.last_dof_vel = dof_vel.clone().detach()
        
        return reward
    
    def _reward_lin_vel_z(self, lin_vel_z):
        # Penalize z axis base linear velocity
        return torch.square(lin_vel_z)
    
    def _reward_ang_vel_xy(self, base_ang_vel_xy):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(base_ang_vel_xy), dim=1)
    
    def _reward_torques(self, torques):
        # Penalize torques
        return torch.sum(torch.square(torques), dim=1)
    
    def _reward_dof_acc(self, dof_vel):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self, actions):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        contact = torch.norm(self._get_net_collision_forces("body", dt=self._timestep)[:, 5:], dim=-1) > 0.1
        return torch.sum(contact, dim=1)
        #return self._get_collision_count("lower_body", "groundplane")
    
    def _reward_dof_pos_limits(self, dof_pos):
        # Penalize dof positions too close to the limit
        out_of_limits = -(dof_pos - self._soft_dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (dof_pos - self._soft_dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_lin_vel(self, lin_vel_xy):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - lin_vel_xy), dim=1)
        return torch.exp(-lin_vel_error/0.25)
    
    def _reward_tracking_ang_vel(self, ang_vel_z):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - ang_vel_z)
        return torch.exp(-ang_vel_error/0.25)

    def _reward_feet_air_time(self):
        # Reward long steps
        #contact = torch.zeros((self.number, 4), device=self._device, dtype=bool)
        contact = self._get_net_collision_forces("body", dt=self._timestep)[:, 1:5, 2] > 1. #1/4 mass of robot * 9.81
        contact_filt = torch.logical_or(contact, self.last_contacts) #contact
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime