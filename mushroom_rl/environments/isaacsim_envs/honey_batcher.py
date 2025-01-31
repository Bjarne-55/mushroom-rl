from mushroom_rl.environments import IsaacSim
from mushroom_rl.environments.isaacsim_envs.isaac_a1_legged_gym import IsaacA1Description
from mushroom_rl.utils.isaac_sim import ActionType, ObservationType
from mushroom_rl.rl_utils.spaces import Box

import numpy as np
import torch

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

        self._default_joint_max_vel = torch.tensor([
            25., 25., 25.,
            25., 25., 25.,
            25., 25., 25.,
            25., 25., 25.
        ],device=device)
        
        observation_spec = [
            ("base_lin_vel", "", ObservationType.BODY_LIN_VEL, None),
            ("base_ang_vel", "", ObservationType.BODY_ANG_VEL, None),
            ("base_pos", "", ObservationType.BODY_POS, None),

            ("joint_pos", "", ObservationType.JOINT_POS, self._action_spec),
            ("joint_vel", "", ObservationType.JOINT_VEL, self._action_spec),
        ]
        additional_data_spec = [#TODO missing joint range
            ("body_rot", "", ObservationType.BODY_ROT, None), 
            ("body_vel", "", ObservationType.BODY_VEL, None),
            ("trunk_mass", "", ObservationType.SUB_BODY_MASS, "body"),
            ("trunk_inertia", "", ObservationType.SUB_BODY_INERTIA, "body"),
            ("trunk_com", "", ObservationType.SUB_BODY_COM_POS, "body"),
            ("FL_foot_scale", "/fl_foot", ObservationType.BODY_SCALE, None),
            ("FR_foot_scale", "/fr_foot", ObservationType.BODY_SCALE, None),
            ("RL_foot_scale", "/rl_foot", ObservationType.BODY_SCALE, None),
            ("RR_foot_scale", "/rr_foot", ObservationType.BODY_SCALE, None),
            ("torque_limit", "", ObservationType.JOINT_MAX_EFFORT, self._action_spec),
            ("max_joint_vel", "", ObservationType.JOINT_MAX_VELOCITY, self._action_spec),
            ("joint_range", "", ObservationType.JOINT_MAX_POS, self._action_spec),
            ("joint_armature", "", ObservationType.JOINT_ARMATURES, self._action_spec),
            ("joint_frictionloss", "", ObservationType.JOINT_FRICTION, self._action_spec),
            ("joint_damping", "", ObservationType.JOINT_GAIN_DAMPING, self._action_spec),
            ("joint_stiffness", "", ObservationType.JOINT_GAIN_STIFFNESS, self._action_spec),
            ("joint_default_pos", "", ObservationType.JOINT_DEFAULT_POS, self._action_spec)
        ]
        collision_groups = [
            ("groundplane", ["/World/groundPlane/collisionPlane"]), 
            ("FL_foot", ["/fl_foot"]), 
            ("FR_foot", ["/fr_foot"]), 
            ("RL_foot", ["/rl_foot"]), 
            ("RR_foot", ["/rr_foot"]),
            ("body", ["/body"]),
            ("lower_body", ["/fl_l2", "/fr_l2", "/rl_l2", "/rr_l2"])
        ]
        sim_params = {
            "gpu_found_lost_aggregate_pairs_capacity": 128*1024, 
            "gpu_total_aggregate_pairs_capacity": 128*1024, 
            "gpu_temp_buffer_capacity": 16777216,
            "gpu_max_rigid_patch_count": 2 * 81920
        }
        collision_between_envs = False
        env_spacing = 3.
        physics_material_spec = self._get_values_for_physics_materials(num_envs)
        IsaacSim.__init__(self, usd_path, self._action_spec, observation_spec, backend, device, collision_between_envs, num_envs, 
                         env_spacing, 0.99, horizon, additional_data_spec=additional_data_spec, collision_groups=collision_groups, 
                         action_type=ActionType.EFFORT, headless=headless, n_intermediate_steps=4, timestep=0.005, 
                         physics_material_spec=physics_material_spec, sim_params=sim_params, camera_position=(105, 0, 4), 
                         camera_target=(95, 0, 0)) 
        self._import_helper_functions()
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
        self._gravity = torch.tensor([0., 0., -1.], device=self._device).repeat((self.number, 1))

        #domain randomization
        self.np_rng = np.random.default_rng()
        self.current_mixed = False
        self.current_nr_delay_steps = 0
    
    def _import_helper_functions(self):
        super()._import_helper_functions()
        global get_euler_xyz
        from omni.isaac.core.utils.torch.rotations import get_euler_xyz

    def _get_noise_scale_vec(self):
        v = torch.zeros((self.observation_helper.obs_length), device=self._device)

        lin_vel = self.observation_helper.obs_idx_map["base_lin_vel"]
        ang_vel = self.observation_helper.obs_idx_map["base_ang_vel"]
        joint_positions = self.observation_helper.obs_idx_map["joint_pos"]
        joint_velocities = self.observation_helper.obs_idx_map["joint_vel"]
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
        joint_positions = self.observation_helper.obs_idx_map["joint_pos"]
        joint_velocities = self.observation_helper.obs_idx_map["joint_vel"]
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

        dof_pos_indices = self.observation_helper.obs_idx_map["joint_pos"]
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
    
    def _step_finalize(self, env_indices):
        super()._step_finalize(env_indices)

        if self.np_rng.uniform() < 0.002:
            self.current_mixed = self.np_rng.uniform() < self.MIXED_CHANCE
            self.current_nr_delay_steps = 0

    def setup(self, env_indices, obs):
        super().setup(env_indices, obs)

        self.action_history = torch.zeros((self.MAX_NR_DELAY_STEPS + 1, self.number, self.NUM_DOFS), device=self._device)
    
    def _preprocess_action(self, action):
        action = self.delay_action(action)
        return super()._preprocess_action(action)
    
    def _compute_torque(self, action, joint_vels, joint_pos):
        action_scaled = action * self._seen_scaling_factor
        target_joint_pos = self._seen_joint_nominal_pos + action_scaled

        self._torques = self._unseen_p_gain * (target_joint_pos - joint_pos + self._joint_position_offset) \
            - self._unseen_d_gain * joint_vels
        self._torques *= self._nf_motor_strength
        self._torques = torch.clip(self._torques, -self._seen_torque_limit, self._seen_torque_limit)

        return self._torques
        

    #domain randomization -------------------------------------
    MAX_NR_DELAY_STEPS = 1
    MIXED_CHANCE = 0.05

    def delay_action(self, action):
        if self.current_mixed:
            self.current_nr_delay_steps = self.np_rng.integers(self.MAX_NR_DELAY_STEPS+1)

        self.action_history = torch.roll(self.action_history, -1, dims=0)
        self.action_history[-1] = action

        chosen_action = self.action_history[-1-self.current_nr_delay_steps]

        return chosen_action
    
    def _init_domain_randomization_parameters(self):
        #init some seen parameters
        self._seen_joint_damping = self._read_data("joint_damping")
        self._seen_joint_stiffness = self._read_data("joint_stiffness")
        self._seen_joint_armature = self._read_data("joint_armature")
        self._seen_joint_frictionloss = self._read_data("joint_frictionloss")

        self._seen_trunk_mass = self._read_data("trunk_mass")
        self._seen_torque_limit = self._read_data("torque_limit")
        self._seen_joint_nominal_pos = self._default_joint_angles.repeat((self.number, 1))
        self._seen_joint_max_vel = self._default_joint_max_vel.repeat((self.number, 1))

        self._seen_p_gain = torch.full((self.number, self.NUM_DOFS), 20., device=self._device)
        self._seen_d_gain = torch.full((self.number, self.NUM_DOFS), 0.5, device=self._device)
        self._seen_scaling_factor = torch.full((self.number, self.NUM_DOFS), 0.25, device=self._device)

        self._unseen_p_gain = torch.full((self.number, self.NUM_DOFS), 20., device=self._device)
        self._unseen_d_gain = torch.full((self.number, self.NUM_DOFS), 0.5, device=self._device)

        self._default_trunk_mass = self._seen_trunk_mass[0]
        self._default_trunk_inertia = self._read_data("trunk_inertia")[0]
        self._default_trunk_com = self._read_data("trunk_com")[0]
        self._default_torque_limit = self._seen_torque_limit[0]
        self._default_joint_nominal_pos = self._seen_joint_nominal_pos[0]#TODO nominal postion missing
        #self._default_joint_max_vel = self._seen_joint_max_vel[0]
        self._default_joint_range = self._read_data("joint_range")[0]
        self._default_joint_damping = self._seen_joint_damping[0]
        self._default_joint_stiffness = self._seen_joint_stiffness[0]
        self._default_joint_armature = self._seen_joint_armature[0]
        self._default_joint_frictionloss = self._seen_joint_frictionloss[0]

        self._nf_trunk_mass = torch.ones((self.number, ), device=self._device)
        self._nf_trunk_com = torch.ones((self.number, ), device=self._device)
        self._nf_foot_size = torch.ones((self.number, ), device=self._device)
        self._nf_joint_damping = torch.ones((self.number, ), device=self._device)
        self._nf_joint_stiffness = torch.ones((self.number, ), device=self._device)
        self._nf_joint_armature = torch.ones((self.number, ), device=self._device)
        self._nf_joint_friction = torch.ones((self.number, ), device=self._device)

        self._nf_p_gain = torch.ones((self.number, ), device=self._device)
        self._nf_d_gain = torch.ones((self.number, ), device=self._device)
        self._nf_motor_strength = torch.ones((self.number, ), device=self._device)
        self._joint_position_offset = torch.ones((self.number, ), device=self._device)

    def sample_unseen_noise_factors(
            self, env_indices,
            trunk_mass_factor=0.25,
            trunk_com_factor=0.25,
            foot_size_factor=0.03,
            joint_damping_factor=0.5,
            joint_armature_factor=0.5,
            joint_stiffness_factor=0.5,
            joint_friction_factor=0.5,
            motor_strength_factor=0.25,
            p_gain_factor=0.25,
            d_gain_factor=0.25,
            position_offset=0.05
        ):
        n_envs = env_indices.shape[0]

        self._nf_trunk_mass[env_indices] = torch_rand_float(1 - trunk_mass_factor, 1 + trunk_mass_factor, (n_envs, ), self._device)
        self._nf_trunk_com[env_indices] = torch_rand_float(1 - trunk_com_factor, 1 + trunk_com_factor, (n_envs, ), self._device)
        self._nf_foot_size[env_indices] = torch_rand_float(1 - foot_size_factor, 1 + foot_size_factor, (n_envs, ), self._device)
        self._nf_joint_damping[env_indices] = torch_rand_float(1 - joint_damping_factor, 1 + joint_damping_factor, (n_envs, ), self._device)
        self._nf_joint_stiffness[env_indices] = torch_rand_float(1 - joint_stiffness_factor, 1 + joint_stiffness_factor, (n_envs, ), self._device)
        self._nf_joint_armature[env_indices] = torch_rand_float(1 - joint_armature_factor, 1 + joint_armature_factor, (n_envs, ), self._device)
        self._nf_joint_friction[env_indices] = torch_rand_float(1 - joint_friction_factor, 1 + joint_friction_factor, (n_envs, ), self._device)

        #control function
        self._nf_p_gain[env_indices] = torch_rand_float(1 - p_gain_factor, 1 + p_gain_factor, (n_envs, ), self._device)
        self._nf_d_gain[env_indices] = torch_rand_float(1 - d_gain_factor, 1 + d_gain_factor, (n_envs, ), self._device)
        self._nf_motor_strength[env_indices] = torch_rand_float(1 - motor_strength_factor, 1 + motor_strength_factor, (n_envs, ), self._device)
        self._joint_position_offset[env_indices] = torch_rand_float(-position_offset, position_offset, (n_envs, ), self._device)#something with model nu
    
    def sample_seen_parameters(
            self, env_indices,
            stay_at_default_percentage=0.3,
            add_trunk_mass_min=-0.8, add_trunk_mass_max=0.8,
            add_com_displacement_min=-0.0025, add_com_displacement_max=0.0025,
            foot_scaling_min=0.975, foot_scaling_max=1.025,
            torque_limit_factor=0.3,
            add_joint_nominal_position_min=-0.01, add_joint_nominal_position_max=0.01,
            joint_velocity_factor=0.15,
            add_joint_range_min=-0.05, add_joint_range_max=0.05,
            joint_damping_min=0.0, joint_damping_max=0.3,
            joint_armature_min=0.009, joint_armature_max=0.023,
            joint_stiffness_min=0.0, joint_stiffness_max=0.5,
            joint_friction_loss_min=0.0, joint_friction_loss_max=1.0,
            add_p_gain_min=-3.0, add_p_gain_max=3.0,
            add_d_gain_min=-0.1, add_d_gain_max=0.1,  
            add_scaling_factor_min=-0.03, add_scaling_factor_max=0.03,  
        ):
        n_envs = env_indices.shape[0]

        #trunk mass
        self._seen_trunk_mass[env_indices] = self._default_trunk_mass \
            + torch_rand_float(add_trunk_mass_min, add_trunk_mass_max, (n_envs, 1), self._device)
        actual_trunk_mass = self._seen_trunk_mass * self._nf_trunk_mass[env_indices]
        self._write_data("trunk_mass", actual_trunk_mass, env_indices)
        actual_trunk_inertia = self._default_trunk_inertia + (actual_trunk_mass / self._default_trunk_mass)
        self._write_data("trunk_inertia", actual_trunk_inertia, env_indices)

        #trunk com
        actual_trunk_com = self._default_trunk_com \
            + torch_rand_float(add_com_displacement_min, add_com_displacement_max, (n_envs, 1), self._device)
        actual_trunk_com *= self._nf_trunk_com[env_indices]
        self._write_data("trunk_com", actual_trunk_com, env_indices)

        #foot scaling
        actual_foot_scaling = torch_rand_float(foot_scaling_min, foot_scaling_max, (n_envs, 4), self._device) \
            * self._nf_foot_size[env_indices]
        for i, name in enumerate(["FL_foot_scale", "FR_foot_scale", "RL_foot_scale", "RR_foot_scale"]):
            self._write_data(name, actual_foot_scaling[env_indices, i], env_indices)
        
        #joint nominal position
        self._seen_joint_nominal_pos[env_indices] = self._default_joint_nominal_pos \
            + torch_rand_float(add_joint_nominal_position_min, add_joint_nominal_position_max, (n_envs, self.NUM_DOFS), self._device)
        #self._write_data("joint_default_pos", self._seen_joint_nominal_pos[env_indices], env_indices)

        #joint torque limit
        self._seen_torque_limit[env_indices] = self._default_torque_limit \
            * (1 + torch_rand_float(-torque_limit_factor, torque_limit_factor, (n_envs, self.NUM_DOFS), self._device))
        self._write_data("torque_limit", self._seen_torque_limit[env_indices], env_indices)

        #joint max velocity
        self._seen_joint_max_vel[env_indices] = self._default_joint_max_vel \
            * (1 + torch_rand_float(-joint_velocity_factor, joint_velocity_factor, (n_envs, self.NUM_DOFS), self._device))
        self._write_data("max_joint_vel", self._seen_joint_max_vel[env_indices], env_indices)

        #joint range
        #TODO can't do that

        stay_at_default_mask = torch_rand_float(0, 1, (n_envs, ), self._device) < stay_at_default_percentage
        stay_at_default_idx = env_indices[stay_at_default_mask]
        self._seen_joint_damping[stay_at_default_idx] = self._default_joint_damping
        self._seen_joint_stiffness[stay_at_default_idx] = self._default_joint_stiffness
        self._seen_joint_armature[stay_at_default_idx] = self._default_joint_armature
        self._seen_joint_frictionloss[stay_at_default_idx] = self._default_joint_frictionloss

        not_stay_at_default_mask = torch.logical_not(stay_at_default_mask)
        not_stay_at_default_idx = env_indices[not_stay_at_default_mask]
        num_envs_not_default = not_stay_at_default_idx.shape[0]
        self._seen_joint_damping[not_stay_at_default_idx] = torch_rand_float(joint_damping_min, joint_damping_max, (num_envs_not_default, self.NUM_DOFS), self._device)
        self._seen_joint_stiffness[not_stay_at_default_idx] = torch_rand_float(joint_stiffness_min, joint_stiffness_max, (num_envs_not_default, self.NUM_DOFS), self._device)
        self._seen_joint_armature[not_stay_at_default_idx] = torch_rand_float(joint_armature_min, joint_armature_max, (num_envs_not_default, self.NUM_DOFS), self._device)
        self._seen_joint_frictionloss[not_stay_at_default_idx] = torch_rand_float(joint_friction_loss_min, joint_friction_loss_max, (num_envs_not_default, self.NUM_DOFS), self._device)

        self._write_data("joint_damping", self._seen_joint_damping[env_indices] * self._nf_joint_damping[env_indices], env_indices)
        self._write_data("joint_stiffness", self._seen_joint_stiffness[env_indices] * self._nf_joint_stiffness[env_indices], env_indices)
        self._write_data("joint_armature", self._seen_joint_armature[env_indices] * self._nf_joint_armature[env_indices], env_indices)
        self._write_data("joint_frictionloss", self._seen_joint_frictionloss[env_indices] * self._nf_joint_friction[env_indices], env_indices)

        #used for control function
        self._seen_p_gain[env_indices] = 20 + torch_rand_float(add_p_gain_min, add_p_gain_max, (n_envs, self.NUM_DOFS), self._device)
        self._seen_d_gain[env_indices] = 0.5 + torch_rand_float(add_d_gain_min, add_d_gain_max, (n_envs, self.NUM_DOFS), self._device)
        self._seen_scaling_factor[env_indices] = 0.25 + torch_rand_float(add_scaling_factor_min, add_scaling_factor_max, (n_envs, self.NUM_DOFS), self._device)

        self._unseen_p_gain[env_indices] = self._seen_p_gain[env_indices] * self._nf_p_gain[env_indices]
        self._unseen_d_gain[env_indices] = self._seen_d_gain[env_indices] * self._nf_d_gain[env_indices]

    #rewards --------------------------------------------------------------------

    def reward(self, obs, action, next_obs, absorbing):
        base_lin_vel = self.observation_helper.get_from_obs(next_obs, "base_lin_vel")
        base_lin_vel_xy = base_lin_vel[:, 0:2]
        base_lin_vel_z = base_lin_vel[:, 2]

        base_ang_vel = self.observation_helper.get_from_obs(next_obs, "base_ang_vel")
        base_ang_vel_xy = base_ang_vel[:, 0:2]
        base_ang_vel_z = base_ang_vel[:, 2]

        base_rot = self._read_data("body_rot")
        base_rot_euler = get_euler_xyz(base_rot)
        base_rot_xy = base_rot_euler[:, :2]

        base_pos = self.observation_helper.get_from_obs(next_obs, "base_pos")
        base_pos_z = base_pos[:, 2]

        dof_vel = self.observation_helper.get_from_obs(next_obs, "joint_vel")
        dof_pos = self.observation_helper.get_from_obs(next_obs, "joint_pos")

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

