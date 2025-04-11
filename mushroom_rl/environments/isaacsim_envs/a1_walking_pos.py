from mushroom_rl.environments import IsaacSim
from mushroom_rl.environments.isaacsim_envs import A1Walking
from mushroom_rl.utils.isaac_sim import ObservationType, ActionType
import numpy as np
import torch
from mushroom_rl.rl_utils.spaces import Box
from pathlib import Path

class A1WalkingPos(A1Walking):
    """
    A learning environment for training the A1 quadroped to walk. IsaacSim interprets actions as positions, so
    no PD-controller in environment.
    
    Resembles environment implemented by Rudin et al. for 
    "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning"
    """
    def __init__(self, num_envs, horizon, headless, domain_randomization=True, camera_position=(105, 0, 4), camera_target=(95, 0, 0)):
        usd_path = str(Path(__file__).resolve().parent / "robots_usds/a1/a1.usd")
        self.NUM_JOINTS = 12

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
            ("body_vel", "", ObservationType.BODY_VEL, None),
            ("joint_damping", "", ObservationType.JOINT_GAIN_DAMPING, self._action_spec),
            ("joint_stiffness", "", ObservationType.JOINT_GAIN_STIFFNESS, self._action_spec),
            ("joint_measured_effort", "", ObservationType.JOINT_MEASURED_EFFORT, self._action_spec),
        ]

        #one collision group is faster, of course it would be cleaner with 3 (feet, body, lower_body)
        collision_groups = [
            ("body", ["/trunk", "/FL_foot", "/FR_foot", "/RL_foot", "/RR_foot", "/FL_thigh", "/FR_thigh", "/RL_thigh", "/RR_thigh", "/FL_calf", "/FR_calf", "/RL_calf", "/RR_calf"]),
        ]
        self._trunk_idx = 0
        self._feet_ids = slice(1, 5)
        self._lower_bodies_ids = slice(5, None)

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
        IsaacSim.__init__(self, usd_path, self._action_spec, observation_spec, backend, device, collision_between_envs, num_envs, 
                         env_spacing, 0.99, horizon, additional_data_spec=additional_data_spec, collision_groups=collision_groups, 
                         action_type=ActionType.POSITION, headless=headless, timestep=0.02, physics_material_spec=physics_material_spec, 
                         sim_params=sim_params, camera_position=camera_position, camera_target=camera_target, solver_pos_it_count=solver_pos, 
                         solver_vel_it_count=solver_vel) 
        self._import_helper_functions()
        self._set_stiffness_damping()
        action_limits = (self._task.get_joint_pos_limits() - self._default_joint_angles) / 0.25
        self._mdp_info.action_space = Box(*action_limits, data_type=action_limits[0].dtype)
        
        #register custom observations
        self.observation_helper.add_obs("projected_gravity", 3, -1, 1)
        commands_upper = torch.tensor([1., 1., np.pi], device=device)
        self.observation_helper.add_obs("commands", 3, -commands_upper, commands_upper)
        self.observation_helper.add_obs("actions", self.NUM_JOINTS, self.info.action_space.low, self.info.action_space.high)

        #get normalization and noise vector
        self._normalization_obs_vec = self._get_obs_normilization_vec()
        self._noise_scale_vec = self._get_noise_scale_vec()
        self._soft_joint_pos_limits = self._get_soft_joint_pos_limit()

        #update observation space
        obs_low, obs_high = self.observation_helper.obs_limits
        joint_pos_indices = self.observation_helper.obs_idx_map["joint_pos"]
        obs_low[joint_pos_indices] -= self._default_joint_angles
        obs_high[joint_pos_indices] -= self._default_joint_angles
        new_obs_low = obs_low * self._normalization_obs_vec - self._noise_scale_vec
        new_obs_high = obs_high * self._normalization_obs_vec + self._noise_scale_vec
        self._mdp_info.observation_space = Box(new_obs_low, new_obs_high, data_type=new_obs_high.dtype)

        self._commands = torch.zeros(num_envs, 4, dtype=torch.float, device=device)
        self._actions = torch.zeros((num_envs, self.NUM_JOINTS), device=device)
        self._feet_air_time = torch.zeros((num_envs, 4), device=device)
        self._last_actions =  torch.zeros((num_envs, self.NUM_JOINTS), device=device)
        self._last_joint_vel = torch.zeros((num_envs, self.NUM_JOINTS), device=device)
        self._last_contacts = torch.zeros((num_envs, 4), device=device, dtype=torch.bool)
        self._episode_length = torch.zeros((num_envs, ), dtype=int, device=device)

        self._forward_vec = torch.tensor([1., 0., 0.], device=device).repeat((num_envs, 1))
        self._gravity = torch.tensor([0., 0., -1.], device=self._device).repeat((self.number, 1))

        self._effort_limit = self._task.get_joint_max_efforts()

    def _set_stiffness_damping(self):
        env_ids = torch.arange(0, self.number, 1, dtype=int, device=self._device)
        self._write_data("joint_damping", torch.full((self.number, self.NUM_JOINTS), 0.5, device=self._device), env_ids, reapply_after_reset=True)
        self._write_data("joint_stiffness", torch.full((self.number, self.NUM_JOINTS), 20.0, device=self._device), env_ids, reapply_after_reset=True)
    
    def _compute_action(self, action):
        desired_position = action * 0.25 + self._default_joint_angles
        return desired_position
    
    def _step_finalize(self, env_indices):
        self._torques = self._read_data("joint_measured_effort")
        return super()._step_finalize(env_indices)