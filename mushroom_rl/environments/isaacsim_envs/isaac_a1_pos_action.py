from mushroom_rl.environments import IsaacSim
from mushroom_rl.environments.isaacsim_envs.isaac_a1_legged_gym import IsaacA1Description
from mushroom_rl.environments.isaac_sim_env import ActionType #TODO
from mushroom_rl.utils.isaac_sim import ObservationType
import numpy as np
import torch
import random
from mushroom_rl.rl_utils.spaces import Box

#from mushroom_rl.environments.isaacsim_envs.isaac_gym import IsaacGym

class A1Pos(IsaacA1Description):
    def __init__(self, num_envs, horizon, headless, domain_randomization=True, camera_position=(105, 0, 4), camera_target=(95, 0, 0),
                 usd_path = "/home/bjarne/GitWorkspace/BachelorThesis/mushroom-rl/isaac_assets/a1/a1.usd"):
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
            ("base_lin_vel", "/base", ObservationType.BODY_LIN_VEL, None),
            ("base_ang_vel", "/base", ObservationType.BODY_ANG_VEL, None),

            ("joint_pos", "", ObservationType.JOINT_POS, self._action_spec),
            ("joint_vel", "", ObservationType.JOINT_VEL, self._action_spec)
        ]
        additional_data_spec = [
            ("body_rot", "/base", ObservationType.BODY_ROT, None), 
            ("body_vel", "/base", ObservationType.BODY_VEL, None),
            ("joint_damping", "", ObservationType.JOINT_GAIN_DAMPING, self._action_spec),
            ("joint_stiffness", "", ObservationType.JOINT_GAIN_STIFFNESS, self._action_spec)
        ]
        collision_groups = [
            ("groundplane", ["/World/groundPlane/collisionPlane"]), 
            ("FL_foot", ["/FL_foot"]), 
            ("FR_foot", ["/FR_foot"]), 
            ("RL_foot", ["/RL_foot"]), 
            ("RR_foot", ["/RR_foot"]),
            ("body", ["/trunk"]),
            ("lower_body", ["/FL_thigh", "/FR_thigh", "/RL_thigh", "/RR_thigh", "/FL_calf", "/FR_calf", "/RL_calf", "/RR_calf"])
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
        IsaacSim.__init__(self, usd_path, self._action_spec, observation_spec, backend, device, collision_between_envs, num_envs, 
                         env_spacing, 0.99, horizon, additional_data_spec=additional_data_spec, collision_groups=collision_groups, 
                         action_type=ActionType.POSITION, headless=headless, n_intermediate_steps=1, n_substeps=4, timestep=0.005, 
                         physics_material_spec=physics_material_spec, sim_params=sim_params, camera_position=camera_position, 
                         camera_target=camera_target) 
        self._import_helper_functions()
        self._set_stiffness_damping()
        action_limit = (self._task.get_joint_pos_limits() - self._default_joint_angles) / 0.25
        self._mdp_info.action_space = Box(*action_limit, data_type=action_limit[0].dtype)
        
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

    def _set_stiffness_damping(self):
        self._write_data("joint_damping", torch.full((self.number, self.NUM_DOFS), 0.5, device=self._device), torch.arange(0, self.number, 1, dtype=int, device=self._device))
        self._write_data("joint_stiffness", torch.full((self.number, self.NUM_DOFS), 20.0, device=self._device), torch.arange(0, self.number, 1, dtype=int, device=self._device))
    
    def _compute_action(self, obs, action):
        #for torque reward
        joint_vels = self.observation_helper.get_from_obs(obs, "joint_vel")
        dof_positions = self.observation_helper.get_from_obs(obs, "joint_pos")
        self._compute_torque(action, joint_vels, dof_positions)

        desired_position = action * 0.25 + self._default_joint_angles
        return desired_position