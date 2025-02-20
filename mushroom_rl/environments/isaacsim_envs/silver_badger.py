from mushroom_rl.environments import IsaacSim
from mushroom_rl.environments.isaacsim_envs.honey_badger import HoneyBadger
from mushroom_rl.utils.isaac_sim import ActionType, ObservationType
from mushroom_rl.rl_utils.spaces import Box

import numpy as np
import torch


class SilverBadger(HoneyBadger):

    def __init__(self, num_envs, horizon, headless):
        self.NUM_DOFS = 13
        
        backend="torch"
        device="cuda:0"

        usd_path = "/home/bjarne/GitWorkspace/BachelorThesis/mushroom-rl/isaac_assets/silver_badger/silver_badger.usd"

        self._action_spec = [
            "fl_j0", "fl_j1", "fl_j2",
            "fr_j0", "fr_j1", "fr_j2",   
            "rl_j0", "rl_j1", "rl_j2", 
            "rr_j0", "rr_j1", "rr_j2", 
            "sp_j0"
        ]
        self._default_joint_angles = torch.tensor([
            0.1, -0.8, 1.5,
            -0.1, 0.8, -1.5,
            0.1, -1., 1.5,
            -0.1, 1., -1.5, 
            0
        ], device=device)
        
        observation_spec = [
            ("base_lin_vel", "", ObservationType.BODY_LIN_VEL),
            ("base_ang_vel", "", ObservationType.BODY_ANG_VEL),
            ("base_pos", "", ObservationType.BODY_POS),

            ("FL_hip_joint_pos", "/body/fl_j0", ObservationType.JOINT_POS),#FL            
            ("FL_thigh_joint_pos", "/fl_l0/fl_j1", ObservationType.JOINT_POS),            
            ("FL_calf_joint_pos", "/fl_l1/fl_j2", ObservationType.JOINT_POS),            

            ("FR_hip_joint_pos", "/body/fr_j0", ObservationType.JOINT_POS),#FR            
            ("FR_thigh_joint_pos", "/fr_l0/fr_j1", ObservationType.JOINT_POS),
            ("FR_calf_joint_pos", "/fr_l1/fr_j2", ObservationType.JOINT_POS),           

            ("RL_hip_joint_pos", "/body/rl_j0", ObservationType.JOINT_POS),#RL
            ("RL_thigh_joint_pos", "/rl_l0/rl_j1", ObservationType.JOINT_POS),
            ("RL_calf_joint_pos", "/rl_l1/rl_j2", ObservationType.JOINT_POS),

            ("RR_hip_joint_pos", "/body/rr_j0", ObservationType.JOINT_POS),#RR
            ("RR_thigh_joint_pos", "/rl_l0/rr_j1", ObservationType.JOINT_POS),
            ("RR_calf_joint_pos", "/rl_l1/rr_j2", ObservationType.JOINT_POS),

            ("middle_joint_pos", "/body/sp_j0", ObservationType.JOINT_POS),#sp

            ("FL_hip_joint_vel", "/body/fl_j0", ObservationType.JOINT_VEL),
            ("FL_thigh_joint_vel", "/fl_l0/fl_j1", ObservationType.JOINT_VEL),
            ("FL_calf_joint_vel", "/fl_l1/fl_j2", ObservationType.JOINT_VEL),

            ("FR_hip_joint_vel", "/body/fr_j0", ObservationType.JOINT_VEL),
            ("FR_thigh_joint_vel", "/fr_l0/fr_j1", ObservationType.JOINT_VEL),
            ("FR_calf_joint_vel", "/fr_l1/fr_j2", ObservationType.JOINT_VEL),

            ("RL_hip_joint_vel", "/body/rl_j0", ObservationType.JOINT_VEL),
            ("RL_thigh_joint_vel", "/rl_l0/rl_j1", ObservationType.JOINT_VEL),
            ("RL_calf_joint_vel", "/rl_l1/rl_j2", ObservationType.JOINT_VEL),
            
            ("RR_hip_joint_vel", "/body/rr_j0", ObservationType.JOINT_VEL),
            ("RR_thigh_joint_vel", "/rl_l0/rr_j1", ObservationType.JOINT_VEL),
            ("RR_calf_joint_vel", "/rl_l1/rr_j2", ObservationType.JOINT_VEL),

            ("middle_joint_vel", "/body/sp_j0", ObservationType.JOINT_VEL),
        ]
        additional_data_spec = [("body_rot", "", ObservationType.BODY_ROT), ("body_vel", "", ObservationType.BODY_VEL)]
        collision_groups = [
            ("groundplane", ["/World/groundPlane/collisionPlane"]), 
            ("FL_foot", ["/fl_foot"]), 
            ("FR_foot", ["/fr_foot"]), 
            ("RL_foot", ["/rl_foot"]), 
            ("RR_foot", ["/rr_foot"]),
            ("body", ["/body", "/rear"]),
            ("lower_body", ["/fl_l2", "/fr_l2", "/rl_l2", "/rr_l2"])
        ]
        collision_between_envs = False
        env_spacing = 3.
        physics_material_spec = self._get_values_for_physics_materials(num_envs)
        IsaacSim.__init__(self, usd_path, self._action_spec, observation_spec, backend, device, collision_between_envs, num_envs, 
                         env_spacing, 0.99, horizon, additional_data_spec=additional_data_spec, collision_groups=collision_groups, 
                         action_type=ActionType.EFFORT, headless=headless, n_intermediate_steps=4, timestep=0.005,
                         physics_material_spec=physics_material_spec) 
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