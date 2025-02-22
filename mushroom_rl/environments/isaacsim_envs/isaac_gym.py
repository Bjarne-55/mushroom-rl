from isaacgym import gymtorch, gymapi, gymutil
from mushroom_rl.environments.isaacsim_envs.isaac_a1_legged_gym import IsaacA1Description
from mushroom_rl.utils.isaac_sim import ObservationType, ActionType, ObservationHelper
from mushroom_rl.core import VectorizedEnvironment, MDPInfo, ArrayBackend
from mushroom_rl.rl_utils.spaces import Box
from mushroom_rl.utils import TorchUtils

import torch
import numpy as np

class IsaacGymTask:
    def __init__(self, num_envs, device):
        self.device = device
        self.num_envs = num_envs
        self.collision_helper = self

        self.gym = gymapi.acquire_gym()

        params = {'sim': {'dt': 0.005, 'gravity': [0.0, 0.0, -9.81], 'physx': {'bounce_threshold_velocity': 0.5, 'contact_collection': 2, 'contact_offset': 0.01, 'default_buffer_size_multiplier': 5, 'max_depenetration_velocity': 1.0, 'max_gpu_contact_pairs': 8388608, 'num_position_iterations': 4, 'num_threads': 10, 'num_velocity_iterations': 0, 'rest_offset': 0.0, 'solver_type': 1}, 'substeps': 1, 'up_axis': 1}}
        sim_params = gymapi.SimParams()
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = True
        gymutil.parse_sim_config(params['sim'], sim_params)

        self.sim = self.gym.create_sim(0, 0, gymapi.SimType.SIM_PHYSX, sim_params)

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.
        plane_params.dynamic_friction = 1.
        plane_params.restitution = 0.
        self.gym.add_ground(self.sim, plane_params)

        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = 3.
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.


        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = 3
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = False
        asset_options.density = 0.001
        asset_options.angular_damping = 0.
        asset_options.linear_damping = 0.
        asset_options.max_angular_velocity = 1000.
        asset_options.max_linear_velocity = 1000.
        asset_options.armature = 0.
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        #robot_asset = self.gym.load_asset(self.sim, "/home/bjarne/GitWorkspace/BachelorThesis/unitree_ros/robots/a1/urdf", "a1.urdf", asset_options)
        robot_asset = self.gym.load_asset(self.sim, "/home/bjarne/GitWorkspace/BachelorThesis/legged_gym/resources/robots/a1/urdf", "a1.urdf", asset_options)

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        dof_names = self.gym.get_asset_dof_names(robot_asset)
        num_bodies = len(body_names)
        
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0, 0, 0.42)
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []

        for i in range(num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(num_envs)))
            
            start_pose.p = gymapi.Vec3(*self.env_origins[i].clone())

            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, f"env_{i}", i, 1, 0)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        feet_names = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        penalized_contact_names = ['FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh', 'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        termination_contact_names = ["base"]
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        self.gym.prepare_sim(self.sim)

        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        cam_pos = gymapi.Vec3(5, 5, 5)
        cam_target = gymapi.Vec3(0, 0, 0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, 12, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, 12, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) 

        init_state_lst = [0., 0., 0.42] + [0.0, 0.0, 0.0, 1.0] + [0.0, 0.0, 0.0] + [0.0, 0.0, 0.0]
        self.init_state = torch.tensor(init_state_lst, device=self.device)

    def get_observation_limits(self):
        obs_high = torch.tensor([torch.inf, torch.inf, torch.inf, torch.inf, torch.inf, torch.inf,  0.8029,  4.1888,
        -0.9163,  0.8029,  4.1888, -0.9163,  0.8029,  4.1888, -0.9163,  0.8029,
         4.1888, -0.9163, 52.4000, 28.6000, 28.6000, 52.4000, 28.6000, 28.6000,
        52.4000, 28.6000, 28.6000, 52.4000, 28.6000, 28.6000], device='cuda:0')
        obs_low = torch.tensor([-torch.inf, -torch.inf, -torch.inf, -torch.inf, -torch.inf, -torch.inf,  -0.8029,
         -1.0472,  -2.6965,  -0.8029,  -1.0472,  -2.6965,  -0.8029,  -1.0472,
         -2.6965,  -0.8029,  -1.0472,  -2.6965, -52.4000, -28.6000, -28.6000,
        -52.4000, -28.6000, -28.6000, -52.4000, -28.6000, -28.6000, -52.4000,
        -28.6000, -28.6000], device='cuda:0')
        return obs_low, obs_high
    
    def get_action_limits(self):
        return torch.zeros((self.num_envs, 12)), torch.zeros((self.num_envs, 12))
    
    def get_joint_pos_limits(self):
        return torch.tensor([[-0.8029, -1.0472, -2.6965, -0.8029, -1.0472, -2.6965, -0.8029, -1.0472,
         -2.6965, -0.8029, -1.0472, -2.6965],
        [ 0.8029,  4.1888, -0.9163,  0.8029,  4.1888, -0.9163,  0.8029,  4.1888,
         -0.9163,  0.8029,  4.1888, -0.9163]], device='cuda:0')
    
    def get_joint_max_efforts(self):
        return torch.tensor([20., 55., 55., 20., 55., 55., 20., 55., 55., 20., 55., 55.], device='cuda:0') 
    
    def write_data(self, name, value, env_indices=None, reapply_after_reset=False):
        if env_indices is not None and env_indices.shape[0] == 0:
            return
        if env_indices is None:
            env_indices = torch.arange(0, self.num_envs, 1, dtype=int, device=self.device)
        if name == "body_vel":
            self.root_states[env_indices, 7:13] = value
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, 
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(env_indices.to(dtype=torch.int32)), 
                env_indices.shape[0]
            )
        elif name == "joint_pos":
            self.dof_pos[env_indices] = value
        elif name == "joint_vel":
            self.dof_vel[env_indices] = value
            self.gym.set_dof_state_tensor_indexed(self.sim,
                gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(env_indices.to(dtype=torch.int32)), env_indices.shape[0])
        elif name == "body_rot":#Careful not good implemented
            self.root_states[env_indices, 3:7] = value
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, 
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(env_indices.to(dtype=torch.int32)), 
                env_indices.shape[0]
            )
        else:
            raise NotImplementedError()
    
    def read_data(self, name, env_indices=None):
        if env_indices is None:
            env_indices = torch.arange(0, self.num_envs, 1, dtype=int, device=self.device)
        if name == "body_rot":
            return self.root_states[env_indices, 3:7]
        elif name == "body_vel":
            return self.root_states[env_indices, 7:13]
        elif name == "base_lin_vel":
            return self.root_states[env_indices, 7:10]
        elif name == "base_ang_vel":
            return self.root_states[env_indices, 10:13]
        elif name == "joint_pos":
            return self.dof_pos[env_indices, :]
        elif name == "joint_vel":
            return self.dof_vel[env_indices, :]
        raise NotImplementedError()
    
    def apply_action(self, action, env_indices=None):
        if env_indices is not None and env_indices.shape[0] == 0:
            return
        if env_indices is None:
            env_indices = torch.arange(0, self.num_envs, 1, dtype=int, device=self.device)
        actions = torch.zeros((self.num_envs, 12), device=self.device)
        actions[env_indices] = action
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim, 
            gymtorch.unwrap_tensor(actions),
            gymtorch.unwrap_tensor(env_indices.to(dtype=torch.int32)), env_indices.shape[0])
    
    def get_observations(self, clone=True):
        obs = {}
        for name in ["base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel"]:
            obs[name] = self.read_data(name)
        return obs
    
    def step(self, render=False):
        self.gym.simulate(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        if render:
            self.render()

    def render(self):
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        #self.gym.sync_frame_time(self.sim)

    def get_net_contact_forces(self, group, dt=1.0):
        if group == "body":
            self.gym.refresh_net_contact_force_tensor(self.sim)
            return self.contact_forces[:, self.termination_contact_indices, :]
        elif group == "lower_body":
            return self.contact_forces[:, self.penalised_contact_indices, :]
        elif group == "FL_foot":
            return self.contact_forces[:, [self.feet_indices[0]], :]
        elif group == "FR_foot":
            return self.contact_forces[:, [self.feet_indices[1]], :]
        elif group == "RL_foot":
            return self.contact_forces[:, [self.feet_indices[2]], :]
        elif group == "RR_foot":
            return self.contact_forces[:, [self.feet_indices[3]], :]
        raise NotImplementedError()
    
    def reset_env(self, env_indices):
        if env_indices.shape[0] == 0:
            return
        if env_indices is None:
            env_indices = torch.arange(0, self.num_envs, 1, dtype=int, device=self.device)
        self.root_states[env_indices] = self.init_state
        self.root_states[env_indices, :3] += self.env_origins[env_indices]

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_indices.to(dtype=torch.int32)), env_indices.shape[0])

    def teleport_away(self, env_indices):
        return
        self.root_states[env_indices, 2] = 10.

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_indices.to(dtype=torch.int32)), env_indices.shape[0])
        
    def reset(self):#world
        pass


class IsaacGym(IsaacA1Description):
    def __init__(self, num_envs, horizon, headless, domain_randomization=True):
        super().__init__(num_envs, horizon, headless, domain_randomization)
        obs_high = torch.full((48, ), torch.inf, device="cuda:0")
        self._mdp_info.observation_space = Box(-obs_high, obs_high, data_type=obs_high.dtype)
        action_high = torch.full((12, ), torch.inf, device="cuda:0")
        self._mdp_info.action_space = Box(-action_high, action_high, data_type=action_high.dtype)

    def _create_simulation_app(self, headless):
        pass

    def _apply_carb_settings(self):
        pass

    def _create_world(self, timestep, custom_sim_params=None):
        self._timestep = timestep
        self._world = None

    def _set_task(self, usd_path, num_envs, env_spacing, collision_between_envs, observation_spec, actuation_spec, additional_data_spec, collision_groups, physics_material_spec, camera_position, camera_target, solver_pos_it_count=None, solver_vel_it_count=None,
                 ground_plane_friction=None):
        self._task = IsaacGymTask(num_envs, "cuda:0")
        self._world = self._task

    def render_all(self, env_mask, record=False):
        self._world.render()
    
    def _import_helper_functions(self):
        from isaacgym.torch_utils import quat_apply, quat_rotate_inverse, torch_rand_float
        self.torch_rand_float = torch_rand_float
        self.quat_apply = quat_apply
        self.quat_rotate_inverse = quat_rotate_inverse

    def stop(self):
        pass

    def __del__(self):
        pass

    def _create_simulation_app(self, headless):
        return None