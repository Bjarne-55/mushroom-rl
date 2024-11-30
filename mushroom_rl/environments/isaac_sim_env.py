from enum import Enum
import numpy as np
import torch
import random

from isaacsim import SimulationApp

from mushroom_rl.core import VectorizedEnvironment, MDPInfo, ArrayBackend
from mushroom_rl.rl_utils.spaces import Box
from mushroom_rl.utils import TorchUtils
from mushroom_rl.utils.viewer import ImageViewer
from mushroom_rl.utils.isaac_sim import ObservationHelper, ObservationType
    
class ActionType(Enum):
    EFFORT = "joint_efforts"
    POSITION = "joint_positions"
    VELOCITY = "joint_velocities"

class IsaacSim(VectorizedEnvironment):
    # TODO add all relevant varibales from mujoco Constructur
    # TODO think about tasks
    def __init__(self, usd_path, action_spec, observation_spec, backend, device, collision_between_envs, 
                 n_envs, env_spacing, gamma, horizon, timestep=None, n_substeps=1, n_intermediate_steps=1, 
                 additional_data_spec=None, collision_groups=None, action_type=ActionType.EFFORT, headless=True):
        """
        Constructor.

        Args:
            usd_path (str): A string with a path to the usd file.
            actuation_spec (list): A list specifying the names of the joints  which should be controllable by the
               agent. Can be left empty when all actuators should be used;
            observation_spec (list): A list containing the names of data that should be made available to the agent as
               an observation and their type (ObservationType). They are combined with a key, which is used to access
               the data. An entry in the list is given by: (key, name, type). The name can later be used to retrieve
               specific observations;
            backend (str)
            device (str)
            collision_between_envs (bool): Whether collisions between environments should be possible or not
            n_envs (int): Number of parallel environments
            env_spacing (int): Distance between environments
        """
        self._simulation_app = SimulationApp({"headless": headless, "hide_ui": False}) 
        self._viewer = None

        self._backend = backend
        self._device = device
        TorchUtils.set_default_device(device)

        self._action_type = action_type

        self._n_intermediate_steps = n_intermediate_steps
        self._n_substeps = n_substeps

        #create world and set task
        self._create_world(timestep)
        self._set_camera()
        self._create_light()
        self._set_task(usd_path, n_envs, env_spacing, collision_between_envs, observation_spec, action_spec, 
                       additional_data_spec, collision_groups)
        self._world.reset()

        observation_limits = self._task.get_observation_limits()
        observation_space = Box(*observation_limits)

        self.observation_helper = ObservationHelper(observation_spec, observation_limits, backend, n_envs, device)

        action_limits = self._task.get_action_limits()
        action_space = Box(action_limits[0].to(self._device), action_limits[1].to(self._device))

        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, self.dt, backend)
        mdp_info = self._modify_mdp_info(mdp_info)
        
        super().__init__(mdp_info, n_envs)

    def _create_world(self, timestep):
        from omni.isaac.core.world import World
        self._world = World(
            stage_units_in_meters=1.0,
            rendering_dt=1.0 / 60.0,
            backend=self._backend,
            device=self._device
        )
        self._physics_context = self._world.get_physics_context()
        if timestep is None:
            self._timestep = self._world.get_physics_dt()
            
            self._physics_context.set_physics_dt(dt=self._timestep, substeps=self._n_substeps)
        else:
            self._physics_context.set_physics_dt(dt=timestep, substeps=self._n_substeps)
            self._timestep = timestep

    def _set_camera(self):
        from omni.kit.viewport.utility import get_viewport_from_window_name
        from omni.kit.viewport.utility.camera_state import ViewportCameraState
        from pxr import Gf
        import omni.replicator.core as rep
        import os

        viewport_api_2 = get_viewport_from_window_name("Viewport")
        viewport_api_2.set_active_camera("/OmniverseKit_Persp")
        camera_state = ViewportCameraState("/OmniverseKit_Persp", viewport_api_2)
        camera_state.set_position_world(Gf.Vec3d(10, 0, 6), True)
        camera_state.set_target_world(Gf.Vec3d(0, 0, 0), True)

        #create annotator
        rp = rep.create.render_product("/OmniverseKit_Persp", (480, 480))
        self.rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_annot.attach(rp)

    def _create_light(self, prim_path="/World/defaultDistantLight", intensity=1000):
        from omni.isaac.core.utils.stage import get_current_stage
        from pxr import UsdLux
        stage = get_current_stage()
        light = UsdLux.DistantLight.Define(stage, prim_path)
        light.CreateIntensityAttr().Set(intensity)

    def _set_task(self, usd_path, n_envs, env_spacing, collision_between_envs, observation_spec, action_spec, 
                  additional_data_spec, collision_groups):
        from mushroom_rl.environments.isaac_sim_task import IsaacSimTask

        self._task = IsaacSimTask(self._physics_context, usd_path, n_envs, env_spacing, 
                                  collision_between_envs, observation_spec, action_spec, additional_data_spec, 
                                  collision_groups, self._backend, self._action_type)
        self._world.add_task(self._task)

    def render_all(self, env_mask, record=False):#TODO add recording
        self._world.render()
        data = self.rgb_annot.get_data()[..., :3]

        if self._viewer is None:
            self._viewer = ImageViewer((480, 480), self.dt)
        self._viewer.display(data)

        if record:
            return data

    def step_all(self, env_mask, action):#TODO intermediate and substeps
        arr_backend = ArrayBackend.get_array_backend(self._mdp_info.backend)

        """
        action = self._bound(action, self.info.action_space.low, self.info.action_space.high)
        if self._action_type == ActionType.POSITION:
            lower_limits = self._max_actions[:, 0]
            upper_limits = self._max_actions[:, 1]
            action = lower_limits + (action + 1) * 0.5 * (upper_limits - lower_limits)
        else:
            action = action * self._max_actions
        """
        action = self._preprocess_action(action)

        env_indices = arr_backend.where(env_mask)[0]
        self._task.apply_action(action[env_indices], env_indices)

        for _ in range(self._n_intermediate_steps):
            self._world.step(render=False)

        cur_obs = self.observation_helper.build_obs(self._task.get_observations(clone=True))
        cur_obs = self._create_observation(cur_obs)

        absorbing = self.is_absorbing(cur_obs)
        reward = self.reward(self._obs, action, cur_obs, absorbing)
        extra_info = self._create_info_dictionary(cur_obs)

        self._obs = cur_obs.clone().detach()

        cur_obs = self._modify_observation(cur_obs)
        
        return cur_obs.clone().detach(), reward.clone().detach(), torch.logical_and(absorbing, env_mask).clone().detach(), extra_info
    
    def reset_all(self, env_mask, state=None):
        arr_backend = ArrayBackend.get_array_backend(self._mdp_info.backend)
        env_indices = arr_backend.where(env_mask)[0]

        self._task.reset_env(env_indices, state)
        self.setup(env_indices, state)
        
        obs = self.observation_helper.build_obs(self._task.get_observations(clone=True))
        obs = self._create_observation(obs)
        self._obs = obs.clone().detach()

        info = self._create_info_dictionary(obs)
        obs = self._modify_observation(obs)

        return obs.clone().detach(), info
    
    """
    def _create_observation(self, obs):
        size = next(reversed(self._obs_idx_map.values()))
        arr = ArrayBackend.get_array_backend(self._backend).empty((self.number, size), self._device)

        for name, indices in self._obs_idx_map.items():
            arr[indices[0]:indices[1]] = obs[name]
            
        return arr
    """
    def _create_observation(self, obs):
        return obs

    def _modify_observation(self, obs):
        """
        This method can be overridden to edit the created observation. This is done after the reward and absorbing
        functions are evaluated. Especially useful to transform the observation into different frames. If the original
        observation order is not preserved, the helper functions in ObervationHelper breaks.

        Args:
            obs (np.ndarray): the generated observation

        Returns:
            The environment observation.

        """
        return obs

    def seed(self, seed=-1):
        from omni.isaac.core.utils.torch.maths import set_seed
        return set_seed(seed)
    
    def stop(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._world.reset()
        print("reset")

    def __del__(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._simulation_app.close()
    
    @property
    def dt(self):
        return self._timestep * self._n_intermediate_steps * self._n_substeps
    
    def reward(self, obs, action, next_obs, absorbing):
        """
        Compute the reward based on the given transition.

        Args:
            obs (np.array): the current state of the system;
            action (np.array): the action that is applied in the current state;
            next_obs (np.array): the state reached after applying the given
                action.
            absorbing (bool): whether next_state is an absorbing state or not.

        Returns:
            The reward as a floating point scalar value.

        """
        raise NotImplementedError

    def is_absorbing(self, obs):
        """
        Check whether the given state is an absorbing state or not.

        Args:
            obs (np.array): the state of the system.

        Returns:
            A boolean flag indicating whether this state is absorbing or not.

        """
        raise NotImplementedError

    def setup(self, env_indices, obs):
        """
        A function that allows to execute setup code after an environment
        reset.

        """
        raise NotImplementedError
    
    def _check_collision(self, group1, group2):
        """
        Check for collision between the specified groups.

        Args:
            group1 (string): A name referring to an entry contained in the
                collision_groups list handed to the constructor;
            group2 (string): A name referring to an entry contained in the
                collision_groups list handed to the constructor.

        Returns:
            A flag indicating whether a collision occurred between the given
            groups or not.

        """
        return self._task.check_collision(group1, group2)

    def _get_collision_force(self, group1, group2):
        """
        Returns the collision force and torques between the specified groups.

        Args:
            group1 (string): A name referring to an entry contained in the
                collision_groups list handed to the constructor;
            group2 (string): A name referring to an entry contained in the
                collision_groups list handed to the constructor.

        Returns:
            A 3D vector specifying the collision forces
        """
        return self._task.get_collision_force(group1, group2)
    
    def _get_collision_count(self, group1, group2):
        return self._task.get_collision_count(group1, group2)
    
    def _read_data(self, name, env_indices=None):
        return self._task.read_data(name, env_indices)

    def _write_data(self, name, value, env_indices=None):
        self._task.write_data(name, value, env_indices)
    
    def _preprocess_action(self, action):
        """
        Compute a transformation of the action provided to the
        environment.

        Args:
            action (np.ndarray): numpy array with the actions
                provided to the environment.

        Returns:
            The action to be used for the current step
        """
        return action
    
    def _modify_mdp_info(self, mdp_info):
        """
        This method can be overridden to modify the automatically generated MDPInfo data structure.
        By default, returns the given mdp_info structure unchanged.

        Args:
            mdp_info (MDPInfo): the MDPInfo structure automatically computed by the environment.

        Returns:
            The modified MDPInfo data structure.

        """
        return mdp_info
    
    def _create_info_dictionary(self, obs):
        """
        This method can be overridden to create a custom info dictionary.

        Args:
            obs (np.ndarray): the generated observation

        Returns:
            The information dictionary.

        """
        return {}