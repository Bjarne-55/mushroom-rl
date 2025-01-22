import numpy as np
import torch
import random

from isaacsim import SimulationApp

from mushroom_rl.core import VectorizedEnvironment, MDPInfo, ArrayBackend
from mushroom_rl.rl_utils.spaces import Box
from mushroom_rl.utils import TorchUtils
from mushroom_rl.utils.viewer import ImageViewer
from mushroom_rl.utils.isaac_sim import ObservationHelper, ObservationType, ActionType

class IsaacSim(VectorizedEnvironment):
    # TODO add all relevant varibales from mujoco Constructur
    # TODO think about tasks
    def __init__(self, usd_path, actuation_spec, observation_spec, backend, device, collision_between_envs, 
                 n_envs, env_spacing, gamma, horizon, timestep=None, n_substeps=1, n_intermediate_steps=1, 
                 additional_data_spec=None, collision_groups=None, action_type=ActionType.EFFORT, headless=True,
                 physics_material_spec=None, sim_params=None, camera_position=(5, 0, 4), camera_target=(0, 0, 0)):
        """
        Constructor.

        Args:
            usd_path (str): Path to the USD file.
            actuation_spec (list): A list specifying the names of the joints  which should be controllable by the
               agent. Can be left empty when all actuators should be used;
            observation_spec (list): A list containing the names of data that should be made available to the agent as
               an observation and their type (ObservationType). They are combined with a key, which is used to access
               the data. An entry in the list is given by: (key, name, type). The name can later be used to retrieve
               specific observations;
            backend (str): Backend for array operations.
            device (str): Compute device (e.g., 'cuda:0').
            collision_between_envs (bool): Whether inter-environment collisions are allowed.
            n_envs (int): Number of parallel environments.
            env_spacing (float): Distance between environments.
            gamma (float): Discount factor for RL.
            horizon (int): Episode horizon.
            timestep (float, optional): Simulation timestep.
            n_substeps (int): Number of substeps per simulation step.
            n_intermediate_steps (int): Intermediate control steps.
            additional_data_spec (list, optional): Additional data specifications.
            collision_groups (dict, optional): Collision groups configuration.
            action_type (ActionType): Type of action (effort, position, velocity).
            headless (bool): Whether to run in headless mode.
            sim_params (dict)
        """
        self._headless = headless
        self._simulation_app = SimulationApp({"headless": headless, "hide_ui": False}) 
        self._viewer = None

        self._apply_carb_settings()

        self._backend = backend
        self._device = device
        TorchUtils.set_default_device(device)

        self._action_type = action_type
        self._n_intermediate_steps = n_intermediate_steps
        self._n_substeps = n_substeps

        # Initialize world and tasks
        self._create_world(timestep, sim_params)
        self._set_task(
            usd_path, 
            n_envs, 
            env_spacing, 
            collision_between_envs, 
            observation_spec, 
            actuation_spec, 
            additional_data_spec, 
            collision_groups,
            physics_material_spec, 
            camera_position,
            camera_target
        )
        self._world.reset()

        observation_limits = self._task.get_observation_limits()
        observation_space = Box(*observation_limits)
        self.observation_helper = ObservationHelper(observation_spec, observation_limits, backend, n_envs, device)

        action_limits = self._task.get_action_limits()
        action_space = Box(action_limits[0].to(self._device), action_limits[1].to(self._device))

        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, self.dt, backend)
        mdp_info = self._modify_mdp_info(mdp_info)

        self._recompute_action_per_step = type(self)._compute_action != IsaacSim._compute_action
        
        super().__init__(mdp_info, n_envs)

    def _apply_carb_settings(self):
        """Apply Carb settings for optimization."""
        import carb

        carb.settings.get_settings().set("/persistent/omnihydra/useSceneGraphInstancing", True)
        carb.settings.get_settings().set_bool("/physics/physxDispatcher", True)
        carb.settings.get_settings().set("/app/viewport/grid/enabled", False)
        carb.settings.get_settings().set("/app/runLoops/main/rateLimitEnabled", False)

    def _create_world(self, timestep, custom_sim_params=None):
        """Create and configure the simulation world."""
        from omni.isaac.core.world import World

        sim_params = {
            'gravity': [0.0, 0.0, -9.81], 
            'use_gpu_pipeline': True, 
            'use_fabric': True, 
            'enable_scene_query_support': True, 
            'use_gpu': True
        }
        if custom_sim_params is not None:
            sim_params.update(custom_sim_params)

        self._world = World(
            stage_units_in_meters=1.0,
            rendering_dt=1.0 / 60.0,
            backend=self._backend,
            device=self._device,
            sim_params=sim_params
        )
        self._physics_context = self._world.get_physics_context()
        self._physics_context.enable_gpu_dynamics(True)

        if timestep is None:
            self._timestep = self._world.get_physics_dt()
            self._physics_context.set_physics_dt(dt=self._timestep, substeps=self._n_substeps)
        else:
            self._physics_context.set_physics_dt(dt=timestep, substeps=self._n_substeps)
            self._timestep = timestep

        self._world.set_simulation_dt(rendering_dt=self.dt)
        print(f"rendering dt: {self._world.get_rendering_dt()}, physix dt: {self._world.get_physics_dt()}")

    def _set_task(self, usd_path, n_envs, env_spacing, collision_between_envs, observation_spec, actuation_spec, 
                  additional_data_spec, collision_groups, physics_material_spec, camera_position, camera_target):
        """Set up the simulation task."""
        from mushroom_rl.environments.isaac_sim_task import IsaacSimTask

        self._task = IsaacSimTask(
            self._physics_context, usd_path, n_envs, env_spacing, observation_spec, actuation_spec, 
            self._backend, self._device, self._action_type, collision_between_envs, additional_data_spec, 
            collision_groups, physics_material_spec, camera_position, camera_target
        )
        self._world.add_task(self._task)

    def render_all(self, env_mask, record=False):
        """Render all environments. Optionally record the frames."""
        self._world.render()
        data = self._task.rgb_annot.get_data()[..., :3]

        if self._viewer is None:
            self._viewer = ImageViewer((1280, 720), 0)
        self._viewer.display(data)

        if record:
            return data

    def step_all(self, env_mask, action):
        arr_backend = ArrayBackend.get_array_backend(self._mdp_info.backend)

        cur_obs = self._obs.clone().detach()

        action = self._preprocess_action(action)

        env_indices = arr_backend.where(env_mask)[0]
        self._task.teleport_away(arr_backend.where(env_mask == False)[0])

        ctrl_action = None

        for _ in range(self._n_intermediate_steps):
            if self._recompute_action_per_step or ctrl_action is None:
                ctrl_action = self._compute_action(cur_obs, action)

            self._simulation_pre_step()

            self._task.apply_action(ctrl_action[env_indices], env_indices)
            self._world.step(render=not self._headless)

            self._simulation_post_step()

            if self._recompute_action_per_step:
                cur_obs = self.observation_helper.build_obs(self._task.get_observations(clone=False))
                cur_obs = self._create_observation(cur_obs)

        if not self._recompute_action_per_step:
            cur_obs = self.observation_helper.build_obs(self._task.get_observations(clone=False))
            cur_obs = self._create_observation(cur_obs)

        self._step_finalize(env_indices)

        absorbing = self.is_absorbing(cur_obs)
        reward = self.reward(self._obs, action, cur_obs, absorbing)
        extra_info = self._create_info_dictionary(cur_obs)

        self._obs = cur_obs.clone().detach()

        cur_obs = self._modify_observation(cur_obs)
        
        return cur_obs.clone().detach(), reward.clone().detach(), torch.logical_and(absorbing, env_mask).clone().detach(), extra_info
    
    def reset_all(self, env_mask, state=None):
        arr_backend = ArrayBackend.get_array_backend(self._mdp_info.backend)
        env_indices = arr_backend.where(env_mask)[0]

        self._task.reset_env(env_indices)
        self.setup(env_indices, state)
        
        obs = self.observation_helper.build_obs(self._task.get_observations(clone=False))
        obs = self._create_observation(obs)
        self._obs = obs.clone().detach()

        info = self._create_info_dictionary(obs)
        obs = self._modify_observation(obs)

        return obs.clone().detach(), info
    
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

    def seed(self, seed=-1):
        from omni.isaac.core.utils.torch.maths import set_seed
        return set_seed(seed)
    
    def stop(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        #self._world.reset() #leads sometimes to an illegal cuda memory access
        #self._task.reset_env(list(range(self.number)))
        self._world.reset(soft=True)

    def __del__(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._simulation_app.close()

    @property
    def dt(self):
        return self._timestep * self._n_intermediate_steps * self._n_substeps
    
    def _check_collision(self, group1, group2, threshold=0., selector=None, dt=1.):
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
        if selector:
            return self._task.collision_helper.check_collision(group1, group2, threshold, selector=selector, dt=dt)
        else:
            return self._task.collision_helper.check_collision(group1, group2, threshold, dt=dt)

    def _get_collision_force(self, group1, group2, selector=None, dt=1.):
        """
        Returns the collision force or impulse between the specified groups.

        Args:
            group1 (string): A name referring to an entry contained in the
                collision_groups list handed to the constructor;
            group2 (string): A name referring to an entry contained in the
                collision_groups list handed to the constructor.
            selector (Callable[[torch.tensor | np.ndarray], torch.tensor | np.ndarray]): 

        Returns:
            A 3D vector specifying the collision forces
        """
        if selector:
            return self._task.collision_helper.get_collision_force(group1, group2, selector, dt)
        else:
            return self._task.collision_helper.get_collision_force(group1, group2, dt=dt)
    
    def _get_collision_count(self, group1, group2, threshold=0., selector=None, dt=1.):
        if selector:
            return self._task.collision_helper.count_collisions(group1, group2, threshold, selector=selector, dt=dt)
        else:
            return self._task.collision_helper.count_collisions(group1, group2, threshold, dt=dt)
    
    def _read_data(self, name, env_indices=None):
        return self._task.read_data(name, env_indices)

    def _write_data(self, name, value, env_indices=None):
        self._task.write_data(name, value, env_indices)

    def _set_joint_data(self, value, type, joint_indices=None, env_indices=None):
        assert type == ObservationType.JOINT_POS or type == ObservationType.JOINT_VEL
        self._task.set_joint_data(value, type, joint_indices, env_indices)

    # callbacks ------------------------------------------------------------------------------------------

    def _create_observation(self, obs):
        """
        This method can be overridden to create a custom observation. Should be used to append observation which have
        been registered via observation_helper.add_obs(self, name, length, min_value, max_value)

        Args:
            obs (np.ndarray, torch.tensor): the generated observation

        Returns:
            The environment observation.

        """
        return obs

    def _modify_observation(self, obs):
        """
        This method can be overridden to edit the created observation. This is done after the reward and absorbing
        functions are evaluated. Especially useful to transform the observation into different frames. If the original
        observation order is not preserved, the helper functions in ObervationHelper breaks.

        Args:
            obs (np.ndarray, torch.tensor): the generated observation

        Returns:
            The environment observation.

        """
        return obs
    
    def _compute_action(self, obs, action):
        """
        Compute a transformation of the action at every intermediate step.
        Useful to add control signals simulated directly in python.

        Args:
            obs (np.ndarray, torch.tensor): current state of the simulation;
            action (np.ndarray, torch.tensor): the actions, provided at every step.

        Returns:
            The action to be set in the actual pybullet simulation.

        """
        return action
        
    def _preprocess_action(self, action):
        """
        Compute a transformation of the action provided to the
        environment.

        Args:
            action (np.ndarray, torch.tensor): the actions provided to the environment.

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
            obs (np.ndarray, torch.tensor): the generated observation

        Returns:
            The information dictionary.

        """
        return {}
    
    def _simulation_pre_step(self):
        """
        Allows information to be accesed and changed at every intermediate step
        before taking a step in the isaac sim simulation.
        Can be usefull to apply an external force/torque to the specified bodies.
        """
        pass

    def _simulation_post_step(self):
        """
        Allows information to be accesed at every intermediate step
        after taking a step in the isaac sim simulation.
        Can be usefull to average forces over all intermediate steps.

        """
        pass

    def _step_finalize(self, env_indices):
        """
        Allows information to be accesed at the end of a step.
        """
        pass