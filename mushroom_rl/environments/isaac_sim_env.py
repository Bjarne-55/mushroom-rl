import numpy as np
import torch
import random

from mushroom_rl.core import VectorizedEnvironment, MDPInfo, ArrayBackend
from mushroom_rl.rl_utils.spaces import Box
from mushroom_rl.utils import TorchUtils
from mushroom_rl.utils.viewer import ImageViewer
from mushroom_rl.utils.isaac_sim import ObservationHelper, ObservationType, ActionType

class IsaacSim(VectorizedEnvironment):
    """
    Class to create a Mushroom environment using the Isaac Sim simulator.
    """

    def __init__(self, usd_path, actuation_spec, observation_spec, backend, device, collision_between_envs, 
                 num_envs, env_spacing, gamma, horizon, timestep=None, n_substeps=1, n_intermediate_steps=1, 
                 additional_data_spec=None, collision_groups=None, action_type=ActionType.EFFORT, headless=True,
                 physics_material_spec=None, sim_params=None, camera_position=(5, 0, 4), camera_target=(0, 0, 0)):
        """
        Constructor.

        Args:
            usd_path (str): Path to usd file of the robot.
            actuation_spec (list): A list specifying the names of the joints  which should be controllable by the
               agent.
            observation_spec (list): A list containing the names of data that should be made available to the agent as
               an observation and their type (ObservationType). They are combined with a path, which is used to access the prim,
               and a list or a single string with name of the subelements of prim which should be accessed. For example a subbody 
               or a joint of an ArticulationView
               An entry in the list is given by: (key, name, type, element). The name can later be used to retrieve
               specific observations.
            backend (str): Backend for array operations.
            device (str): Compute device (e.g., 'cuda:0').
            collision_between_envs (bool): Whether inter-environment collisions are allowed.
            num_envs (int): Number of parallel environments.
            env_spacing (float): Distance between environments.
            gamma (float): The discounting factor of the environment.
            horizon (int): The maximum horizon for the environment.
            timestep (float, None): Simulation timestep.
            n_substeps (int, None): Number of substeps per simulation step. If None default timestep 
                of isaac sim is used
            n_intermediate_steps (int): Number of intermediate control steps. Defaults to 1.
            additional_data_spec (list, None): A list containing the data fields of interest, which should be read from
               or written to during simulation. The entries are given as the following tuples: (key, path, type) key
               is a string for later referencing in the "read_data" and "write_data" methods.
            collision_groups (dict, None): A list containing groups of prims for which collisions should be checked during
                simulation. The entries are given as ``(key, prim_paths)``, where key is a string for later reference and 
                prim_paths is a list of paths to the prims.
            action_type (ActionType): Control type of the joints (effort, position, velocity).
            headless (bool): Whether to run in headless mode.
            physics_material_spec (list, None): A list containing all data to create a custom physics material for each environment, which 
                will be applied to all rigidbodies. 
                The entries are given as the following tuples: (name, dynamic_friction, static_friction, restitution)
            sim_params (dict): Dictionary of simulation parameters for the physics context. 
                Intended to set gpu_collision_stack_size, gpu_found_lost_aggregate_pairs_capacity, 
                gpu_found_lost_pairs_capacity, gpu_heap_capacity, gpu_max_num_partitions, gpu_max_particle_contacts, 
                gpu_max_rigid_contact_count, gpu_max_rigid_patch_count, gpu_max_soft_body_contacts, 
                gpu_temp_buffer_capacity, gpu_total_aggregate_pairs_capacity.
            camera_position (tuple): The position where the camera is placed.
            camera_target (tuple): The position the camera is aimed at.
        """
        
        self._headless = headless
        self._simulation_app = self._create_simulation_app(headless)
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
            num_envs, 
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
        assert observation_limits[0].dtype == observation_limits[1].dtype
        observation_space = Box(*observation_limits, data_type=observation_limits[0].dtype)
        self.observation_helper = ObservationHelper(observation_spec, observation_limits, backend, num_envs, device)

        action_limits = self._task.get_action_limits()
        assert action_limits[0].dtype == action_limits[1].dtype
        action_space = Box(*action_limits, data_type=action_limits[0].dtype)

        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, self.dt, backend)
        mdp_info = self._modify_mdp_info(mdp_info)

        self._recompute_action_per_step = type(self)._compute_action != IsaacSim._compute_action
        self._obs = None #TODO rework
        
        super().__init__(mdp_info, num_envs)
    
    def _create_simulation_app(self, headless):
        from isaacsim import SimulationApp
        return SimulationApp({"headless": headless, "hide_ui": False}) 

    def _apply_carb_settings(self):
        """Apply Carb settings for optimization."""
        import carb

        carb.settings.get_settings().set("/persistent/omnihydra/useSceneGraphInstancing", True)
        carb.settings.get_settings().set_bool("/physics/physxDispatcher", True)
        carb.settings.get_settings().set("/app/viewport/grid/enabled", False)
        carb.settings.get_settings().set("/app/runLoops/main/rateLimitEnabled", False)

    def _create_world(self, timestep, custom_sim_params=None):
        """
        Create and configure the simulation world.

        Args:
            timestep (float, None): The physics timestep. the default physics timestep is used.
            custom_sim_params (dict, None): A dictionary of simulation parameters to override the default ones.
        """
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
            self._physics_context.set_physics_dt(dt=self._timestep * self._n_substeps, substeps=self._n_substeps)
        else:
            self._physics_context.set_physics_dt(dt=timestep * self._n_substeps, substeps=self._n_substeps)
            self._timestep = timestep

        self._world.set_simulation_dt(rendering_dt=self.dt)
        print(f"rendering dt: {self._world.get_rendering_dt()}, physix dt: {self._world.get_physics_dt()}")
        print(f"uses Fabric: {self._physics_context.use_fabric}, uses gpu_pipeline: {self._physics_context.use_gpu_pipeline}, use_gpu_sim: {self._physics_context.use_gpu_sim}")

    def _set_task(self, usd_path, num_envs, env_spacing, collision_between_envs, observation_spec, actuation_spec, 
                  additional_data_spec, collision_groups, physics_material_spec, camera_position, camera_target):
        """Set up the simulation task."""
        from mushroom_rl.environments.isaac_sim_task import IsaacSimTask

        self._task = IsaacSimTask(
            self._physics_context, usd_path, num_envs, env_spacing, observation_spec, actuation_spec, 
            self._backend, self._device, self._action_type, self._n_intermediate_steps, collision_between_envs, 
            additional_data_spec, collision_groups, physics_material_spec, camera_position, camera_target
        )
        self._world.add_task(self._task)

    def render_all(self, env_mask, record=False):
        """
        Render all environments. Optionally record the frames.

        Args:
            record (bool): If True, the function returns the rendered image data.
                Defaults to False.
        """
        self._world.render()
        data = self._task.rgb_annot.get_data()[..., :3]

        if self._viewer is None:
            self._viewer = ImageViewer((1280, 720), 0)
        self._viewer.display(data)

        if record:
            return data

    def step_all(self, env_mask, action):
        """
        Performs a simulation step for all active environments.

        Args:
            env_mask (torch.tensor, np.ndarray): A boolean mask indicating which environments 
                are active for this step.
            action (torch.tensor, np.ndarray): The actions to be applied to the active environments.

        Returns:
            cur_obs (torch.tensor, np.ndarray): The updated observations after the step.
            reward (torch.tensor, np.ndarray): The computed rewards for each environment.
            absorbing (torch.tensor, np.ndarray): A boolean tensor indicating if an environment is 
                in an absorbing state.
            extra_info (dict): Additional information about the simulation step.
        """
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
                cur_obs = self.observation_helper.build_obs(self._task.get_observations(clone=True))
                cur_obs = self._create_observation(cur_obs)

        if not self._recompute_action_per_step:
            cur_obs = self.observation_helper.build_obs(self._task.get_observations(clone=True))
            cur_obs = self._create_observation(cur_obs)

        self._step_finalize(env_indices)

        absorbing = self.is_absorbing(cur_obs)
        reward = self.reward(self._obs, action, cur_obs, absorbing)
        extra_info = self._create_info_dictionary(cur_obs)

        self._obs = cur_obs.clone().detach()#TODO maybe only update active envs

        cur_obs = self._modify_observation(cur_obs)
        
        return cur_obs.clone().detach(), reward.clone().detach(), arr_backend.logical_and(absorbing, env_mask).clone().detach(), extra_info
    
    def reset_all(self, env_mask, state=None):
        """
        Resets the specified environments and initializes their states.

        Args:
            env_mask (torch.tensor, np.ndarray): A boolean mask indicating which environments 
                should be reset.
            state (torch.tensor, np.ndarray): The initial state to set for the reset environments. 
                Defaults to None.
        
        Returns:
            obs (torch.tensor, np.ndarray): The observations after resetting the environments.
            info (dict): Additional information about the reset environments.
        """
        arr_backend = ArrayBackend.get_array_backend(self._mdp_info.backend)
        env_indices = arr_backend.where(env_mask)[0]

        self._task.reset_env(env_indices)
        self.setup(env_indices, state)
        
        obs = self.observation_helper.build_obs(self._task.get_observations(clone=True))
        obs = self._create_observation(obs)
        if self._obs is None:
            self._obs = obs.clone().detach()
        else:
            self._obs[env_mask] = obs.clone().detach()[env_mask]

        info = self._create_info_dictionary(obs)
        obs = self._modify_observation(obs)

        return obs.clone().detach(), info
    
    def reward(self, obs, action, next_obs, absorbing):
        """
        Compute the rewards based on the given transitions.

        Args:
            obs (torch.tensor, np.array): the current states of the parallel environments.
            action (torch.tensor, np.array): the actions that are applied in the current states.
            next_obs (torch.tensor, np.array): the states reached after applying the given
                actions.
            absorbing (torch.tensor, np.array): whether next_state is an absorbing state or not.

        Returns:
            The rewards as a array or tensor.

        """
        raise NotImplementedError

    def is_absorbing(self, obs):
        """
        Check whether the given states are an absorbing states or not.

        Args:
            obs (torch.tensor, np.array): the states of the parallel environments.

        Returns:
            A tensor of booleans indicating whether the corresponding states are absorbing or not.

        """
        raise NotImplementedError

    def setup(self, env_indices, obs):
        """
        A function that allows to execute setup code after an environment reset.
        """
        raise NotImplementedError

    def seed(self, seed=-1):
        """
        Sets the random seed for a deterministic behavior.

        Args:
            seed (int, optional): The seed value to set. If -1, a random seed is used. 
                Defaults to -1.
        
        Returns:
            int: The seed value that was set.
        """
        from omni.isaac.core.utils.torch.maths import set_seed
        return set_seed(seed)
    
    def stop(self):
        """
        Resets simulation and closes viewer.
        """
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._world.reset(soft=True)

    def __del__(self):
        """
        Ends simulation.
        """
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._simulation_app.close()

    @property
    def dt(self):
        return self._timestep * self._n_intermediate_steps * self._n_substeps
    
    def _check_collision(self, group1, group2, threshold=0., selector=None, dt=1.):
        """
        Checks whether the collision force between two collision groups exceeds a given threshold.

        Args:
            group1 (str): The name of the first collision group.
            group2 (str): The name of the second collision group.
            threshold (float, torch.tensor, np.ndarray): The threshold value to compare against.
                Can be a scalar or a tensor/array of the same shape as the computed forces.
            selector (Callable[[torch.tensor | np.ndarray], torch.tensor | np.ndarray], optional): 
                A function that processes the collision force tensor or array.
                If None, a default selector is used. Defaults to None.
            dt (float, optional): The time step duration used for computing forces. 
                The function uses impulses if the default dt is used

        Returns:
            A boolean tensor or array indicating where the computed forces exceed the given threshold.
        """
        if selector:
            return self._task.collision_helper.check_collision(group1, group2, threshold, selector=selector, dt=dt)
        else:
            return self._task.collision_helper.check_collision(group1, group2, threshold, dt=dt)

    def _get_collision_force(self, group1, group2, selector=None, dt=1.):
        """
        Computes the collision forces or impulses between two collision groups.

        Args:
            group1 (str): The name of the first collision group.
            group2 (str): The name of the second collision group.
            selector (Callable[[torch.tensor | np.ndarray], torch.Tensor | np.ndarray], optional): 
                A function that processes the collision force tensor. 
                Defaults to selecting the maximum force of each environment
            dt (float, optional): The time step duration used for computing forces. 
                The function returns contact impulses if the default dt is used

        Returns:
            A tensor or array containing the computed collision forces between the groups, 
            processed by the `selector` function.
        """
        if selector:
            return self._task.collision_helper.get_collision_force(group1, group2, selector, dt)
        else:
            return self._task.collision_helper.get_collision_force(group1, group2, dt=dt)
    
    def _get_collision_count(self, group1, group2, threshold=0., selector=None, dt=1.):
        """
        Counts the number of collisions between two groups, considering at most one collision between each 
        possible pair of objects from the two groups. 
        For example, the maximum collision count between a group with 3 objects and a group with 4 objects 
        would be 12.

        Args:
            group1 (str): The name of the first collision group.
            group2 (str): The name of the second collision group.
            threshold (float, torch.tensor, np.ndarray): The threshold value for detecting collisions.
                Can be a scalar or a tensor/array of the same shape as the computed forces.
            selector (Callable[[torch.tensor | np.ndarray], torch.tensor | np.ndarray], optional): 
                A function that processes the collision force tensor or array.
                If None, the default selector computes the norm along the collision dimension. Defaults to None.
            dt (float, optional): The time step duration used for computing forces. Defaults to 1.0.

        Returns:
            A tensor or array containing the count of collisions
        """
        if selector:
            return self._task.collision_helper.count_collisions(group1, group2, threshold, selector=selector, dt=dt)
        else:
            return self._task.collision_helper.count_collisions(group1, group2, threshold, dt=dt)
        
    def _get_net_collision_forces(self, group, dt=1.):
        return self._task.collision_helper.get_net_contact_forces(group, dt)
    
    def _read_data(self, name, env_indices=None):
        """
        Read data from isaac sim.

        Args: 
            name (str): A name referring to an entry contained in additional_data_spec or observation_spec.

        Returns:
            The desired data as a tensor or array.
        """
        return self._task.read_data(name, env_indices)

    def _write_data(self, name, value, env_indices=None):
        """
        Writes data to isaac sim.

        Args: 
            name (str): A name referring to an entry contained in additional_data_spec or observation_spec.
            value (torch.tensor, np.ndarra): The data that should be written.
        """
        self._task.write_data(name, value, env_indices)

    def _set_joint_data(self, value, type, joint_indices=None, env_indices=None):
        """
        Sets the joint properties for the specified joints and environments.

        Args:
            value (torch.tensor, np.ndarray): The value to set for the specified joint property.
            type (ObservationType): The type of joint property to be set (e.g., "position", "velocity").
            joint_indices (torch.tensor, np.ndarray, list[int], optional): The indices of the joints to update.
                If None, defaults to the controlled joints.
            env_indices (torch.tensor, np.ndarray, list[int], optional): The indices of the environments where
                the joint data should be updated. If None, applies to all environments.
        """
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
            The action to be applied in the isaac sim simulation

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