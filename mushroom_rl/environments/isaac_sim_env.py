from enum import Enum
import numpy as np
import torch

from isaacsim import SimulationApp

from mushroom_rl.core import VectorizedEnvironment, MDPInfo, ArrayBackend
from mushroom_rl.rl_utils.spaces import Box
from mushroom_rl.utils import TorchUtils

class ObservationType(Enum):
    __order__ = "BODY_POS BODY_ROT BODY_LIN_VEL BODY_ANG_VEL JOINT_POS JOINT_VEL"
    BODY_POS = 0
    BODY_ROT = 1
    BODY_LIN_VEL = 2
    BODY_ANG_VEL = 3
    JOINT_POS = 4
    JOINT_VEL = 5

    def is_body(self):
        return self in {
            ObservationType.BODY_POS, 
            ObservationType.BODY_ROT, 
            ObservationType.BODY_LIN_VEL, 
            ObservationType.BODY_ANG_VEL
        }

    def is_joint(self):
        return self in {
            ObservationType.JOINT_POS, 
            ObservationType.JOINT_VEL
        }

class IsaacSim(VectorizedEnvironment):
    # TODO add all relevant varibales from mujoco Constructur
    # TODO think about tasks
    def __init__(self, usd_path, action_spec, observation_spec, backend, device, collision_between_envs, 
                 n_envs, env_spacing, gamma, horizon, timestep=None, n_substeps=1, n_intermediate_steps=1, 
                 additional_data_spec=None):
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
        self._simulation_app = SimulationApp({"headless": False}) 

        self._backend = backend
        self._device = device
        TorchUtils.set_default_device(device)

        self._n_intermediate_steps = n_intermediate_steps
        self._n_substeps = n_substeps

        #create world and set task
        self._create_world(timestep)
        self._set_task(usd_path, n_envs, env_spacing, collision_between_envs, observation_spec, action_spec, 
                       additional_data_spec)
        self._world.reset()

        observation_limits = self._task.get_observation_limits()
        observation_space = Box(*observation_limits)

        action_limits = self._task.get_action_limits()
        action_space = Box(*action_limits)

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
        if timestep is None:
            self._timestep = self._world.get_physics_dt()
        else:
            self._world.set_simulation_dt(physics_dt=timestep)
            self._timestep = timestep

    def _set_task(self, usd_path, n_envs, env_spacing, collision_between_envs, observation_spec, action_spec, 
                  additional_data_spec):
        from mushroom_rl.environments.isaac_sim_task import IsaacSimTask

        self._task = IsaacSimTask(self._world.get_physics_context(), usd_path, n_envs, env_spacing, 
                                  collision_between_envs, observation_spec, action_spec, additional_data_spec, 
                                  self._backend)
        self._world.add_task(self._task)
    
    def step_all(self, env_mask, action):#TODO intermediate and substeps
        arr_backend = ArrayBackend.get_array_backend(self._mdp_info.backend)

        action = self._preprocess_action(action)

        env_indices = arr_backend.where(env_mask)[0]
        self._task.apply_action(action[env_indices], env_indices)

        self._world.step(render=True)

        cur_obs = self._task.get_observation(clone=True)
        cur_obs = arr_backend.concatenate(list(cur_obs.values()), dim=1)
        absorbing = self.is_absorbing(cur_obs)
        reward = self.reward(cur_obs, action, self._obs, absorbing)
        info = self._create_info_dictionary(cur_obs)

        self._obs = cur_obs
        
        return cur_obs, reward, torch.logical_and(absorbing, env_mask), info
    
    def reset_all(self, env_mask, state=None):
        arr_backend = ArrayBackend.get_array_backend(self._mdp_info.backend)
        env_indices = arr_backend.where(env_mask)[0]

        self._task.reset_env(env_indices, state)
        self.setup(env_indices, state)
        
        obs = self._task.get_observation(clone=True)
        obs = arr_backend.concatenate(list(obs.values()), dim=1)
        self._obs = obs

        info = self._create_info_dictionary(obs)

        return obs, info

    def seed(self, seed=-1):
        from omni.isaac.core.utils.torch.maths import set_seed
        return set_seed(seed)
    
    def stop(self):#TODO
        pass
    
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