import numpy as np
import torch
from enum import Enum
from mushroom_rl.core.array_backend import ArrayBackend


class ObservationType(Enum):
    """
    Enumeration for different types of observations in a simulation.

    Each observation type is represented as a tuple containing:
    - A unique identifier (`id`).
    - A category (`category`), either 'body' or 'joint'.
    - The length of the observation (`length`).
    """

    BODY_POS = (0, 'body', 3)
    BODY_ROT = (1, 'body', 4)
    BODY_LIN_VEL = (2, 'body', 3)
    BODY_ANG_VEL = (3, 'body', 3)
    BODY_VEL = (4, 'body', 6) #combination of lin_vel and ang_vel
    JOINT_POS = (5, 'joint', 1)
    JOINT_VEL = (7, 'joint', 1)

    def __init__(self, id, category, length):
        """
        Constructor.

        Args:
            id (int): A unique identifier to ensure each tuple is distinct
            category (str): The category of the observation type, either 'body' or 'joint'.
            length (int): The length of the observation, representing the number of values 
                associated with this observation type.
        """
        self.category = category
        self.length = length

    def is_body(self):
        """
        Checks whether the observation type belongs to the 'body' category.

        Returns:
            bool: True if the observation type is category 'body', False otherwise.
        """
        return self.category == 'body'

    def is_joint(self):
        """
        Checks whether the observation type belongs to the 'joint' category.

        Returns:
            bool: True if the observation type is is category 'joint', False otherwise.
        """
        return self.category == 'joint'


class ObservationHelper:
    """
    A helper class for managing and processing observation data.
    """

    def __init__(self, observation_spec, observation_limits, backend, num_env, device):
        """
        Constructor.

        Args: 
            observation_spec (list): A list containing the names of data that should be made available to the agent as
               an observation and their type (ObservationType). They are combined with a path, which is used to access
               the data. An entry in the list is given by: (key, name, type). The name can later be used to retrieve
               specific observations.
            observation_limits (torch.tensor, np.ndarray): A tensor or array containing the maximum and minimum value
                for all observations in observation_spec. First element are the minimum values and second element the
                maximum values.
            backend (str): name of the backend for array operations.
            n_envs (int): Number of parallel environments.
            device (str): Compute device (e.g., 'cuda:0').
        """
        self._observation_spec = observation_spec
        self._obs_low = observation_limits[0]
        self._obs_high = observation_limits[1]

        self._backend = backend
        self._num_env = num_env
        self._device = device

        self.obs_idx_map = self._compute_obs_idx_map()
        self.obs_types_idx_map = self._compute_type_idx_map()

    def build_obs(self, data):
        """
        Constructs a tensor or array for all observations and fills it with the data of the dictionary.

        Args:
            data (dict): A dictionary where keys are obseravtions names and values are their corresponding
                data arrays or tensors. Should contain all data for all observations specified in 
                observation_spec
        
        Returns:
            An array or tensor of shape (num_env, combined length of all observations), containing all
            data given in the dictionary.
        """
        size = self.obs_length
        obs = ArrayBackend.get_array_backend(self._backend).empty((self._num_env, size), self._device)

        for name, indices in self.obs_idx_map.items():
            if name in data:
                obs[:, indices] = data[name]

        return obs

    def get_from_obs(self, obs, name):
        """
        Retrieves the data from obs that corresponds to the observation specified by name.

        Args:
            obs (torch.tensor, np.ndarray): The observation tensor or array.
            name (str): The name specifing the specific observation to extract.

        Returns:
            A tensor or array containing the specified observation data
        """
        indices = self.obs_idx_map[name]
        return obs[:, indices]

    def get_by_type_from_obs(self, obs, obs_type):
        """
        Retrieves all data from obs of a specific observation type.

        Args:
            obs (torch.tensor): The observation tensor or array.
            obs_type (ObservationType): The type of observation to retrieve.

        Returns:
            A tensor or array containing the specified observation data
        """
        indices = self.obs_types_idx_map[obs_type]
        return obs[:, indices]

    def add_obs(self, name, length, min_value, max_value):
        """
        Adds a new observation to the observation space.

        Args:
            name (str): The name of the new observation
            length (int): The length of the observation
            min_value (float, np.ndarray, torch.tensor): The lower bound for the observation.
            max_value (float, np.ndarray, torch.tensor): The upper bound for the observation
        """
        array_backend = ArrayBackend.get_array_backend(self._backend)
        idx = self.obs_length
        self.obs_idx_map[name] = list(range(idx, idx + length))

        if hasattr(min_value, "__len__"): 
            low = ArrayBackend.convert(min_value, to=self._backend)
        else:
            low = array_backend.full((length, ), min_value)
        self._obs_low = array_backend.concatenate([self._obs_low, low])

        if hasattr(max_value, "__len__"): 
            high = ArrayBackend.convert(max_value, to=self._backend)
        else:
            high = array_backend.full((length, ), max_value)
        self._obs_high = array_backend.concatenate([self._obs_high, high])
    
    def remove_obs_idx(self, name, index):#TODO maybe add
        pass

    def _compute_obs_idx_map(self):
        """
        Computes a mapping from observation names to index ranges.

        Returns:
            A dictionary mapping observation names to lists of index values.
        """
        index = 0
        mapping = {}
        for name, _, obs_type in self._observation_spec:
            mapping[name] = list(range(index, index + obs_type.length))
            index += obs_type.length
        mapping = {key: torch.tensor(value, device=self._device) for key, value in mapping.items()}
        return mapping
    
    def _compute_type_idx_map(self):
        """
        Computes a mapping from observation types to their corresponding indices.

        Returns:
            A dictionary mapping ObservationType instances to lists of index values.
        """
        mapping = {}
        for obs_type in ObservationType.__members__.values():
            names = [x[0] for x in self._observation_spec if x[2] == obs_type]
            indices = []
            for name in names:
                indices.extend(self.obs_idx_map[name])
            mapping[obs_type] = indices
        mapping = {key: torch.tensor(value, device=self._device) for key, value in mapping.items()}
        return mapping

    @property
    def obs_limits(self):
        return self._obs_low, self._obs_high
    
    @property
    def obs_length(self):
        return max(map(lambda x: x[-1], self.obs_idx_map.values())) + 1


