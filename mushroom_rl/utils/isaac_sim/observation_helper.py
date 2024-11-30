import numpy as np
from enum import Enum
from mushroom_rl.core.array_backend import ArrayBackend


class ObservationType(Enum):
    BODY_POS = (0, 'body', 3)
    BODY_ROT = (1, 'body', 4)
    BODY_LIN_VEL = (2, 'body', 3)
    BODY_ANG_VEL = (3, 'body', 3)
    JOINT_POS = (4, 'joint', 1)
    JOINT_VEL = (5, 'joint', 1)

    def __init__(self, id, category, length):
        self.category = category
        self.length = length

    def is_body(self):
        return self.category == 'body'

    def is_joint(self):
        return self.category == 'joint'


class ObservationHelper:
    def __init__(self, observation_spec, observation_limits, backend, num_env, device):
        self._observation_spec = observation_spec
        self._backend = backend
        self._num_env = num_env
        self._device = device

        self._obs_low = observation_limits[0]
        self._obs_high = observation_limits[1]
        self.obs_idx_map = self._compute_obs_idx_map()
        self.obs_types_idx_map = self._compute_type_idx_map()

    def build_obs(self, data):
        size = self.obs_length
        obs = ArrayBackend.get_array_backend(self._backend).empty((self._num_env, size), self._device)

        for name, indices in self.obs_idx_map.items():
            if name in data:
                obs[:, indices] = data[name]

            
        return obs

    def get_from_obs(self, obs, name):
        indices = self.obs_idx_map[name]
        return obs[:, indices]

    def get_by_type_from_obs(self, obs, obs_type):
        indices = self.obs_types_idx_map[obs_type]
        return obs[:, indices]

    def add_obs(self, name, length, min_value, max_value):
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
        index = 0
        mapping = {}
        for name, _, obs_type in self._observation_spec:
            mapping[name] = list(range(index, index + obs_type.length))
            index += obs_type.length
        return mapping
    
    def _compute_type_idx_map(self):
        mapping = {}
        for obs_type in ObservationType.__members__.values():
            names = [x[0] for x in self._observation_spec if x[2] == obs_type]
            indices = []
            for name in names:
                indices.extend(self.obs_idx_map[name])
            mapping[obs_type] = indices
        return mapping

    @property
    def obs_limits(self):
        return self._obs_low, self._obs_high
    
    @property
    def obs_length(self):
        return max(map(lambda x: x[-1], self.obs_idx_map.values())) + 1


