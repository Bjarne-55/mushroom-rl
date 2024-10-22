import numpy as np
import torch

from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.cloner import GridCloner
from omni.isaac.core.robots.robot import Robot
import omni.usd
from pxr import Gf, UsdGeom, UsdLux
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import RigidPrimView

from mushroom_rl.environments.isaac_sim_env import ObservationType #TODO
from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.utils import TorchUtils

class IsaacSimTask(BaseTask):
    BASE_ENV_PATH = "/World/envs"
    TEMPLATE_ENV_PATH = BASE_ENV_PATH + "/env"
    ZERO_ENV_PATH = TEMPLATE_ENV_PATH + "_0"

    def __init__(self, physic_context, usd_path, num_envs, env_spacing, collision_between_envs, observation_spec, 
                 action_spec, additional_data_spec, backend):
        self.usd_path = usd_path
        self._physic_context = physic_context
        self._num_envs = num_envs
        self._env_spacing = env_spacing
        self._collisions_between_envs = collision_between_envs
        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self._additional_data_spec = additional_data_spec
        self._backend = backend

        super().__init__("CustomNameTask")#TODO
    
    def set_up_scene(self, scene):
        super().set_up_scene(scene)

        scene.add_default_ground_plane()

        #Define env_0
        add_reference_to_stage(self.usd_path, self.ZERO_ENV_PATH + "/Robot")
        Robot(#used for translation, TODO replace it
            prim_path=self.ZERO_ENV_PATH + "/Robot", 
            name="Robot", 
            translation=torch.tensor([0.0, 0, 2.0])
        )
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, self.ZERO_ENV_PATH)

        #clone env_0
        self._cloner = GridCloner(spacing=self._env_spacing)
        self._cloner.define_base_env(self.BASE_ENV_PATH)
        prim_paths = self._cloner.generate_paths(self.TEMPLATE_ENV_PATH, self._num_envs)
        self._env_pos = self._cloner.clone(
            source_prim_path=self.ZERO_ENV_PATH, 
            prim_paths=prim_paths, 
            replicate_physics=True, 
            copy_from_source=False #Faster, but changes made to source prim will also reflect in the cloned prims
        )
        self._env_pos = ArrayBackend.convert(self._env_pos, to=self._backend)
        
        #handle collisions between environments
        if not self._collisions_between_envs:
            self._cloner.filter_collisions(
                self._physic_context.prim_path,
                "/World/collisions",
                prim_paths
            )
        
        self.robots = ArticulationView(
            prim_paths_expr= self.BASE_ENV_PATH + "/.*/Robot", 
            name="robot_view", 
            reset_xform_properties=False
        )
        scene.add(self.robots)
        
        #register view
        self._views = {}
        if self._additional_data_spec is None:
            specifications = self._observation_spec 
        else:
            specifications = self._observation_spec + self._additional_data_spec
        
        for name, path, obs_type in specifications:
            if obs_type.is_body() and path not in self._views:
                view = RigidPrimView(
                    prim_paths_expr=self.BASE_ENV_PATH + "/.*/Robot" + path,
                    name=path.replace("/", "_") + "_view",
                    reset_xform_properties=False
                )
                scene.add(view)
                self._views[path] = view
    
    def get_observation(self, clone=True):
        obs = {}
        for name, view, obs_type, joint_index in self._observers:
            obs[name] = self._read_property(view, obs_type, joint_indices=joint_index, clone=clone)
        return obs
    
    def apply_action(self, action, env_indices=None):
        self.robots.set_joint_efforts(action, indices=env_indices, joint_indices=self._controlled_joints)

    def get_observation_limits(self):
        obs_low = []
        obs_high = []
        obs = self.get_observation()

        for name, _, obs_type, joint_index in self._observers:
            obs_count = ArrayBackend.get_array_backend(self._backend).size(obs[name][0, ...])

            if obs_type == ObservationType.JOINT_POS:
                limits = self.robots.get_dof_limits().to(TorchUtils.get_device())
                obs_low.append(limits[0, joint_index, 0])
                obs_high.append(limits[0, joint_index, 1])
            elif obs_type == ObservationType.JOINT_VEL:
                zero = ArrayBackend.get_array_backend(self._backend).zeros(1)
                limit = self.robots.get_joint_max_velocities(indices=zero, joint_indices=joint_index)[0]
                obs_low.append(-limit)
                obs_high.append(limit)
            else:
                inf = ArrayBackend.get_array_backend(self._backend).inf()
                obs_low.append(ArrayBackend.get_array_backend(self._backend).full((obs_count, ), -inf))
                obs_high.append(ArrayBackend.get_array_backend(self._backend).full((obs_count, ), inf))

        obs_low = ArrayBackend.get_array_backend(self._backend).concatenate(obs_low)
        obs_high = ArrayBackend.get_array_backend(self._backend).concatenate(obs_high)

        return obs_low, obs_high
    
    def get_action_limits(self):
        limit = self.robots.get_max_efforts(indices=[0], joint_indices=self._controlled_joints)[0]
        
        for index in range(len(self._controlled_joints)):
            if limit[index] == 0:
                limit[index] = ArrayBackend.get_array_backend(self._backend).inf()

        return -limit, limit
    
    def reset_env(self, env_indices, state=None):
        joints_defaults = self.robots.get_joints_default_state()
        dof_pos = joints_defaults.positions[env_indices]
        dof_vel = joints_defaults.velocities[env_indices]
        dof_eff = joints_defaults.efforts[env_indices]

        self.robots.set_joint_positions(dof_pos, indices=env_indices)
        self.robots.set_joint_velocities(dof_vel, indices=env_indices)
        self.robots.set_joint_efforts(dof_eff, indices=env_indices)

    def cleanup(self):
        pass

    def post_reset(self):
        """
        Called as the last step when resetting the world. 
        """
        self._controlled_joints = []
        for joint_name in self._action_spec:
            joint_index = self.robots.get_dof_index(joint_name)
            self._controlled_joints.append(joint_index)

        self._observers = []
        for name, path, obs_type in self._observation_spec:
            if obs_type.is_joint():
                view = self.robots
                joint_name = path.split('/')[-1]
                joint_index = self.robots.get_dof_index(joint_name)
                joint_index = ArrayBackend.get_array_backend(self._backend).from_list([joint_index])
            else:
                view = self._views[path]
                joint_index = None

            self._observers.append((name, view, obs_type, joint_index))

        self._additionals = {}
        for name, path, obs_type in self._additional_data_spec:
            if obs_type.is_joint():
                view = self.robots
                joint_name = path.split('/')[-1]
                joint_index = self.robots.get_dof_index(joint_name)
                joint_index = ArrayBackend.get_array_backend(self._backend).from_list([joint_index])
            else:
                view = self._views[path]
                joint_index = None
            self._additionals[name] = (view, obs_type, joint_index)

    def _set_property(self, view, obs_type, value, joint_indices=None, env_indices=None):
        """
        Will set values immediately
        """
        if obs_type == ObservationType.BODY_POS:
            pos = value + self._env_pos[env_indices]
            view.set_world_poses(positions=pos, indices=env_indices)
        elif obs_type == ObservationType.BODY_ROT:
            view.set_world_poses(orientations=value, indices=env_indices)
        elif obs_type == ObservationType.BODY_LIN_VEL:
            view.set_linear_velocities(value, indices=env_indices)
        elif obs_type == ObservationType.BODY_ANG_VEL:
            view.set_angular_velocities(value, indices=env_indices)
        elif obs_type == ObservationType.JOINT_POS:
            view.set_joint_positions(value, indices=env_indices, joint_indices=joint_indices)
        elif obs_type == ObservationType.JOINT_VEL:
            view.set_joint_velocities(value, indices=env_indices, joint_indices=joint_indices)

    def _read_property(self, view, obs_type, joint_indices=None, env_indices=None, clone=True):
        if obs_type == ObservationType.BODY_POS:
            return view.get_world_poses(indices=env_indices, clone=clone)[0] - self._env_pos
        elif obs_type == ObservationType.BODY_ROT:
            return view.get_world_poses(indices=env_indices, clone=clone)[1]
        elif obs_type == ObservationType.BODY_LIN_VEL:
            return view.get_velocities(indices=env_indices, clone=clone)[:, :3]
        elif obs_type == ObservationType.BODY_ANG_VEL:
            return view.get_velocities(indices=env_indices, clone=clone)[:, 3:]
        elif obs_type == ObservationType.JOINT_POS:
            return view.get_joint_positions(indices=env_indices, joint_indices=joint_indices, clone=clone)
        elif obs_type == ObservationType.JOINT_VEL:
            return view.get_joint_velocities(indices=env_indices, joint_indices=joint_indices, clone=clone)

    def write_data(self, name, value, env_indices=None):
        view, obs_type, joint_index = self._additionals[name]
        self._set_property(view, obs_type, value, joint_indices=joint_index, env_indices=env_indices)

    def read_data(self, name, env_indices=None):
        view, obs_type, joint_index = self._additionals[name]
        return self._read_property(view, obs_type, joint_indices=joint_index, env_indices=env_indices)


    
