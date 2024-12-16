import numpy as np
import torch
import hydra
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.stage import add_reference_to_stage, print_stage_prim_paths
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.cloner import GridCloner
from omni.usd import get_context
from pxr import UsdGeom, PhysxSchema, UsdPhysics, Sdf
from omni.isaac.core.prims import RigidPrimView

from omni.physx.scripts.physicsUtils import *
from omni.physx import get_physx_simulation_interface

from mushroom_rl.environments.isaac_sim_env import ActionType #TODO
from mushroom_rl.utils.isaac_sim import ObservationType, CollisionHelper
from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.utils import TorchUtils

from omni.isaac.core.utils.types import ArticulationActions
from omni.isaac.core.robots.robot import Robot

class IsaacSimTask(BaseTask):
    BASE_ENV_PATH = "/World/envs"
    TEMPLATE_ENV_PATH = BASE_ENV_PATH + "/env"
    ZERO_ENV_PATH = TEMPLATE_ENV_PATH + "_0"

    def __init__(self, physic_context, usd_path, num_envs, env_spacing, collision_between_envs, observation_spec, 
                 action_spec, additional_data_spec, collision_groups, backend, action_type, intermediate_steps, device):
        self.usd_path = usd_path
        self._physic_context = physic_context
        self._num_envs = num_envs
        self._env_spacing = env_spacing
        self._collisions_between_envs = collision_between_envs
        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self._additional_data_spec = additional_data_spec
        self._backend = backend
        self._device = device
        self._action_type = action_type

        self.collision_helper = CollisionHelper(collision_groups, backend, num_envs, device)

        super().__init__("CustomNameTask")#TODO
    
    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        self._views = {}

        #Define env_0
        add_reference_to_stage(self.usd_path, self.ZERO_ENV_PATH + "/Robot")

        #clone env_0
        self._cloner = GridCloner(spacing=self._env_spacing)#check with automatic spacing
        self._cloner.define_base_env(self.BASE_ENV_PATH)

        stage = get_context().get_stage()
        UsdGeom.Xform.Define(stage, self.ZERO_ENV_PATH)

        prim_paths = self._cloner.generate_paths(self.TEMPLATE_ENV_PATH, self._num_envs)
        self.env_pos = self._cloner.clone(
            source_prim_path=self.ZERO_ENV_PATH, 
            prim_paths=prim_paths, 
            replicate_physics=True, 
            copy_from_source=False #Faster, but changes made to source prim will also reflect in the cloned prims
        )
        self.env_pos = ArrayBackend.convert(self.env_pos, to=self._backend)
        
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

        scene.add_default_ground_plane()
        
        self._views[""] = self.robots

        #register view
        self._views.update(self.collision_helper.set_up(scene, stage))

        if self._additional_data_spec is None:
            specifications = self._observation_spec 
        else:
            specifications = self._observation_spec + self._additional_data_spec
        
        for name, path, obs_type in specifications:
            if obs_type.is_body() and path not in self._views:
                view = RigidPrimView(
                    prim_paths_expr=self.BASE_ENV_PATH + "/.*/Robot" + path,
                    name=path.replace("/", "_") + "_view",
                    reset_xform_properties=False,
                )
                scene.add(view)
                self._views[path] = view
    
    def get_observations(self, clone=True):
        obs = {}
        for name, (view, obs_type, joint_index) in self._observers.items():
            obs[name] = self._read_property(view, obs_type, joint_indices=joint_index, clone=clone)
        return obs
    
    def apply_action(self, action, env_indices=None):
        kwargs = {'joint_indices': self._controlled_joints, self._action_type.value: action}
        art_action = ArticulationActions(**kwargs)
        self.robots.apply_action(art_action, indices=env_indices)

    def get_observation_limits(self):
        obs_low = []
        obs_high = []
        obs = self.get_observations()

        for name, (_, obs_type, joint_index) in self._observers.items():
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
    
    def get_joint_max_efforts(self):
        return self.robots.get_max_efforts(indices=[0], joint_indices=self._controlled_joints)[0].to(self._device)
    
    def get_joint_pos_limits(self):
        return self.robots.get_dof_limits()[0].to(self._device)[self._controlled_joints].T
    
    def get_joint_max_velocities(self):
        return self.robots.get_joint_max_velocities(indices=[0], joint_indices=self._controlled_joints, clone=True)[0]

    def get_action_limits(self):
        if self._action_type == ActionType.EFFORT:
            limit = self.get_joint_max_efforts()
            return -limit, limit
        elif self._action_type == ActionType.POSITION:
            limit = self.get_joint_pos_limits()
            return limit[0], limit[1]
        else:
            limit = self.get_joint_max_velocities()
            return -limit, limit
    
    def reset_env(self, env_indices, state=None):
        joints_defaults = self.robots.get_joints_default_state()
        dof_pos = joints_defaults.positions[env_indices]
        dof_vel = joints_defaults.velocities[env_indices]
        dof_eff = joints_defaults.efforts[env_indices]

        self.robots.set_joint_positions(dof_pos, indices=env_indices)
        self.robots.set_joint_velocities(dof_vel, indices=env_indices)
        self.robots.set_joint_efforts(dof_eff, indices=env_indices)

        default_state = self.robots.get_default_state()
        default_positions = default_state.positions[env_indices]
        default_orientations = default_state.orientations[env_indices]
        self.robots.set_world_poses(default_positions, default_orientations, indices=env_indices)
        velocity = ArrayBackend.get_array_backend(self._backend).zeros((len(env_indices), 6))
        self.robots.set_velocities(velocity, indices=env_indices)

    def post_reset(self):
        """
        Called as the last step when resetting the world. 
        """
        self._controlled_joints = []
        for joint_name in self._action_spec:
            joint_index = self.robots.get_dof_index(joint_name)
            self._controlled_joints.append(joint_index)
        self._controlled_joints = torch.tensor(self._controlled_joints, device=self._device)

        self._observers = {}
        for name, path, obs_type in self._observation_spec:
            if obs_type.is_joint():
                view = self.robots
                joint_name = path.split('/')[-1]
                joint_index = self.robots.get_dof_index(joint_name)
                joint_index = ArrayBackend.get_array_backend(self._backend).from_list([joint_index])
            else:
                view = self._views[path]
                joint_index = None

            self._observers[name] = (view, obs_type, joint_index)

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

        #self.contact_view.initialize()

    def _set_property(self, view, obs_type, value, joint_indices=None, env_indices=None):
        """
        Will set values immediately
        """
        if obs_type == ObservationType.BODY_POS:
            pos = value + self.env_pos[env_indices]
            view.set_world_poses(positions=pos, indices=env_indices)
        elif obs_type == ObservationType.BODY_ROT:
            view.set_world_poses(orientations=value, indices=env_indices)
        elif obs_type == ObservationType.BODY_LIN_VEL:
            view.set_linear_velocities(value, indices=env_indices)
        elif obs_type == ObservationType.BODY_ANG_VEL:
            view.set_angular_velocities(value, indices=env_indices)
        elif obs_type == ObservationType.JOINT_POS:
            if joint_indices is None:
                joint_indices = self._controlled_joints
            view.set_joint_positions(value, indices=env_indices, joint_indices=joint_indices)
        elif obs_type == ObservationType.JOINT_VEL:
            if joint_indices is None:
                joint_indices = self._controlled_joints
            view.set_joint_velocities(value, indices=env_indices, joint_indices=joint_indices)
        elif obs_type == ObservationType.BODY_VEL:
            view.set_velocities(value, indices=env_indices)

    def _read_property(self, view, obs_type, joint_indices=None, env_indices=None, clone=True):
        if obs_type == ObservationType.BODY_POS:
            return view.get_world_poses(indices=env_indices, clone=clone)[0] - self.env_pos
        elif obs_type == ObservationType.BODY_ROT:
            return view.get_world_poses(indices=env_indices, clone=clone)[1]
        elif obs_type == ObservationType.BODY_LIN_VEL:
            return view.get_velocities(indices=env_indices, clone=clone)[:, :3]
        elif obs_type == ObservationType.BODY_ANG_VEL:
            return view.get_velocities(indices=env_indices, clone=clone)[:, 3:]
        elif obs_type == ObservationType.JOINT_POS:
            if joint_indices is None:
                joint_indices = self._controlled_joints
            return view.get_joint_positions(indices=env_indices, joint_indices=joint_indices, clone=clone)
        elif obs_type == ObservationType.JOINT_VEL:
            if joint_indices is None:
                joint_indices = self._controlled_joints
            return view.get_joint_velocities(indices=env_indices, joint_indices=joint_indices, clone=clone)
        elif obs_type == ObservationType.BODY_VEL:
            view.get_velocities(indices=env_indices, clone=clone)

    def write_data(self, name, value, env_indices=None):
        if name in self._additionals:
            view, obs_type, joint_index = self._additionals[name]
        else:
            view, obs_type, joint_index = self._observers[name]
        self._set_property(view, obs_type, value, joint_indices=joint_index, env_indices=env_indices)

    def read_data(self, name, env_indices=None):
        if name in self._additionals:
            view, obs_type, joint_index = self._additionals[name]
        else:
            view, obs_type, joint_index = self._observers[name]
        return self._read_property(view, obs_type, joint_indices=joint_index, env_indices=env_indices)
    
    def set_joint_data(self, value, type, joint_indices=None, env_indices=None):
        self._set_property(self.robots, type, value, joint_indices, env_indices)
