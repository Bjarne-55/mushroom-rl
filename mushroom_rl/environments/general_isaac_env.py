from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False}) 

from mushroom_rl.core import Environment
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationView
import numpy as np
from omni.isaac.core.tasks import BaseTask
from omni.isaac.cloner import GridCloner
from omni.isaac.core.robots.robot import Robot
import omni.usd
from pxr import Gf, UsdGeom, UsdLux
from enum import Enum
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import RigidPrimView

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


class GeneralIsaac(Environment):#Change to IsaacSim maybe Env
    # TODO add all relevant varibales from mujoco Constructur
    # TODO think about tasks
    def __init__(self, usd_path, action_spec, observation_spec, backend, device, collision_between_envs, num_envs, env_spacing):
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
            num_envs (int): Number of parallel environments
            env_spacing (int): Distance between environments
        """
        self._model = self.load_model(usd_path, num_envs, env_spacing, collision_between_envs, observation_spec, action_spec)
        self._backend = backend
        self._device = device


    def load_model(self, usd_path, num_envs, env_spacing, collision_between_envs, observation_spec, action_spec):
        from omni.isaac.core.world import World
        self.world = World(**{"physics_dt": 1.0 / 60.0, "stage_units_in_meters": 1.0, "rendering_dt": 1.0 / 60.0})

        self._task = GeneralTask("CustomName", self, usd_path, num_envs, env_spacing, collision_between_envs, observation_spec)
        self.world.add_task(self._task)
        self.world.reset()
        self._task.set_controlled_joints(action_spec)

    def step(self, action):
        self._task.apply_action(action)
        self.world.step(render=True)
        obs = self._task.get_observation(clone=True)
        return obs

class GeneralTask(BaseTask):
    BASE_ENV_PATH = "/World/envs"
    TEMPLATE_ENV_PATH = BASE_ENV_PATH + "/env"
    ZERO_ENV_PATH = TEMPLATE_ENV_PATH + "_0"

    def __init__(self, name, env, usd_path, num_envs, env_spacing, collision_between_envs, observation_spec):
        self.usd_path = usd_path
        self._env = env
        self._num_envs = num_envs
        self._env_spacing = env_spacing
        self._collisions_between_envs = collision_between_envs
        self._observation_spec = observation_spec
        self._views = []

        super().__init__(name, env)
    
    def set_up_scene(self, scene):
        super().set_up_scene(scene)

        scene.add_default_ground_plane()

        #Define env_0
        add_reference_to_stage(self.usd_path, self.ZERO_ENV_PATH + "/Robot")
        Robot(#used for translation, TODO replace it
            prim_path=self.ZERO_ENV_PATH + "/Robot", 
            name="Robot", 
            translation=np.array([0.0, 0, 2.0])
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
        
        #handle collisions between environments
        if not self._collisions_between_envs:
            self._cloner.filter_collisions(
                self._env.world.get_physics_context().prim_path,
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
        for name, path, obs_type in self._observation_spec:
            if obs_type.is_body():
                view = RigidPrimView(
                    prim_paths_expr=self.BASE_ENV_PATH + "/.*/Robot" + path,
                    name=path.replace("/", "_") + "_view",
                    reset_xform_properties=False
                )
                scene.add(view)
                self._views.append(view)
            else:
                self._views.append(self.robots)

    def set_controlled_joints(self, action_spec):
        self._controlled_joints = []
        for joint_name in action_spec:
            joint_index = self.robots.get_dof_index(joint_name)
            self._controlled_joints.append(joint_index)

    
    def get_observation(self, clone):
        obs = {}
        for (name, path, obs_type), view in zip(self._observation_spec, self._views):
            if obs_type.is_joint():
                joint_name = path.split('/')[-1]
                joint_index = self.robots.get_dof_index(joint_name)

            if obs_type == ObservationType.BODY_POS:
                obs[name] = view.get_world_poses(clone=clone)[0] - self._env_pos
            elif obs_type == ObservationType.BODY_ROT:
                obs[name] = view.get_world_poses(clone=clone)[1]
            elif obs_type == ObservationType.BODY_LIN_VEL:
                obs[name] = view.get_velocities(clone=clone)[:, :3]
            elif obs_type == ObservationType.BODY_ANG_VEL:
                obs[name] = view.get_velocities(clone=clone)[:, 3:]
            elif obs_type == ObservationType.JOINT_POS:
                obs[name] = view.get_joint_positions(joint_indices=joint_index, clone=clone)
            elif obs_type == ObservationType.JOINT_VEL:
                obs[name] = view.get_joint_velocities(joint_indices=joint_index, clone=clone)
        return obs
    
    def apply_action(self, action):
        self.robots.set_joint_efforts(action, joint_indices=self._controlled_joints)


if __name__ == '__main__':
    usd_path = "/home/bjarne/GitWorkspace/BachelorThesis/mushroom-rl/isaac_assets/cartpole.usd"
    obs_spec = [
        ("cart", "/cart", ObservationType.BODY_POS),
        ("cartJointVel", "/cartJoint", ObservationType.JOINT_VEL),
        ("cartJointPos", "/cartJoint", ObservationType.JOINT_POS)
    ]
    act_spec = ["cartJoint"]
    num_envs = 3
    env = GeneralIsaac(usd_path, act_spec, obs_spec, "torch", "gpu", False, num_envs, 7)
    while True:
        action = np.random.rand(num_envs, 1) * 10 - 5
        obs = env.step(action)
        for name, _, _ in obs_spec:
            print(f"{name}: {obs[name]}")