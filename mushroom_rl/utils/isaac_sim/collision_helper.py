from mushroom_rl.core import ArrayBackend
import torch
from functools import reduce

class CollisionHelper:
    BASE_ENV_PATH = "/World/envs"
    TEMPLATE_ENV_PATH = BASE_ENV_PATH + "/env"
    ZERO_ENV_PATH = TEMPLATE_ENV_PATH + "_0"

    def __init__(self, collision_groups, backend, num_envs, device):
        self._backend = backend
        self._device = device
        self._num_envs = num_envs
        self.collision_groups = {key: group for key, group in collision_groups} if collision_groups is not None else {}
        self._first_set_up = True

    def prepare_env(self, scene, stage):
        from pxr import PhysxSchema
        from omni.isaac.core.prims import RigidPrim, RigidPrimView
        for group_name, group in self.collision_groups.items():
            for path in group:
                if path.startswith("/World/"):
                    continue
                prim = stage.GetPrimAtPath(self.ZERO_ENV_PATH + "/Robot" + path)
                if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                    PhysxSchema.PhysxContactReportAPI.Apply(prim)

    def set_up(self, scene, stage):
        from omni.isaac.core.prims import RigidContactView
        self._views = {}
        self._collision_groups_indices = {}
        self._collision_group_contains_world = {key: False for key in self.collision_groups}

        for group_name, group in self.collision_groups.items():
            possible_partners = reduce(
                lambda acc, val: acc + val if val not in acc else acc, [value for key, value in self.collision_groups.items() if key != group_name], []
            )
            self._collision_groups_indices[group_name] = {key: torch.tensor([possible_partners.index(value) for value in self.collision_groups[key]], device=self._device) for key in self.collision_groups if key != group_name}
            possible_partners = [self.BASE_ENV_PATH + "/.*/Robot" + partner if not partner.startswith("/World/") else partner for partner in possible_partners]
            for path in group:#TODO check same view two times
                if path in self._views:
                    continue
                if path.startswith("/World/"):
                    self._collision_group_contains_world[group_name] = True
                    continue
                view = RigidContactView(
                    prim_paths_expr= self.BASE_ENV_PATH + "/.*/Robot" + path,
                    name=path.replace("/", "_") + "_view",
                    filter_paths_expr=possible_partners,
                    prepare_contact_sensors=False
                )
                self._views[path] = view
    
    def post_reset(self):
        if self._first_set_up:
            for path in self._views:
                self._views[path].initialize()
            self._first_set_up = False
            

    def get_collision_force(self, group1, group2, selector=lambda x: torch.max(torch.norm(x, dim=2), dim=1).values, dt=1.0):
        
        if self._collision_group_contains_world[group2]:
            prims = self.collision_groups[group1]
            indices_prims2 = self._collision_groups_indices[group1][group2]
        else:
            prims = self.collision_groups[group2]
            indices_prims2 = self._collision_groups_indices[group2][group1]
        
        forces = torch.cat([self._views[p].get_contact_force_matrix(clone=False, dt=dt)[:, indices_prims2] for p in prims], dim=1)

        return selector(forces)
    
    def check_collision(self, group1, group2, threshold, selector=lambda x: torch.max(torch.norm(x, dim=2), dim=1).values, dt=1.0):
        forces = self.get_collision_force(group1, group2, selector, dt)
        return forces > threshold
    
    def count_collisions(self, group1, group2, threshold, selector=lambda x: torch.norm(x, dim=2), dt=1.0):
        forces = self.get_collision_force(group1, group2, selector, dt)
        return torch.sum(forces > threshold, dim=1)