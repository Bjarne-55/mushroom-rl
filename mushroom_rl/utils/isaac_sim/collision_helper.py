from mushroom_rl.core import ArrayBackend
import torch

class CollisionHelper:
    def __init__(self, collision_groups, num_steps, backend, num_envs, device):
        self._backend = backend
        self._device = device
        self._num_envs = num_envs
        self.collision_groups = {key: group for key, group in collision_groups} if collision_groups is not None else {}
        self.collision_dict = {}
        self.num_steps = num_steps

    def clear(self):
        self.collision_dict = {}
        self.num_steps_stored = 0
        
    def gather_collisions(self):
        from omni.physx.scripts.physicsUtils import PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        #https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/index.html#contact-reports
        collision_report = get_physx_simulation_interface().get_contact_report()
        for contact_header in collision_report[0]:
            if str(contact_header.type) != "ContactEventType.CONTACT_LOST":
                collider1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0)) # First prim involved in collision
                collider2 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1)) # Second prim involved in collision

                collider1, env1 = self._extract_name_and_env(collider1)
                collider2, env2 = self._extract_name_and_env(collider2)
                
                if env1 is None and env2 is None:
                    continue
                elif env1 is None:
                    env = env2
                elif env2 is None:
                    env = env1
                elif env1 != env2:
                    continue
                else:
                    env = env1

                if collider1 <= collider2:
                    pair = (collider1, collider2)
                else:
                    pair = (collider2, collider1)
                
                contact_data_offset = contact_header.contact_data_offset
                contact_data = collision_report[1]

                if pair not in self.collision_dict:
                    self.collision_dict[pair] = {env: [(contact_header, contact_data[contact_data_offset])]}
                elif env not in self.collision_dict[pair]:
                    self.collision_dict[pair][env] = [(contact_header, contact_data[contact_data_offset])]
                else:
                    self.collision_dict[pair][env].append((contact_header, contact_data[contact_data_offset]))

    def check_collision(self, group1, group2):#TODO add collision count
        collision_in_env = ArrayBackend.get_array_backend(self._backend).zeros(self._num_envs, dtype=bool)

        prims1 = self.collision_groups[group1]
        prims2 = self.collision_groups[group2]

        for p1 in prims1:
            for p2 in prims2:
                if p1 <= p2:
                    pair = (p1, p2)
                else:
                    pair = (p2, p1)
                
                if pair in self.collision_dict:
                    collision_in_env[list(self.collision_dict[pair].keys())] = True

        return collision_in_env
    
    def get_collision_force(self, group1, group2):# Uses last collision force found as for each env
        collision_force = ArrayBackend.get_array_backend(self._backend).zeros(self._num_envs, 3)

        prims1 = self.collision_groups[group1]
        prims2 = self.collision_groups[group2]

        for p1 in prims1:
            for p2 in prims2:
                if p1 <= p2:
                    pair = (p1, p2)
                else:
                    pair = (p2, p1)
                
                if pair in self.collision_dict:
                    normal = ArrayBackend.convert([self.collision_dict[pair][env][1].normal for env in self.collision_dict[pair].keys()], to=self._backend)
                    collision_force[list(self.collision_dict[pair].keys())] = normal

        return collision_force
    
    def get_collision_count(self, group1, group2):
        collision_in_env = ArrayBackend.get_array_backend(self._backend).zeros(self._num_envs, dtype=int)

        prims1 = self.collision_groups[group1]
        prims2 = self.collision_groups[group2]

        for p1 in prims1:
            for p2 in prims2:
                if p1 <= p2:
                    pair = (p1, p2)
                else:
                    pair = (p2, p1)
                
                if pair in self.collision_dict:
                    collision_in_env[list(self.collision_dict[pair].keys())] += 1

        return collision_in_env
    
    def _extract_name_and_env(self, path):
        if not path.startswith("/World/envs/"):
            return path, None
        path = path.split("/", 5)  
        name = "/" + path[5]
        env = int(path[3][4:])
        return name, env
    """
    def gather_collisions(self):
        from omni.physx.scripts.physicsUtils import PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        #https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/index.html#contact-reports
        collision_report = get_physx_simulation_interface().get_contact_report()
        for contact_header in collision_report[0]:
            if str(contact_header.type) != "ContactEventType.CONTACT_LOST":
                collider1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0)) # First prim involved in collision
                collider2 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1)) # Second prim involved in collision

                collider1, env1 = self._extract_name_and_env(collider1)
                collider2, env2 = self._extract_name_and_env(collider2)
                
                if env1 is None and env2 is None:
                    continue
                elif env1 is None:
                    env = env2
                elif env2 is None:
                    env = env1
                elif env1 != env2:
                    continue
                else:
                    env = env1

                if collider1 <= collider2:
                    pair = (collider1, collider2)
                else:
                    pair = (collider2, collider1)
                
                contact_data_offset = contact_header.contact_data_offset
                contact_data = collision_report[1]

                if pair not in self.collision_dict:
                    self.collision_dict[pair] = {
                        "contact": ArrayBackend.get_array_backend(self._backend).zeros((self._num_envs, ), dtype=bool, device=self._device),
                        "force": ArrayBackend.get_array_backend(self._backend).zeros((self._num_envs, self.num_steps, 3), device=self._device)
                    }
                self.collision_dict[pair]["contact"][env] = True
                self.collision_dict[pair]["force"][env, self.num_steps_stored] = torch.tensor([contact_data[contact_data_offset].normal.x, contact_data[contact_data_offset].normal.y, contact_data[contact_data_offset].normal.z], device=self._device)
        self.num_steps_stored += 1

    def check_collision(self, group1, group2):#TODO add collision count
        collision_in_env = ArrayBackend.get_array_backend(self._backend).zeros(self._num_envs, dtype=bool)

        prims1 = self.collision_groups[group1]
        prims2 = self.collision_groups[group2]

        for p1 in prims1:
            for p2 in prims2:
                if p1 <= p2:
                    pair = (p1, p2)
                else:
                    pair = (p2, p1)
                
                if pair in self.collision_dict:
                    collision_in_env[self.collision_dict[pair]["contact"]] = True

        return collision_in_env
    
    def get_collision_force(self, group1, group2):# Uses last collision force found as for each env
        collision_force = ArrayBackend.get_array_backend(self._backend).zeros(self._num_envs, self.num_steps_stored, 3)

        prims1 = self.collision_groups[group1]
        prims2 = self.collision_groups[group2]

        for p1 in prims1:
            for p2 in prims2:
                if p1 <= p2:
                    pair = (p1, p2)
                else:
                    pair = (p2, p1)
                
                if pair in self.collision_dict:
                    collision_force[self.collision_dict[pair]["contact"]] = self.collision_dict[pair]["force"]
                    collision_force[env][:len(self.collision_dict[pair][env])] = ArrayBackend.convert([contact[1].normal for contact in self.collision_dict[pair][env]], to=self._backend)

                    normals_list = [ArrayBackend.convert([contact[1].normal for contact in self.collision_dict[pair][env]], to=self._backend) for env in self.collision_dict[pair].keys()]
                    normal = ArrayBackend.get_array_backend(self._backend).from_list(normals_list)
                    collision_force[list(self.collision_dict[pair].keys())] = normal

        return collision_force
    
    def get_collision_count(self, group1, group2):
        collision_in_env = ArrayBackend.get_array_backend(self._backend).zeros(self._num_envs, dtype=int)

        prims1 = self.collision_groups[group1]
        prims2 = self.collision_groups[group2]

        for p1 in prims1:
            for p2 in prims2:
                if p1 <= p2:
                    pair = (p1, p2)
                else:
                    pair = (p2, p1)
                
                if pair in self.collision_dict:
                    collision_in_env[self.collision_dict[pair]["contact"]] += 1

        return collision_in_env
    
    def _extract_name_and_env(self, path):
        if not path.startswith("/World/envs/"):
            return path, None
        path = path.split("/", 5)  
        name = "/" + path[5]
        env = int(path[3][4:])
        return name, env
    """
    """
    def check_collision(self, group1, group2):#TODO add collision count
        #https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/index.html#contact-reports
        collision_report = get_physx_simulation_interface().get_contact_report()
        collision_in_env = ArrayBackend.get_array_backend(self._backend).zeros(self._num_envs, dtype=bool)

        prims1 = self.collision_groups[group1]
        prims2 = self.collision_groups[group2]

        for contact_header in collision_report[0]:
            collision, env = self._collision_occurs(prims1, prims2, contact_header)

            if collision:
                collision_in_env[env] = True

        return collision_in_env
    
    def get_collision_force(self, group1, group2):
        collision_report = get_physx_simulation_interface().get_contact_report()
        collision_force = ArrayBackend.get_array_backend(self._backend).zeros(self._num_envs, 3)

        prims1 = self.collision_groups[group1]
        prims2 = self.collision_groups[group2]

        for contact_header in collision_report[0]:
            collision, env = self._collision_occurs(prims1, prims2, contact_header)

            if collision:
                contact_data_offset = contact_header.contact_data_offset
                contact_data = collision_report[1]
                collision_force[env] = ArrayBackend.convert(list(contact_data[contact_data_offset].normal), to=self._backend)

        return collision_force
    
    def get_collision_count(self, group1, group2):
        collision_report = get_physx_simulation_interface().get_contact_report()
        collision_in_env = ArrayBackend.get_array_backend(self._backend).zeros(self._num_envs, dtype=int)

        prims1 = self.collision_groups[group1]
        prims2 = self.collision_groups[group2]

        for contact_header in collision_report[0]:
            collision, env = self._collision_occurs(prims1, prims2, contact_header)

            if collision:
                collision_in_env[env] += 1

        return collision_in_env
    
    def get_all_collision(self):
        collision_report = get_physx_simulation_interface().get_contact_report()
        lst = []
        for contact_header in collision_report[0]:
            if str(contact_header.type) != "ContactEventType.CONTACT_LOST":
                collider1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0)) # First prim involved in collision
                collider2 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1)) # Second prim involved in collision

                collider1, env1 = self._extract_name_and_env(collider1)
                collider2, env2 = self._extract_name_and_env(collider2)

                lst.append((f"{env1}_{collider1}", f"{env2}_{collider2}"))
        
        return lst.sort()"""