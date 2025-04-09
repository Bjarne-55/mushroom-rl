try:
    import isaacsim
    from mushroom_rl.environments.isaacsim_envs import A1Walking, HoneyBadgerWalking, SilverBadgerWalking
    import torch
    import numpy as np
    import random

    def test_A1():
        class DeterministicA1(A1Walking):
            def _create_simulation_app(self, headless):
                r = super()._create_simulation_app(headless)
                torch.manual_seed(0)
                np.random.seed(0)
                random.seed(0)
                import omni.replicator.core as rep
                import isaacsim.core.utils.torch as torch_utils
                rep.set_global_seed(0)
                torch_utils.set_seed(0)
                from isaacsim.core.utils.torch.maths import set_seed
                set_seed(0, True)
                return r
            
            def _create_world(self, timestep, custom_sim_params=None):
                r = super()._create_world(timestep, custom_sim_params)
                physx_scene_api = self._physics_context._physx_scene_api
                physx_scene_api.CreateEnableEnhancedDeterminismAttr(True)
                return r

        torch.set_printoptions(precision=8)
        N_ENVS = 2
        obs_lst = []
        mdp = DeterministicA1(N_ENVS, 1000, True)
        mask = torch.ones(N_ENVS, device="cuda:0")
        mdp.reset_all(mask)

        for i in range(20):
            if i < 10:
                action = torch.tensor([[0.] * 12] * N_ENVS, device="cuda:0")
            else:
                action = torch.tensor([[2.] * 12] * N_ENVS, device="cuda:0")
            
            obs, _, _, _ = mdp.step_all(mask, action)
            
            for j in range(N_ENVS):
                assert len(obs[j]) == len(mdp._mdp_info.observation_space.low)
                assert len(obs[j]) == len(mdp._mdp_info.observation_space.high)
            obs_lst.append(obs)

        #torch.save(obs_lst, "tests/environments/isaacsim_envs/a1_data.pt")
        obs_test_lst = torch.load("tests/environments/isaacsim_envs/a1_data.pt")
        
        assert len(obs_lst) == len(obs_test_lst)
        for i, (obs, obs_test) in enumerate(zip(obs_lst, obs_test_lst)):
            assert torch.allclose(obs, obs_test, atol=1e-5), f"\n{obs[~torch.isclose(obs, obs_test)]} \n!=\n {obs_test[~torch.isclose(obs, obs_test)]} \n {obs[~torch.isclose(obs, obs_test)] - obs_test[~torch.isclose(obs, obs_test)]} at index {i}"
            

except ImportError:
    pass