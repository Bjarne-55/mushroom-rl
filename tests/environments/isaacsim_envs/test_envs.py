try:
    import isaacsim
    from mushroom_rl.environments.isaacsim_envs import CartPole
    from mushroom_rl.environments.isaacsim_envs import A1Walking, HoneyBadgerWalking, SilverBadgerWalking

    import torch
    import numpy as np
    import multiprocessing

    # Run tests for all environments sequentially, as only one instance of Isaac Sim
    # can be executed at a time.
    def test_envs():
        multiprocessing.set_start_method('spawn', force=True)

        lst_functions = [cartpole_numpy, cartpole_torch_cpu, cartpole_torch_cuda,
                            a1, silver_badger, honey_badger]
        for test_func in lst_functions:
            process = multiprocessing.Process(target=test_func)
            process.start()
            process.join()

except ImportError:
    pass

# quadropeds ---------------------------------------------------------------------

def helper_env(mdp):
    n_envs = mdp.number
    mask = torch.ones(n_envs, device="cuda:0")

    obs, _ = mdp.reset_all(mask)
    assert obs.shape == (n_envs, len(mdp.info.observation_space.low))
    assert obs.shape == (n_envs, len(mdp.info.observation_space.high))

    for i in range(20):
        if i < 10:
            action = torch.tensor([[0.] * 12] * n_envs, device="cuda:0")
        else:
            action = torch.tensor([[1.] * 12] * n_envs, device="cuda:0")
        
        obs, reward, absorbing, _ = mdp.step_all(mask, action)
        
        assert obs.shape == (n_envs, len(mdp.info.observation_space.low))
        assert obs.shape == (n_envs, len(mdp.info.observation_space.high))
        assert reward.shape == (n_envs, )
        assert absorbing.shape == (n_envs, )

def a1():
    N_ENVS = 2
    mdp = A1Walking(N_ENVS, 1000, True)
    assert mdp.number == N_ENVS
    helper_env(mdp)

def honey_badger():
    N_ENVS = 2
    mdp = HoneyBadgerWalking(N_ENVS, 1000, True, True)
    assert mdp.number == N_ENVS
    helper_env(mdp)

def silver_badger():
    N_ENVS = 2
    mdp = SilverBadgerWalking(N_ENVS, 1000, True, True)
    assert mdp.number == N_ENVS
    helper_env(mdp)

# cartpole -----------------------------------------------------------------------

def cartpole_torch_cuda():
    n_envs = 2
    mdp = CartPole(n_envs, "torch", "cuda:0")

    assert mdp.number == n_envs
    assert isinstance(mdp.info.observation_space.low, torch.Tensor)
    assert isinstance(mdp.info.observation_space.high, torch.Tensor)
    assert isinstance(mdp.info.action_space.low, torch.Tensor)
    assert isinstance(mdp.info.action_space.high, torch.Tensor)
    assert mdp.info.observation_space.low.is_cuda
    assert mdp.info.observation_space.high.is_cuda
    assert mdp.info.action_space.low.is_cuda
    assert mdp.info.action_space.high.is_cuda

    mask = torch.ones(n_envs, device="cuda:0")

    obs, _ = mdp.reset_all(mask)

    assert isinstance(obs, torch.Tensor) and obs.is_cuda
    assert obs.shape == (n_envs, len(mdp.info.observation_space.low))
    assert obs.shape == (n_envs, len(mdp.info.observation_space.high))

    for i in range(20):
        if i < 10:
            action = torch.tensor([[0.] * 1] * n_envs, device="cuda:0")
        else:
            action = torch.tensor([[1.] * 1] * n_envs, device="cuda:0")
        
        obs, reward, absorbing, _ = mdp.step_all(mask, action)

        assert isinstance(obs, torch.Tensor) and obs.is_cuda
        assert isinstance(reward, torch.Tensor) and reward.is_cuda
        assert isinstance(absorbing, torch.Tensor) and absorbing.is_cuda
        
        assert obs.shape == (n_envs, len(mdp.info.observation_space.low))
        assert obs.shape == (n_envs, len(mdp.info.observation_space.high))
        assert reward.shape == (n_envs, )
        assert absorbing.shape == (n_envs, )

def cartpole_torch_cpu():
    n_envs = 2
    mdp = CartPole(n_envs, "torch", "cpu")

    assert mdp.number == n_envs
    assert isinstance(mdp.info.observation_space.low, torch.Tensor)
    assert isinstance(mdp.info.observation_space.high, torch.Tensor)
    assert isinstance(mdp.info.action_space.low, torch.Tensor)
    assert isinstance(mdp.info.action_space.high, torch.Tensor)
    assert mdp.info.observation_space.low.is_cpu
    assert mdp.info.observation_space.high.is_cpu
    assert mdp.info.action_space.low.is_cpu
    assert mdp.info.action_space.high.is_cpu

    mask = torch.ones(n_envs, device="cpu")

    obs, _ = mdp.reset_all(mask)
    
    assert isinstance(obs, torch.Tensor) and obs.is_cpu
    assert obs.shape == (n_envs, len(mdp.info.observation_space.low))
    assert obs.shape == (n_envs, len(mdp.info.observation_space.high))

    for i in range(20):
        if i < 10:
            action = torch.tensor([[0.] * 1] * n_envs, device="cpu")
        else:
            action = torch.tensor([[1.] * 1] * n_envs, device="cpu")
        
        obs, reward, absorbing, _ = mdp.step_all(mask, action)

        assert isinstance(obs, torch.Tensor) and obs.is_cpu
        assert isinstance(reward, torch.Tensor) and reward.is_cpu
        assert isinstance(absorbing, torch.Tensor) and absorbing.is_cpu
        
        assert obs.shape == (n_envs, len(mdp.info.observation_space.low))
        assert obs.shape == (n_envs, len(mdp.info.observation_space.high))
        assert reward.shape == (n_envs, )
        assert absorbing.shape == (n_envs, )

def cartpole_numpy():
    n_envs = 2
    mdp = CartPole(n_envs, "numpy", None)

    assert mdp.number == n_envs
    assert isinstance(mdp.info.observation_space.low, np.ndarray)
    assert isinstance(mdp.info.observation_space.high, np.ndarray)
    assert isinstance(mdp.info.action_space.low, np.ndarray)
    assert isinstance(mdp.info.action_space.high, np.ndarray)

    mask = np.ones(n_envs)

    obs, _ = mdp.reset_all(mask)
    
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (n_envs, len(mdp.info.observation_space.low))
    assert obs.shape == (n_envs, len(mdp.info.observation_space.high))

    for i in range(20):
        if i < 10:
            action = np.array([[0.] * 1] * n_envs)
        else:
            action = np.array([[1.] * 1] * n_envs)
        
        obs, reward, absorbing, _ = mdp.step_all(mask, action)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, np.ndarray)
        assert isinstance(absorbing, np.ndarray)
        
        assert obs.shape == (n_envs, len(mdp.info.observation_space.low))
        assert obs.shape == (n_envs, len(mdp.info.observation_space.high))
        assert reward.shape == (n_envs, )
        assert absorbing.shape == (n_envs, )