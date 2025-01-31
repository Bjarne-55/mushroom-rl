import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import trange

import random
import time

from mushroom_rl.core import VectorCore, Logger
from mushroom_rl.algorithms.actor_critic import TRPO, PPO

from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.environments.isaacsim_envs.isaac_a1_legged_gym import IsaacA1Description
from mushroom_rl.utils import TorchUtils

class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Network, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self.actor = nn.Sequential(
            nn.Linear(n_input, n_features[0]),
            nn.ELU(alpha=1.),
            nn.Linear(n_features[0], n_features[1]),
            nn.ELU(alpha=1.),
            nn.Linear(n_features[1], n_features[2]),
            nn.ELU(alpha=1.),
            nn.Linear(n_features[2], n_output)
        )

    def forward(self, state, **kwargs):
        state = torch.squeeze(state, 1).float()
        return self.actor(state)

class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(CriticNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features[0])
        self._h2 = nn.Linear(n_features[0], n_features[1])
        self._h3 = nn.Linear(n_features[1], n_features[2])
        self._h4 = nn.Linear(n_features[2], n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, **kwargs):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        features3 = F.relu(self._h3(features2))
        a = self._h4(features3)

        return a
    
class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self.actor = nn.Sequential(
            nn.Linear(n_input, n_features[0]),
            nn.ELU(alpha=1.),
            nn.Linear(n_features[0], n_features[1]),
            nn.ELU(alpha=1.),
            nn.Linear(n_features[1], n_features[2]),
            nn.ELU(alpha=1.),
            nn.Linear(n_features[2], n_output)
        )

        #nn.init.xavier_uniform_(self.actor[0].weight, gain=nn.init.calculate_gain('relu'))
        #nn.init.xavier_uniform_(self.actor[2].weight, gain=nn.init.calculate_gain('relu'))
        #nn.init.xavier_uniform_(self.actor[4].weight, gain=nn.init.calculate_gain('relu'))
        #nn.init.xavier_uniform_(self.actor[6].weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, state, **kwargs):
        state = torch.squeeze(state, 1).float()
        return self.actor(state)
    
class A1Custom(IsaacA1Description):
    def _preprocess_action(self, action):
        #action = torch.zeros((self.number, 12), device="cuda:0")
        self.counter += 1
        if self.counter < 50:
            action = torch.tensor([-2.3172,  3.1434, -3.7754,  3.3198,  6.5241, -2.4077,  0.9242,  3.9530,
        -2.8774, -2.5747,  1.9155, -2.3527], device='cuda:0').repeat(self.number, 1)
        else:
            action = torch.tensor([0,  0, 0,  0,  0, 0,  0,  0, 0, 0, 0, 0], device='cuda:0').repeat(self.number, 1)
        return super()._preprocess_action(action)
    
class A1LeggedGymActor(IsaacA1Description):
    counter = 0
    def _modify_observation(self, obs):
        obs = super()._modify_observation(obs)
        new_obs = obs.clone().detach()
        new_obs[:, 6:9] = self.observation_helper.get_from_obs(obs, "projected_gravity")
        new_obs[:, 9:12] = self.observation_helper.get_from_obs(obs, "commands")
        new_obs[:, 12:24] = self.observation_helper.get_from_obs(obs, "joint_pos")
        new_obs[:, 24:36] = self.observation_helper.get_from_obs(obs, "joint_vel")
        new_obs[:, 36:48] = self.observation_helper.get_from_obs(obs, "actions")

        return new_obs

    def _preprocess_action(self, action):
        self.counter += 1
        if self.counter < 50:
            action = torch.tensor([-2.3172,  3.1434, -3.7754,  3.3198,  6.5241, -2.4077,  0.9242,  3.9530,
        -2.8774, -2.5747,  1.9155, -2.3527], device='cuda:0').repeat(self.number, 1)
        else:
            action = torch.tensor([0,  0, 0,  0,  0, 0,  0,  0, 0, 0, 0, 0], device='cuda:0').repeat(self.number, 1)
        return super()._preprocess_action(action)
    
class MyPPO(PPO):
    def draw_action(self, state, policy_state=None):
        return self.policy._mu.model.network(state).detach(), None


def experiment(alg, num_envs, n_epochs, n_steps, n_steps_per_fit, n_episodes_test,
               alg_params, policy_params):

    logger = Logger(alg.__name__ + "_1_legged_gym", results_dir="./logs/", log_console=True, use_timestamp=True)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    mdp = A1LeggedGymActor(num_envs, 1000, True, True, (5, 0, 4), (0, 0, 0), 
        usd_path = "/home/bjarne/GitWorkspace/BachelorThesis/mushroom-rl/isaac_assets/fixed/a1/a1.usd")
    
    #mdp = A1LeggedGymActor(num_envs, 1000, True, True, (5, 0, 4), (0, 0, 0))

    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 1e-4}},
                         loss=F.mse_loss,
                         n_features=[512, 256, 128],
                         batch_size=int((4096*24) / 4),
                         use_cuda=True,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    policy = GaussianTorchPolicy(ActorNetwork,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    alg_params['critic_params'] = critic_params

    loaded_dict = torch.load("/home/bjarne/GitWorkspace/BachelorThesis/legged_gym/logs/rough_a1/Dec11_10-37-14_/model_1500.pt")
    policy._mu.model.network.load_state_dict(loaded_dict["model_state_dict"], strict=False)
    s = loaded_dict["model_state_dict"]["std"]
    policy._log_sigma = nn.Parameter(s)
    policy._mu.model.network.to("cuda:0")

    #agent = alg(mdp.info, policy, **alg_params)
    #agent = agent.load("/home/bjarne/GitWorkspace/BachelorThesis/mushroom-rl/stored_agents/a1_ppo/1738081594.2672067.zip")
    #agent.set_logger(logger)
    agent = PPO.load("/home/bjarne/GitWorkspace/BachelorThesis/mushroom-rl/stored_agents/a1_ppo/1738194207.8245382.zip")

    core = VectorCore(agent, mdp)
    """
    mdp.reset_all(torch.ones((num_envs,), dtype=bool), None)
    action = torch.tensor([ 
        0.2777,  2.2838,  0.1837,  0.6804, -0.1905,  0.7365, -1.1487,  0.4724, -0.5471,  2.0708,  
        1.4633, -1.5805
        ], device="cuda:0")
    while True:
        act = action.repeat((num_envs, 1))
        mdp.step_all(torch.ones((num_envs,), dtype=bool, device="cuda:0"), act)
    """
    INFOs = []
    for i in range(1):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False, record=False)

        #J = torch.mean(dataset.discounted_return).item()
        #R = torch.mean(dataset.undiscounted_return).item()
        #E = agent.policy.entropy().item()
        INFO = {key: torch.mean(value).to("cpu").item() for key, value in dataset.info.items()}
        INFOs.append(INFO)

        print(INFO["r_collision"])

        #logger.epoch_info(0, J=J, R=R, entropy=E)
        logger.epoch_info(i)
    print(INFOs)


if __name__ == '__main__':
    TorchUtils.set_default_device('cuda:0')
    ppo_params = dict(
        actor_optimizer={'class': optim.Adam,
        'params': {'lr': 1e-4}},#changed from 1e-3
        n_epochs_policy=5,
        batch_size=int((4096*24) / 4),
        eps_ppo=.2,
        lam=.95
    )
    policy_params = dict(
        std_0=1.,
        n_features=[512, 256, 128],
        use_cuda=True,
        ent_coeff=0.01
    )
    num_envs = 256
    experiment(alg=PPO, num_envs=num_envs, n_epochs=80, n_steps=4096*24*20, n_steps_per_fit=4096*24,
                   n_episodes_test=num_envs, alg_params=ppo_params, policy_params=policy_params)
