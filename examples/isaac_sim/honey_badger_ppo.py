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
from mushroom_rl.algorithms.actor_critic.deep_actor_critic.ppo_nikita import NikitaPPO

from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.environments.isaacsim_envs.honey_badger import HoneyBadger#TODO fix name
from mushroom_rl.utils import TorchUtils


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Network, self).__init__()

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


def experiment(alg, num_envs, n_epochs, n_steps, n_steps_per_fit, n_episodes_test,
               alg_params, policy_params):

    logger = Logger(alg.__name__ + "_honey_batcher", results_dir="./logs/", log_console=True, use_timestamp=True)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    mdp = HoneyBadger(num_envs, 1000, True, True)
    
    critic_params = dict(network=Network,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=[512, 256, 128],
                         batch_size=int((4096*24) / 32),
                         use_cuda=True,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    policy = GaussianTorchPolicy(Network,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    alg_params['critic_params'] = critic_params

    agent = alg(mdp.info, policy, **alg_params)
    #agent = agent.load("/home/bjarne/GitWorkspace/BachelorThesis/mushroom-rl/stored_agents/NikitaPPO_honey_batcher/1739569422.1717234/iteration_2.zip")
    #agent.set_logger(logger)

    core = VectorCore(agent, mdp)
    
    dataset = core.evaluate(n_episodes=n_episodes_test, render=True, record=True)

    J = torch.mean(dataset.discounted_return).item()
    R = torch.mean(dataset.undiscounted_return).item()
    E = agent.policy.entropy().item()
    A = dataset.absorbing.sum().item()
    penalities = torch.mean(dataset.info["penalties"]).item()
    tracking_reward = torch.mean(dataset.info["tracking_reward"]).item()
    
    logger.epoch_info(0, J=J, R=R, entropy=E, A=A, penalities=penalities, tracking=tracking_reward)

    dir = str(time.time())
    agent.save(f"stored_agents/{alg.__name__}_honey_batcher/{dir}/iteration_{0}.zip", True)
    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=True, record=True)
        agent.save(f"stored_agents/{alg.__name__}_honey_batcher/{dir}/iteration_{it}.zip", True)

        J = torch.mean(dataset.discounted_return).item()
        R = torch.mean(dataset.undiscounted_return).item()
        E = agent.policy.entropy().item()
        A = dataset.absorbing.sum().item()
        penalities = torch.mean(dataset.info["penalties"]).item()
        tracking_reward = torch.mean(dataset.info["tracking_reward"]).item()
    
        logger.epoch_info(it + 1, J=J, R=R, entropy=E, A=A, penalities=penalities, tracking=tracking_reward)

    #logger.info('Press a button to visualize')
    #input()
    #core.evaluate(n_episodes=n_episodes_test, render=True, record=True)
    #agent.save(f"stored_agents/a1_ppo/{str(time.time())}.zip", True)


if __name__ == '__main__':
    TorchUtils.set_default_device('cuda:0')
    ppo_params = dict(
        actor_optimizer={'class': optim.Adam,
        'params': {'lr': 3e-4}},#changed from 1e-3
        n_epochs_policy=5,
        batch_size=int((4096*24) / 32),
        eps_ppo=.2,
        lam=.95
    )
    policy_params = dict(
        std_0=1.,
        n_features=[512, 256, 128],
        use_cuda=True,
        ent_coeff=0.01
    )
    num_envs = 4096
    experiment(alg=NikitaPPO, num_envs=num_envs, n_epochs=10, n_steps=4096*24*50*15, n_steps_per_fit=4096*24,
                   n_episodes_test=256, alg_params=ppo_params, policy_params=policy_params)
