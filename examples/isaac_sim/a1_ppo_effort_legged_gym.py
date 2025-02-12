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
from mushroom_rl.environments.isaacsim_envs.isaac_a1_pos_action import A1Pos
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


def experiment(alg, num_envs, n_epochs, n_steps, n_steps_per_fit, n_episodes_test,
               alg_params, policy_params):

    logger = Logger(alg.__name__ + "_1_legged_gym", results_dir="./logs/", log_console=True, use_timestamp=True)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    mdp = A1Pos(num_envs, 1000, True, True)
    
    critic_params = dict(network=Network,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 1e-4}},
                         loss=F.mse_loss,
                         n_features=[512, 256, 128],
                         batch_size=int((4096*24) / 4),
                         use_cuda=True,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    policy = GaussianTorchPolicy(Network,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    alg_params['critic_params'] = critic_params

    agent = alg(mdp.info, policy, **alg_params)

    core = VectorCore(agent, mdp)

    dataset = core.evaluate(n_episodes=n_episodes_test, render=False, record=False)
    J = torch.mean(dataset.discounted_return).item()
    R = torch.mean(dataset.undiscounted_return).item()
    E = agent.policy.entropy().item()

    logger.epoch_info(0, J=J, R=R, entropy=E)

    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        if it == 4:
            dataset = core.evaluate(n_episodes=n_episodes_test, render=True, record=True)
        else:
            dataset = core.evaluate(n_episodes=n_episodes_test, render=False, record=False)
        agent.save(f"stored_agents/a1_ppo/{str(time.time())}.zip", True)

        J = torch.mean(dataset.discounted_return).item()
        R = torch.mean(dataset.undiscounted_return).item()
        E = agent.policy.entropy().item()

        logger.epoch_info(it+1, J=J, R=R, entropy=E)

    #logger.info('Press a button to visualize')
    #input()
    core.evaluate(n_episodes=n_episodes_test, render=True, record=True)
    agent.save(f"stored_agents/a1_ppo/{str(time.time())}.zip", True)

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
    num_envs = 4096

    experiment(alg=PPO, num_envs=num_envs, n_epochs=5, n_steps=4096*24*50*6, n_steps_per_fit=4096*24,
                   n_episodes_test=256, alg_params=ppo_params, policy_params=policy_params)