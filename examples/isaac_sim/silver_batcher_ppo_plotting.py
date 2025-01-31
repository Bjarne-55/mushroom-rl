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
from mushroom_rl.environments.isaacsim_envs.silver_batcher import Silver_Batcher
from mushroom_rl.utils import TorchUtils

from mushroom_rl.utils.plot import plot_mean_conf
import matplotlib.pyplot as plt
import os


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
               alg_params, policy_params, seed):

    logger = Logger(alg.__name__ + "_1_legged_gym", results_dir="./logs/", log_console=True, use_timestamp=True)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    mdp = Silver_Batcher(num_envs, 1000, True)
    mdp.seed(seed)
    
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
    #agent = agent.load("/home/bjarne/GitWorkspace/BachelorThesis/mushroom-rl/stored_agents/a1_ppo/1737244844.636734.zip")
    #agent.set_logger(logger)

    core = VectorCore(agent, mdp)

    Js = []
    Rs = []
    Es = []
    Vs = []
    INFOs = []

    dataset = core.evaluate(n_episodes=n_episodes_test, render=True, record=True)

    J = torch.mean(dataset.discounted_return).to("cpu").item()
    R = torch.mean(dataset.undiscounted_return.to("cpu")).item()
    E = agent.policy.entropy().to("cpu").item()
    V = torch.mean(agent._V(dataset.get_init_states())).detach().to("cpu").item()
    INFO = {key: torch.mean(value).to("cpu").item() for key, value in dataset.info.items()}
    Js.append(J)
    Rs.append(R)
    Es.append(E)
    Vs.append(V)
    INFOs.append(INFO)

    logger.epoch_info(0, J=J, R=R, entropy=E, V=V)

    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=True, record=True)
        agent.save(f"stored_agents/a1_ppo/{str(time.time())}.zip", True)

        J = torch.mean(dataset.discounted_return).to("cpu").item()
        R = torch.mean(dataset.undiscounted_return).to("cpu").item()
        E = agent.policy.entropy().to("cpu").item()
        V = torch.mean(agent._V(dataset.get_init_states())).detach().to("cpu").item()
        INFO = {key: torch.mean(value).to("cpu").item() for key, value in dataset.info.items()}
        Js.append(J)
        Rs.append(R)
        Es.append(E)
        Vs.append(V)
        INFOs.append(INFO)

        logger.epoch_info(it+1, J=J, R=R, entropy=E, V=V)

    #logger.info('Press a button to visualize')
    #input()
    #core.evaluate(n_episodes=n_episodes_test, render=True, record=True)
    #agent.save(f"stored_agents/a1_ppo/{str(time.time())}.zip", True)

    return Js, Rs, Es, Vs, INFOs

def create_plot(data, directory, name, title):
    fig, ax = plt.subplots()
    plt.title(title)
    plot_mean_conf(data, ax)
    plt.savefig(f"{directory}/{name}.png") 


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

    seed = np.random.randint(0, 10000)
    Js, Rs, Es, Vs, INFOs = experiment(alg=PPO, num_envs=num_envs, n_epochs=40, n_steps=4096*24*50, n_steps_per_fit=4096*24,
                   n_episodes_test=256, alg_params=ppo_params, policy_params=policy_params, seed=seed)
    
    dir = "plots/a1_effort_ppo/" + str(time.time())
    os.makedirs(dir)
    create_plot([Js], dir, "J", f"PPO - discounted Return: {seed}")
    create_plot([Rs], dir, "R", f"PPO - undiscounted Return: {seed}")
    create_plot([Es], dir, "E", f"PPO - Entropy: {seed}")
    create_plot([Vs], dir, "V", f"PPO - value of intial states: {seed}")
    for key in INFOs[0]:
        lst_info = []
        for epi_info in INFOs:
            lst_info.append(epi_info[key])
        create_plot([lst_info], dir, key, f"PPO - {key}: {seed}")