import os
from mushroom_rl.utils.plot import plot_mean_conf
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import trange

from mushroom_rl.core import VectorCore, Logger, MultiprocessEnvironment
from mushroom_rl.environments import Gym
from mushroom_rl.algorithms.actor_critic import PPO, TRPO

from mushroom_rl.policy import GaussianTorchPolicy

torch.set_num_threads(1)
torch.set_num_interop_threads(4)


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Network, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, state, **kwargs):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a


def experiment(alg, env_id, horizon, gamma, n_epochs, n_steps, n_steps_per_fit, n_episodes_test,
               alg_params, policy_params):

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    mdp = MultiprocessEnvironment(Gym, env_id, horizon, gamma, n_envs=15)

    critic_params = dict(network=Network,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=32,
                         batch_size=64,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    policy = GaussianTorchPolicy(Network,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    alg_params['critic_params'] = critic_params

    agent = alg(mdp.info, policy, **alg_params)
    #agent.set_logger(logger)

    core = VectorCore(agent, mdp)

    dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

    Js = []
    Rs = []
    Es = []
    Vs = []
    Infos = []

    J = np.mean(dataset.discounted_return)
    R = np.mean(dataset.undiscounted_return)
    E = agent.policy.entropy()
    V = torch.mean(agent._V(dataset.get_init_states())).detach().to("cpu")
    Js.append(J)
    Rs.append(R)
    Es.append(E)
    Vs.append(V)
    Infos.append(dataset.info)

    logger.epoch_info(0, J=J, R=R, entropy=E)

    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

        J = np.mean(dataset.discounted_return)
        R = np.mean(dataset.undiscounted_return)
        E = agent.policy.entropy()
        V = torch.mean(agent._V(dataset.get_init_states())).detach().to("cpu")
        Js.append(J)
        Rs.append(R)
        Es.append(E)
        Vs.append(V)
        Infos.append(dataset.info)

        logger.epoch_info(it+1, J=J, R=R, entropy=E)

    #logger.info('Press a button to visualize')
    #input()
    #core.evaluate(n_episodes=5, render=True)

    return Js, Rs, Es, Vs, Infos

def create_plot(data, directory, name, title):
    fig, ax = plt.subplots()
    plt.title(title)
    plot_mean_conf(data, ax)
    plt.savefig(f"{directory}/{name}.png")

if __name__ == '__main__':
    max_kl = .015

    policy_params = dict(
        std_0=1.,
        n_features=32
    )

    ppo_params = dict(actor_optimizer={'class': optim.Adam,
                                       'params': {'lr': 3e-4}},
                      n_epochs_policy=4,
                      batch_size=64,
                      eps_ppo=.2,
                      lam=.95)

    algs_params = [
        (PPO, 'ppo', ppo_params)
    ]

    run_Js = []
    run_Rs = []
    run_Es = []
    run_Vs = []
    run_Is = []
    seeds = []
    
    num_runs = 3
    for i in range(num_runs):
        seed = np.random.randint(0, 10000)
        seeds.append(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Start run {i} with seed {seed}")

        Js, Rs, Es, Vs, Infos = experiment(alg=PPO, env_id='HalfCheetah-v4', horizon=200, gamma=.99,
                   n_epochs=40, n_steps=30000, n_steps_per_fit=3000,
                   n_episodes_test=25, alg_params=ppo_params,
                   policy_params=policy_params)
        
        run_Js.append(Js)
        run_Rs.append(Rs)
        run_Es.append(Es)
        run_Vs.append(Vs)
        run_Is.append(Infos)
        
    
    dir = "plots/ppo/" + str(time.time())
    os.makedirs(dir)
    create_plot(run_Js, dir, "J", f"PPO - discounted Return: {seeds}")
    create_plot(run_Rs, dir, "R", f"PPO - undiscounted Return: {seeds}")
    create_plot(run_Es, dir, "E", f"PPO - Entropy: {seeds}")
    create_plot(run_Vs, dir, "V", f"PPO - value of intial states: {seeds}")

    for key in run_Is[0][0].keys():
        if key == 'x_position':
            prop = [[np.max(np.nan_to_num(one_episode[key])) for one_episode in one_run] for one_run in run_Is]
        else:
            prop = [[np.mean(np.nan_to_num(one_episode[key])) for one_episode in one_run] for one_run in run_Is]
        create_plot(prop, dir, str(key), f"PPO - {str(key)}: {seeds}") # DO average of maximum position for x_position
        