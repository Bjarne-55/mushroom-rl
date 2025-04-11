import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import trange

import random
import time
import csv
import statistics as stat

from mushroom_rl.core import VectorCore, Logger
from mushroom_rl.algorithms.actor_critic.deep_actor_critic.ppo_nikita import NikitaPPO
from mushroom_rl.algorithms.actor_critic.deep_actor_critic import PPO

from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.environments.isaacsim_envs.isaac_a1_legged_gym import IsaacA1Description
from mushroom_rl.environments.isaacsim_envs.honey_badger import HoneyBadger
from mushroom_rl.environments.isaacsim_envs.silver_badger import SilverBadger
from mushroom_rl.utils import TorchUtils

from mushroom_rl.utils.plot import plot_mean_conf
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

times_collection = []
times_complete_fit = []
times_env_stop = []
class TimeMeasuringVectorCore(VectorCore):
    def _run(self, dataset, n_steps, n_episodes, render, quiet, record, initial_states=None):
        global times_collection, times_complete_fit, times_env_stop
        times_complete_fit = []
        times_collection = []

        self._core_logic.initialize_run(n_steps, n_episodes, initial_states, quiet)

        last = self._core_logic.converter.ones(self.env.number, dtype=bool)
        mask = None

        start_coll = time.perf_counter()

        while self._core_logic.move_required():
            if last.any():
                mask = self._core_logic.get_mask(last)
                current_theta, reset_mask = self._reset(initial_states, last, mask)

                if self.agent.info.is_episodic and reset_mask.any():
                    dataset.append_theta_vectorized(current_theta, reset_mask)

            samples, step_infos = self._step(render, record, mask)

            self.callback_step(samples)
            self._core_logic.after_step(samples[5] & mask)

            dataset.append_vectorized(samples, step_infos, mask)

            last = samples[5]

            if self._core_logic.fit_required():
                end_coll = time.perf_counter()
                times_collection.append(end_coll - start_coll)
                start_fit = time.perf_counter()
                fit_dataset = dataset.flatten(self._core_logic.n_steps_per_fit)
                self.agent.fit(fit_dataset)

                for c in self.callbacks_fit:
                    c(dataset)

                n_carry_forward_steps = dataset.clear(self._core_logic.n_steps_per_fit)
                last = self._core_logic.after_fit_vectorized(last, n_carry_forward_steps)

                end_fit = time.perf_counter()
                times_complete_fit.append(end_fit - start_fit)
                start_coll = time.perf_counter()
        end_coll = time.perf_counter()
        times_collection.append(end_coll - start_coll)

        self.agent.stop()

        start_env_stop = time.perf_counter()
        self.env.stop()
        end_env_stop = time.perf_counter()
        times_env_stop.append(end_env_stop - start_env_stop)

        self._end(record)

        return dataset.flatten()
    
def store_config(folder, filename, **variables):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)

    with open(file_path, "w") as file:
        for name, value in variables.items():
            file.write(f"{name}: {value}\n")

def store_csv(lst, path, header):
    folder = "/".join(path.split("/")[:-1])
    os.makedirs(folder, exist_ok=True)

    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for values in lst:
            writer.writerow(values)

def experiment(mdp, alg, run_idx, n_epochs, n_steps, n_steps_per_fit, n_episodes_test, batch_size, seed, dir_path, lr, ent_coeff):
    run_name = f"run_{run_idx}"
    path = dir_path + run_name

    logger = Logger(run_name, results_dir=dir_path, log_console=True, use_timestamp=False, seed=seed)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    store_config(path, "config.txt", run_idx=run_idx, n_epochs=n_epochs, n_steps=n_steps, 
                 n_steps_per_fit=n_steps_per_fit, n_episodes_test=n_episodes_test, batch_size=batch_size, seed=seed)

    alg_params = dict(
        actor_optimizer={'class': optim.Adam,
        'params': {'lr': lr}},#changed from 1e-3
        n_epochs_policy=5,
        batch_size=batch_size,
        eps_ppo=.2,
        lam=.95,
        ent_coeff=ent_coeff
    )
    policy_params = dict(
        std_0=1.,
        n_features=[512, 256, 128],
        use_cuda=True
    )
    
    critic_params = dict(network=Network,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': lr}},
                         loss=F.mse_loss,
                         n_features=[512, 256, 128],
                         batch_size=batch_size,
                         use_cuda=True,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    policy = GaussianTorchPolicy(Network,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    alg_params['critic_params'] = critic_params

    agent = alg(mdp.info, policy, **alg_params)
    #agent = agent.load("/home/bjarne/GitWorkspace/BachelorThesis/mushroom-rl/stored_agents/a1_ppo/1739340890.6037955.zip")
    #agent.set_logger(logger)

    record_dict = {"path": path, "tag": "recordings"}
    core = TimeMeasuringVectorCore(agent, mdp, record_dictionary=record_dict)

    values = []

    start_eval = time.perf_counter()
    dataset = core.evaluate(n_episodes=n_episodes_test, render=True, record=False)
    end_eval = time.perf_counter()

    J = torch.mean(dataset.discounted_return).to("cpu").item()
    R = torch.mean(dataset.undiscounted_return.to("cpu")).item()
    E = agent.policy.entropy().to("cpu").item()
    V = torch.mean(agent._V(dataset.get_init_states())).detach().to("cpu").item()
    A = dataset.absorbing.sum().to("cpu").item()
    T_EVAL = end_eval - start_eval
    values.append((J, R, E, V, A, T_EVAL))

    logger.epoch_info(0, J=J, R=R, entropy=E, V=V, A=A, T_EVAL=T_EVAL)

    for it in trange(n_epochs, leave=False):
        global alg_fit_times, env_step_times, world_step_times
        alg_fit_times = []
        env_step_times = []
        world_step_times = []
        start_learn = time.perf_counter()
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        end_learn = time.perf_counter()
        detailed_times = list(zip(alg_fit_times, env_step_times, world_step_times, times_collection, times_complete_fit))
        alg_fit_mean = stat.mean(alg_fit_times)
        env_step_mean = stat.mean(env_step_times)
        world_step_mean = stat.mean(world_step_times)
        collection_mean = stat.mean(times_collection)
        complete_fit_mean = stat.mean(times_complete_fit)

        headers = ("Alg Fit Times", "Env Step times", "World step times", "Collection Time", "Complete Fit Time")
        store_csv(detailed_times, path + "/detailed_learning_times/" + f"it_{it + 1}.csv", headers)
        torch.save(detailed_times, path + "/detailed_learning_times/" + f"torch_it_{it + 1}.pt")
        alg_fit_times = []
        env_step_times = []
        world_step_times = []

        start_eval = time.perf_counter()
        render = False #(it + 1) % 10 == 0 or it == n_epochs - 1
        if render:
            dataset = core.evaluate(n_episodes=n_episodes_test, render=True, record=True)
        else:
            dataset = core.evaluate(n_episodes=n_episodes_test, render=False, record=False)
        end_eval = time.perf_counter()

        J = torch.mean(dataset.discounted_return).to("cpu").item()
        R = torch.mean(dataset.undiscounted_return).to("cpu").item()
        E = agent.policy.entropy().to("cpu").item()
        V = torch.mean(agent._V(dataset.get_init_states())).detach().to("cpu").item()
        A = dataset.absorbing.sum().to("cpu").item()
        T_EVAL = end_eval - start_eval
        T_LEARN = end_learn - start_learn
        T_SINCE_START = end_eval - start_run
        values.append((J, R, E, V, A, T_EVAL, T_LEARN, T_SINCE_START,
                       alg_fit_mean, env_step_mean, world_step_mean, collection_mean, complete_fit_mean))

        logger.epoch_info(it+1, J=J, R=R, entropy=E, V=V, A=A, T_EVAL=T_EVAL, render=render, T_LEARN=T_LEARN, 
                          T_SINCE_START=T_SINCE_START, T_MEAN_ALG_FIT=alg_fit_mean, T_MEAN_ENV_STEP=env_step_mean, 
                          T_MEAN_WORLD_STEP=world_step_mean, T_MEAN_COLL=collection_mean, 
                          T_MEAN_COMPLETE_FIT=complete_fit_mean)
        agent.save(f"{path}/stored_agents/a1_ppo/agent_{it}.zip", True)

    headers = ("Discounted Return", "Undiscounted Return", "Entropy", "Value function", "Absorbing", "Time needed for Evaluation", 
               "Time needed for Learning", "Time since start", "Average Time needed for Fit of PPO", 
               "Average Time needed for Step in Environment", "Average Time needed for Step in Simulation", 
               "Average Time needed for Collection per Fit",  "Average Time needed for Fit in Core")
    store_csv(values, path + "/values.csv", headers)
    torch.save(values, path + "/torch_values.pt")

    store_csv([(x, ) for x in times_env_stop], path + "/time_env_stop.csv", ("Time needed for Env stop", ))

    dataset = core.evaluate(n_episodes=n_episodes_test, render=True, record=True)

    return values

def create_plot(data, directory, name, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.title(title)
    plot_mean_conf(data, ax)
    plt.savefig(f"{directory}/{name}.png") 

alg_fit_times = []
class TimeMeasuringAlgorithm(NikitaPPO):
    def fit(self, dataset):
        global alg_fit_times
        start = time.perf_counter()
        super().fit(dataset)
        end = time.perf_counter()
        alg_fit_times.append(end - start)

env_step_times = []
world_step_times = []
start_world_step = 0
def create_time_measuring_env(env):
    class TimeMeasuringEnv(env):
        def step_all(self, mask, action):
            global env_step_times

            start = time.perf_counter()
            r = super().step_all(mask, action)
            end = time.perf_counter()
            env_step_times.append(end - start)

            return r
        
        def _simulation_pre_step(self):
            global start_world_step
            start_world_step = time.perf_counter()

        def _simulation_post_step(self):
            global world_step_times
            end = time.perf_counter()
            world_step_times.append(end - start_world_step)

    return TimeMeasuringEnv

start_run = 0
if __name__ == '__main__':
    TorchUtils.set_default_device('cuda:0')

    robot = SilverBadger
    env_class = create_time_measuring_env(robot)
    alg_class = TimeMeasuringAlgorithm

    num_runs = 5
    num_envs = 4096 #TODO TODO Why bug when 1028

    torch.set_printoptions(precision=4)
    start_startup = time.perf_counter()
    mdp = env_class(num_envs, 1000, True, True)
    end_startup = time.perf_counter()
    time_startup = end_startup - start_startup

    n_epochs = 30
    n_steps = 4096*24*50*2
    n_steps_per_fit = int(4096*24)
    n_episodes_test = 256
    batch_size = int((4096*24) / 32)
    lr = 3e-4
    ent_coeff = 0.

    parent_dir_name = "exp_results"
    dir_name = f"{robot.__name__}_{alg_class.__bases__[0].__name__}_{str(time.time())}"
    dir_path = parent_dir_name + "/" + dir_name + "/"

    store_config(dir_path, "config.txt", algorithm=alg_class.__bases__[0].__name__, n_runs=num_runs, n_envs=num_envs, n_epochs=n_epochs, n_steps=n_steps, 
                 n_steps_per_fit=n_steps_per_fit, n_episodes_test=n_episodes_test, batch_size=batch_size,
                 time_startup=time_startup, lr=lr, ent_coeff=ent_coeff)
    
    vs = []
    seeds = []
    time_for_exp = []
    for i in range(num_runs):
        start_run = time.perf_counter()
        alg_fit_times = []
        env_step_times = []
        world_step_times = []
        times_env_stop = []
        seed = mdp.seed(i)
        start_reset = time.perf_counter()
        mdp.stop(soft=False)
        end_reset = time.perf_counter()
        start_exp = time.perf_counter()
        values = experiment(mdp, alg=alg_class, run_idx=i, n_epochs=n_epochs, n_steps=n_steps, 
                   n_steps_per_fit=n_steps_per_fit, n_episodes_test=n_episodes_test, batch_size=batch_size,
                   seed=seed, dir_path=dir_path, lr=lr, ent_coeff=ent_coeff)
        end_exp = time.perf_counter()
        time_for_exp.append((end_exp - start_exp, end_reset - start_reset))
        vs.append(values)
        seeds.append(seed)
    
    store_csv(time_for_exp, dir_path + "times_exp.csv", ("Time needed per Experiment", "Time needer per Hard Reset"))
    
    create_plot([[it[0] for it in v] for v in vs], dir_path, "J", f"PPO - discounted Return: {seeds}")
    create_plot([[it[1] for it in v] for v in vs], dir_path, "R", f"PPO - undiscounted Return: {seeds}")
    create_plot([[it[2] for it in v] for v in vs], dir_path, "E", f"PPO - Entropy: {seeds}")
    create_plot([[it[3] for it in v] for v in vs], dir_path, "V", f"PPO - value of intial states: {seeds}")
    create_plot([[it[4] for it in v] for v in vs], dir_path, "A", f"PPO - Absorbing: {seeds}")

    #times -----------------------------------------------------------------------------------
    create_plot([[it[5] for it in v] for v in vs], dir_path, "Time_Eval", f"Time needed for Evaluation: {seeds}")
    create_plot([[it[6] for it in v[1:]] for v in vs], dir_path, "Time_Learn", f"Time needed for Learning: {seeds}")
    create_plot([[it[7] for it in v[1:]] for v in vs], dir_path, "Time_Since_Start", f"Time since start: {seeds}")
    create_plot([[it[8] for it in v[1:]] for v in vs], dir_path, "Time_Alg_Fit", f"Average Time needed for Fit of PPO: {seeds}")
    create_plot([[it[9] for it in v[1:]] for v in vs], dir_path, "Time_Env_Step", f"Average Time needed for Step in Environment: {seeds}")
    create_plot([[it[10] for it in v[1:]] for v in vs], dir_path, "Time_World_Step", f"Average Time needed for Step in Simulation: {seeds}")
    create_plot([[it[11] for it in v[1:]] for v in vs], dir_path, "Time_Collection", f"Average Time needed for Collection per Fit: {seeds}")
    create_plot([[it[12] for it in v[1:]] for v in vs], dir_path, "Time_Core_Fit", f"Average Time needed for Fit in Core: {seeds}")
