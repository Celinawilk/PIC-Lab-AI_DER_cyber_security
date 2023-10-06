# Copyright 2021 Siemens Corporation
# SPDX-License-Identifier: MIT

"""Random agent to probe enviroment
"""
import matplotlib.pyplot as plt
import numpy as np
import imageio
import glob

from powergym.env_register import make_env, remove_parallel_dss

from powergym.env import Env

import argparse
import random
import itertools
import sys, os
import multiprocessing as mp

from datetime import datetime

from stable_baselines3 import A2C

models_dir = "models/A2C"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--env_name', default='13Bus_PV_Attack')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed')
    parser.add_argument('--num_steps', type=int, default=500, metavar='N',
                        help='maximum number of steps')
    parser.add_argument('--num_workers', type=int, default=3, metavar='N',
                        help='number of parallel processes')
    parser.add_argument('--use_plot', type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument('--do_testing', type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument('--mode', type=str, default='episodic',
                        help="running mode, random, parallele, episodic or dss")
    args = parser.parse_args()
    return args

def seeding(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def a2c_evaluate(model, env, profiles, episodes=None):
    returns = np.zeros(len(profiles)) if episodes is None else np.zeros(min(episodes, len(profiles)))
    for i in range(len(returns)):
        pidx = profiles[i] if episodes is None else random.choice(profiles)
        obs = env.reset(load_profile_idx=pidx)
        episode_reward = 0
        episode_steps = 0
        done = False
        while not done:
            action, _states = model.predict(obs)
            next_obs, reward, done, _ = env.step(action)
            episode_reward += reward
         #   violated_nodes = []
          #  total_violation = 0
            #for name, voltages in env.obs['bus_voltages'].items():
             #   max_penalty = min(0, 1.05 - max(voltages))  # penalty is negative if above max
              #  min_penalty = min(0, min(voltages) - 0.95)  # penalty is negative if below min
               # total_violation += (max_penalty + min_penalty)
                #if max_penalty != 0 or min_penalty != 0:
                 #   violated_nodes.append(name)
                  #  print('profile: {}'.format(profiles[i]))
              #      print('step: {}'.format(episode_steps))
               #     print(violated_nodes)
                #    print(total_violation)
            episode_steps += 1
            mask = 1 if episode_steps == env.horizon else float(not done)
            ## check voltage violations here, and print out the load profile index, which bus index
            #if env.obs['bus_voltages'] < 0.95 or env.obs['bus_voltages'] > 1.05:
            obs = next_obs
        returns[i] = episode_reward
    return returns.mean(), returns.std()

def run_episodic_a2c_agent(args, worker_idx=None):
    """Run a episodic random agent with train-test split

    """
    # output file
    if args.do_testing:
        fout = open('C:/Users/wilke/Downloads/powergym-master/systems/13_Bus_VVC.csv', 'w')



    # get environment
    env = make_env(args.env_name, worker_idx=worker_idx)
    env.reset()

    env.seed(args.seed + 0 if worker_idx is None else worker_idx)

    # get obs, act
    obs_dim = env.observation_space.shape[0]
    CRB_num = (env.cap_num, env.reg_num, env.bat_num)
    CRB_dim = (2, env.reg_act_num, env.bat_act_num)
    print('NumCap, NumReg, NumBat: {}'.format(CRB_num))
    print('ObsDim, ActDim: {}, {}'.format(obs_dim, sum(CRB_num)))
    print('-' * 80)

    # train-test split
    if args.do_testing:
        train_profiles = random.sample(range(env.num_profiles), k=env.num_profiles // 2)
        test_profiles = [i for i in range(env.num_profiles) if i not in train_profiles]
    else:
        train_profiles = list(range(env.num_profiles))

    # Training Loop
    total_numsteps = 0

    model = A2C("MlpPolicy", env, verbose=1, seed=4)



    for i_episode in itertools.count(start=1):
        model.learn(total_timesteps=total_numsteps)
        model.save(f"{models_dir}")
        episode_reward = 0
        episode_steps = 0
        done = False
        load_profile_idx = random.choice(train_profiles)
        obs = env.reset(load_profile_idx=load_profile_idx)

        while not done:
            #action = env.dummy_action()  # Sample random action
            action, _states = model.predict(obs)
            next_obs, reward, done, info = env.step(action)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            mask = 1 if episode_steps == env.horizon else float(not done)
            obs = next_obs

        print("episode: {}, profile: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode,
                                                                                                   load_profile_idx,
                                                                                                   total_numsteps,
                                                                                                   episode_steps,
                                                                                                   round(episode_reward,
                                                                                                         2)))

        total_numsteps += 24

        if total_numsteps >= args.num_steps: break

    if args.do_testing:
        avg_reward, std = a2c_evaluate(model, env, test_profiles)
        fout.write('{},{},{}\n'.format(total_numsteps, avg_reward, std))
        fout.flush()
        print("----------------------------------------")
        print("Avg., Std. Reward: {}, {}".format(round(avg_reward, 2), round(std, 2)))
        print("----------------------------------------")

        #seed = seed + 1

        #args.seed = args.seed + 1


    # save the model
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
    model_name = models_dir+"_"+date_time
    model.save(model_name)


if __name__ == '__main__':
    args = parse_arguments()
    seeding(args.seed)
    if args.mode.lower() == 'episodic':
        run_episodic_a2c_agent(args)

    else:
        raise NotImplementedError("Running mode not implemented")