"""Runner script for plot correlation experiments.

Choice of hyper-parameters can be seen and adjusted from the code below.
"""

import multiprocessing
import argparse
import json
from functools import partial
import os
from time import strftime
from copy import deepcopy
import sys
import random
import time
import datetime
from collections import deque

import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from myppo.a2c_ppo_acktr.arguments import get_args
from myppo.a2c_ppo_acktr.envs import make_vec_envs
from myppo.a2c_ppo_acktr import utils
from flow.core.util import ensure_dir
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from flow.utils.registry import make_create_env


def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a Flow simulation.",
        epilog="python train.py EXP_CONFIG")

    # required input parameters
    parser.add_argument(
        'exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/rl/singleagent or exp_configs/rl/multiagent.')
    parser.add_argument(
        '--checkpoint', type=int)

    return parser.parse_known_args(args)[0]

def plot_congestion(submodule):

    flow_params = submodule.flow_params
    flow_params['sim'].render = False
    args = get_args(sys.argv[2:])

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)

    a = datetime.datetime.now().microsecond % 5

    time.sleep(a * 5)
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    device = torch.device("cuda:{}".format(np.argmax(memory_gpu)) if args.cuda else "cpu")

    print("*" * 10)
    print("Device: {}".format(device))
    print("*" * 10)

    save_path = os.path.join(os.path.join(args.save_dir, args.algo), args.experiment_name)
    save_path_2 = os.path.join(os.path.join(args.save_dir, args.algo), args.experiment_name_2)
    save_path_3 = os.path.join(os.path.join(args.save_dir, args.algo), args.experiment_name_3)
    pt = os.path.join(save_path, str(args.eval_ckpt) + ".pt")
    pt_2 = os.path.join(save_path_2, str(args.eval_ckpt) + ".pt")
    pt_3 = os.path.join(save_path_3, str(args.eval_ckpt) + ".pt")
    actor_critic, ob_rms = torch.load(pt, map_location='cpu')
    actor_critic.to(device)
    actor_critic_2 = None
    ob_rms_2 = None
    actor_critic_3 = None
    ob_rms_3 = None
    if args.experiment_name_2 != 'default':
        actor_critic_2, ob_rms_2 = torch.load(pt_2, map_location='cpu')
        actor_critic_2.to(device)
    if args.experiment_name_3 != 'default':
        actor_critic_3, ob_rms_3 = torch.load(pt_3, map_location='cpu')
        actor_critic_3.to(device)

    screenshot_path = os.path.join(save_path, "images") if args.save_screenshot else None

    flow_params['sim'].render = not args.disable_render_during_eval
    flow_params['sim'].save_render = screenshot_path
    if args.random_rate != None:
        flow_params['env'].additional_params['distribution_random_ratio'] = float(args.random_rate) / 100
    eval_envs = make_vec_envs(args.env_name, args.seed, args.num_processes, \
        None, save_path, True, device=device, flow_params=flow_params, verbose=args.verbose, evaluate_id=args.random_rate)
    evaluate(actor_critic, eval_envs, ob_rms, args.num_processes, device, save_path=save_path, \
        do_plot_congestion=args.plot_congestion, ckpt=args.eval_ckpt, verbose=True, \
             actor_critic_2=actor_critic_2, actor_critic_3=actor_critic_3,
             ob_rms_2=ob_rms_2, ob_rms_3=ob_rms_3, random_rate=args.random_rate)
    eval_envs.close()

def evaluate(actor_critic, eval_envs, ob_rms, num_processes, device, save_path=None, writer=None, \
    total_num_steps=None, do_plot_congestion=False, ckpt=None, verbose=False, ob_rms_2=None, \
             ob_rms_3=None, actor_critic_2=None, actor_critic_3=None, random_rate=None):

    # # flow_params['sim'].render = True
    # eval_envs = make_vec_envs(env_name, seed, num_processes,
    #                           None, eval_log_dir, device, True, flow_params=flow_params, port=port, verbose=verbose)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []
    nums_orders = []
    nums_complete_orders = []
    reservation_before_end = []
    total_pickup_distances = []
    total_pickup_times = []
    total_valid_distances = []
    total_valid_times = []
    total_wait_times = []
    total_congestion_rates = []
    mean_velocities = []
    reward_composition = {'wait_penalty': [], 'exist_penalty': [], 'pickup_reward': [],
                       'miss_penalty': [], 'tle_penalty': [], 'time_reward': [],
                       'distance_reward': []}
    background_velocities = [[] for _ in range(eval_envs.num_envs)]
    background_co2s = [[] for _ in range(eval_envs.num_envs)]
    taxi_velocities = [[] for _ in range(eval_envs.num_envs)]
    taxi_co2s = [[] for _ in range(eval_envs.num_envs)]
    background_cos = [[] for _ in range(eval_envs.num_envs)]
    taxi_cos = [[] for _ in range(eval_envs.num_envs)]
    total_taxi_distances = [[] for _ in range(eval_envs.num_envs)]
    total_back_distances = [[] for _ in range(eval_envs.num_envs)]
    edge_position = None
    statistics = []
    rev_count = None
    nearest_prob = None
    nearest_dist_per_rev = None
    true_dist_per_rev = None

    obs = eval_envs.reset()

    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)
    action_masks = None
    values = []
    while len(eval_episode_rewards) < num_processes:
        with torch.no_grad():
            if actor_critic_2 == None:
                value, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    action_masks=action_masks,
                    deterministic=True)
            else:
                obs_1 = np.clip((obs.cpu().numpy() - ob_rms.mean) / np.sqrt(ob_rms.var + eval_envs.epsilon),
                        -eval_envs.clipob, eval_envs.clipob)
                obs_1 = torch.from_numpy(obs_1).to(device).float()
                obs_2 = np.clip((obs.cpu().numpy() - ob_rms_2.mean) / np.sqrt(ob_rms_2.var + eval_envs.epsilon),
                        -eval_envs.clipob, eval_envs.clipob)
                obs_2 = torch.from_numpy(obs_2).to(device).float()
                obs_3 = np.clip((obs.cpu().numpy() - ob_rms_3.mean) / np.sqrt(ob_rms_3.var + eval_envs.epsilon),
                        -eval_envs.clipob, eval_envs.clipob)
                obs_3 = torch.from_numpy(obs_3).to(device).float()
                value_1, action_1, _, eval_recurrent_hidden_states = actor_critic.act(
                    obs_1,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    action_masks=action_masks,
                    deterministic=True)
                value_2, action_2, _, eval_recurrent_hidden_states = actor_critic_2.act(
                    obs_2,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    action_masks=action_masks,
                    deterministic=True)
                value_3, action_3, _, eval_recurrent_hidden_states = actor_critic_3.act(
                    obs_3,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    action_masks=action_masks,
                    deterministic=True)
                action_1 = action_1[:, 0].unsqueeze(1)
                action_2 = action_2[:, 1].unsqueeze(1)
                action_3 = action_3[:, 2].unsqueeze(1)
                action = torch.cat((action_1, action_2, action_3), 1)


        # values.append(value) # To be used

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)
        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)
        try:
            action_masks = torch.cat([info['action_mask'] for info in infos], dim=0)
        except KeyError:
            action_masks = None

        for i, info in enumerate(infos):
            background_velocities[i].append(info['background_velocity'])
            background_co2s[i].append(info['background_co2'])
            taxi_velocities[i].append(info['taxi_velocity'])
            taxi_co2s[i].append(info['taxi_co2'])
            background_cos[i].append(info['background_co'])
            taxi_cos[i].append(info['taxi_co'])
            total_taxi_distances[i].append(info['total_taxi_distance'])
            # total_back_distances[i].append(info['total_back_distance'])
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
                nums_orders.append(info['episode']['num_orders'])
                nums_complete_orders.append(info['episode']['num_complete_orders'])
                reservation_before_end.append(info['episode']['reservation_before_end'])
                reward_composition['wait_penalty'].append(info['episode']['reward_composition']['wait_penalty'])
                reward_composition['exist_penalty'].append(info['episode']['reward_composition']['exist_penalty'])
                reward_composition['pickup_reward'].append(info['episode']['reward_composition']['pickup_reward'])
                reward_composition['miss_penalty'].append(info['episode']['reward_composition']['miss_penalty'])
                reward_composition['time_reward'].append(info['episode']['reward_composition']['time_reward'])
                reward_composition['distance_reward'].append(info['episode']['reward_composition']['distance_reward'])
                reward_composition['tle_penalty'].append(info['episode']['reward_composition']['tle_penalty'])
                total_pickup_distances.append(info['episode']['total_pickup_distance'])
                total_pickup_times.append(info['episode']['total_pickup_time'])
                total_valid_distances.append(info['episode']['total_valid_distance'])
                # total_distances[i].append(info['episode']['total_distance'])
                total_valid_times.append(info['episode']['total_valid_time'])
                total_wait_times.append(info['episode']['total_wait_time'])
                total_congestion_rates.append(np.mean(info['episode']['congestion_rates']))
                mean_velocities.append(np.mean(info['episode']['mean_velocities'], axis=0))
                edge_position = info['episode']['edge_position']
                statistics.append(info['episode']['statistics'])
                rev_count = info['episode']['rev_count']
                nearest_prob = info['episode']['prob_nearest']
                nearest_dist_per_rev = info['episode']['nearest_dist_per_rev']
                true_dist_per_rev = info['episode']['true_dist_per_rev']

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
    if save_path:
        with open(os.path.join(save_path, 'eval.txt'), 'a') as f:
            f.write(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
            len(eval_episode_rewards), np.mean(eval_episode_rewards)))
    # writer.add_scalar('rewards/eval', np.mean(eval_episode_rewards), total_num_steps)
    denom = np.array(nums_complete_orders, dtype=int)
    denom2 = np.array(nums_orders, dtype=int)
    if writer and total_num_steps:
        writer.add_scalars(
                "rewards/eval",
                {
                    "mean": np.mean(eval_episode_rewards),
                    "median": np.median(eval_episode_rewards),
                    "max": np.max(eval_episode_rewards),
                    "min": np.min(eval_episode_rewards)
                },
            total_num_steps
            )
    if verbose:
        print('rewards/eval mean {:.2f} median {:.2f} std {:.2f}'.format(\
            np.mean(eval_episode_rewards), np.median(eval_episode_rewards), np.std(eval_episode_rewards)))
    if writer and total_num_steps:
        writer.add_scalars(
                "eval/order",
                {
                    "mean": np.mean(nums_complete_orders),
                    "median": np.median(nums_complete_orders),
                    "max": np.max(nums_complete_orders),
                    "min": np.min(nums_complete_orders)
                },
            total_num_steps
            )
    if verbose:
        print('eval/order mean {:.2f} median {:.2f} std {:.2f}'.format(\
            np.mean(nums_complete_orders), np.median(nums_complete_orders), np.std(nums_complete_orders)))
        print('eval/reward_per_order {:.2f} '.format(np.mean(eval_episode_rewards) / np.mean(nums_complete_orders)))
        print('eva/reward_per_order (single) mean {:.2f} median {:.2f} std {:.2f}'.format(
            (np.array(eval_episode_rewards) / np.array(nums_complete_orders)).mean(),
            np.median(np.array(eval_episode_rewards) / np.array(nums_complete_orders)),
            (np.array(eval_episode_rewards) / np.array(nums_complete_orders)).std()
        ))
    if not np.alltrue(denom == 0):
        array = np.ma.masked_invalid(total_pickup_distances / denom)
        if writer and total_num_steps:
            writer.add_scalars(
                    "eval/pickup_distance",
                    {
                        "mean": np.nanmean(array),
                        "median": np.nanmedian(array),
                        "max": np.nanmax(array),
                        "min": np.nanmin(array)
                    },
                total_num_steps
                )
        if verbose:
            print('eval/pickup_distance mean {:.2f} median {:.2f}'.format(\
                np.nanmean(array), np.nanmedian(array)))
        array = np.ma.masked_invalid(total_pickup_times / denom)
        if writer and total_num_steps:
            writer.add_scalars(
                    "eval/pickup_time",
                    {
                        "mean": np.nanmean(array),
                        "median": np.nanmedian(array),
                        "max": np.nanmax(array),
                        "min": np.nanmin(array)
                    },
                total_num_steps
                )
        if verbose:
            print('eval/pickup_time mean {:.2f} median {:.2f}'.format(\
                np.nanmean(array), np.nanmedian(array)))
        array = np.ma.masked_invalid(total_valid_distances / denom)
        if writer and total_num_steps:
            writer.add_scalars(
                    "eval/valid_distance",
                    {
                        "mean": np.nanmean(array),
                        "median": np.nanmedian(array),
                        "max": np.nanmax(array),
                        "min": np.nanmin(array)
                    },
                total_num_steps
                )
        if verbose:
            print('eval/valid_distance mean {:.2f} median {:.2f}'.format(\
                np.nanmean(array), np.nanmedian(array)))
        array = np.ma.masked_invalid(total_valid_times / denom)
        if writer and total_num_steps:
            writer.add_scalars(
                    "eval/valid_time",
                    {
                        "mean": np.nanmean(array),
                        "median": np.nanmedian(array),
                        "max": np.nanmax(array),
                        "min": np.nanmin(array)
                    },
                total_num_steps
                )
        if verbose:
            print('eval/valid_time mean {:.2f} median {:.2f}'.format(\
                np.nanmean(array), np.nanmedian(array)))
    if not np.alltrue(denom2 == 0):
        array = np.ma.masked_invalid(total_wait_times / denom2)
        if writer and total_num_steps:
            writer.add_scalars(
                    "eval/wait_time",
                    {
                        "mean": np.nanmean(array),
                        "median": np.nanmedian(array),
                        "max": np.nanmax(array),
                        "min": np.nanmin(array)
                    },
                total_num_steps
                )
        if verbose:
            print('eval/wait_time mean {:.2f} median {:.2f}'.format(\
                np.nanmean(array), np.nanmedian(array)))
    if writer and total_num_steps:
        writer.add_scalars(
                "eval/congestion_rate",
                {
                    "mean": np.mean(total_congestion_rates),
                    "median": np.median(total_congestion_rates),
                    "max": np.max(total_congestion_rates),
                    "min": np.min(total_congestion_rates)
                },
            total_num_steps
            )
    if verbose:
        print('eval/congestion_rate mean {:.2f} median {:.2f}'.format(\
            np.mean(total_congestion_rates), np.median(total_congestion_rates)))
    if verbose:
        print('eval/pickup_reward mean {:.2f} median {:.2f}'.format(\
            np.mean(reward_composition['pickup_reward']), np.median(reward_composition['pickup_reward'])))
        print('eval/time_reward mean {:.2f} median {:.2f}'.format(\
            np.mean(reward_composition['time_reward']), np.median(reward_composition['time_reward'])))
        print('eval/distance_reward mean {:.2f} median {:.2f}'.format(\
            np.mean(reward_composition['distance_reward']), np.median(reward_composition['distance_reward'])))
        print('eval/reservation_before_end mean {:.2f} median {:.2f}'.format(\
            np.mean(reservation_before_end), np.median(reservation_before_end)))
    if verbose:
        print('eval/rev_count mean {:.2f} median {:.2f}'.format(\
            np.mean(rev_count), np.median(rev_count)))
        print('eval/nearest_prob mean {:.2f} median {:.2f}'.format(\
            np.mean(nearest_prob), np.median(nearest_prob)))
        print('eval/nearest_dist_per_rev mean{: .2f} median {: .2f}'.format(\
            np.mean(nearest_dist_per_rev), np.median(nearest_dist_per_rev)))
        print('eval/true_dist_per_rev mean{: .2f} median {: .2f}'.format(\
            np.mean(true_dist_per_rev), np.median(true_dist_per_rev)))

    df = pd.DataFrame({'reward': [], "order_num": []})
    df.to_csv("./data/plot_{}.csv".format(int(random_rate)), index=False, sep=',')
    reward = np.mean(eval_episode_rewards)
    orders = np.mean(nums_complete_orders)
    reward = round(reward, 6)

    data = [reward, orders]
    df = pd.read_csv('./data/plot_{}.csv'.format(int(random_rate)))
    df.loc[1] = data
    df.to_csv("./data/plot_{}.csv".format(int(random_rate)), index=False, sep=',')

def main(args):
    """Perform the training operations."""
    # Parse script-level arguments (not including package arguments).
    flags = parse_args(args)
    # Import relevant information from the exp_config script.
    module = __import__(
        "exp_configs.rl.singleagent", fromlist=[flags.exp_config])
    module_ma = __import__(
        "exp_configs.rl.multiagent", fromlist=[flags.exp_config])

    # Import the sub-module containing the specified exp_config and determine
    # whether the environment is single agent or multi-agent.
    if hasattr(module, flags.exp_config):
        submodule = getattr(module, flags.exp_config)
        multiagent = False
    elif hasattr(module_ma, flags.exp_config):
        submodule = getattr(module_ma, flags.exp_config)
        assert flags.rl_trainer.lower() in ["rllib", "h-baselines"], \
            "Currently, multiagent experiments are only supported through "\
            "RLlib. Try running this experiment using RLlib: " \
            "'python train.py EXP_CONFIG'"
        multiagent = True
    else:
        raise ValueError("Unable to find experiment config.")

    plot_congestion(submodule)


if __name__ == "__main__":
    main(sys.argv[1:])