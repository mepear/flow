"""Runner script for simple random-policy experiments.

This script performs a random policy on flow environment. Choice of
hyperparameters can be seen and adjusted from the code below.

Usage
    python train.py EXP_CONFIG
"""
import argparse
import json
import os
from time import strftime
from copy import deepcopy
import sys
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from myppo.a2c_ppo_acktr.arguments import get_args
from myppo.a2c_ppo_acktr.envs import make_vec_envs
from myppo.a2c_ppo_acktr import utils
from myppo.a2c_ppo_acktr.model import Policy
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

    return parser.parse_known_args(args)[0]

def eval_random_policy(submodule):
    flow_params = submodule.flow_params
    flow_params['sim'].render = False
    args = get_args(sys.argv[2:])

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    save_path = os.path.join(os.path.join(args.save_dir, args.algo), args.experiment_name)

    flow_params['sim'].render = not args.disable_render_during_eval
    eval_envs = make_vec_envs(args.env_name, args.seed, args.num_processes,\
        None, save_path, True, device=device, flow_params=flow_params, verbose=True)
    evaluate(eval_envs, args.num_processes, device, save_path=None,\
        verbose=True)
    eval_envs.close()

def evaluate(eval_envs, num_processes, device, save_path=None, writer=None, \
    total_num_steps=None, verbose=False):

    eval_episode_rewards = []
    nums_orders = []
    nums_complete_orders = []
    total_pickup_distances = []
    total_pickup_times = []
    total_valid_distances = []
    total_valid_times = []
    total_wait_times = []
    total_congestion_rates = []
    mean_velocities = []
    edge_position = None
    statistics = []

    obs = eval_envs.reset()

    action_masks = None
    values = []
    while len(eval_episode_rewards) < num_processes:

        actor_critic = Policy(
            eval_envs.observation_space.shape,
            eval_envs.action_space,
            base_kwargs={'recurrent': False})
        actor_critic.to(device)

        eval_recurrent_hidden_states = torch.zeros(
            actor_critic.recurrent_hidden_state_size, device=device)

        eval_masks = torch.zeros(1, device=device)
        value, action, _, eval_recurrent_hidden_states = actor_critic.act(
            obs,
            eval_recurrent_hidden_states,
            eval_masks,
            action_masks=action_masks,
            deterministic=True)

        # Observe reward and next obs

        # Add Edge Index 0,13,38,51
        ob, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        try:
            action_masks = torch.cat([info['action_mask'] for info in infos], dim=0)
        except KeyError:
            action_masks = None

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
                nums_orders.append(info['episode']['num_orders'])
                nums_complete_orders.append(info['episode']['num_complete_orders'])
                total_pickup_distances.append(info['episode']['total_pickup_distance'])
                total_pickup_times.append(info['episode']['total_pickup_time'])
                total_valid_distances.append(info['episode']['total_valid_distance'])
                total_valid_times.append(info['episode']['total_valid_time'])
                total_wait_times.append(info['episode']['total_wait_time'])
                total_congestion_rates.append(info['episode']['total_congestion_rate'])
                mean_velocities.append(info['episode']['mean_velocity'])
                edge_position = info['episode']['edge_position']
                statistics.append(info['episode']['statistics'])

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
        print('rewards/eval mean {:.2f} median {:.2f}'.format( \
            np.mean(eval_episode_rewards), np.median(eval_episode_rewards)))
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
        print('eval/order mean {:.2f} median {:.2f}'.format( \
            np.mean(nums_complete_orders), np.median(nums_complete_orders)))
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
            print('eval/pickup_distance mean {:.2f} median {:.2f}'.format( \
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
            print('eval/pickup_time mean {:.2f} median {:.2f}'.format( \
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
            print('eval/valid_distance mean {:.2f} median {:.2f}'.format( \
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
            print('eval/valid_time mean {:.2f} median {:.2f}'.format( \
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
            print('eval/wait_time mean {:.2f} median {:.2f}'.format( \
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
        print('eval/congestion_rate mean {:.2f} median {:.2f}'.format( \
            np.mean(total_congestion_rates), np.median(total_congestion_rates)))

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

    eval_random_policy(submodule)

if __name__ == "__main__":
    main(sys.argv[1:])
