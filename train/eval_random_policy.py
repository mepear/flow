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
    evaluate(eval_envs, args.num_processes, device)
    eval_envs.close()

def evaluate(eval_envs, num_processes, device):

    eval_episode_rewards = []
    reward_composition = {'wait_penalty': [], 'exist_penalty': [], 'pickup_reward': [],
                       'miss_penalty': [], 'tle_penalty': [], 'time_reward': [],
                       'distance_reward': []}
    obs = eval_envs.reset()

    action_masks = None
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
        ob, _, done, infos = eval_envs.step(action)
        try:
            action_masks = torch.cat([info['action_mask'] for info in infos], dim=0)
        except KeyError:
            action_masks = None

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
                reward_composition['wait_penalty'].append(info['episode']['reward_composition']['wait_penalty'])
                reward_composition['exist_penalty'].append(info['episode']['reward_composition']['exist_penalty'])
                reward_composition['pickup_reward'].append(info['episode']['reward_composition']['pickup_reward'])
                reward_composition['miss_penalty'].append(info['episode']['reward_composition']['miss_penalty'])
                reward_composition['time_reward'].append(info['episode']['reward_composition']['time_reward'])
                reward_composition['distance_reward'].append(info['episode']['reward_composition']['distance_reward'])
                reward_composition['tle_penalty'].append(info['episode']['reward_composition']['tle_penalty'])

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
    print('eval/pickup_reward mean {:.2f} median {:.2f}'.format(\
            np.mean(reward_composition['pickup_reward']), np.median(reward_composition['pickup_reward'])))
    print('eval/time_reward mean {:.2f} median {:.2f}'.format(\
            np.mean(reward_composition['time_reward']), np.median(reward_composition['time_reward'])))
    print('eval/distance_reward mean {:.2f} median {:.2f}'.format(\
            np.mean(reward_composition['distance_reward']), np.median(reward_composition['distance_reward'])))


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
