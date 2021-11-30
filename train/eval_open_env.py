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
import random
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.spaces.multi_discrete import MultiDiscrete

from myppo.a2c_ppo_acktr.arguments import get_args
from myppo.a2c_ppo_acktr.envs import make_vec_envs
from myppo.a2c_ppo_acktr import utils
from myppo.a2c_ppo_acktr.model import Policy
from flow.core.util import ensure_dir
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from flow.utils.registry import make_create_env

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from functools import partial

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

    area_idx = submodule.flow_params['net'].additional_params['grid_array']['row_idx'] * submodule.flow_params['net'].additional_params['grid_array']['col_idx']
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    save_path = os.path.join(os.path.join(args.save_dir, args.algo), args.experiment_name)
    pt = os.path.join(save_path, str(args.eval_ckpt) + ".pt")
    actor_critic, ob_rms = torch.load(pt, map_location='cpu')
    actor_critic.to(device)

    screenshot_path = os.path.join(save_path, "images") if args.save_screenshot else None

    flow_params['sim'].render = not args.disable_render_during_eval
    flow_params['sim'].save_render = screenshot_path
    eval_envs = make_vec_envs(args.env_name, args.seed, args.num_processes, \
        None, save_path, True, device=device, flow_params=flow_params, verbose=args.verbose, is_open=True)
    evaluate(actor_critic, eval_envs, ob_rms, args.num_processes, device, area_idx, save_path=save_path,
             ckpt=args.eval_ckpt, do_plot_congestion=args.plot_congestion)
    eval_envs.close()

def evaluate(actor_critic, eval_envs, ob_rms, num_processes, device, area_idx, save_path,
             ckpt, do_plot_congestion = False):

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = [[] for _ in range(area_idx)]
    nums_complete_orders = []
    nums_orders = []
    statistics = []
    eval_recurrent_hidden_states = torch.zeros(area_idx,
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(area_idx * num_processes, 1, device=device) # torch.zeros(num_processes, 1, device=device)

    obs = eval_envs.reset()
    action_masks = [None for _ in range(area_idx)]
    while len(eval_episode_rewards[0]) < num_processes:
        action = [None for _ in range(area_idx)]
        for i in range(area_idx):
            _, action[i], _, _ = actor_critic.act(
                obs[:, i],
                eval_recurrent_hidden_states[i],
                eval_masks,
                action_masks=action_masks[i],
                deterministic=True)

        # Observe reward and next obs
        action = torch.stack(action, dim=1)

        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0] if done_ else [1.0, 1.0, 1.0, 1.0] for done_ in done],
            dtype=torch.float32,
            device=device)
        eval_masks = torch.cat([eval_masks[i] for i in range(num_processes)], dim=0)
        try:
            action_masks = [None for _ in range(area_idx)]
            for i in range(area_idx):
                action_masks[i] = torch.cat([info['action_mask'][i] for info in infos], dim=0)
        except KeyError:
            pass

        for info in infos:
            if 'episode' in info.keys():
                for i in range(area_idx):
                    eval_episode_rewards[i].append(info['episode']['r'][i])
                nums_complete_orders.append(info['episode']['num_complete_orders'])
                nums_orders.append(info['episode']['num_orders'])
                statistics.append(info['episode']['statistics'])
                edge_position = info['episode']['edge']

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards[0]), np.mean([np.mean(eval_episode_rewards[i]) for i in range(area_idx)])))
    for i in range(area_idx):
        print("Area_{} mean reward {:.5f}\n".format(
            i, np.mean(eval_episode_rewards[i])))
    print('eval/order mean {:.2f} median {:.2f} std {:.2f}'.format( \
        np.mean(nums_complete_orders), np.median(nums_complete_orders), np.std(nums_complete_orders)))

    # if do_plot_congestion:
    #     plot_congestion(edge_position, statistics, save_path, ckpt)

def get_corners(s, e, w):
    s, e = np.array(s), np.array(e)
    se = e - s
    p = np.array([-se[1], se[0]])
    p = p / np.linalg.norm(p) * w / 2
    return [s + p, e + p, e - p, s - p]


def plot(statistics, edge_position, key1, key2, name, n, m, idx, cmap, tp=None, norm=False, vmin=None, vmax=None):
    ax = plt.subplot(n, m, idx)

    if tp is None:
        cnts = np.mean([sta[key1][key2] for sta in statistics], axis=0)
    else:
        cnts = np.mean([sta[key1][key2][tp] for sta in statistics], axis=0)
    if norm:
        cnts /= cnts.max()
    colors = []
    patches = []
    x_min, x_max, y_min, y_max = 1e9, -1e9, 1e9, -1e9
    for cnt, (start, end, width) in zip(cnts, edge_position):
        vertices = get_corners(start, end, width)
        # assert cnt >= 0 and cnt <= 1, cnts
        poly = Polygon(vertices)
        colors.append(cnt)
        patches.append(poly)
        for vertex in vertices:
            x_min, y_min = min(x_min, vertex[0]), min(y_min, vertex[1])
            x_max, y_max = max(x_max, vertex[0]), max(y_max, vertex[1])
    if norm:
        colors = np.array(colors) * 100
    else:
        colors = np.array(colors)
    p = PatchCollection(patches)
    p.set_cmap(cmap)
    p.set_array(colors)
    if not norm:
        p.set_clim(vmin=vmin, vmax=vmax)
    ax.add_collection(p)
    plt.xlim(x_min - 5, x_max + 5)
    plt.ylim(y_min - 5, y_max + 5)
    plt.title(name)
    plt.xticks([]), plt.yticks([])
    if not norm:
        plt.colorbar(p)


def draw(plotter, n_tp, cmap, ckpt, save_path, norm=True):
    # route and location
    ## background
    # plotter('route', 'background', 'background', 3, 2, 3, cmap)
    ## free
    plotter = partial(plotter, norm=norm)
    plotter('route', 'free', 'free', 2, n_tp + 1, n_tp + 1, cmap, vmin=0, vmax=30)
    ## reposition
    plotter('location', 'reposition', 'reposition location', 2, n_tp + 1, 2 * n_tp + 2, cmap, vmin=0, vmax=20)

    for i in range(n_tp):
        plotter('route', 'pickup', 'pickup {}'.format(i), 2, n_tp + 1, i + 1, cmap, tp=i, vmin=0, vmax=10)
        plotter('route', 'occupied', 'occupied {}'.format(i), 2, n_tp + 1, i + n_tp + 2, cmap, tp=i, vmin=0, vmax=10)
    ## pickup0
    # plotter('route', 'pickup', 'pickup 0', 2, 2, 1, cmap, tp=0)
    ## pickup1
    # plotter('route', 'pickup', 'pickup 1', 3, 3, 2, cmap, tp=1)
    ## occupied0
    # plotter('route', 'occupied', 'occupied 0', 2, 2, 2, cmap, tp=0)
    ## occupied1
    # plotter('route', 'occupied', 'occupied 1', 3, 3, 5, cmap, tp=1)

    plt.suptitle('routes_and_locations_in_open_env from ckpt {}'.format(ckpt), y=0.00)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'routes_and_locations_in_open_env{}.{}.jpg'.format(ckpt, norm)), \
        dpi=500, bbox_inches='tight')


def plot_congestion(mean_velocities, edge_position, statistics, save_path, ckpt):
    # fig, ax = plt.subplots()
    # cmap = plt.get_cmap('YlGn')
    # mean_vels = np.mean(mean_velocities, axis=0)
    # for vel, (start, end, width) in zip(mean_vels, edge_position):
    #     vertices = get_corners(start, end, width)
    #     assert vel > 0 and vel <= 1
    #     poly = Polygon(vertices, color=cmap(1 - vel))
    #     ax.add_patch(poly)
    # plt.xlim(-5., 160.)
    # plt.ylim(-5., 160.)
    # plt.savefig(os.path.join(save_path, 'congestion_{}.jpg'.format(ckpt)), dpi=500)

    # fig, ax = plt.subplots()
    # plt.bar(np.arange(len(mean_vels)), np.sort(1 - mean_vels))
    # plt.ylim(0., 1.)
    # plt.savefig(os.path.join(save_path, 'distribution_{}.jpg'.format(ckpt)), dpi=500)
    # cmap1 = plt.get_cmap('Greys')
    cmap2 = plt.get_cmap('YlGn')
    plotter = partial(plot, statistics, edge_position)
    # draw(plotter, 2, cmap1, ckpt, save_path, norm=True)
    draw(plotter, 2, cmap2, ckpt, save_path, norm=False)


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
