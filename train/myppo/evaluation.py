import numpy as np
import torch
import os
import traci

from .a2c_ppo_acktr import utils
from .a2c_ppo_acktr.envs import make_vec_envs

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def evaluate(actor_critic, eval_envs, ob_rms, num_processes, device, \
    save_path=None, writer=None, total_num_steps=None, do_plot_congestion=False, ckpt=None):
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

    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)
    action_masks = None
    values = []
    while len(eval_episode_rewards) < num_processes:
        with torch.no_grad():
            value, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                action_masks=action_masks,
                deterministic=True)
        
        values.append(value) # To be used

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
    if writer:
        assert total_num_steps is not None
        # writer.add_scalar('rewards/eval', np.mean(eval_episode_rewards), total_num_steps)
        denom = np.array(nums_complete_orders, dtype=int)
        denom2 = np.array(nums_orders, dtype=int)
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
        if not np.alltrue(denom == 0):
            array = np.ma.masked_invalid(total_pickup_distances / denom)
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
            array = np.ma.masked_invalid(total_pickup_times / denom)
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
            array = np.ma.masked_invalid(total_valid_distances / denom)
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
            array = np.ma.masked_invalid(total_valid_times / denom)
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
        if not np.alltrue(denom2 == 0):
            array = np.ma.masked_invalid(total_wait_times / denom2)
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
    
    if do_plot_congestion:
        plot_congestion(mean_velocities, edge_position, statistics, save_path, ckpt)


def get_corners(s, e, w):
    s, e = np.array(s), np.array(e)
    se = e - s
    p = np.array([-se[1], se[0]])
    p = p / np.linalg.norm(p) * w / 2
    return [s + p, e + p, e - p, s - p]


def plot(key1, key2, name, n, m, idx, cmap, tp=None):
    ax = plt.subplot(n, m, idx)
    if tp is None:
        cnts = np.mean([sta[key1][key2] for sta in statistics], axis=0)
    else:
        cnts = np.mean([sta[key1][key2][tp] for sta in statistics], axis=0)
    cnts /= cnts.max()
    colors = []
    patches = []
    for cnt, (start, end, width) in zip(cnts, edge_position):
        vertices = get_corners(start, end, width)
        assert cnt >= 0 and cnt <= 1
        poly = Polygon(vertices)
        colors.append(cnt)
        patches.append(poly)
    colors = np.array(colors) * 100
    p = PatchCollection(patches)
    p.set_cmap(cmap)
    p.set_array(colors)
    ax.add_collection(p)
    plt.xlim(-5., 160.)
    plt.ylim(-5., 160.)
    plt.title(name)
    plt.xticks([]), plt.yticks([])


def plot_congestion(mean_velocities, edge_position, statistics, save_path, ckpt):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('Greys')
    mean_vels = np.mean(mean_velocities, axis=0)
    for vel, (start, end, width) in zip(mean_vels, edge_position):
        vertices = get_corners(start, end, width)
        assert vel > 0 and vel <= 1
        poly = Polygon(vertices, color=cmap(1 - vel))
        ax.add_patch(poly)
    plt.xlim(-5., 160.)
    plt.ylim(-5., 160.)
    plt.savefig(os.path.join(save_path, 'congestion_{}.jpg'.format(ckpt)), dpi=500)

    fig, ax = plt.subplots()
    plt.bar(np.arange(len(mean_vels)), np.sort(1 - mean_vels))
    plt.ylim(0., 1.)
    plt.savefig(os.path.join(save_path, 'distribution_{}.jpg'.format(ckpt)), dpi=500)

    # route and location
    ## background
    plot('route', 'background', 'background', 3, 3, 3, cmap)
    ## free
    plot('route', 'free', 'free', 3, 3, 7, cmap)
    ## reposition
    plot('location', 'reposition', 'reposition location', 3, 3, 8, cmap)
    ## pickup0
    plot('route', 'pickup', 'pickup 0', 3, 3, 1, cmap, tp=0)
    ## pickup1
    plot('route', 'pickup', 'pickup 1', 3, 3, 2, cmap, tp=1)
    ## occupied0
    plot('route', 'occupied', 'occupied 0', 3, 3, 4, cmap, tp=0)
    ## occupied1
    plot('route', 'occupied', 'occupied 1', 3, 3, 5, cmap, tp=1)

    plt.suptitle('routes_and_locations from ckpt {}'.format(ckpt), y=0.00)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'routes_and_locations_{}.jpg'.format(ckpt)), \
        dpi=500, bbox_inches='tight')

    # # location
    # ## pickup
    # ax = plt.subplot(2, 1, 1)
    # poses = sum((sta['location']['pickup'] for sta in statistics), [])
    # plt.scatter(*zip(*poses))
    # plt.xlim(-5., 160.)
    # plt.ylim(-5., 160.)
    # plt.title('pickup location')
    # plt.xticks([]), plt.yticks([])

    # ## reposition
    # ax = plt.subplot(2, 1, 2)
    # poses = sum((sta['location']['reposition'] for sta in statistics), [])
    # plt.scatter(*zip(*poses), s=2)
    # plt.xlim(-5., 160.)
    # plt.ylim(-5., 160.)
    # plt.title('reposition location')
    # plt.xticks([]), plt.yticks([])

    # plt.tight_layout()
    # plt.savefig(os.path.join(save_path, 'locations.jpg'), dpi=500)
    