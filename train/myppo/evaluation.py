from numpy.lib.type_check import _nan_to_num_dispatcher
from flow.core.params import VehicleParams
import numpy as np
import torch
import os
import traci
from functools import partial
from tqdm import tqdm

from .a2c_ppo_acktr import utils
from .a2c_ppo_acktr.envs import make_vec_envs

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def evaluate(actor_critic, eval_envs, ob_rms, num_processes, device, save_path=None, writer=None, \
    total_num_steps=None, do_plot_congestion=False, ckpt=None, verbose=False):
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
        print('rewards/eval mean {:.2f} median {:.2f}'.format(\
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
        print('eval/order mean {:.2f} median {:.2f}'.format(\
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
    
    if do_plot_congestion:
        plot_congestion(mean_velocities, edge_position, statistics, save_path, ckpt)
        # plot_co_emission(np.array(background_velocities), np.array(background_cos), np.array(taxi_velocities), np.array(taxi_cos), save_path, ckpt, num_processes=num_processes)
        plot_emission(np.array(background_velocities), np.array(background_co2s), np.array(taxi_velocities), np.array(taxi_co2s), save_path, ckpt, np.array(total_taxi_distances), num_processes=num_processes)


def get_corners(s, e, w):
    s, e = np.array(s), np.array(e)
    se = e - s
    p = np.array([-se[1], se[0]])
    p = p / np.linalg.norm(p) * w / 2
    return [s + p, e + p, e - p, s - p]


def plot(statistics, edge_position, key1, key2, name, n, m, idx, cmap, tp=None):
    ax = plt.subplot(n, m, idx)
    if tp is None:
        cnts = np.mean([sta[key1][key2] for sta in statistics], axis=0)
    else:
        cnts = np.mean([sta[key1][key2][tp] for sta in statistics], axis=0)
    cnts /= cnts.max()
    colors = []
    patches = []
    x_min, x_max, y_min, y_max = 1e9, -1e9, 1e9, -1e9
    for cnt, (start, end, width) in zip(cnts, edge_position):
        vertices = get_corners(start, end, width)
        assert cnt >= 0 and cnt <= 1, cnts
        poly = Polygon(vertices)
        colors.append(cnt)
        patches.append(poly)
        for vertex in vertices:
            x_min, y_min = min(x_min, vertex[0]), min(y_min, vertex[1])
            x_max, y_max = max(x_max, vertex[0]), max(y_max, vertex[1])
    colors = np.array(colors) * 100
    p = PatchCollection(patches)
    p.set_cmap(cmap)
    p.set_array(colors)
    ax.add_collection(p)
    plt.xlim(x_min - 5, x_max + 5)
    plt.ylim(y_min - 5, y_max + 5)
    plt.title(name)
    plt.xticks([]), plt.yticks([])


def draw(plotter, n_tp, cmap, ckpt, save_path):
    # route and location
    ## background
    # plotter('route', 'background', 'background', 3, 2, 3, cmap)
    ## free
    plotter('route', 'free', 'free', 2, n_tp + 1, n_tp + 1, cmap)
    ## reposition
    plotter('location', 'reposition', 'reposition location', 2, n_tp + 1, 2 * n_tp + 2, cmap)

    for i in range(n_tp):
        plotter('route', 'pickup', 'pickup {}'.format(i), 2, n_tp + 1, i + 1, cmap, tp=i)
        plotter('route', 'occupied', 'occupied {}'.format(i), 2, n_tp + 1, i + n_tp + 2, cmap, tp=i)
    ## pickup0
    # plotter('route', 'pickup', 'pickup 0', 2, 2, 1, cmap, tp=0)
    ## pickup1
    # plotter('route', 'pickup', 'pickup 1', 3, 3, 2, cmap, tp=1)
    ## occupied0
    # plotter('route', 'occupied', 'occupied 0', 2, 2, 2, cmap, tp=0)
    ## occupied1
    # plotter('route', 'occupied', 'occupied 1', 3, 3, 5, cmap, tp=1)

    plt.suptitle('routes_and_locations from ckpt {}'.format(ckpt), y=0.00)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'routes_and_locations_{}.jpg'.format(ckpt)), \
        dpi=500, bbox_inches='tight')


def plot_congestion(mean_velocities, edge_position, statistics, save_path, ckpt):
    # fig, ax = plt.subplots()
    cmap = plt.get_cmap('Greys')
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

    plotter = partial(plot, statistics, edge_position)
    draw(plotter, 2, cmap, ckpt, save_path)


def plot_emission(background_velocities, background_co2s, taxi_velocities, taxi_co2s, save_path, ckpt, total_distances, num_processes=100):

    # background_velocities = background_velocities[:, 1:, :]
    # taxi_velocities = taxi_velocities[:, 1:, :]
    # background_co2s = background_co2s[:, :-1, :]
    # taxi_co2s = taxi_co2s[:, :-1, :]
    
    def _plot_vel(ax, velocities, title):
        num_env, num_step, num_vel = velocities.shape

        mean_vel = velocities.transpose(0, 2, 1).reshape(-1, num_step).mean(axis=-1)
        print(f'eval/velocity of {title} mean {mean_vel.mean()} median {np.median(mean_vel)}')
        ax.hist(mean_vel)
        ax.set_title('velocities of vehicles')
        ax.set_xlabel('Velocity (m/s)')
        ax.set_ylabel('Number')
    
    def _plot_co2(ax, co2s, title):

        num_env, num_step, num_vel = co2s.shape

        mean_co2 = co2s.transpose(0, 2, 1).reshape(-1, num_step).mean(axis=-1)
        print(f'eval/co2 of {title} mean {mean_co2.mean()} median {np.median(mean_co2)}')
        ax.set_title('co2 of vehicles')
        ax.set_xlabel('Co2 (mg/s)')
        ax.set_ylabel('Number')

        ax.hist(mean_co2)

    def _plot_co2_over_distance(ax, co2s, valid_distances, title):

        num_env, num_step, num_vel = co2s.shape

        co2 = co2s.transpose(0, 2, 1).reshape(-1, num_step).sum(axis=-1)
        distance = valid_distances.transpose(0, 2, 1).reshape(-1, num_step).sum(axis=-1)

        co2_over_dis = co2 / distance
        co2_over_dis = co2_over_dis[~ np.isinf(co2_over_dis)]
        print(f'eval/co2@distance of {title} mean {round(co2.sum() / distance.sum(), 3)}')
        ax.set_title('co2@distance of vehicles')
        ax.set_xlabel('Co2 (mg/m)')
        ax.set_ylabel('Number')

        ax.hist(co2_over_dis, 100, range=(0, 500))
    
    def _plot(velocities, co2s, title):
        fig, axs = plt.subplots(1, 2)
        
        _plot_vel(axs[0], velocities, title)
        _plot_co2(axs[1], co2s, title)
        fig.savefig(os.path.join(save_path, 'emission_{}_{}.jpg'.format(title, ckpt)), dpi=500, bbox_inches='tight')

    def _plot_over_distance(velocities, co2s, valid_distances, title):
        fig, axs = plt.subplots(1, 2)
        
        _plot_vel(axs[0], velocities, title)
        _plot_co2_over_distance(axs[1], co2s, valid_distances, title)
        fig.savefig(os.path.join(save_path, 'emission_{}_{}.jpg'.format(title, ckpt)), dpi=500, bbox_inches='tight')

    _plot(background_velocities, background_co2s, 'background')
    _plot(taxi_velocities, taxi_co2s, 'taxi')
    _plot(np.concatenate([background_velocities, taxi_velocities], axis=-1), 
    np.concatenate([background_co2s, taxi_co2s], axis=-1), 'all_vehicles')

    _plot_over_distance(taxi_velocities, taxi_co2s, total_distances, 'CO2@valid_distance')

    fig, axs = plt.subplots(2, 1)
    velocity = taxi_velocities[0, :, 0]
    co2 = taxi_co2s[0, :, 0]
    num_step = velocity.shape[0]

    axs[0].scatter(range(num_step), velocity, s=1)
    axs[0].set_ylabel('velocity')

    axs2 = axs[0].twinx()
    axs2.scatter(range(num_step), co2, s=1, c='r')
    axs2.set_ylabel('co2')


    axs[1].plot(range(num_step), velocity)
    axs[1].set_ylabel('velocity')

    axs3 = axs[1].twinx()
    axs3.plot(range(num_step), co2, c='r')
    axs3.set_ylabel('co2')

    fig.savefig('vel_co2.jpg', dpi=500, bbox_inches='tight')


    fig, axs = plt.subplots(2, 1, sharex=True)
    velocity = np.concatenate([background_velocities, taxi_velocities], axis=-1).reshape(-1)
    co2 = np.concatenate([background_co2s, taxi_co2s], axis=-1).reshape(-1)
    sorted_index = np.argsort(velocity)

    # mv_avg_vel = []
    mv_avg_co2 = co2[sorted_index].reshape(-1, num_processes).mean(axis=1)
    mv_avg_vel = velocity[sorted_index].reshape(-1, num_processes).mean(axis=1)

    axs[0].plot(velocity[sorted_index], co2[sorted_index])
    axs[1].plot(mv_avg_vel, mv_avg_co2)

    axs[0].set_xlabel('velocity')
    axs[1].set_xlabel('velocity')
    axs[1].set_ylabel('co2')
    fig.savefig('vel@co2.jpg')


def plot_co_emission(background_velocities, background_cos, taxi_velocities, taxi_cos, save_path, ckpt, num_processes=100):

    # background_velocities = background_velocities[:, 1:, :]
    # taxi_velocities = taxi_velocities[:, 1:, :]
    # background_cos = background_cos[:, :-1, :]
    # taxi_cos = taxi_cos[:, :-1, :]
    
    # def _plot_vel(ax, velocities, title):
    #     num_env, num_step, num_vel = velocities.shape

    #     mean_vel = velocities.transpose(0, 2, 1).reshape(-1, num_step).mean(axis=-1)
    #     print(f'eval/velocity of {title} mean {mean_vel.mean()} median {np.median(mean_vel)}')
    #     ax.hist(mean_vel)
    #     ax.set_title('velocities of vehicles')
    #     ax.set_xlabel('Velocity (m/s)')
    #     ax.set_ylabel('Number')
    
    # def _plot_co(ax, cos, title):

    #     num_env, num_step, num_vel = cos.shape

    #     mean_co = cos.transpose(0, 2, 1).reshape(-1, num_step).mean(axis=-1)
    #     print(f'eval/co of {title} mean {mean_co.mean()} median {np.median(mean_co)}')
    #     ax.set_title('co of vehicles')
    #     ax.set_xlabel('co (mg/s)')
    #     ax.set_ylabel('Number')

    #     ax.hist(mean_co)
    
    # def _plot(velocities, cos, title):
    #     fig, axs = plt.subplots(1, 2)
        
    #     _plot_vel(axs[0], velocities, title)
    #     _plot_co(axs[1], cos, title)
    #     fig.savefig(os.path.join(save_path, 'emission_{}_{}.jpg'.format(title, ckpt)), dpi=500, bbox_inches='tight')

    # _plot(background_velocities, background_cos, 'background')
    # _plot(taxi_velocities, taxi_cos, 'taxi')
    # _plot(np.concatenate([background_velocities, taxi_velocities], axis=-1), 
    # np.concatenate([background_cos, taxi_cos], axis=-1), 'all_vehicles')

    fig, axs = plt.subplots(2, 1)
    velocity = taxi_velocities[0, :, 0]
    co = taxi_cos[0, :, 0]
    num_step = velocity.shape[0]

    axs[0].scatter(range(num_step), velocity, s=1)
    axs[0].set_ylabel('velocity')

    axs2 = axs[0].twinx()
    axs2.scatter(range(num_step), co, s=1, c='r')
    axs2.set_ylabel('co')


    axs[1].plot(range(num_step), velocity)
    axs[1].set_ylabel('velocity')

    axs3 = axs[1].twinx()
    axs3.plot(range(num_step), co, c='r')
    axs3.set_ylabel('co')

    fig.savefig('vel_co.jpg', dpi=500, bbox_inches='tight')


    fig, axs = plt.subplots(2, 1, sharex=True)
    velocity = np.concatenate([background_velocities, taxi_velocities], axis=-1).reshape(-1)
    co = np.concatenate([background_cos, taxi_cos], axis=-1).reshape(-1)
    sorted_index = np.argsort(velocity)

    # mv_avg_vel = []
    mv_avg_co = co[sorted_index].reshape(-1, num_processes).mean(axis=1)
    mv_avg_vel = velocity[sorted_index].reshape(-1, num_processes).mean(axis=1)

    axs[0].plot(velocity[sorted_index], co[sorted_index])
    axs[1].plot(mv_avg_vel, mv_avg_co)

    axs[0].set_xlabel('velocity')
    axs[1].set_xlabel('velocity')
    axs[1].set_ylabel('co')
    fig.savefig('vel@co.jpg')
