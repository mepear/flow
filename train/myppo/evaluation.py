import numpy as np
import torch
import os
import traci

from .a2c_ppo_acktr import utils
from .a2c_ppo_acktr.envs import make_vec_envs


def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir, 
    device, flow_params, save_path=None, writer=None, total_num_steps=None, port=None):
    # flow_params['sim'].render = True
    eval_envs = make_vec_envs(env_name, seed, num_processes,
                              None, eval_log_dir, device, True, flow_params=flow_params, port=port)

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

    obs = eval_envs.reset()

    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)
    action_mask = None
    values = []
    while len(eval_episode_rewards) < num_processes:
        with torch.no_grad():
            value, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                action_mask=action_mask,
                deterministic=True)
        
        values.append(value) # To be used

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)
        
        try:
            action_mask = torch.cat([info['action_mask'] for info in infos], dim=0)
        except KeyError:
            action_mask = None

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

    eval_envs.close()

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
