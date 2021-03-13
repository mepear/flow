import copy
import glob
import os
import sys
import time
from collections import deque
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .a2c_ppo_acktr import algo, utils
from .a2c_ppo_acktr.algo import gail
from .a2c_ppo_acktr.arguments import get_args
from .a2c_ppo_acktr.envs import make_vec_envs
from .a2c_ppo_acktr.model import Policy
from .a2c_ppo_acktr.storage import RolloutStorage
from .evaluation import evaluate

def train_ppo(flow_params=None):
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
    save_path = os.path.join(os.path.join(args.save_dir, args.algo), args.experiment_name)
    try:
        os.makedirs(save_path)
    except OSError:
        pass
    
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard_logs'))

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, \
                        args.gamma, save_path, device, False, \
                        port=args.port, popart_reward=args.popart_reward, \
                        flow_params=flow_params, reward_scale=args.reward_scale, \
                        verbose=args.verbose)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=args.num_processes)
    nums_orders = deque(maxlen=args.num_processes)
    nums_complete_orders = deque(maxlen=args.num_processes)
    total_pickup_distances = deque(maxlen=args.num_processes)
    total_pickup_times = deque(maxlen=args.num_processes)
    total_valid_distances = deque(maxlen=args.num_processes)
    total_valid_times = deque(maxlen=args.num_processes)
    total_wait_times = deque(maxlen=args.num_processes)
    total_congestion_rates = deque(maxlen=args.num_processes)

    if args.port is not None:
        eval_envs = make_vec_envs(args.env_name, args.seed, args.eval_num_processes, \
            None, save_path, device, True, flow_params=flow_params, port=args.port + args.num_processes)
    else:
        eval_envs = make_vec_envs(args.env_name, args.seed, args.eval_num_processes, \
            None, save_path, device, True, flow_params=flow_params)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        action_mask = None
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step], action_mask=action_mask)

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            # action_mask
            try:
                action_mask = torch.cat([info['action_mask'] for info in infos], dim=0)
            except KeyError:
                action_mask = None

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    nums_orders.append(info['episode']['num_orders'])
                    nums_complete_orders.append(info['episode']['num_complete_orders'])
                    total_pickup_distances.append(info['episode']['total_pickup_distance'])
                    total_pickup_times.append(info['episode']['total_pickup_time'])
                    total_valid_distances.append(info['episode']['total_valid_distance'])
                    total_valid_times.append(info['episode']['total_valid_time'])
                    total_wait_times.append(info['episode']['total_wait_time'])
                    total_congestion_rates.append(info['episode']['total_congestion_rate'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        writer.add_scalar('training loss/value loss', value_loss, (j + 1) * args.num_processes * args.num_steps)
        writer.add_scalar('training loss/action loss', action_loss, (j + 1) * args.num_processes * args.num_steps)
        writer.add_scalar('training loss/dist_entropy', dist_entropy, (j + 1) * args.num_processes * args.num_steps)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            # save_path = os.path.join(os.path.join(args.save_dir, args.algo), args.experiment_name)
            # try:
            #     os.makedirs(save_path)
            # except OSError:
            #     pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, str(j // args.save_interval) + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                '\n' + '=' * 20, 
                "Updates {}, num timesteps {}, FPS {}, Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
            denom = np.array(nums_complete_orders, dtype=int)
            denom2 = np.array(nums_orders, dtype=int)
            writer.add_scalars(
                    "rewards/train", 
                    {
                        "mean": np.mean(episode_rewards), 
                        "median": np.median(episode_rewards),
                        "max": np.max(episode_rewards),
                        "min": np.min(episode_rewards)
                    },
                total_num_steps
                )
            writer.add_scalars(
                    "train/order", 
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
                        "train/pickup_distance", 
                        {
                            "mean": np.mean(array),
                            "median": np.median(array),
                            "max": np.max(array),
                            "min": np.min(array)
                        },
                    total_num_steps
                    )
                array = np.ma.masked_invalid(total_pickup_times / denom)
                writer.add_scalars(
                        "train/pickup_time", 
                        {
                            "mean": np.mean(array),
                            "median": np.median(array),
                            "max": np.max(array),
                            "min": np.min(array)
                        },
                    total_num_steps
                    )
                array = np.ma.masked_invalid(total_valid_distances / denom)
                writer.add_scalars(
                        "train/valid_distance", 
                        {
                            "mean": np.mean(array),
                            "median": np.median(array),
                            "max": np.max(array),
                            "min": np.min(array)
                        },
                    total_num_steps
                    )
                array = np.ma.masked_invalid(total_valid_times / denom)
                writer.add_scalars(
                        "train/valid_time", 
                        {
                            "mean": np.mean(array),
                            "median": np.median(array),
                            "max": np.max(array),
                            "min": np.min(array)
                        },
                    total_num_steps
                    )
            if not np.alltrue(denom2 == 0):
                array = np.ma.masked_invalid(total_wait_times / denom2)
                writer.add_scalars(
                        "train/wait_time", 
                        {
                            "mean": np.mean(array),
                            "median": np.median(array),
                            "max": np.max(array),
                            "min": np.min(array)
                        },
                    total_num_steps
                    )
            writer.add_scalars(
                    "train/congestion_rate", 
                    {
                        "mean": np.mean(total_congestion_rates), 
                        "median": np.median(total_congestion_rates),
                        "max": np.max(total_congestion_rates),
                        "min": np.min(total_congestion_rates)
                    },
                total_num_steps
                )


        if (args.eval_interval is not None and len(episode_rewards) > 0
                and j % args.eval_interval == 0):
            # save_path = os.path.join(os.path.join(args.save_dir, args.algo), args.experiment_name)
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, eval_envs, ob_rms, args.eval_num_processes, device, save_path, writer, total_num_steps)

    eval_envs.close()
            


if __name__ == "__main__":
    train_ppo()
