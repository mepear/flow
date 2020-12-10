import numpy as np
import torch
import os

from .a2c_ppo_acktr import utils
from .a2c_ppo_acktr.envs import make_vec_envs


def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device, flow_params, save_path=None, writer=None, total_num_steps=None):
    # flow_params['sim'].render = True
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True, flow_params=flow_params)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()

    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)
    action_mask = None
    while len(eval_episode_rewards) < num_processes:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                action_mask=action_mask,
                deterministic=True)

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

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
    if save_path:
        with open(os.path.join(save_path, 'eval.txt'), 'a') as f:
            f.write(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
            len(eval_episode_rewards), np.mean(eval_episode_rewards)))
    if writer:
        assert total_num_steps is not None
        writer.add_scalar('rewards/eval', np.mean(eval_episode_rewards), total_num_steps)
