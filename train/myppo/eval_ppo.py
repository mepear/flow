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

from .a2c_ppo_acktr import algo, utils
from .a2c_ppo_acktr.algo import gail
from .a2c_ppo_acktr.arguments import get_args
from .a2c_ppo_acktr.envs import make_vec_envs
from .a2c_ppo_acktr.model import Policy
from .a2c_ppo_acktr.storage import RolloutStorage
from .evaluation import evaluate

def eval_ppo(flow_params=None):
    args = get_args(sys.argv[2:])

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
    save_path_2 = os.path.join(os.path.join(args.save_dir, args.algo), args.experiment_name_2)
    save_path_3 = os.path.join(os.path.join(args.save_dir, args.algo), args.experiment_name_3)
    pt =  os.path.join(save_path, str(args.eval_ckpt) + ".pt")
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
        None, save_path, True, device=device, flow_params=flow_params, verbose=args.verbose)
    evaluate(actor_critic, eval_envs, ob_rms, args.num_processes, device, save_path=save_path, \
        do_plot_congestion=args.plot_congestion, ckpt=args.eval_ckpt, verbose=True, \
             actor_critic_2=actor_critic_2, actor_critic_3=actor_critic_3,
             ob_rms_2=ob_rms_2, ob_rms_3=ob_rms_3, random_rate=args.random_rate)
    eval_envs.close()
