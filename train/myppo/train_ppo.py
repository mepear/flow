import copy
import glob
import os
import sys
import time
from collections import deque
import random
from functools import partial

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed import rpc
import torch.multiprocessing as mp

import sumolib

from .a2c_ppo_acktr.arguments import get_args
from .a2c_ppo_acktr.envs import make_vec_envs
from .a2c_ppo_acktr.model import Policy
from .a2c_ppo_acktr.storage import RolloutStorage
from .a2c_ppo_acktr.trainer import Trainer
from .evaluation import evaluate

def train(rank, args, flow_params=None):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.master_port)
    if rank == 0:
        rpc_opt = rpc.TensorPipeRpcBackendOptions(num_worker_threads=\
            max(16, args.num_splits * args.num_actors), rpc_timeout=300)
        rpc.init_rpc('agent', rank=rank, world_size=args.num_actors + 1, rpc_backend_options=rpc_opt)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        # envs = make_vec_envs(args.env_name, args.seed, 1, \
        #                     args.gamma, save_path, device, False, \
        #                     port=args.port, popart_reward=args.popart_reward, \
        #                     flow_params=flow_params, reward_scale=args.reward_scale, \
        #                     verbose=args.verbose)

        trainer = Trainer(args, flow_params)
        trainer.run()
    else:
        rpc_opt = rpc.TensorPipeRpcBackendOptions(rpc_timeout=300)
        rpc.init_rpc('actor_' + str(rank - 1), rank=rank, world_size=args.num_actors + 1, rpc_backend_options=rpc_opt)
    rpc.shutdown()


def train_ppo(flow_params=None):
    mp.set_start_method('spawn')
    args = get_args(sys.argv[2:])
    args.master_port = sumolib.miscutils.getFreeSocketPort()
    procs = []
    for i in range(args.num_actors + 1):
        p = mp.Process(target=train, args=(i, args, flow_params))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    train_ppo()
