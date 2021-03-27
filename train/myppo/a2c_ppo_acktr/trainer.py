import os
import copy
import time
from multiprocessing import Lock
from functools import partial

import torch
from torch.distributed import rpc
from torch.utils.tensorboard import SummaryWriter

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from .actor import Actor
from .algo import PPO
from .model import Policy
from .storage import ReplayBuffer, RolloutStorage
from .utils import tocpu, update_linear_schedule
from .envs import make_vec_envs
from ..evaluation import evaluate
from flow.utils.registry import env_constructor

class Trainer:
    def __init__(self, args, flow_params=None):
        self.args = args
        # Initialize saving location and tensorboard writer
        self.save_path = os.path.join(os.path.join(args.save_dir, args.algo), args.experiment_name)
        os.makedirs(self.save_path, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.save_path, 'tensorboard_logs'))
        
        # Set device
        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        
        # Create partial env_fn
        env_params = copy.deepcopy(flow_params)
        env_params['sim'].seed = args.seed
        self.env_fn = partial(env_constructor, params=env_params, verbose=args.verbose, \
            popart_reward=args.popart_reward, gamma=args.gamma, reward_scale=args.reward_scale, \
            save_path=self.save_path)
        self.example_env = DummyVecEnv([self.env_fn(version=0)])

        # Create eval envs
        self.eval_envs = make_vec_envs(args.env_name, args.seed, args.eval_num_processes, \
                None, self.save_path, True, device=self.device, flow_params=flow_params)

        # Actor critic network
        self.actor_critic = Policy(
            self.example_env.observation_space.shape,
            self.example_env.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
        self.actor_critic.to(self.device)

        self.rollout_policy = Policy(
            self.example_env.observation_space.shape,
            self.example_env.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
        self.rollout_policy.to(self.device)
        self.rollout_policy.load_state_dict(self.actor_critic.state_dict())

        # PPO
        self.agent = PPO(self.actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,\
            args.value_loss_coef, args.entropy_coef, lr=args.lr, eps=args.eps, \
            max_grad_norm=args.max_grad_norm)

        # Actor
        self.n_actor = args.num_actors
        self.n_env = args.num_envs
        assert args.num_envs % args.num_actors == 0
        self.n_env_per_actor = args.num_envs // args.num_actors
        self.n_split = args.num_splits
        assert self.n_env_per_actor % self.n_split == 0
        self.n_env_per_split = self.n_env_per_actor // self.n_split
        ## remote reference to actors
        self.actor_rrefs = []
        self.rref = rpc.RRef(self)
        self.future_outputs = [torch.futures.Future() for _ in range(self.n_split)]
        self.locks = [Lock() for _ in range(self.n_split)]
        self.split_cnt = [0] * self.n_split

        # Training parameters
        self.n_env_steps = args.num_env_steps
        self.n_step_per_ep = args.num_steps
        self.batch_size = self.n_env_per_split * self.n_actor * self.n_step_per_ep
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.eval_interval = args.eval_interval
        self.eval_num_processes = args.eval_num_processes
        self.use_linear_lr_decay = args.use_linear_lr_decay
        self.lr = args.lr

        # Training monitor
        self.log_lock = Lock()
        self.global_steps = 0
        self.rollout_rewards = torch.zeros(self.n_split, self.n_actor, self.n_env_per_split).float()
        
        self.buffer = ReplayBuffer(args, self.example_env.observation_space.shape,
            self.example_env.action_space, self.actor_critic.recurrent_hidden_state_size)

    def setup_actors(self):
        self.actor_rrefs = []
        for i in range(self.n_actor):
            name = 'actor_{}'.format(i)
            actor_rref = rpc.remote(name, Actor, args=(i, self.env_fn, self.rref, self.args))
            actor_rref.remote().run()
            self.actor_rrefs.append(actor_rref)

    @rpc.functions.async_execution
    def select_action(self, actor_id, split_id, model_inputs, init=False):
        # if split_id == 0: print('select action for actor {} split {}'.format(actor_id, split_id))
        if init == True:
            obs = model_inputs
            reward = torch.zeros(self.n_env_per_split, 1).float()
            done = [False] * self.n_env_per_split
            infos = [{} for i in range(self.n_env_per_split)]
            action_masks = None
        else:
            obs, reward, done, infos = model_inputs
            action_masks = torch.cat([info['action_mask'] for info in infos], dim=0)
        # if split_id == 0: print('after select action for actor {} split {}'.format(actor_id, split_id))
        obs = obs.to(self.device)

        # if split_id == 0: print('obs to device for actor {} split {}'.format(actor_id, split_id))
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] \
            for info in infos])
        done = torch.tensor(done, dtype=bool)

        # if split_id == 0: print('insert before inference for actor {} split {}'.format(actor_id, split_id))
        self.buffer.insert_before_inference(actor_id, split_id, obs, reward, action_masks, masks, \
            bad_masks, done, init)

        # collect rollout information
        if init == False:
            self.rollout_rewards[split_id, actor_id] += torch.tensor([info['reward'] for info in infos])
        # END

        def _unpack(action_batch_futures):
            action_batch = action_batch_futures.wait()
            batch_slice = slice(actor_id * self.n_env_per_split, (actor_id + 1) * \
                self.n_env_per_split)
            return action_batch[batch_slice]
        
        # if split_id == 0: print('future for actor {} split {}'.format(actor_id, split_id))

        fut = self.future_outputs[split_id].then(_unpack)

        # if split_id == 0: print('before lock for actor {} split {}'.format(actor_id, split_id))

        with self.locks[split_id]:
            self.split_cnt[split_id] += 1
            # if split_id == 0: print('split cnt', self.split_cnt[split_id])
            if self.split_cnt[split_id] == self.n_actor:
                obs, recurrent_hidden_states, masks, action_masks = \
                    self.buffer.get_policy_inputs(split_id, self.device)
                with torch.no_grad():
                    outputs = self.rollout_policy.act(obs, recurrent_hidden_states, masks, \
                        action_masks=action_masks)
                value, action, action_log_prob, recurrent_hidden_states = tocpu(outputs)
                self.buffer.insert_after_inference(split_id, recurrent_hidden_states, action, \
                    action_log_prob, value)

                if self.buffer.current_rollouts[split_id].step == 0 and init == False:
                    print("log done actor {} split {} step {}".format(actor_id, split_id, self.buffer.current_rollouts[split_id].step))
                    self.log_rollout(split_id)

                self.split_cnt[split_id] = 0
                cur_fut = self.future_outputs[split_id]
                cur_fut.set_result(action)
                self.future_outputs[split_id] = torch.futures.Future()

        return fut

    def log_rollout(self, split_id):
        with self.log_lock:
            rewards = self.rollout_rewards[split_id]
            self.writer.add_scalars(
                "rewards/train", 
                {
                    "mean": rewards.mean(), 
                    "median": rewards.median(),
                    "max": rewards.max(),
                    "min": rewards.min()
                },
                self.global_steps
            )
            print("mean reward of rollout from split {}: {}".format(split_id, rewards.mean()))
            self.rollout_rewards[split_id] = 0.0
            self.global_steps += self.batch_size

    def train(self, idx):
        train_rollouts = self.buffer.get()
        value_loss, action_loss, dist_entropy = self.agent.update(train_rollouts)
        self.rollout_policy.load_state_dict(self.actor_critic.state_dict())

        self.log_train(idx, value_loss, action_loss, dist_entropy)
        self.save_train(idx)

    def log_train(self, idx, value_loss, action_loss, dist_entropy):
        self.writer.add_scalar('training loss/value loss', value_loss, (idx + 1) * self.batch_size)
        self.writer.add_scalar('training loss/action loss', action_loss, (idx + 1) * self.batch_size)
        self.writer.add_scalar('training loss/dist_entropy', dist_entropy, (idx + 1) * self.batch_size)
        
        if idx % self.log_interval == 0:
            total_num_steps = (idx + 1) * self.batch_size
            end_time = time.time()
            print('\n' + '=' * 20, "Updates {}, num timesteps {}, FPS {}, cur FPS {}\n".format(idx, \
                total_num_steps, int(total_num_steps / (end_time - self.start_time)), \
                    int(total_num_steps / (end_time - self.up_start_time))))

    def save_train(self, idx):
        if idx % self.save_interval == 0:
            torch.save([self.actor_critic, None], \
                os.path.join(self.save_path, str(idx // self.save_interval) + ".pt"))

    def eval(self, idx):
        total_num_steps = (idx + 1) * self.batch_size
        evaluate(self.actor_critic, self.eval_envs, self.eval_num_processes, self.device, \
            self.save_path, self.writer, total_num_steps)

    def run(self):
        self.setup_actors()

        self.start_time = time.time()
        num_updates = self.n_env_steps // self.n_step_per_ep // (self.n_env_per_split * self.n_actor)
        
        for j in range(num_updates):
            self.up_start_time = time.time()
            if self.use_linear_lr_decay:
                # decrease learning rate linearly
                update_linear_schedule(self.agent.optimizer, j, num_updates, self.lr)

            self.train(j)

            if self.eval_interval is not None and j % self.eval_interval == 0:
                self.eval(j)
            
        self.eval_envs.close()
