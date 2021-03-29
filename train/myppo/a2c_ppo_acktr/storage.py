import copy
import queue

from multiprocessing import Condition, Lock

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class ReplayBuffer:
    def __init__(self, args, obs_shape, action_space, recurrent_hidden_state_size):
        # repeated definition of numbers
        self.n_step = args.num_steps
        self.n_actor = args.num_actors
        self.n_env = args.num_envs
        assert args.num_envs % args.num_actors == 0
        self.n_env_per_actor = args.num_envs // args.num_actors
        self.n_split = args.num_splits
        assert self.n_env_per_actor % self.n_split == 0
        self.n_env_per_split = self.n_env_per_actor // self.n_split

        self.obs_shape = obs_shape
        self.action_space = action_space
        self.recurrent_hidden_state_size = recurrent_hidden_state_size

        self.qsize = args.queue_size
        self.slot_queue = [queue.Queue(maxsize=self.qsize) for i in range(self.n_split)]
        self.slot_cnt = [0 for i in range(self.n_split)]
        self.reuse = args.reuse
        self.current_rollouts = [RolloutStorage(self.n_step, self.n_env_per_split * self.n_actor, \
            self.n_actor, obs_shape, action_space, recurrent_hidden_state_size) \
            for i in range(self.n_split)]
        self.use_which_split = 0

        self.condition = Condition(Lock())

        self.use_gae = args.use_gae
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.use_proper_time_limits = args.use_proper_time_limits
        self.device = torch.device("cuda:0" if args.cuda else "cpu")

    def _push(self, split_id, rollout):
        rollout.to(self.device)
        q = self.slot_queue[split_id]
        with self.condition:
            if q.qsize() == self.qsize:
                q.get()
                self.slot_cnt[split_id] = 0
            print('putting rollout of split', split_id)
            q.put(rollout)
            if self.use_which_split == split_id:
                self.condition.notify(1)

    def insert_before_inference(self, actor_id, split_id, obs, rewards, action_masks, masks, \
        bad_masks, done, init):
        # print('replay_buffer.insert_before_inference')
        rollout = self.current_rollouts[split_id]
        span = slice(actor_id * self.n_env_per_split, (actor_id + 1) * self.n_env_per_split)
        if init is True:
            rollout.obs[0, span] = obs
            return
        
        # assert done.all() or (~done).all()
        rollout.insert_before_inference(span, obs, rewards, action_masks, done, masks, bad_masks)
        # print('return replay_buffer.insert_before_inference')

    def get_policy_inputs(self, split_id, device):
        # get the state out
        rollout = self.current_rollouts[split_id]
        return rollout.get_inputs(device)

    def insert_after_inference(self, split_id, recurrent_hidden_states, actions, action_log_probs, \
        value_preds):       
        
        rollout = self.current_rollouts[split_id]

        if rollout.step == self.n_step - 1:
            # compute returns
            rollout.compute_returns(torch.zeros(self.n_env_per_split * self.n_actor, 1), \
                self.use_gae, self.gamma, self.gae_lambda, self.use_proper_time_limits)
            # store the rollout
            self._push(split_id, copy.deepcopy(rollout))
            # reset the rollout
            rollout.after_update()

        rollout.step = (rollout.step + 1) % self.n_step

        # insert the rest of the step
        rollout.insert_after_inference(recurrent_hidden_states, actions, action_log_probs, \
            value_preds)

    def get(self):
        # take one rollout out, counter++
        split_id = self.use_which_split
        q = self.slot_queue[split_id]
        with self.condition:
            self.condition.wait_for(lambda: q.qsize() > 0)
            rollout = q.queue[0]
            self.slot_cnt[split_id] += 1
            if self.slot_cnt[split_id] == self.reuse:
                q.get()
                self.slot_cnt[split_id] = 0
            self.use_which_split = (self.use_which_split + 1) % self.n_split
        return rollout


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, n_actor, obs_shape, action_space,
                 recurrent_hidden_state_size):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        if hasattr(action_space, 'nvec'):
            self.action_dim = sum(action_space.nvec)
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.action_masks = torch.zeros((num_steps + 1, num_processes, sum(action_space.nvec)), \
            dtype=bool)
        self.dones = torch.zeros((num_steps, num_processes), dtype=bool)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = -1

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.device = device

    def insert_before_inference(self, span, obs, rewards, action_masks, done, masks, bad_masks):
        # print('rollout.insert_before_inference')
        self.obs[self.step + 1, span] = obs.clone()
        self.rewards[self.step, span] = rewards.clone()
        self.action_masks[self.step + 1, span] = action_masks.clone()
        self.dones[self.step, span] = done.clone()
        self.masks[self.step + 1, span] = masks.clone()
        self.bad_masks[self.step + 1, span] = bad_masks.clone()

    def get_inputs(self, device):
        return self.obs[self.step + 1].to(device), \
            self.recurrent_hidden_states[self.step + 1].to(device), \
            self.masks[self.step + 1].to(device), \
            self.action_masks[self.step + 1].to(device)

    def insert_after_inference(self, recurrent_hidden_states, actions, action_log_probs, \
        value_preds):
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[step + 1] * \
                        self.masks[step + 1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * gamma * self.masks[step + 1] + \
                        self.rewards[step]) * self.bad_masks[step + 1] + \
                        (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[step + 1] * \
                        self.masks[step + 1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + \
                        self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
