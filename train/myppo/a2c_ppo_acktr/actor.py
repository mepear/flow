import time

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv

from .envs import make_vec_envs, VecNormalize, Converter

class Actor:
    def __init__(self, actor_id, env_fn, agent_rref, args):
        self.id = actor_id
        self.n_env_per_actor = args.num_envs // args.num_actors
        self.n_split = args.num_splits
        self.n_env_per_split = self.n_env_per_actor // self.n_split
        self.n_step = args.num_steps

        self.agent_rref = agent_rref

        self.envs = []
        for i in range(self.n_split):
            idx = actor_id * self.n_env_per_actor + i * self.n_env_per_split
            env = [env_fn(version=idx + j) for j in range(self.n_env_per_split)]
            env = ShmemVecEnv(env) if self.n_env_per_split > 1 else DummyVecEnv(env)
            env = VecNormalize(env, gamma=args.gamma)
            env = Converter(env)
            self.envs.append(env)
        self.action_futures = []
        print('actor {} init completes'.format(actor_id))

    def run(self):
        for i, env in enumerate(self.envs):
            obs = env.reset()
            action_fut = self.agent_rref.rpc_async().select_action(self.id, i, obs, init=True)
            self.action_futures.append(action_fut)

        while True:
            for j in range(self.n_step):
                for i, env in enumerate(self.envs):
                    actions = self.action_futures[i].wait()
                    model_inputs = env.step(actions)
                    if j == self.n_step - 1:
                        obs, reward, done, infos = model_inputs
                        model_inputs = env.reset(), reward, done, infos
                    self.action_futures[i] = self.agent_rref.rpc_async().select_action(self.id, i, model_inputs)