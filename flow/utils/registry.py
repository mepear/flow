"""Utility method for registering environments with OpenAI gym."""
import time

import gym
from gym.envs.registration import register

from copy import deepcopy

import flow.envs
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams, PersonParams
from flow.utils.runningstat import RunningStat

import numpy as np
import random
from typing import List, Optional, Tuple, Union
class Monitor(gym.Wrapper):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.
    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    :param allow_early_resets: allows the reset of the environment before it is done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step()
    """

    EXT = "monitor.csv"

    def __init__(
        self,
        env: gym.Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: Tuple[str, ...] = (),
        info_keywords: Tuple[str, ...] = (),
    ):
        # if not hasattr(env, 'reward_range'):
        #     setattr(env, 'reward_range', (-float('inf'), float('inf')))
        super(Monitor, self).__init__(env=env)
        self.t_start = time.time()
        if filename is None:
            self.file_handler = None
            self.logger = None
        else:
            if not filename.endswith(Monitor.EXT):
                if os.path.isdir(filename):
                    filename = os.path.join(filename, Monitor.EXT)
                else:
                    filename = filename + "." + Monitor.EXT
            self.file_handler = open(filename, "wt")
            self.file_handler.write("#%s\n" % json.dumps({"t_start": self.t_start, "env_id": env.spec and env.spec.id}))
            self.logger = csv.DictWriter(self.file_handler, fieldnames=("r", "l", "t") + reset_keywords + info_keywords)
            self.logger.writeheader()
            self.file_handler.flush()

        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.mean_velocities = []
        self.total_co2s = []
        self.congestion_rates = []
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {}  # extra info about the current episode, that was passed in during reset()

    def reset(self, **kwargs):
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True
        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. If you want to allow early resets, "
                "wrap your env with Monitor(env, path, allow_early_resets=True)"
            )
        self.rewards = []
        self.mean_velocities = []
        self.total_co2s = []
        self.congestion_rates = []
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError("Expected you to pass kwarg {} into reset".format(key))
            self.current_reset_info[key] = value
        observation = None
        while observation is None:
            try:
                observation = self.env.reset(**kwargs)
            except Exception as e:
                print("reset error with {}, reset again".format(e))
        return observation

    def step(self, action: Union[np.ndarray, int]):
        """
        Step the environment with the given action
        :param action: the action
        :return: observation, reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        self.mean_velocities.append(self.env.mean_velocity.copy())
        self.total_co2s.append(np.concatenate([self.env.background_co2, self.env.taxi_co2]))
        self.congestion_rates.append(self.env.congestion_rate)
        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_rewards.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info['num_orders'] = len(self.env.k.person.get_ids())
            ep_info['num_complete_orders'] = self.env.num_complete_orders
            ep_info['total_pickup_distance'] = self.env.total_pickup_distance
            ep_info['total_pickup_time'] = self.env.total_pickup_time
            ep_info['total_valid_distance'] = self.env.total_valid_distance
            ep_info['total_distance'] = self.env.total_distance
            ep_info['total_valid_time'] = self.env.total_valid_time
            ep_info['total_wait_time'] = self.env.total_wait_time
            ep_info['congestion_rates'] = self.congestion_rates
            ep_info['mean_velocities'] = self.mean_velocities
            ep_info['total_co2s'] = self.total_co2s
            ep_info['edge_position'] = self.env.edge_position
            ep_info['statistics'] = self.env.statistics
            ep_info.update(self.current_reset_info)
            if self.logger:
                self.logger.writerow(ep_info)
                self.file_handler.flush()
            info["episode"] = ep_info
        info['action_mask'] = self.env.get_action_mask()
        info['reward'] = reward

        info['background_velocity'] = self.env.background_velocity.copy()
        info['background_co2'] = self.env.background_co2.copy()
        info['taxi_velocity'] = self.env.taxi_velocity.copy()
        info['taxi_co2'] = self.env.taxi_co2.copy()
        info['background_co'] = self.env.background_co.copy()
        info['taxi_co'] = self.env.taxi_co.copy()
        info['total_taxi_distance'] = self.env.total_taxi_distances
        info['total_back_distance'] = self.env.total_back_distances

        self.total_steps += 1
        return observation, reward, done, info

    def close(self) -> None:
        """
        Closes the environment
        """
        super(Monitor, self).close()
        if self.file_handler is not None:
            self.file_handler.close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps
        :return:
        """
        return self.total_steps

    def get_episode_rewards(self) -> List[float]:
        """
        Returns the rewards of all the episodes
        :return:
        """
        return self.episode_rewards

    def get_episode_lengths(self) -> List[int]:
        """
        Returns the number of timesteps of all the episodes
        :return:
        """
        return self.episode_lengths

    def get_episode_times(self) -> List[float]:
        """
        Returns the runtime in seconds of all the episodes
        :return:
        """
        return self.episode_times
    
class RewardScaling(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        popart_reward: bool,
        gamma: int,
        reward_scale=None,
        clip=None
    ):
        super().__init__(env=env)
        shape = ()
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.popart_reward = popart_reward
        self.rs = RunningStat(shape=shape)
        self.ret = np.zeros(shape)
        self.clip = clip
    
    def reset(self, **kwargs):
        self.ret = np.zeros_like(self.ret)
        return self.env.reset(**kwargs)
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        scaled_reward = reward / self.reward_scale if self.reward_scale else reward
        
        if self.popart_reward:
            self.ret = self.ret * self.gamma + scaled_reward
            self.rs.push(self.ret)
            scaled_reward = scaled_reward / (self.rs.std + 1e-8)
        if self.clip:
            scaled_reward = np.clip(scaled_reward, -self.clip, self.clip)
        return observation, scaled_reward, done, info
    
    def close(self):
        super().close()


def make_create_env(params, version=0, render=None, popart_reward=False, gamma=0.99, reward_scale=None, \
    port=None, verbose=False, save_path=None):
    """Create a parametrized flow environment compatible with OpenAI gym.

    This environment creation method allows for the specification of several
    key parameters when creating any flow environment, including the requested
    environment and network classes, and the inputs needed to make these
    classes generalizable to networks of varying sizes and shapes, and well as
    varying forms of control (e.g. AVs, automated traffic lights, etc...).

    This method can also be used to recreate the environment a policy was
    trained on and assess it performance, or a modified form of the previous
    environment may be used to profile the performance of the policy on other
    types of networks.

    Parameters
    ----------
    params : dict
        flow-related parameters, consisting of the following keys:

         - exp_tag: name of the experiment
         - env_name: environment class of the flow environment the experiment
           is running on. (note: must be in an importable module.)
         - network: network class the experiment uses.
         - simulator: simulator that is used by the experiment (e.g. aimsun)
         - sim: simulation-related parameters (see flow.core.params.SimParams)
         - env: environment related parameters (see flow.core.params.EnvParams)
         - net: network-related parameters (see flow.core.params.NetParams and
           the network's documentation or ADDITIONAL_NET_PARAMS component)
         - veh: vehicles to be placed in the network at the start of a rollout
           (see flow.core.params.VehicleParams)
         - per: persons to be placed in the network at the start of a rollout
           (see flow.core.params.PersonParams)
         - initial (optional): parameters affecting the positioning of vehicles
           upon initialization/reset (see flow.core.params.InitialConfig)
         - tls (optional): traffic lights to be introduced to specific nodes
           (see flow.core.params.TrafficLightParams)

    version : int, optional
        environment version number
    render : bool, optional
        specifies whether to use the gui during execution. This overrides
        the render attribute in SumoParams

    Returns
    -------
    function
        method that calls OpenAI gym's register method and make method
    str
        name of the created gym environment
    """

    # print('We are in registry_with_person now.')  # TEST: this info should be printed if experiment uses this function

    exp_tag = params["exp_tag"]

    if isinstance(params["env_name"], str):
        print("""Passing of strings for env_name will be deprecated.
        Please pass the Env instance instead.""")
        base_env_name = params["env_name"]
    else:
        base_env_name = params["env_name"].__name__

    # deal with multiple environments being created under the same name
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]
    while "{}-v{}".format(base_env_name, version) in env_ids:
        version += 1
    env_name = "{}-v{}".format(base_env_name, version)

    if isinstance(params["network"], str):
        print("""Passing of strings for network will be deprecated.
        Please pass the Network instance instead.""")
        module = __import__("flow.networks", fromlist=[params["network"]])
        network_class = getattr(module, params["network"])
    else:
        network_class = params["network"]

    env_params = params['env']
    net_params = params['net']
    initial_config = params.get('initial', InitialConfig())
    traffic_lights = params.get("tls", TrafficLightParams())

    env_params.verbose = verbose

    def create_env(*_):
        sim_params = deepcopy(params['sim'])
        sim_params.port = port
        vehicles = deepcopy(params['veh'])

        # print(sim_params.seed)
        random.seed(sim_params.seed)
        np.random.seed(sim_params.seed)

        # print(params['per'])
        persons = deepcopy(params.get('per', PersonParams()))


        network = network_class(
            name=exp_tag,
            vehicles=vehicles,
            persons=persons,
            net_params=net_params,
            initial_config=initial_config,
            traffic_lights=traffic_lights,
        )

        # accept new render type if not set to None
        sim_params.render = render or sim_params.render

        # save path
        env_params.save_path = save_path

        # check if the environment is a single or multiagent environment, and
        # get the right address accordingly
        single_agent_envs = [env for env in dir(flow.envs)
                             if not env.startswith('__')]

        if isinstance(params["env_name"], str):
            if params['env_name'] in single_agent_envs:
                env_loc = 'flow.envs'
            else:
                env_loc = 'flow.envs.multiagent'
            entry_point = env_loc + ':{}'.format(params["env_name"])
        else:
            entry_point = params["env_name"].__module__ + ':' + params["env_name"].__name__

        # register the environment with OpenAI gym
        register(
            id=env_name,
            entry_point=entry_point,
            kwargs={
                "env_params": env_params,
                "sim_params": sim_params,
                "network": network,
                "simulator": params['simulator']
            })

        env =  Monitor( gym.envs.make(env_name) )
        env = RewardScaling(env, popart_reward=popart_reward, gamma=gamma, reward_scale=reward_scale)
        return env

    return create_env, env_name


def env_constructor(params, version=0, render=None, port=None, verbose=False, popart_reward=False, \
    gamma=0.99, reward_scale=None, save_path=None):
    """Return a constructor from make_create_env."""
    create_env, env_name = make_create_env(params, version, render, popart_reward, gamma, \
        reward_scale, port, verbose, save_path)
    return create_env
