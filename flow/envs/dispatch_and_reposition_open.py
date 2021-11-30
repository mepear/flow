import numpy as np
from flow.envs.base import Env
from copy import deepcopy
import re
import random
from numpy.core.fromnumeric import _nonzero_dispatcher
import torch
import os
import time
from queue import Queue

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces import Tuple

from flow.core import rewards

from flow.utils.distributions import gen_request
from traci.exceptions import TraCIException, FatalTraCIError

import threading
from exclusiveprocess import Lock, CannotAcquireLock

ADDITIONAL_ENV_PARAMS = {
    "max_num_order": 10,
    "pickup_price": 0,
    "starting_distance": 100,
    "time_price": 0.00, # in second
    "distance_price": 0.0, # in meter
    "co2_penalty": 0.0, # in g
    "miss_penalty": 0, # miss a reservation
    "wait_penalty": 0.0, # in second
    "tle_penalty": 0.00, # in second
    "exist_penalty": 0.00, # in second
    "person_prob": 0.03,
    "max_waiting_time": 10, # in second
    "free_pickup_time": 20, # in second
    "max_stop_time": 1, # in second, intentionally waiting time (deprecated)
    "stop_distance_eps": 1, # in meter, a threshold to determine whether the car is stopping (deprecated)
    "distribution": 'random', # random, mode-1, mode-2, mode-3
    "reservation_order": 'fifo', # random or fifo
    "n_mid_edge": 0, # number of mid point for an order
    "use_tl": False, # whether using traffic light info
    "max_detour": 1.5, # detour length / minimal length <= max_detour
}


class DispatchAndRepositionEnv_with_index(Env):

    def __init__(self, env_params, sim_params, network, simulator='traci'):

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # network related parameter
        self.grid_array = network.net_params.additional_params["grid_array"]
        self.rows = self.grid_array["row_num"]
        self.cols = self.grid_array["col_num"]
        self.col_idx = self.grid_array["row_idx"]
        self.row_idx = self.grid_array["col_idx"]
        self.inner_length = self.grid_array["inner_length"]
        self.outer_length = self.grid_array["outer_length"]
        assert self.outer_length == self.inner_length // 2

        # price related parameter
        self.pickup_price = env_params.additional_params['pickup_price']
        self.time_price = env_params.additional_params['time_price']
        self.distance_price = env_params.additional_params['distance_price']
        self.co2_penalty = env_params.additional_params['co2_penalty']
        self.miss_penalty = env_params.additional_params['miss_penalty']
        self.wait_penalty = env_params.additional_params['wait_penalty']
        self.tle_penalty = env_params.additional_params['tle_penalty']
        self.exist_penalty = env_params.additional_params['exist_penalty']
        self.starting_distance = env_params.additional_params['starting_distance']
        self.max_detour = env_params.additional_params['max_detour']

        # person related parameter
        self.person_prob = env_params.additional_params['person_prob']

        # whether verbose info will be printed out
        self.verbose = env_params.verbose

        # orders related parameter
        self.max_num_order = env_params.additional_params['max_num_order']
        self.max_waiting_time = env_params.additional_params['max_waiting_time']
        self.free_pickup_time = env_params.additional_params['free_pickup_time']
        self.reservation_order = env_params.additional_params['reservation_order']

        # rl actions related parameter
        self.n_mid_edge = env_params.additional_params['n_mid_edge']
        self.use_tl = env_params.additional_params['use_tl']
        self.tl_params = network.traffic_lights
        self.n_tl = int(len(self.tl_params.get_properties()) / (self.row_idx * self.col_idx))
        self.n_phase = 4 # the default has 4 phases

        # environment statics
        self.num_complete_orders = np.zeros(self.row_idx * self.col_idx)
        self.total_valid_distance = np.zeros(self.row_idx * self.col_idx)
        self.total_distance = 0
        self.total_valid_time = np.zeros(self.row_idx * self.col_idx)
        self.total_pickup_distance = np.zeros(self.row_idx * self.col_idx)
        self.total_pickup_time = np.zeros(self.row_idx * self.col_idx)
        self.total_wait_time = np.zeros(self.row_idx * self.col_idx)
        self.congestion_rate = np.zeros(self.row_idx * self.col_idx)

        # car stop parameter
        self.stop_distance_eps = env_params.additional_params['stop_distance_eps']

        # distribution parameter
        self.distribution = env_params.additional_params['distribution']
        self.distribution_random_ratio = env_params.additional_params.get('distribution_random_ratio', 0.5)

        super().__init__(env_params, sim_params, network, simulator)

        # Saving env variables for plotting
        self.steps = env_params.horizon
        self.obs_var_labels = {
            'edges': np.zeros((self.steps, self.k.vehicle.num_vehicles)),
            'velocities': np.zeros((self.steps, self.k.vehicle.num_vehicles)),
            'positions': np.zeros((self.steps, self.k.vehicle.num_vehicles))
        }

        # edges initiation of all index area
        self.edges = []
        for idx in range(self.row_idx * self.col_idx):
            self.edges.append([])
        for i in self.k.network.get_edge_list():
            for idx in range(self.col_idx * self.row_idx):
                if(i[-(len(str(idx)) + 1):] == ("_" + str(idx))):
                    self.edges[idx].append(i)

        # in/out and flow edges can't be visited by taxis
        self.flow_edges = []
        self.in_edges = []
        self.out_edges = []
        self.real_out_edges = []
        self.real_in_edges = []
        for idx in range(self.row_idx * self.col_idx):
            self.flow_edges.append([])
            self.in_edges.append([])
            self.out_edges.append([])
        for idx in range(self.row_idx * self.col_idx):
            for i, edge in enumerate(self.edges[idx]):
                if 'in' in edge:
                    self.in_edges[idx].append(i)
                elif 'out' in edge:
                    self.out_edges[idx].append(i)
                elif 'flow' in edge:
                    self.flow_edges[idx].append(i)
        for idx in range(self.col_idx * self.cols):
            col = idx % self.cols
            ind = idx // self.cols
            self.real_out_edges.append("out_left{}_{}".format(col, ind))
            self.real_out_edges.append("out_right{}_{}".format(col, (self.row_idx - 1) * self.col_idx + ind))
            self.real_in_edges.append("in_right{}_{}".format(col, ind))
            self.real_in_edges.append("in_left{}_{}".format(col, (self.row_idx - 1) * self.col_idx + ind))
        for idx in range(self.rows * self.row_idx):
            row = idx % self.rows
            ind = idx // self.rows
            self.real_out_edges.append("out_top{}_{}".format(row, ind * self.col_idx))
            self.real_out_edges.append("out_bot{}_{}".format(row, ind * self.col_idx + self.col_idx - 1))
            self.real_in_edges.append("in_bot{}_{}".format(row, ind * self.col_idx))
            self.real_in_edges.append("in_top{}_{}".format(row, ind * self.col_idx + self.col_idx - 1))
        self._preprocess()
        # vehicles related environment setting
        self.num_taxi = network.vehicles.num_rl_vehicles

        self.taxis = []
        for idx in range(self.row_idx * self.col_idx):
            self.taxis.append([taxi for taxi in network.vehicles.ids if
                               taxi[0: (6 + len(str(idx)))] == ("taxi_" + str(idx) + "_")])
        assert self.num_taxi == len(self.taxis[0]) * self.col_idx * self.row_idx
        self.outside_taxis = []
        for idx in range(self.row_idx * self.col_idx):
            self.outside_taxis.append([])
        self.background_cars = [car for car in network.vehicles.ids if car[0:4] != 'taxi']
        self.num_vehicles = network.vehicles.num_vehicles

        # reward related environment statics(or something changed just in compute reward function)
        self.mean_velocity = np.zeros((self.col_idx * self.row_idx, len(self.edges[0])))
        self.valid_distance = np.zeros(self.row_idx * self.col_idx)
        self.total_taxi_distances = np.zeros((self.row_idx * self.col_idx, len(self.taxis[0])))
        self.background_velocity = np.zeros(len(self.background_cars))
        self.background_co2 = np.zeros(len(self.background_cars))
        self.taxi_velocity = np.zeros((self.row_idx * self.col_idx, len(self.taxis[0])))
        self.taxi_co2 = np.zeros((self.row_idx * self.col_idx, len(self.taxis[0])))
        self.background_co = np.zeros(len(self.background_cars))
        self.taxi_co = np.zeros((self.col_idx * self.row_idx, len(self.taxis[0])))
        self.last_edge = dict([(veh_id, None) for veh_id in self.k.vehicle.get_ids()])

        self.statistics = {
            'route': {},
            'location': {}
        }

        # Edge position
        self.edge_position = []
        for idx in range(self.col_idx * self.row_idx):
            self.edge_position.append([(self.k.kernel_api.simulation.convert2D(edge, 0), \
                    self.k.kernel_api.simulation.convert2D(edge, self.k.kernel_api.lane.getLength(edge + '_0')), \
                    self.k.kernel_api.lane.getWidth(edge + '_0')) for edge in self.edges[idx]])

        # dispatch related lists
        self.__dispatched_orders = []
        self.__pending_orders = []
        self.__reservations = []
        self.__need_reposition = []
        self.__need_mid_edge = []
        for idx in range(self.row_idx * self.col_idx):
            self.__dispatched_orders.append([])
            self.__pending_orders.append([])
            self.__reservations.append([])
            self.__need_reposition.append(None)
            self.__need_mid_edge.append(None)

        self.taxi_states = []
        for idx in range(self.col_idx * self.row_idx):
            self.taxi_states.append(dict([[taxi, {"empty": True, "distance": 0, "pickup_distance": None}] for taxi in self.taxis[idx]]))
        self.background_states = dict([[veh_id, {'distance': 0}] for veh_id in self.background_cars])
        self.stop_time = []
        for idx in range(self.row_idx * self.col_idx):
            self.stop_time.append([])
            self.stop_time[idx] = [None] * len(self.taxis[0])

        # self.hist_dist = [Queue(maxsize=int(self.env_params.additional_params['max_stop_time'] / self.sim_params.sim_step) + 1) for i in range(self.num_taxi)]

        self.reward_info = {'pickup_reward': np.zeros(self.col_idx * self.row_idx), 'time_reward': np.zeros(self.col_idx * self.row_idx),
                       'distance_reward': np.zeros(self.col_idx * self.row_idx)}
        self.action_mask = torch.zeros((self.row_idx * self.col_idx, len(self.taxis[0]) + 1, sum(self.action_space.nvec[0])), dtype=bool)

    def setup_initial_state(self):
        """Store information on the initial state of vehicles in the network.

        This information is to be used upon reset. This method also adds this
        information to the self.vehicles class and starts a subscription with
        sumo to collect state information each step.
        """

        initial_ids = [[] for _ in range(self.col_idx * self.row_idx)]
        for id in self.initial_ids:
            idx = int(id[id.index("_") + 1])
            initial_ids[idx].append(id)

        # generate starting position for vehicles in the network
        for idx in range(self.row_idx * self.col_idx):
            initial_config = deepcopy(self.initial_config)
            initial_config.index = idx
            start_pos, start_lanes = self.k.network.generate_starting_positions(
                initial_config=initial_config,
                num_vehicles=len(initial_ids[0]),
                net_params=self.net_params,
                )

            # save the initial state. This is used in the _reset function
            for i, veh_id in enumerate(initial_ids[idx]):
                type_id = self.k.vehicle.get_type(veh_id)
                pos = start_pos[i][1]
                lane = start_lanes[i]
                speed = self.k.vehicle.get_initial_speed(veh_id)
                edge = start_pos[i][0]

                self.initial_state[veh_id] = (type_id, edge, lane, pos, speed)

    def _preprocess(self):
        def _add_center(edges):
            if len(edges) == 0:
                return []
            ret = []
            for i in range(len(edges) - 1):
                ret.append(edges[i])
                ret.append(self.centers[edges[i] + '&' + edges[i + 1]])
            ret.append(edges[-1])
            return ret
        n_edge = len(self.edges[0])
        save_path = os.path.join(self.env_params.save_path, "preprocess.pt")
        while True:
            try:
                with Lock(name='preprocess'):
                    if os.path.exists(save_path):
                        self.paired_routes, self.paired_complete_routes, self.banned_mid_edges = \
                            torch.load(save_path)
                    else:
                        self.paired_routes, self.paired_complete_routes, self.banned_mid_edges = [], [], []
                        self.centers = {}
                        for idx in range(self.col_idx * self.row_idx):
                            self.paired_routes.append([[self.k.kernel_api.simulation.findRoute(s, t) for t in self.edges[idx]] \
                                                        for s in self.edges[idx]])
                            self.paired_complete_routes.append([])
                            self.banned_mid_edges.append(torch.zeros((n_edge, n_edge, n_edge), dtype=bool))

                        for edge1 in self.network.edges:
                            for edge2 in self.network.edges:
                                if edge1['to'] == edge2['from']:
                                    self.centers[edge1['id'] + '&' + edge2['id']] = edge1['to']

                        for idx in range(self.col_idx * self.row_idx):
                            self.paired_complete_routes[idx] = [
                                [_add_center(route.edges) for route in routes] \
                                for routes in self.paired_routes[idx]]

                        for idx in range(self.row_idx * self.col_idx):
                            for i in range(n_edge):
                                for j in range(n_edge):
                                    if i!= j:
                                        l = len(self.paired_complete_routes[idx][i][j])
                                        for k in range(n_edge):
                                            if k != i and k != j:
                                                l1 = len(self.paired_routes[idx][i][k].edges)
                                                l2 = len(self.paired_routes[idx][k][j].edges)
                                                r1 = set(self.paired_complete_routes[idx][i][k][:-1])
                                                r2 = set(self.paired_complete_routes[idx][k][j][1:])
                                                if len(r1 & r2) > 0 or l1 + l2 - 1 > self.max_detour * l \
                                                        or l1 == 0 or l2 == 0:  # This is for unreachable path
                                                    self.banned_mid_edges[idx][i, j, k] = True

                        torch.save([self.paired_routes, self.paired_complete_routes, self.banned_mid_edges], \
                                   save_path)
                break
            except CannotAcquireLock:
                pass


    def get_action_mask(self):
        """
        :return: [mask0, mask1, ...]
        a list of length of total segmentation in the network, every mask is a torch matrix
        """
        mask_ret = []
        for idx in range(self.col_idx * self.row_idx):
            mask = torch.zeros_like(self.action_mask[0][0])
            num_edges = len(self.edges[0])
            for i in (self.out_edges[idx] + self.in_edges[idx]):
                mask[i] = True
                mask[len(self.taxis[0]) + 1 + num_edges + i] = True
            if self.__need_reposition[idx]:
                taxi_id = self.taxis[idx].index(self.__need_reposition[idx])
                # print(self.action_mask[taxi_id].unsqueeze(0))
                mask = torch.logical_or(mask, self.action_mask[idx][taxi_id])
            if self.__need_mid_edge[idx]:
                taxi_id = self.taxis[idx].index(self.__need_mid_edge[idx])
                mask = torch.logical_or(mask, self.action_mask[idx][taxi_id])
            if len(self.__reservations[idx]) > 0:
                mask = torch.logical_or(mask, self.action_mask[idx][len(self.taxis[0])])
            mask_ret.append(mask.unsqueeze(0))
        return mask_ret


    @property
    def action_space(self):
        """See class definition."""
        one_dim = [len(self.edges[0]), len(self.taxis[0]) + 1] + [len(self.edges[0])] * self.n_mid_edge
        total = []
        for idx in range(self.col_idx * self.row_idx):
            total.append(one_dim)
        try:
            return MultiDiscrete(total, dtype=np.float32)
        except:
            return MultiDiscrete(total)


    @property
    def observation_space(self):
        """See class definition."""
        state_box = Box(
            low=np.float32(-500),
            high=np.float32(500),
            shape=(self.row_idx * self.col_idx, 1 + len(self.edges[0]) + int(self.num_taxi * 9 / (self.col_idx * self.row_idx)) + \
                   int(self.use_tl) * self.n_tl * (self.n_phase + 1) + \
                   self.max_num_order * 5 + (4 + len(self.taxis[0])) + int(self.num_taxi / (self.col_idx * self.row_idx)) + 2,),
            dtype=np.float32
        )
        return state_box


    def get_state(self):
        """
        :return:[state1, state2, ...]
        a list of length with total segmentation in the network, every state is a numpy array
        """
        state_ret = []
        x_length = self.inner_length * self.cols
        y_length = self.inner_length * self.rows
        for idx in range(self.col_idx * self.row_idx):
            col_idx = idx % self.col_idx
            row_idx = idx // self.col_idx
            time_feature = [self.time_counter / (self.env_params.horizon * self.env_params.sims_per_step)]

            edges_feature = [
                self.k.kernel_api.edge.getLastStepVehicleNumber(edge) for edge in self.edges[idx]
            ]

            taxi_feature = []
            empty_taxi = self.k.vehicle.get_taxi_fleet(0)
            pickup_taxi = self.k.vehicle.get_taxi_fleet(1)

            for taxi in self.taxis[idx]:
                while taxi not in self.k.vehicle.get_rl_ids():
                    raise KeyError
                x, y = self.k.vehicle.get_2d_position(taxi, error=(-1, -1))
                from_x, from_y = self.k.kernel_api.simulation.convert2D(self.k.kernel_api.vehicle.getRoute(taxi)[0], 0)
                to_edge = self.k.kernel_api.vehicle.getRoute(taxi)[-1]
                to_pos = self.inner_length - 2 if 'out' not in to_edge else self.outer_length - 2
                to_x, to_y = self.k.kernel_api.simulation.convert2D(to_edge, to_pos)
                # cur_taxi_feature = [0, x, y, self.edges.index(self.k.kernel_api.vehicle.getRoute(taxi)[0]), self.edges.index(self.k.kernel_api.vehicle.getRoute(taxi)[-1])]
                cur_taxi_feature = [0, 0, 0, x - x_length * col_idx, y - y_length * row_idx,
                                    from_x - x_length * col_idx, from_y - y_length * row_idx,
                                    to_x - x_length * col_idx, to_y - y_length * row_idx]  # use (x, y) or edge id
                cur_taxi_feature[0 if taxi in empty_taxi else 1 if taxi in pickup_taxi else 2] = 1
                taxi_feature += cur_taxi_feature

            # use traffic light info
            tl_feature = []
            if self.use_tl:
                tl = self.k.kernel_api.trafficlight
                for tl_id in tl.getIDList():
                    if tl_id[-(len(str(idx)) + 1):] == ("_" + str(idx)):
                        phase = tl.getPhase(tl_id)
                        t_next = tl.getNextSwitch(tl_id)
                        links = tl.getControlledLinks(tl_id)
                        logic = tl.getAllProgramLogics(tl_id)[-1]
                        state = tl.getRedYellowGreenState(tl_id)

                        len_phases = 4  # TODO: len(logic.phases)
                        ft = [0] * (len_phases + 1)
                        ft[phase % len_phases] = 1

                        durations = [phase.duration for phase in logic.phases]
                        cur_time = self.time_counter * self.sim_params.sim_step
                        phase_time = cur_time % sum(durations)
                        for t in durations:
                            if phase_time > t:
                                phase_time -= t
                            else:
                                res_time = t - phase_time
                                break
                        ft[-1] = res_time / 100
                        tl_feature += ft

            order_feature = self._get_order_state[idx].tolist()

            mid_edge_feature = [-1] * (4 + len(self.taxis[0]))
            if self.__need_mid_edge[idx]:
                taxi = self.__need_mid_edge[idx]
                res = self.k.vehicle.reservation[taxi]
                from_x, from_y = self.k.kernel_api.simulation.convert2D(res.fromEdge, res.departPos)
                to_x, to_y = self.k.kernel_api.simulation.convert2D(res.toEdge, 25)
                veh_id = self.taxis[idx].index(taxi)
                index = [0] * len(self.taxis[0])
                index[veh_id] = 1
                mid_edge_feature = [from_x - x_length * col_idx, from_y - y_length * row_idx,
                                    to_x - x_length * col_idx, to_y - y_length * row_idx] + index

            self._update_action_mask(idx)

            index = [0] * len(self.taxis[0])
            empty_taxi_fleet = self.k.vehicle.get_taxi_fleet(0)
            self.__need_reposition[idx] = None
            for taxi in empty_taxi_fleet:
                if taxi in self.taxis[idx]:
                    edge = self.k.vehicle.get_edge(taxi)
                    edge_id = self.edges[idx].index(edge) if edge in self.edges[idx] else -1
                    # Don't reposit the taxi in flow_edges, in_edges and out_edges
                    if edge_id in self.flow_edges[idx] + self.out_edges[idx]:
                        continue
                    elif self.k.kernel_api.vehicle.isStopped(taxi):
                        self.__need_reposition[idx] = taxi
                        break

            if self.__need_reposition[idx]:
                x, y = self.k.vehicle.get_2d_position(self.__need_reposition[idx], error=(-1, -1))
                index[self.taxis[idx].index(self.__need_reposition[idx])] = 1
                need_reposition_taxi_feature = index + [x - x_length * col_idx, y - y_length * row_idx]
            else:
                need_reposition_taxi_feature = index + [-1, -1]

            state = time_feature + edges_feature + taxi_feature + tl_feature + order_feature + mid_edge_feature + need_reposition_taxi_feature
            state_ret.append(np.array(state))

        return state_ret


    def _get_infos(self):
        return {}


    def reset(self):
        self.__need_mid_edge = []
        for idx in range(self.col_idx * self.row_idx):
            self.__need_mid_edge.append(None)
        observation = super().reset()

        # dispatch related lists
        self.__dispatched_orders = []
        self.__pending_orders = []
        self.__reservations = []
        self.__need_reposition = []
        for idx in range(self.row_idx * self.col_idx):
            self.__dispatched_orders.append([])
            self.__pending_orders.append([])
            self.__reservations.append([])
            self.__need_reposition.append(None)

        self.taxi_states = []
        for idx in range(self.col_idx * self.row_idx):
            self.taxi_states.append(dict([[taxi, {"empty": True, "distance": 0, "pickup_distance": None}] for taxi in self.taxis[idx]]))
        self.background_states = dict([[veh_id, {'distance': 0}] for veh_id in self.background_cars])
        # self.hist_dist = [Queue(maxsize=int(self.env_params.additional_params['max_stop_time'] / self.sim_params.sim_step) + 1) for i in range(self.num_taxi)]

        # environment statistics
        self.action_mask = torch.zeros((self.row_idx * self.col_idx, len(self.taxis[0]) + 1, sum(self.action_space.nvec[0])), dtype=bool)
        self.num_complete_orders = np.zeros(self.row_idx * self.col_idx)
        self.total_valid_distance = np.zeros(self.row_idx * self.col_idx)
        self.total_distance = 0
        self.total_valid_time = np.zeros(self.row_idx * self.col_idx)
        self.total_pickup_distance = np.zeros(self.row_idx * self.col_idx)
        self.total_pickup_time = np.zeros(self.row_idx * self.col_idx)
        self.total_wait_time = np.zeros(self.row_idx * self.col_idx)
        self.congestion_rate = np.zeros(self.row_idx * self.col_idx)

        # reward related environment statistics(or something changed just in compute reward function)
        self.mean_velocity = np.zeros((self.col_idx * self.row_idx, len(self.edges[0])))
        self.valid_distance = np.zeros(self.row_idx * self.col_idx)
        self.total_taxi_distances = np.zeros((self.row_idx * self.col_idx, len(self.taxis[0])))
        self.background_velocity = np.zeros(len(self.background_cars))
        self.background_co2 = np.zeros(len(self.background_cars))
        self.taxi_velocity = np.zeros((self.row_idx * self.col_idx, len(self.taxis[0])))
        self.taxi_co2 = np.zeros((self.row_idx * self.col_idx, len(self.taxis[0])))
        self.background_co = np.zeros(len(self.background_cars))
        self.taxi_co = np.zeros((self.col_idx * self.row_idx, len(self.taxis[0])))
        self.last_edge = dict([(veh_id, None) for veh_id in self.k.vehicle.get_ids()])

        # Appended statistics
        self.statistics = {
            'route': {},
            'location': {}
        }
        self.outside_taxis = []
        for idx in range(self.row_idx * self.col_idx):
            self.outside_taxis.append([])

        self.stop_time = []
        for idx in range(self.row_idx * self.col_idx):
            self.stop_time.append([])
            self.stop_time[idx] = [None] * len(self.taxis[0])
        self.reward_info = {'pickup_reward': np.zeros(self.col_idx * self.row_idx), 'time_reward': np.zeros(self.col_idx * self.row_idx),
                       'distance_reward': np.zeros(self.col_idx * self.row_idx)}
        return observation

    @property
    def _get_order_state(self):
        ord_ret = []
        for idx in range(self.row_idx * self.col_idx):
            col_idx = idx % self.col_idx
            row_idx = idx // self.col_idx
            x_length = self.inner_length * self.cols
            y_length = self.inner_length * self.rows
            orders = [[-1, -1, -1, -1, -1]] * self.max_num_order
            reservations_raw = self.k.person.get_reservations()
            reservations = []
            for res in reservations_raw:
                if res.fromEdge[-(len(str(idx)) + 1):] == ("_" + str(idx)):
                    reservations.append(res)
            cur_time = self.time_counter * self.sim_params.sim_step
            issued_orders = self.__dispatched_orders[idx] + self.__pending_orders[idx]
            self.__reservations[idx] = [res for res in reservations if res.id not in map(lambda x: x[0].id, issued_orders)
                                   and cur_time - res.reservationTime < self.max_waiting_time]
            if self.reservation_order == 'random':
                random.shuffle(self.__reservations)
            count = 0
            for res in self.__reservations[idx]:
                waiting_time = self.k.kernel_api.person.getWaitingTime(res.persons[0])
                # form_edge = res.fromEdge
                # to_edge = res.toEdge
                from_x, from_y = self.k.kernel_api.simulation.convert2D(res.fromEdge, res.departPos)
                to_x, to_y = self.k.kernel_api.simulation.convert2D(res.toEdge, res.arrivalPos)
                # orders[count] = [ waiting_time, self.edges.index(form_edge), self.edges.index(to_edge) ]
                orders[count] = [waiting_time, from_x - x_length * col_idx, from_y - y_length * row_idx,
                                 to_x - x_length * col_idx, to_y - y_length * row_idx]
                count += 1
                if count == self.max_num_order:
                    break
            ord_ret.append(np.array(orders).reshape(-1))
        return ord_ret

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        if self.env_params.sims_per_step > 1 and self.time_counter % self.env_params.sims_per_step != 1:
            return
        if self.verbose:
            print(self.time_counter)

        if 'reposition' not in self.statistics['location']:
            self.statistics['location']['reposition'] = np.zeros((self.row_idx * self.col_idx, len(self.edges[0])))
        reposition_stat = self.statistics['location']['reposition']

        for idx in range(self.row_idx * self.col_idx):
            if self.__need_reposition[idx]:
                if not self.__reservations[idx] or rl_actions[idx][1] == len(self.taxis[0]) or \
                    self.taxis[idx][rl_actions[idx][1]] != self.__need_reposition[idx]:
                    # taxi = self.__need_reposition
                    # stop = self.k.kernel_api.vehicle.getStops(taxi, limit=1)[0]
                    # print(self.k.vehicle.get_edge(taxi), stop.lane, self.k.vehicle.get_position(taxi), stop.startPos, stop.endPos)
                    if self.verbose:
                        print('reposition {} to {}, cur_edge {}'.format(self.__need_reposition[idx], self.edges[idx][rl_actions[idx][0]], self.k.vehicle.get_edge(self.__need_reposition[idx])))
                    self.k.vehicle.reposition_taxi_by_road(self.__need_reposition[idx], self.edges[idx][rl_actions[idx][0]])
                    reposition_stat[idx, rl_actions[idx][0]] += 1
                    self.__need_reposition[idx] = None
            if self.__reservations[idx]:
                if rl_actions[idx][1] < len(self.taxis[0]): # do not dispatch when the special action is selected
                    # check if the dispatch is valid
                    # cur_taxi = self.taxis[rl_actions[0]]
                    # cur_edge = self.k.vehicle.get_edge(cur_taxi)
                    # cur_pos = self.k.vehicle.get_position(cur_taxi)
                    # cur_res = self.__reservations[0]
                    # if not (cur_edge == cur_res.fromEdge and cur_pos > cur_res.departPos):
                    self.__pending_orders[idx].append([self.__reservations[idx][0], self.taxis[idx][rl_actions[idx][1]]]) # notice that we may dispach a order to a occupied_taxi
            if self.__need_mid_edge[idx]:
                mid_edges = [self.edges[idx][edge_id] for edge_id in rl_actions[idx][2:]]
                if 'flow' not in mid_edges[0]:
                    if -1 in rl_actions[idx][2:]:
                        mid_edges = []
                    self.k.vehicle.pickup(self.__need_mid_edge[idx], mid_edges)
                    res = self.k.vehicle.reservation[self.__need_mid_edge[idx]]
                    self.k.person.set_color(res.persons[0], (255, 255, 255)) # White
                    if self.verbose:
                        print('arrange mid edges', mid_edges, 'for', res, 'on', self.__need_mid_edge[idx])
                    self.__need_mid_edge[idx] = None
                else:
                    action_mask = self.get_action_mask()
                    print("Error in apply rl action")

        self._dispatch_taxi()

    def _dispatch_taxi(self):

        remain_pending_orders = []
        for idx in range(self.col_idx * self.row_idx):
            remain_pending_orders.append([])

        if 'pickup' not in self.statistics['location']:
            self.statistics['location']['pickup'] = []
            for idx in range(self.col_idx * self.row_idx):
                self.statistics['location']['pickup'].append([[], [], []])
        pickup_stat = self.statistics['location']['pickup']

        for idx in range(self.row_idx * self.col_idx):
            for res, veh_id in self.__pending_orders[idx]:
                # there would be some problems if the a taxi is on the road started with ":"
                # if the taxi is occupied now, we should dispatch this order later
                if self.k.kernel_api.vehicle.getRoadID(veh_id).startswith(':') or veh_id not in self.k.vehicle.get_taxi_fleet(0):
                    remain_pending_orders[idx].append([res, veh_id])
                elif self.k.kernel_api.person.getWaitingTime(res.persons[0]) <= self.max_waiting_time:
                    self.__dispatched_orders[idx].append((res, veh_id))
                    self.k.vehicle.dispatch_taxi(veh_id, res, tp=self.k.person.get_type(res.persons[0]))
                    self.k.person.match(res.persons[0], veh_id)
                    pickup_stat[idx][self.k.vehicle.get_res_type(veh_id)].append(self.k.vehicle.get_2d_position(veh_id))
                    if self.verbose:
                        print('dispatch {} to {}, remaining {} available taxis, cur_edge {}, cur_route {}'.format(\
                            res, veh_id, len(self.k.vehicle.get_taxi_fleet(0)), self.k.vehicle.get_edge(veh_id), self.k.vehicle.get_route(veh_id)))
                else:
                    if self.verbose:
                        print('order {} dispatch tle'.format(res))
        self.__pending_orders = remain_pending_orders

    def compute_reward(self, rl_actions, fail=False):
        """See class definition."""
        if fail:
            return np.zeros(self.row_idx * self.col_idx)
        reward = np.zeros(self.col_idx * self.row_idx)
        total_taxi = []
        for idx in range(self.col_idx * self.row_idx):
            for taxi in self.taxis[idx]:
                total_taxi.append(taxi)
        free_taxi = [[] for _ in range(self.col_idx * self.row_idx)]
        pickup_taxi = [[] for _ in range(self.col_idx * self.row_idx)]
        occupied_taxi = [[] for _ in range(self.col_idx * self.row_idx)]
        free_taxi_raw = self.k.vehicle.get_taxi_fleet(0)
        pickup_taxi_raw = self.k.vehicle.get_taxi_fleet(1)
        occupied_taxi_raw = self.k.vehicle.get_taxi_fleet(2)
        for idx in range(self.row_idx * self.col_idx):
            for taxi in free_taxi_raw:
                if (taxi[0: (6 + len(str(idx)))] == ("taxi_" + str(idx) + "_")):
                    free_taxi[idx].append(taxi)
            for taxi in occupied_taxi_raw:
                if (taxi[0: (6 + len(str(idx)))] == ("taxi_" + str(idx) + "_")):
                    occupied_taxi[idx].append(taxi)
            for taxi in pickup_taxi_raw:
                if (taxi[0: (6 + len(str(idx)))] == ("taxi_" + str(idx) + "_")):
                    pickup_taxi[idx].append(taxi)
        distances = []
        for idx in range(self.row_idx * self.col_idx):
            distances.append([self.k.vehicle.get_distance(taxi) for taxi in self.taxis[idx]])
        background_distances = [self.k.vehicle.get_distance(veh) for veh in self.background_cars]
        cur_time = self.time_counter * self.sim_params.sim_step
        timestep = self.sim_params.sim_step * self.env_params.sims_per_step
        n_edge = len(self.edges[0])

        # collect the mean velocity and total emission of edges
        for idx in range(self.col_idx * self.row_idx):
            num_congestion = 0.0
            for i, edge in enumerate(self.edges[idx]):
                n_veh = self.k.kernel_api.edge.getLastStepVehicleNumber(edge)
                mean_vel = self.k.kernel_api.edge.getLastStepMeanSpeed(edge)  # if n_veh > 0 else 10.0 # MAX_SPEED = 10.0
                self.mean_velocity[idx][i] = mean_vel  # / 10.0 / self.env_params.horizon
                if n_veh > 0 and mean_vel < 3.0:  # a threshold for congestion
                    num_congestion += 1
            self.congestion_rate[idx] = num_congestion / len(self.edges[0])

        #  collect the velocities and co2 emissions of vehicles
        for i, vehicle in enumerate(self.background_cars):
            try:
                speed = self.k.kernel_api.vehicle.getSpeed(vehicle)
                co2 = self.k.vehicle.get_co2_Emission(vehicle)
            except TraCIException:
                speed = 0
                co2 = 0
            self.background_velocity[i] = speed
            self.background_co2[i] = co2

        for idx in range(self.row_idx * self.col_idx):
            for i, vehicle in enumerate(self.taxis[idx]):
                try:
                    speed = self.k.kernel_api.vehicle.getSpeed(vehicle)
                    co2 = self.k.vehicle.get_co2_Emission(vehicle)
                except TraCIException:
                    speed = 0
                    co2 = 0
                self.taxi_velocity[idx][i] = speed
                self.taxi_co2[idx][i] = co2

        # collect the free vehicle density
        if 'free' not in self.statistics['route']:
            self.statistics['route']['free'] = np.zeros((self.col_idx * self.row_idx, n_edge))
        cnt = self.statistics['route']['free']
        for idx in range(self.row_idx * self.col_idx):
            for taxi in free_taxi[idx]:
                edge = self.k.vehicle.get_edge(taxi)
                if edge in self.edges[idx] and self.last_edge[taxi] != edge:
                    cnt[idx][self.edges[idx].index(edge)] += 1
                self.last_edge[taxi] = edge

        # collect the background vehicle density
        if 'background' not in self.statistics['route']:
            self.statistics['route']['background'] = np.zeros((self.row_idx * self.col_idx, n_edge))
        cnt = self.statistics['route']['background']
        for veh in self.k.vehicle.get_ids():
            if veh in total_taxi:
                continue
            edge = self.k.vehicle.get_edge(veh)
            for idx in range(self.row_idx * self.col_idx):
                if edge in self.edges[idx] and (veh not in self.last_edge or self.last_edge[veh] != edge):
                    cnt[idx][self.edges[idx].index(edge)] += 1
                self.last_edge[veh] = edge

        # collect the pickup vehicle density
        if 'pickup' not in self.statistics['route']:
            self.statistics['route']['pickup'] = np.zeros((self.col_idx * self.row_idx, 3, n_edge))
        cnt = self.statistics['route']['pickup']
        for idx in range(self.col_idx * self.row_idx):
            for taxi in pickup_taxi[idx]:
                edge = self.k.vehicle.get_edge(taxi)
                tp = self.k.vehicle.get_res_type(taxi)
                if edge in self.edges[idx] and self.last_edge[taxi] != edge:
                    cnt[idx][tp][self.edges[idx].index(edge)] += 1
                self.last_edge[taxi] = edge

        # collect the on-service vehicle density
        if 'occupied' not in self.statistics['route']:
            self.statistics['route']['occupied'] = np.zeros((self.col_idx * self.row_idx, 3, n_edge))
        cnt = self.statistics['route']['occupied']
        for idx in range(self.col_idx * self.row_idx):
            for taxi in occupied_taxi[idx]:
                edge = self.k.vehicle.get_edge(taxi)
                tp = self.k.vehicle.get_res_type(taxi)
                if edge in self.edges[idx] and self.last_edge[taxi] != edge:
                    cnt[idx][tp][self.edges[idx].index(edge)] += 1
                self.last_edge[taxi] = edge

        pre_reward = reward
        for person in self.k.person.get_ids():
            if not self.k.person.is_matched(person) and not self.k.person.is_removed(person):
                idx = int(person[person.rindex("_") + 1:])
                reward[idx] -= self.wait_penalty * timestep
                self.total_wait_time[idx] += timestep

        if self.verbose:
            print('-' * 10, 'un wait', [idx for idx in self.k.person.get_ids() if
                                        self.k.person.is_matched(idx) or self.k.person.is_removed(idx)])
            print('-' * 10, 'waiting', [idx for idx in self.k.person.get_ids() if
                                        not self.k.person.is_matched(idx) and not self.k.person.is_removed(idx)])
            print('-' * 10, 'need_reposition', self.__need_reposition)
            print('-' * 10, reward - pre_reward)

        for person in self.k.person.get_ids():
            if not self.k.person.is_removed(person):
                idx = int(person[person.rindex("_") + 1:])
                reward[idx] -= self.exist_penalty * timestep

        # pickup price
        self.reward_info['pickup_reward'] = np.zeros(self.row_idx * self.col_idx)
        for idx in range(self.row_idx * self.col_idx):
            for i, taxi in enumerate(self.taxis[idx]):
                # import pdb; pdb.set_trace();
                if self.taxi_states[idx][taxi]['empty'] and taxi in occupied_taxi[idx] and distances[idx][i] > 0:
                    assert self.taxi_states[idx][taxi]['pickup_distance'] is None
                    self.taxi_states[idx][taxi]['pickup_distance'] = distances[idx][i]
                    self.taxi_states[idx][taxi]['empty'] = False
                    reward[idx] += self.pickup_price
                    self.reward_info['pickup_reward'][idx] += self.pickup_price
                    if self.verbose:
                        print('taxi {} pickup successfully'.format(taxi))

        # miss penalty
        persons = self.k.kernel_api.person.getIDList()
        for person in persons:
            if self.k.kernel_api.person.getWaitingTime(person) > self.max_waiting_time:
                if not self.k.person.is_matched(person) and not self.k.person.is_removed(person):
                    idx = int(person[person.rindex("_") + 1:])
                    reward[idx] -= self.miss_penalty
                    self.k.person.remove(person)
                    self.k.person.set_color(person, (0, 255, 255))  # Cyan
                    if self.verbose:
                        print('tle request', person)

        # tle price
        for idx in range(self.row_idx * self.col_idx):
            for taxi in pickup_taxi[idx]:
                if cur_time - self.k.vehicle.reservation[taxi].reservationTime > self.free_pickup_time:
                    reward[idx] -= self.tle_penalty * timestep

        # price about time
        self.reward_info['time_reward'] = np.zeros(self.row_idx * self.col_idx)
        for idx in range(self.row_idx * self.col_idx):
            reward[idx] += len(occupied_taxi[idx]) * self.time_price * timestep
            self.reward_info['time_reward'][idx] += len(occupied_taxi[idx]) * self.time_price * timestep

        self.valid_distance = 0
        self.total_taxi_distances = np.zeros((self.col_idx * self.row_idx, len(self.taxis[0])))
        self.reward_info['distance_reward'] = np.zeros(self.row_idx * self.col_idx)
        for idx in range(self.row_idx * self.col_idx):
            for i, taxi in enumerate(self.taxis[idx]):
                # price about distance
                if taxi in occupied_taxi[idx] and self.taxi_states[idx][taxi]['pickup_distance'] and distances[idx][i] - \
                        self.taxi_states[idx][taxi]['pickup_distance'] > self.starting_distance:
                    if (distances[idx][i] - self.taxi_states[idx][taxi]['distance']) * self.distance_price > 100:
                        print(distances[idx][i], self.taxi_states[idx][taxi]['distance'])
                        raise Exception
                    reward[idx] += (distances[idx][i] - self.taxi_states[idx][taxi]['distance']) * self.distance_price
                    self.reward_info['distance_reward'][idx] += (distances[idx][i] - self.taxi_states[idx][taxi][
                        'distance']) * self.distance_price

                if distances[idx][i] > 0:
                    assert distances[idx][i] >= self.taxi_states[idx][taxi]['distance'], (
                    distances[idx][i], self.taxi_states[idx][taxi]['distance'])
                    if taxi in occupied_taxi[idx]:
                        self.total_valid_distance[idx] += (distances[idx][i] - self.taxi_states[idx][taxi]['distance'])
                        self.total_valid_time[idx] += timestep
                    elif taxi in pickup_taxi[idx]:
                        self.total_pickup_distance[idx] += (distances[idx][i] - self.taxi_states[idx][taxi]['distance'])
                        self.total_pickup_time[idx] += timestep
                    # self.valid_distance += (distances[i] - self.taxi_states[taxi]['distance'])
                    self.total_taxi_distances[idx][i] = (distances[idx][i] - self.taxi_states[idx][taxi]['distance'])
                    self.taxi_states[idx][taxi]['distance'] = distances[idx][i]

                # check empty
                if taxi not in occupied_taxi[idx] and not self.taxi_states[idx][taxi]['empty']:
                    self.taxi_states[idx][taxi]['empty'] = True
                    self.taxi_states[idx][taxi]['pickup_distance'] = None
                    self.num_complete_orders[idx] += 1

        self.total_back_distances = np.zeros(len(self.background_cars))
        for i, veh in enumerate(self.background_cars):
            if background_distances[i] > 0:
                assert background_distances[i] >= self.background_states[veh]['distance'], (
                background_distances[i], self.background_states[veh])
                self.total_back_distances[i] = (background_distances[i] - self.background_states[veh]['distance'])
                # self.total_distance += (background_distances[i] - self.background_states[veh]['distance'])
                self.background_states[veh]['distance'] = background_distances[i]


        self.total_distance = sum([self.background_states[veh]['distance'] for veh in self.background_cars])
        for idx in range(self.row_idx * self.col_idx):
            self.total_distance += sum([self.taxi_states[idx][taxi]['distance'] for taxi in self.taxis[idx]])

        # nonzero_distance = (self.total_taxi_distances.sum().sum() + self.total_back_distances.sum()) or 0.01
        # reward -= (self.taxi_co2.sum().sum() + self.background_co2.sum()) * 1e-3 / nonzero_distance * self.co2_penalty

        return reward

    def _remove_tle_request(self):
        idlist = self.k.kernel_api.person.getIDList()
        for person in idlist:
            if self.k.kernel_api.person.getWaitingTime(person) > self.max_waiting_time:
                if not self.k.person.is_matched(person) and not self.k.person.is_removed(person):
                    self.k.person.remove(person)
                    self.k.person.set_color(person, (0, 255, 255)) # Cyan
                    if self.verbose:
                        print('tle request', person)

    def additional_command(self):
        """See parent class."""
        self._check_route_valid()
        self._add_request()
        self._check_arrived()
        self._check_outside()

    def _check_route_valid(self):
        for idx in range(self.row_idx * self.col_idx):
            for veh_id in self.taxis[idx]:
                is_route_valid = self.k.kernel_api.vehicle.isRouteValid(veh_id)
                if not is_route_valid:
                    # if the route is not valid, we need to reset the route
                    self.k.kernel_api.vehicle.rerouteTraveltime(veh_id)

    def _check_outside(self):
        candidate_edges_back = self.real_in_edges.copy()
        candidate_edges_taxi = self.in_edges.copy()
        for i, veh_id in enumerate(self.background_cars):
            edge = self.k.vehicle.get_edge(veh_id)

            if edge in self.real_out_edges:
                in_edge = np.random.choice(candidate_edges_back)
                candidate_edges_back.remove(in_edge)
                idx = int(in_edge[in_edge.rindex("_") + 1:])
                in_edge_idx = self.edges[idx].index(in_edge)
                x, y = self.edge_position[idx][in_edge_idx][0]
                self.k.kernel_api.vehicle.moveToXY(veh_id, in_edge, lane='0', x=x, y=y, keepRoute=0)

        for idx in range(self.col_idx * self.row_idx):
            for i, taxi in enumerate(self.taxis[idx]):
                if taxi in self.outside_taxis[idx]:
                    in_edge_idx = np.random.choice(candidate_edges_taxi[idx])
                    in_edge = self.edges[idx][in_edge_idx]
                    x, y = self.edge_position[idx][in_edge_idx][0]
                    try:
                        self.k.kernel_api.vehicle.moveToXY(taxi, in_edge, lane='0', x=x, y=y, keepRoute=0)
                        self.outside_taxis[idx].remove(taxi)
                        pass
                    except TraCIException:
                        self.k.vehicle.remove(taxi)
                        self.k.kernel_api.vehicle.remove(taxi)
                        self.k.vehicle.add(
                            taxi,
                            'taxi',
                            edge=in_edge,
                            pos=0,
                            lane=0,
                            speed=0,
                        )

    def _check_arrived(self):

        def is_arrived(veh_id, edge, pos):
            cur_edge = self.k.vehicle.get_edge(veh_id)
            if cur_edge == edge:
                cur_pos = self.k.vehicle.get_position(veh_id)
                if abs(cur_pos - pos) < 10:
                    return True
            return False

        for idx in range(self.row_idx * self.col_idx):
            for i, taxi in enumerate(self.taxis[idx]):
                if self.k.vehicle.is_pickup(taxi):
                    tgt_edge, tgt_pos = self.k.vehicle.pickup_stop[taxi]
                elif self.k.vehicle.is_occupied(taxi):
                    mid_edges = self.k.vehicle.mid_edges[taxi]
                    if len(mid_edges) > 0 and mid_edges[0] == self.k.vehicle.get_edge(taxi):
                        self.k.vehicle.checkpoint(taxi)
                        continue
                    tgt_edge, tgt_pos = self.k.vehicle.dropoff_stop[taxi]
                elif self.k.vehicle.is_free(taxi):
                    if len(self.k.kernel_api.vehicle.getStops(taxi)) == 0 and taxi not in self.outside_taxis[idx]:
                        self.k.vehicle.stop(taxi)
                    continue
                else:
                    continue

                if is_arrived(taxi, tgt_edge, tgt_pos):
                    if self.k.vehicle.is_pickup(taxi):
                        self.k.kernel_api.vehicle.setSpeed(taxi, 0)
                        self.__need_mid_edge[idx] = taxi
                    elif self.k.vehicle.is_occupied(taxi):
                        if self.stop_time[idx][i] is None:
                            self.k.kernel_api.vehicle.setSpeed(taxi, 0)
                            self.stop_time[idx][i] = 0
                        else:
                            self.stop_time[idx][i] += 1

                        assert self.stop_time[idx][i] is None or self.stop_time[idx][i] <= 3 * self.env_params.sims_per_step
                        if self.stop_time[idx][i] == 3 * self.env_params.sims_per_step:
                            self.stop_time[idx][i] = None
                            edge = self.k.vehicle.get_edge(taxi)
                            edge_idx = self.edges[idx].index(edge)
                            is_outside = edge_idx in self.out_edges[idx]
                            self.k.vehicle.dropoff(taxi, is_outside)
                            if is_outside:
                                self.outside_taxis[idx].append(taxi)
                            res = self.k.vehicle.reservation[taxi]
                            self.k.person.remove(res.persons[0])

    def _update_action_mask(self, idx):

        self.action_mask[idx] = torch.zeros((len(self.taxis[0]) + 1, sum(self.action_space.nvec[0])), dtype=bool)

        n_edge = len(self.edges[0])
        n_taxi = len(self.taxis[0])

        # mid point mask
        if self.__need_mid_edge[idx] is not None:
            res = self.k.vehicle.reservation[self.__need_mid_edge[idx]]
            from_id, to_id = self.edges[idx].index(res.fromEdge), self.edges[idx].index(res.toEdge)
            taxi_id = self.taxis[idx].index(self.__need_mid_edge[idx])

            # mask from edge, to edge
            for i in range(self.n_mid_edge):
                self.action_mask[idx][taxi_id][n_edge + n_taxi + 1 + i * n_edge + from_id] = True
                self.action_mask[idx][taxi_id][n_edge + n_taxi + 1 + i * n_edge + to_id] = True
                # mask flow edges, in edges and out edges
                for j in self.flow_edges[idx] + self.in_edges[idx] + self.out_edges[idx]:
                    self.action_mask[idx][taxi_id][n_edge + n_taxi + 1 + i * n_edge + j] = True

            if self.n_mid_edge == 0:
                pass
            elif self.n_mid_edge == 1:
                self.action_mask[idx][taxi_id][n_edge + n_taxi + 1:] |= self.banned_mid_edges[from_id, to_id]
            else:
                raise NotImplementedError

        for i, taxi in enumerate(self.taxis[idx]):
            cur_edge = self.k.vehicle.get_edge(taxi)
            assert cur_edge != ""

            # reposition mask, mask current edge
            if cur_edge in self.edges[idx]:
                edge_id = self.edges[idx].index(cur_edge)
                self.action_mask[idx][i][edge_id] = True

            # reposition mask, mask flow edges, in edges and out edges
            for j in self.flow_edges[idx] + self.in_edges[idx] + self.out_edges[idx]:
                self.action_mask[idx][i][j] = True

            # taxi mask
            if len(self.__reservations[idx]) > 0:
                res = self.__reservations[idx][0]
                edge = self.k.vehicle.get_edge(taxi)
                pos = self.k.vehicle.get_position(taxi)
                # Do not dispath the order to the taxi on out edges
                edge_idx = self.edges[idx].index(edge) if edge in self.edges[idx] else -1
                if edge == res.fromEdge and pos > res.departPos:
                    self.action_mask[idx][len(self.taxis[0])][n_edge + i] = True
                if edge_idx in self.out_edges[idx]:
                    self.action_mask[idx][len(self.taxis[0])][n_edge + i] = True
                for edge in self.k.vehicle.get_route(taxi):
                    if self.edges[idx].index(edge) in self.out_edges[idx]:
                        self.action_mask[idx][len(self.taxis[0])][n_edge + i] = True
                        break

    def _add_request(self):
        if np.random.rand() > self.person_prob * self.sim_params.sim_step and self.distribution != "mode-4" and self.distribution != "mode-5":
            return
        for idx in range(self.col_idx * self.row_idx):
            per_id, edge_id1, edge_id2, pos, tp = gen_request(self, area=idx)
            self.k.person.add_request(per_id, edge_id1, edge_id2, pos, tp=tp)
            if self.verbose:
                print('add request from', edge_id1, 'to', edge_id2, 'total', self.k.person.total)
