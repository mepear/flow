"""Environments for networks with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m traffic light grid.
"""

import numpy as np
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
from flow.envs.base import Env
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
    "reservation_order": 'random', # random or fifo 
    "n_mid_edge": 0, # number of mid point for an order
    "use_tl": False, # whether using traffic light info
    "max_detour": 1.5, # detour length / minimal length <= max_detour
}

class DispatchAndRepositionEnv(Env):
 
    def __init__(self, env_params, sim_params, network, simulator='traci'):

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.grid_array = network.net_params.additional_params["grid_array"]
        self.rows = self.grid_array["row_num"]
        self.cols = self.grid_array["col_num"]
        self.inner_length = self.grid_array['inner_length']
        self.outer_length = self.grid_array.get("outer_length", None)

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

        self.person_prob = env_params.additional_params['person_prob']
        
        self.max_num_order = env_params.additional_params['max_num_order']
        self.max_waiting_time = env_params.additional_params['max_waiting_time']
        self.free_pickup_time = env_params.additional_params['free_pickup_time']

        self.reservation_order = env_params.additional_params['reservation_order']

        self.n_mid_edge = env_params.additional_params['n_mid_edge']
        self.use_tl = env_params.additional_params['use_tl']
        self.tl_params = network.traffic_lights
        self.n_tl = len(self.tl_params.get_properties())
        self.n_phase = 4 # the default has 4 phases

        self.verbose = env_params.verbose

        self.num_complete_orders = 0
        self.total_valid_distance = 0
        self.total_distance = 0
        self.total_valid_time = 0
        self.total_pickup_distance = 0
        self.total_pickup_time = 0
        self.total_wait_time = 0
        self.congestion_rate = 0

        self.stop_distance_eps = env_params.additional_params['stop_distance_eps']
        
        self.distribution = env_params.additional_params['distribution']
        self.distribution_random_ratio = env_params.additional_params.get('distribution_random_ratio', 0.5)
        # print(self.distribution)
        super().__init__(env_params, sim_params, network, simulator)

        # Saving env variables for plotting
        self.steps = env_params.horizon
        self.obs_var_labels = {
            'edges': np.zeros((self.steps, self.k.vehicle.num_vehicles)),
            'velocities': np.zeros((self.steps, self.k.vehicle.num_vehicles)),
            'positions': np.zeros((self.steps, self.k.vehicle.num_vehicles))
        }

        self.edges = self.k.network.get_edge_list()
        # in/out flow edge cannot be visited by taxi
        self.flow_edges = []
        self.in_edges = []
        self.out_edges = []
        for i, edge in enumerate(self.edges):
            if 'flow' in edge:
                self.flow_edges.append(i)
            elif 'in' in edge:
                self.in_edges.append(i)
            elif 'out' in edge:
                self.out_edges.append(i)

        self._preprocess()

        self.num_taxi = network.vehicles.num_rl_vehicles
        self.taxis = [taxi for taxi in network.vehicles.ids if network.vehicles.get_type(taxi) == 'taxi']
        self.outside_taxis = []
        self.background_cars = [car for car in network.vehicles.ids if network.vehicles.get_type(car) != 'taxi']
        assert self.num_taxi == len(self.taxis)
        self.num_vehicles = network.vehicles.num_vehicles

        self.mean_velocity = np.zeros(len(self.edges))
        self.valid_distance = 0
        self.total_taxi_distances = np.zeros(len(self.taxis))
        self.background_velocity = np.zeros(len(self.background_cars))
        self.background_co2 = np.zeros(len(self.background_cars))
        self.taxi_velocity = np.zeros(len(self.taxis))
        self.taxi_co2 = np.zeros(len(self.taxis))
        self.background_co = np.zeros(len(self.background_cars))
        self.taxi_co = np.zeros(len(self.taxis))
        self.edge_position = [(self.k.kernel_api.simulation.convert2D(edge, 0), \
            self.k.kernel_api.simulation.convert2D(edge, self.k.kernel_api.lane.getLength(edge + '_0')), \
            self.k.kernel_api.lane.getWidth(edge + '_0')) \
            for edge in self.edges]
        self.statistics = {
            'route': {},
            'location': {}
        }
        self.last_edge = dict([(veh_id, None) for veh_id in self.k.vehicle.get_ids()])

        self.__dispatched_orders = []
        self.__pending_orders = []
        self.__reservations = []
        self.__need_reposition = None
        self.__need_mid_edge = None

        self.taxi_states = dict([[taxi, {"empty": True, "distance": 0, "pickup_distance": None}] for taxi in self.taxis])
        self.background_states = dict([[veh_id, {'distance': 0}] for veh_id in self.background_cars])
        self.stop_time = [None] * len(self.taxis)

        self.hist_dist = [Queue(maxsize=int(self.env_params.additional_params['max_stop_time'] / self.sim_params.sim_step) + 1) for i in range(self.num_taxi)]
        self.action_mask = torch.zeros((self.num_taxi + 1, sum(self.action_space.nvec)), dtype=bool)

        # test for several functions    
        if 'ENV_TEST' in os.environ and os.environ['ENV_TEST'] == '1':
            self._test()

    def _test(self):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection

        def get_corners(s, e, w):
            s, e = np.array(s), np.array(e)
            se = e - s
            p = np.array([-se[1], se[0]])
            p = p / np.linalg.norm(p) * w / 2
            return [s + p, e + p, e - p, s - p]
        
        edge1 = 'bot2_1_0'
        edge2 = 'bot1_3_0'
        edge3 = 'top2_3_0'

        id1 = self.edges.index(edge1)
        id2 = self.edges.index(edge2)
        id3 = self.edges.index(edge3)

        route1 = set(self.paired_complete_routes[id1][id2])
        route2 = set(self.paired_complete_routes[id2][id3])
        # print('part 1', route1)
        # print('part 2', route2)
        # print('intersect', route1 & route2)

        fig, ax = plt.subplots()
        patches = []
        colors = []
        for i in range(len(self.edges)):
            poly = Polygon(get_corners(*self.edge_position[i]), True)
            patches.append(poly)
            if self.banned_mid_edges[id1, id2, i]:
                colors.append(100.0)
            else:
                colors.append(0.0)
        p = PatchCollection(patches)
        p.set_array(np.array(colors))
        ax.add_collection(p)
        fig.colorbar(p, ax=ax)
        plt.xlim(-5, 160)
        plt.ylim(-5, 160)
        plt.show()
        
        exit(0)

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

        n_edge = len(self.edges)
        save_path = os.path.join(self.env_params.save_path, 'preprocess.pt')
        while True:
            try:
                with Lock(name='preprocess'):
                    if os.path.exists(save_path):
                        self.paired_routes, self.paired_complete_routes, self.banned_mid_edges = \
                            torch.load(save_path)
                    else:
                        self.paired_routes = [
                            [self.k.kernel_api.simulation.findRoute(s, t) for t in self.edges] \
                            for s in self.edges
                        ]
                        self.centers = {}
                        for edge1 in self.network.edges:
                            for edge2 in self.network.edges:
                                if edge1['to'] == edge2['from']:
                                    self.centers[edge1['id'] + '&' + edge2['id']] = edge1['to']
                        self.paired_complete_routes = [
                            [_add_center(route.edges) for route in routes] \
                            for routes in self.paired_routes
                        ]
                        self.banned_mid_edges = torch.zeros((n_edge, n_edge, n_edge), dtype=bool)
                        for i in range(n_edge):
                            for j in range(n_edge):
                                if i != j:
                                    l = len(self.paired_routes[i][j].edges)
                                    for k in range(n_edge):
                                        if k != i and k != j:
                                            l1 = len(self.paired_routes[i][k].edges)
                                            l2 = len(self.paired_routes[k][j].edges)
                                            r1 = set(self.paired_complete_routes[i][k][:-1])
                                            r2 = set(self.paired_complete_routes[k][j][1:])
                                            if len(r1 & r2) > 0 or l1 + l2 - 1 > self.max_detour * l \
                                                or l1 == 0 or l2 == 0: # This is for unreachable path
                                                self.banned_mid_edges[i, j, k] = True
                        torch.save([self.paired_routes, self.paired_complete_routes, self.banned_mid_edges], \
                            save_path)
                break
            except CannotAcquireLock:
                pass


    def get_action_mask(self):
        mask = torch.zeros_like(self.action_mask[0])
        if self.__need_reposition:
            taxi_id = self.taxis.index(self.__need_reposition)
            # print(self.action_mask[taxi_id].unsqueeze(0))
            mask = torch.logical_or(mask, self.action_mask[taxi_id])
        if self.__need_mid_edge:
            taxi_id = self.taxis.index(self.__need_mid_edge)
            mask = torch.logical_or(mask, self.action_mask[taxi_id])
        if len(self.__reservations) > 0:
            mask = torch.logical_or(mask, self.action_mask[self.num_taxi])
        return mask.unsqueeze(0)

    @property
    def action_space(self):
        """See class definition."""
        try:
            return MultiDiscrete([len(self.edges), self.num_taxi + 1] + [len(self.edges)] * self.n_mid_edge, dtype=np.float32)
        except:
            return MultiDiscrete([len(self.edges), self.num_taxi + 1] + [len(self.edges)] * self.n_mid_edge)
        # return MultiDiscrete([self.num_taxi + 1, len(self.edges)])

    @property
    def observation_space(self):
        """See class definition."""
        # edges_feature = Box(
        #     low=0,
        #     high=self.num_vehicles,
        #     shape=( len(self.edges), ) # number of cars in the road
        # )
        # taxi_feature = Box(
        #     low=0,
        #     high=len(self.edges),
        #     shape=( self.num_taxi * 3, ) # cur_state, from edge, to edge
        # )
        # order_feature = Box(
        #     low=0,
        #     high=len(self.edges),
        #     shape=( self.max_num_order * 3, ) # patience, from edge, to edge
        # )
        # return Tuple((edges_feature, taxi_feature, order_feature))
        state_box = Box(
            low=np.float32(-500),
            high=np.float32(500),
            shape=( 1 + len(self.edges) + self.num_taxi * 9 + \
                int(self.use_tl) * self.n_tl * (self.n_phase + 1) + \
                self.max_num_order * 5 + (4 + len(self.taxis)) + self.num_taxi + 2, ),
            dtype=np.float32
        )
        return state_box

    def get_state(self):
        """See class definition."""

        time_feature = [self.time_counter / (self.env_params.horizon * self.env_params.sims_per_step)]

        edges_feature = [
            self.k.kernel_api.edge.getLastStepVehicleNumber(edge) for edge in self.edges
        ]
        taxi_feature = []
        empty_taxi = self.k.vehicle.get_taxi_fleet(0)
        pickup_taxi = self.k.vehicle.get_taxi_fleet(1)               
        
        for taxi in self.taxis:
            while taxi not in self.k.vehicle.get_rl_ids():
                raise KeyError
            x, y = self.k.vehicle.get_2d_position(taxi, error=(-1, -1))
            from_x, from_y = self.k.kernel_api.simulation.convert2D(self.k.kernel_api.vehicle.getRoute(taxi)[0], 0)
            to_edge = self.k.kernel_api.vehicle.getRoute(taxi)[-1]
            to_pos = self.inner_length - 2 if 'out' not in to_edge else self.outer_length - 2
            to_x, to_y = self.k.kernel_api.simulation.convert2D(to_edge, to_pos)
            # cur_taxi_feature = [0, x, y, self.edges.index(self.k.kernel_api.vehicle.getRoute(taxi)[0]), self.edges.index(self.k.kernel_api.vehicle.getRoute(taxi)[-1])]
            cur_taxi_feature = [0, 0, 0, x, y, from_x, from_y, to_x, to_y] # use (x, y) or edge id
            cur_taxi_feature[0 if taxi in empty_taxi else 1 if taxi in pickup_taxi else 2] = 1
            taxi_feature += cur_taxi_feature

        # use traffic light info
        tl_feature = []
        if self.use_tl:
            tl = self.k.kernel_api.trafficlight
            for tl_id in tl.getIDList():
                phase = tl.getPhase(tl_id)
                t_next = tl.getNextSwitch(tl_id)
                links = tl.getControlledLinks(tl_id)
                logic = tl.getAllProgramLogics(tl_id)[-1]
                state = tl.getRedYellowGreenState(tl_id)

                len_phases = 4 # TODO: len(logic.phases)
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
        
        order_feature = self._get_order_state.tolist()

        mid_edge_feature = [-1] * (4 + len(self.taxis))
        if self.__need_mid_edge:
            taxi = self.__need_mid_edge
            res = self.k.vehicle.reservation[taxi]
            from_x, from_y = self.k.kernel_api.simulation.convert2D(res.fromEdge, res.departPos)
            to_x, to_y = self.k.kernel_api.simulation.convert2D(res.toEdge, 25)
            veh_id = self.taxis.index(taxi)
            index = [0] * len(self.taxis)
            index[veh_id] = 1
            mid_edge_feature = [from_x, from_y, to_x, to_y] + index
        
        self._update_action_mask()
    
        index = [0] * len(self.taxis)
        # if self.__reservations:
        #     need_reposition_taxi_feature = index + [-1, -1]
        # else:
        empty_taxi_fleet = self.k.vehicle.get_taxi_fleet(0)
        self.__need_reposition = None
        for taxi in empty_taxi_fleet:
            edge = self.k.vehicle.get_edge(taxi)
            edge_id = self.edges.index(edge) if edge in self.edges else -1
            # Don't reposit the taxi in flow_edges, in_edges and out_edges
            if edge_id in self.flow_edges + self.out_edges:
                continue
            elif self.k.kernel_api.vehicle.isStopped(taxi):
                self.__need_reposition = taxi
                break
        
        if self.__need_reposition:
            # need_reposition_taxi_feature = [self.edges.index(self.k.kernel_api.vehicle.getRoadID(self.__need_reposition)), self.k.vehicle.get_position(self.__need_reposition)]
            x, y = self.k.vehicle.get_2d_position(self.__need_reposition, error=(-1, -1))
            index[self.taxis.index(self.__need_reposition)] = 1
            need_reposition_taxi_feature = index + [x, y]
        else:
            need_reposition_taxi_feature = index + [-1, -1]

        state = time_feature + edges_feature + taxi_feature + tl_feature + order_feature + mid_edge_feature + need_reposition_taxi_feature
        return np.array(state)
    
    def _get_info(self):
        return {}

    def reset(self):
        self.__need_mid_edge = None
        observation = super().reset()
        self.__dispatched_orders = []
        self.__pending_orders = []
        self.__reservations = []
        self.__need_reposition = None
        self.taxi_states = dict([[taxi, {"empty": True, "distance": 0, "pickup_distance": None}] for taxi in self.taxis])
        self.background_states = dict([[veh_id, {'distance': 0}] for veh_id in self.background_cars])
        self.hist_dist = [Queue(maxsize=int(self.env_params.additional_params['max_stop_time'] / self.sim_params.sim_step) + 1) for i in range(self.num_taxi)]
        self.action_mask = torch.zeros((self.num_taxi + 1, sum(self.action_space.nvec)), dtype=bool)
        self.num_complete_orders = 0
        self.total_valid_distance = 0
        self.total_distance = 0
        self.total_valid_time = 0
        self.total_pickup_distance = 0
        self.total_pickup_time = 0
        self.total_wait_time = 0
        self.congestion_rate = 0
        self.mean_velocity = np.zeros(len(self.edges))
        self.valid_distance = 0
        self.total_taxi_distances = np.zeros(len(self.taxis))
        self.background_velocity = np.zeros(len(self.background_cars))
        self.background_co2 = np.zeros(len(self.background_cars))
        self.taxi_velocity = np.zeros(len(self.taxis))
        self.taxi_co2 = np.zeros(len(self.taxis))
        self.background_co = np.zeros(len(self.background_cars))
        self.taxi_co = np.zeros(len(self.taxis))
        self.stop_time = [None] * len(self.taxis)
        self.outside_taxis = []
        self.statistics = {
            'route': {},
            'location': {}
        }
        self.last_edge = dict([(veh_id, None) for veh_id in self.k.vehicle.get_ids()])
        return observation

    @property
    def _get_order_state(self):
        # if np.random.rand() < self.person_prob:
        #     person_ids = [int(per_id[4:]) for per_id in self.k.person.get_ids()]
        #     idx = max(person_ids) + 1 if len(person_ids) > 0 else 1
        #     edge_list = self.edges.copy()
        #     edge_id1 = np.random.choice(edge_list)
        #     edge_list.remove(edge_id1)
        #     edge_id2 = np.random.choice(edge_list)

        #     per_id = 'per_' + str(idx)
        #     pos = np.random.uniform(self.inner_length)
        #     self.k.kernel_api.person.add(per_id, edge_id1, pos)
        #     self.k.kernel_api.person.appendDrivingStage(per_id, edge_id2, 'taxi')
        #     self.k.kernel_api.person.setColor(per_id, (255, 0, 0))
        # print('add_request', per_id, 'from', str(edge_id1), 'to', str(edge_id2))
        orders = [[-1, -1, -1, -1, -1]] * self.max_num_order
        reservations = self.k.person.get_reservations()
        cur_time = self.time_counter * self.sim_params.sim_step
        issued_orders = self.__dispatched_orders + self.__pending_orders
        self.__reservations = [res for res in reservations if res.id not in map(lambda x: x[0].id, issued_orders) and cur_time - res.reservationTime < self.max_waiting_time]
        if self.reservation_order == 'random':
            random.shuffle(self.__reservations)

        count = 0
        for res in self.__reservations:            
            waiting_time = self.k.kernel_api.person.getWaitingTime(res.persons[0])
            # form_edge = res.fromEdge
            # to_edge = res.toEdge
            from_x, from_y = self.k.kernel_api.simulation.convert2D(res.fromEdge, res.departPos)
            to_x, to_y = self.k.kernel_api.simulation.convert2D(res.toEdge, res.arrivalPos)
            # orders[count] = [ waiting_time, self.edges.index(form_edge), self.edges.index(to_edge) ]
            orders[count] = [waiting_time, from_x, from_y, to_x, to_y]
            count += 1
            if count == self.max_num_order:
                break
        return np.array(orders).reshape(-1)

    # def _apply_rl_actions(self, rl_actions):
        # pass

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        if self.env_params.sims_per_step > 1 and self.time_counter % self.env_params.sims_per_step != 1:
            return
        if self.verbose:
            print(self.time_counter)
        if 'reposition' not in self.statistics['location']:
            self.statistics['location']['reposition'] = np.zeros(len(self.edges))
        reposition_stat = self.statistics['location']['reposition']
        if self.__need_reposition:
            if not self.__reservations or rl_actions[1] == self.num_taxi or \
                self.taxis[rl_actions[1]] != self.__need_reposition:
                # taxi = self.__need_reposition
                # stop = self.k.kernel_api.vehicle.getStops(taxi, limit=1)[0]
                # print(self.k.vehicle.get_edge(taxi), stop.lane, self.k.vehicle.get_position(taxi), stop.startPos, stop.endPos)
                if self.verbose:
                    print('reposition {} to {}, cur_edge {}'.format(self.__need_reposition, self.edges[rl_actions[0]], self.k.vehicle.get_edge(self.__need_reposition)))
                self.k.vehicle.reposition_taxi_by_road(self.__need_reposition, self.edges[rl_actions[0]])
                reposition_stat[rl_actions[0]] += 1
                self.__need_reposition = None
        if self.__reservations:
            if rl_actions[1] < self.num_taxi: # do not dispatch when the special action is selected
                # check if the dispatch is valid
                # cur_taxi = self.taxis[rl_actions[0]]
                # cur_edge = self.k.vehicle.get_edge(cur_taxi)
                # cur_pos = self.k.vehicle.get_position(cur_taxi)
                # cur_res = self.__reservations[0]
                # if not (cur_edge == cur_res.fromEdge and cur_pos > cur_res.departPos):
                self.__pending_orders.append([self.__reservations[0], self.taxis[rl_actions[1]]]) # notice that we may dispach a order to a occupied_taxi
        if self.__need_mid_edge:
            mid_edges = [self.edges[edge_id] for edge_id in rl_actions[2:]]
            if 'flow' not in mid_edges[0]:
                if -1 in rl_actions[2:]:
                    mid_edges = []
                self.k.vehicle.pickup(self.__need_mid_edge, mid_edges)
                res = self.k.vehicle.reservation[self.__need_mid_edge]
                self.k.person.set_color(res.persons[0], (255, 255, 255)) # White
                if self.verbose:
                    print('arrange mid edges', mid_edges, 'for', res, 'on', self.__need_mid_edge)
                self.__need_mid_edge = None
            else:
                action_mask = self.get_action_mask()
                print(action_mask[0, len(self.edges) + len(self.taxis) + 1 + 12], \
                    action_mask[0, len(self.edges) + len(self.taxis) + 1 + 25])

        self._dispatch_taxi()

    def compute_reward(self, rl_actions, fail=False):
        """See class definition."""
        if fail:
            return 0.
        reward = 0
        free_taxi = self.k.vehicle.get_taxi_fleet(0)
        pickup_taxi = self.k.vehicle.get_taxi_fleet(1)
        occupied_taxi = self.k.vehicle.get_taxi_fleet(2)
        distances = [self.k.vehicle.get_distance(taxi) for taxi in self.taxis]
        background_distances = [self.k.vehicle.get_distance(veh) for veh in self.background_cars]
        cur_time = self.time_counter * self.sim_params.sim_step
        timestep = self.sim_params.sim_step * self.env_params.sims_per_step
        n_edge = len(self.edges)

        # collect the mean velocity and total emission of edges
        num_congestion = 0.0
        for i, edge in enumerate(self.edges):
            n_veh = self.k.kernel_api.edge.getLastStepVehicleNumber(edge)
            mean_vel = self.k.kernel_api.edge.getLastStepMeanSpeed(edge) # if n_veh > 0 else 10.0 # MAX_SPEED = 10.0
            self.mean_velocity[i] = mean_vel #/ 10.0 / self.env_params.horizon
            if n_veh > 0 and mean_vel < 3.0: # a threshold for congestion
                num_congestion += 1
        self.congestion_rate = num_congestion / len(self.edges)

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
        
        for i, vehicle in enumerate(self.taxis):
            try:
                speed = self.k.kernel_api.vehicle.getSpeed(vehicle)
                co2 = self.k.vehicle.get_co2_Emission(vehicle)
            except TraCIException:
                speed = 0
                co2 = 0
            self.taxi_velocity[i] = speed
            self.taxi_co2[i] = co2

        # collect the free vehicle density
        if 'free' not in self.statistics['route']:
            self.statistics['route']['free'] = np.zeros((n_edge))
        cnt = self.statistics['route']['free']
        for taxi in free_taxi:
            edge = self.k.vehicle.get_edge(taxi)
            if edge in self.edges and self.last_edge[taxi] != edge:
                cnt[self.edges.index(edge)] += 1
            self.last_edge[taxi] = edge

        # collect the background vehicle density
        if 'background' not in self.statistics['route']:
            self.statistics['route']['background'] = np.zeros((n_edge))
        cnt = self.statistics['route']['background']
        for veh in self.k.vehicle.get_ids():
            if veh in self.taxis:
                continue
            edge = self.k.vehicle.get_edge(veh)
            if edge in self.edges and (veh not in self.last_edge or self.last_edge[veh] != edge):
                cnt[self.edges.index(edge)] += 1
            self.last_edge[veh] = edge


        # collect the pickup vehicle density
        if 'pickup' not in self.statistics['route']:
            self.statistics['route']['pickup'] = \
                np.zeros((1 if 'mode-X' not in self.distribution else 3, n_edge))
        cnt = self.statistics['route']['pickup']
        for taxi in pickup_taxi:
            edge = self.k.vehicle.get_edge(taxi)
            tp = self.k.vehicle.get_res_type(taxi)
            if edge in self.edges and self.last_edge[taxi] != edge:
                cnt[tp][self.edges.index(edge)] += 1
            self.last_edge[taxi] = edge

        # collect the on-service vehicle density
        if 'occupied' not in self.statistics['route']:
            self.statistics['route']['occupied'] = \
                np.zeros((1 if 'mode-X' not in self.distribution else 3, n_edge))
        cnt = self.statistics['route']['occupied']
        for taxi in occupied_taxi:
            edge = self.k.vehicle.get_edge(taxi)
            tp = self.k.vehicle.get_res_type(taxi)
            if edge in self.edges and self.last_edge[taxi] != edge:
                cnt[tp][self.edges.index(edge)] += 1
            self.last_edge[taxi] = edge

        pre_reward = reward
        for person in self.k.person.get_ids():
            if not self.k.person.is_matched(person) and not self.k.person.is_removed(person):
                reward -= self.wait_penalty * timestep
                self.total_wait_time += timestep

        if self.verbose:
            print('-' * 10, 'un wait', [idx for idx in self.k.person.get_ids() if self.k.person.is_matched(idx) or self.k.person.is_removed(idx)])
            print('-' * 10, 'waiting', [idx for idx in self.k.person.get_ids() if not self.k.person.is_matched(idx) and not self.k.person.is_removed(idx)])
            print('-' * 10, 'need_reposition', self.__need_reposition)
            print('-' * 10, reward - pre_reward)

        for person in self.k.person.get_ids():
            if not self.k.person.is_removed(person):
                reward -= self.exist_penalty * timestep

        # pickup price  
        for i, taxi in enumerate(self.taxis):
            if self.taxi_states[taxi]['empty'] and taxi in occupied_taxi and distances[i] > 0:
                assert self.taxi_states[taxi]['pickup_distance'] is None
                self.taxi_states[taxi]['pickup_distance'] = distances[i]
                self.taxi_states[taxi]['empty'] = False
                reward += self.pickup_price
                if self.verbose:
                    print('taxi {} pickup successfully'.format(taxi))

        # miss penalty
        persons = self.k.kernel_api.person.getIDList()
        for person in persons:
            if self.k.kernel_api.person.getWaitingTime(person) > self.max_waiting_time:
                if not self.k.person.is_matched(person) and not self.k.person.is_removed(person):
                    reward -= self.miss_penalty
                    self.k.person.remove(person)
                    self.k.person.set_color(person, (0, 255, 255)) # Cyan
                    if self.verbose:
                        print('tle request', person)

        # tle price
        for taxi in pickup_taxi:
            if cur_time - self.k.vehicle.reservation[taxi].reservationTime > self.free_pickup_time:
                reward -= self.tle_penalty * timestep

        # price about time 
        reward += len(occupied_taxi) * self.time_price * timestep
        

        self.valid_distance = 0
        self.total_taxi_distances = np.zeros(len(self.taxis))
        for i, taxi in enumerate(self.taxis):
            # price about distance
            if taxi in occupied_taxi and self.taxi_states[taxi]['pickup_distance'] and distances[i] - self.taxi_states[taxi]['pickup_distance'] > self.starting_distance:
                if (distances[i] - self.taxi_states[taxi]['distance']) * self.distance_price > 100:
                    print(distances[i], self.taxi_states[taxi]['distance'])
                    raise Exception
                reward += (distances[i] - self.taxi_states[taxi]['distance']) * self.distance_price

            # update distance
            if distances[i] > 0:
                assert distances[i] >= self.taxi_states[taxi]['distance'], (distances[i], self.taxi_states[taxi]['distance'])
                if taxi in occupied_taxi:
                    self.total_valid_distance += (distances[i] - self.taxi_states[taxi]['distance'])
                    self.total_valid_time += timestep
                elif taxi in pickup_taxi:
                    self.total_pickup_distance += (distances[i] - self.taxi_states[taxi]['distance'])
                    self.total_pickup_time += timestep
                # self.valid_distance += (distances[i] - self.taxi_states[taxi]['distance'])
                self.total_taxi_distances[i] = (distances[i] - self.taxi_states[taxi]['distance'])
                self.taxi_states[taxi]['distance'] = distances[i]
        
            # check empty
            if taxi not in occupied_taxi and not self.taxi_states[taxi]['empty']:
                self.taxi_states[taxi]['empty'] = True
                self.taxi_states[taxi]['pickup_distance'] = None
                self.num_complete_orders += 1

        self.total_back_distances = np.zeros(len(self.background_cars))
        for i, veh in enumerate(self.background_cars):
            if background_distances[i] > 0:
                assert background_distances[i] >= self.background_states[veh]['distance'], (background_distances[i], self.background_states[veh])
                self.total_back_distances[i] = (background_distances[i] - self.background_states[veh]['distance'])
                # self.total_distance += (background_distances[i] - self.background_states[veh]['distance'])
                self.background_states[veh]['distance'] = background_distances[i]
        
        self.total_distance = sum([self.taxi_states[taxi]['distance'] for taxi in self.taxis] + [self.background_states[veh]['distance'] for veh in self.background_cars])
        # co2 penalty
        # reward -= self.total_co2.sum() * 1e-3 * self.co2_penalty
        # nonzero_distance = self.valid_distance or 0.01
        # reward -= self.total_co2.sum() * 1e-3 / nonzero_distance * self.co2_penalty

        nonzero_distance = (self.total_taxi_distances.sum() + self.total_back_distances.sum()) or 0.01
        reward -= (self.taxi_co2.sum() + self.background_co2.sum()) * 1e-3 / nonzero_distance * self.co2_penalty

        # normalizing_term = len(self.taxis) * \
            # (self.pickup_price + timestep * self.time_price + 55.55 * timestep * self.distance_price) # default maxSpeed = 55.55 m/s
        # reward -= normalizing_term * 0.5
        # reward = reward / self.env_params.horizon
        return reward

    def additional_command(self):
        """See parent class."""
        self._check_route_valid()
        # self._update_action_mask()
        self._add_request()
        # self._remove_tle_request()
        self._check_arrived()
        self._check_outside()


    def _check_route_valid(self):
        for veh_id in self.taxis:
            is_route_valid = self.k.kernel_api.vehicle.isRouteValid(veh_id)
            if not is_route_valid:
                # if the route is not valid, we need to reset the route
                self.k.kernel_api.vehicle.rerouteTraveltime(veh_id)
    
    def _check_outside(self):
        candidate_edges = self.in_edges.copy()
        for i, veh_id in enumerate(self.background_cars):
            edge = self.k.vehicle.get_edge(veh_id)
            edge_idx = self.edges.index(edge) if edge in self.edges else -1
            if edge_idx in self.out_edges:
                in_edge_idx = np.random.choice(candidate_edges)
                candidate_edges.remove(in_edge_idx)
                in_edge = self.edges[in_edge_idx]
                x, y = self.edge_position[in_edge_idx][0]
                self.k.kernel_api.vehicle.moveToXY(veh_id, in_edge, lane='0', x=x, y=y, keepRoute=0)

        for i, taxi in enumerate(self.taxis):
            edge = self.k.vehicle.get_edge(taxi)
            edge_idx = self.edges.index(edge) if edge in self.edges else -1
            if taxi in self.outside_taxis:
                in_edge_idx = np.random.choice(candidate_edges)
                in_edge = self.edges[in_edge_idx]
                # in_edge = self.k.vehicle.get_route(taxi)[0]
                # in_edge_idx = self.edges.index(in_edge)
                x, y = self.edge_position[in_edge_idx][0]

                # self.k.vehicle.remove(taxi)
                try:
                    # self.k.vehicle.add(
                    #     taxi,
                    #     'taxi',
                    #     edge=in_edge,
                    #     pos=0,
                    #     lane=0,
                    #     speed=0,
                    # )

                    # self.k.kernel_api.vehicle.setSpeed(taxi, -1)
                    self.k.kernel_api.vehicle.moveToXY(taxi, in_edge, lane='0', x=x, y=y, keepRoute=0)
                    self.outside_taxis.remove(taxi)
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

        #     elif edge_idx in self.out_edges and taxi in self.k.vehicle.get_taxi_fleet(0) and self.k.kernel_api.vehicle.isStopped(taxi):
        #         in_edge_idx = np.random.choice(candidate_edges)
        #         in_edge = self.edges[in_edge_idx]
        #         self.k.kernel_api.vehicle.resume(taxi)
        #         self.k.kernel_api.vehicle.setRoute(taxi, [edge])
        #         # self.k.kernel_api.vehicle.changeTarget(taxi, edge)
        #         # try:
        #         #     self.k.vehicle.remove(taxi)
        #         #     self.k.vehicle.add(
        #         #         taxi,
        #         #         'taxi',
        #         #         edge=in_edge,
        #         #         pos=0,
        #         #         lane=0,
        #         #         speed=0,
        #         #     )
        #         # except TraCIException:
        #         #     self.k.vehicle.remove(taxi)
        #         #     self.k.kernel_api.vehicle.remove(taxi)
        #         #     self.k.vehicle.add(
        #         #         taxi,
        #         #         'taxi',
        #         #         edge=in_edge,
        #         #         pos=0,
        #         #         lane=0,
        #         #         speed=0,
        #         #     )
        #         self.k.kernel_api.vehicle.setSpeed(taxi, 0)
        #         self.outside_taxis.append(taxi)
   
    def _check_arrived(self):
        
        def is_arrived(veh_id, edge, pos):
            cur_edge = self.k.vehicle.get_edge(veh_id)
            if cur_edge == edge:
                cur_pos = self.k.vehicle.get_position(veh_id)
                if abs(cur_pos - pos) < 10:
                    return True
            return False

        for i, taxi in enumerate(self.taxis):
            if self.k.vehicle.is_pickup(taxi):
                tgt_edge, tgt_pos = self.k.vehicle.pickup_stop[taxi]
            elif self.k.vehicle.is_occupied(taxi):
                mid_edges = self.k.vehicle.mid_edges[taxi]
                if len(mid_edges) > 0 and mid_edges[0] == self.k.vehicle.get_edge(taxi):
                    self.k.vehicle.checkpoint(taxi)
                    continue
                tgt_edge, tgt_pos = self.k.vehicle.dropoff_stop[taxi]
            elif self.k.vehicle.is_free(taxi):
                if len(self.k.kernel_api.vehicle.getStops(taxi)) == 0 and taxi not in self.outside_taxis:
                    self.k.vehicle.stop(taxi)
                continue
            else:
                continue

            if is_arrived(taxi, tgt_edge, tgt_pos):
                if self.k.vehicle.is_pickup(taxi):
                    self.k.kernel_api.vehicle.setSpeed(taxi, 0)
                    self.__need_mid_edge = taxi
                elif self.k.vehicle.is_occupied(taxi):
                    if self.stop_time[i] is None:
                        self.k.kernel_api.vehicle.setSpeed(taxi, 0)
                        self.stop_time[i] = 0
                    else:
                        self.stop_time[i] += 1

                    assert self.stop_time[i] is None or self.stop_time[i] <= 3 * self.env_params.sims_per_step
                    if self.stop_time[i] == 3 * self.env_params.sims_per_step:
                        self.stop_time[i] = None
                        edge = self.k.vehicle.get_edge(taxi)
                        edge_idx = self.edges.index(edge)
                        is_outside = edge_idx in self.out_edges
                        self.k.vehicle.dropoff(taxi, is_outside)
                        if is_outside:
                            self.outside_taxis.append(taxi)
                            # self.k.vehicle.remove(taxi)
                            # self.k.simulation.simulation_step()
                            # self.k.vehicle.sim
                            # self.k.kernel_api.vehicle.changeTarget(taxi, edge)
                            # self.k.kernel_api.vehicle.setStop(taxi, edge, self.outer_length - 0.1, 0)
                            # self.k.kernel_api.vehicle.rerouteTraveltime(taxi)
                            # in_edge_idx = np.random.choice(self.in_edges)
                            # in_edge = self.edges[in_edge_idx]
                            # x, y = self.edge_position[in_edge_idx][0]
                            # self.k.vehicle.move2xy(taxi, x, y, in_edge,)
                        res = self.k.vehicle.reservation[taxi]
                        self.k.person.remove(res.persons[0])
            

    def _update_action_mask(self):

        # def _is_blocking(stop, taxi):
        #     if self.k.vehicle.get_edge(taxi) in stop.lane:
        #         p = self.k.vehicle.get_position(taxi)
        #         p0, p1 = stop.startPos, stop.endPos
        #         if min(abs(p0 - p), abs(p1 - p)) < self.stop_distance_eps:
        #             return True
        #     return False

        self.action_mask = torch.zeros((self.num_taxi + 1, sum(self.action_space.nvec)), dtype=bool)
        # distances = [self.k.vehicle.get_distance(taxi) for taxi in self.taxis]
        # unavailable = self.k.vehicle.get_taxi_fleet(1) + self.k.vehicle.get_taxi_fleet(2)

        n_edge = len(self.edges)
        n_taxi = self.num_taxi

        # mid point mask
        if self.__need_mid_edge is not None:
            res = self.k.vehicle.reservation[self.__need_mid_edge]
            from_id, to_id = self.edges.index(res.fromEdge), self.edges.index(res.toEdge)
            taxi_id =  self.taxis.index(self.__need_mid_edge)
            
            # mask from edge, to edge
            for i in range(self.n_mid_edge):
                self.action_mask[taxi_id][n_edge + n_taxi + 1 + i * n_edge + from_id] = True
                self.action_mask[taxi_id][n_edge + n_taxi + 1 + i * n_edge + to_id] = True
                # mask flow edges, in edges and out edges
                for j in self.flow_edges + self.in_edges + self.out_edges:
                    self.action_mask[taxi_id][n_edge + n_taxi + 1 + i * n_edge + j] = True

            if self.n_mid_edge == 0:
                pass
            elif self.n_mid_edge == 1:
                # for i, edge in enumerate(self.edges):
                #     if edge != res.fromEdge and edge != res.toEdge:
                #         r1 = set(self.k.kernel_api.simulation.findRoute(res.fromEdge, edge).edges[1:-1])
                #         r2 = set(self.k.kernel_api.simulation.findRoute(edge, res.toEdge).edges[1:-1])
                #         if len(r1 & r2) > 0:
                #             self.action_mask[taxi_id][n_edge + n_taxi + 1 + i] = True
                self.action_mask[taxi_id][n_edge + n_taxi + 1:] |= self.banned_mid_edges[from_id, to_id]
            else:
                raise NotImplementedError

        for i, taxi in enumerate(self.taxis):
            # q = self.hist_dist[i]
            # if q.qsize() == q.maxsize:
            #     d0 = q.get()
            # else:
            #     d0 = None
            # d1 = distances[i]
            # q.put(d1)
            
            # if d0 is not None and d1 - d0 < self.stop_distance_eps: 
            #     # check whether it stops for a long time intentionally
            #     # print('stopping', taxi)
            #     stops = self.k.kernel_api.vehicle.getStops(taxi, limit=1)
            #     if len(stops) > 0:
            #         next_stop = stops[0]
            #         if _is_blocking(next_stop, taxi):
            #             print(taxi, 'is blocking')
            #             edge_id = self.edges.index(self.k.vehicle.get_edge(taxi))
            #             self.action_mask[i][self.num_taxi + 1 + edge_id] = True

            cur_edge = self.k.vehicle.get_edge(taxi)
            assert cur_edge != ""

            # reposition mask, mask current edge
            if cur_edge in self.edges:
                edge_id = self.edges.index(cur_edge)
                self.action_mask[i][edge_id] = True

            # reposition mask, mask flow edges, in edges and out edges
            for j in self.flow_edges + self.in_edges + self.out_edges:
                self.action_mask[i][j] = True

            # taxi mask
            if len(self.__reservations) > 0:
                res = self.__reservations[0]
                edge = self.k.vehicle.get_edge(taxi)
                pos = self.k.vehicle.get_position(taxi)
                # Do not dispath the order to the taxi on out edges
                edge_idx = self.edges.index(edge) if edge in self.edges else -1
                if edge == res.fromEdge and pos > res.departPos:
                    self.action_mask[self.num_taxi][n_edge + i] = True
                if edge_idx in self.out_edges:
                    self.action_mask[self.num_taxi][n_edge + i] = True
                for edge in self.k.vehicle.get_route(taxi):
                    if self.edges.index(edge) in self.out_edges:
                        self.action_mask[self.num_taxi][n_edge + i] = True
                        break
                
                # from_id = self.edges.index(res.fromEdge)    
                # to_id = self.edges.index(res.toEdge)
                # for i in range(self.n_mid_edge):
                #     self.action_mask[self.num_taxi][n_edge + n_taxi + 1 + i * n_edge + from_id] = True
                #     self.action_mask[self.num_taxi][n_edge + n_taxi + 1 + i * n_edge + to_id] = True

            # if taxi in unavailable:
            #     self.action_mask[self.num_taxi][i] = True

    def _add_request(self):
        if np.random.rand() > self.person_prob * self.sim_params.sim_step and self.distribution != "mode-4" and self.distribution != "mode-5":
            return 
        per_id, edge_id1, edge_id2, pos, tp = gen_request(self)
        self.k.person.add_request(per_id, edge_id1, edge_id2, pos, tp=tp)
        if self.verbose:
            print('add request from', edge_id1, 'to', edge_id2, 'total', self.k.person.total)


    def _remove_tle_request(self):
        idlist = self.k.kernel_api.person.getIDList()
        for person in idlist:
            if self.k.kernel_api.person.getWaitingTime(person) > self.max_waiting_time:
                if not self.k.person.is_matched(person) and not self.k.person.is_removed(person):
                    self.k.person.remove(person)
                    self.k.person.set_color(person, (0, 255, 255)) # Cyan
                    if self.verbose:
                        print('tle request', person)

    
    def _dispatch_taxi(self):
        remain_pending_orders = []

        if 'pickup' not in self.statistics['location']:
            self.statistics['location']['pickup'] = [[], [], []] if 'mode-X' in self.distribution else [[]]
        pickup_stat = self.statistics['location']['pickup']
        for res, veh_id in self.__pending_orders:
            # there would be some problems if the a taxi is on the road started with ":"
            # if the taxi is occupied now, we should dispatch this order later
            if self.k.kernel_api.vehicle.getRoadID(veh_id).startswith(':') or veh_id not in self.k.vehicle.get_taxi_fleet(0):
                remain_pending_orders.append([res, veh_id])
            elif self.k.kernel_api.person.getWaitingTime(res.persons[0]) <= self.max_waiting_time:
                self.__dispatched_orders.append((res, veh_id))
                self.k.vehicle.dispatch_taxi(veh_id, res, tp=self.k.person.get_type(res.persons[0]))
                self.k.person.match(res.persons[0], veh_id)
                pickup_stat[self.k.vehicle.get_res_type(veh_id)].append(self.k.vehicle.get_2d_position(veh_id))
                if self.verbose:
                    print('dispatch {} to {}, remaining {} available taxis, cur_edge {}, cur_route {}'.format(\
                        res, veh_id, len(self.k.vehicle.get_taxi_fleet(0)), self.k.vehicle.get_edge(veh_id), self.k.vehicle.get_route(veh_id)))
            else:
                if self.verbose:
                    print('order {} dispatch tle'.format(res))
        self.__pending_orders = remain_pending_orders
