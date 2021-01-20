"""Environments for networks with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m traffic light grid.
"""

import numpy as np
import re
import random
import torch
from queue import Queue

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces import Tuple

from flow.core import rewards
from flow.envs.base import Env
from traci.exceptions import TraCIException

ADDITIONAL_ENV_PARAMS = {
    "max_num_order": 10,
    "pickup_price": 6,
    "starting_distance": 100,
    "time_price": 0.01, # in second
    "distance_price": 0.03, # in meter
    "wait_penalty": 0.0, # in second
    "tle_penalty": 0.005, # in second
    "person_prob": 0.03,
    "max_waiting_time": 10, # in second
    "max_pickup_time": 20, # in second
    "max_stop_time": 1, # in second, intentionally waiting time
    "stop_distance_eps": 1, # in meter, a threshold to determine whether the car is stopping
    "distribution": 'random', # random, mode-1, mode-2, mode-3
    "reservation_order": 'random', # random or fifo 
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

        self.pickup_price = env_params.additional_params['pickup_price']
        self.time_price = env_params.additional_params['time_price']
        self.distance_price = env_params.additional_params['distance_price']
        self.wait_penalty = env_params.additional_params['wait_penalty']
        self.tle_penalty = env_params.additional_params['tle_penalty']
        self.starting_distance = env_params.additional_params['starting_distance']

        self.person_prob = env_params.additional_params['person_prob']
        
        self.max_num_order = env_params.additional_params['max_num_order']
        self.max_waiting_time = env_params.additional_params['max_waiting_time']
        self.max_pickup_time = env_params.additional_params['max_pickup_time']

        self.reservation_order = env_params.additional_params['reservation_order']

        self.num_complete_orders = 0
        self.total_valid_distance = 0
        self.total_valid_time = 0
        self.total_pickup_distance = 0
        self.total_pickup_time = 0
        self.total_wait_time = 0
        self.total_congestion_rate = 0

        self.stop_distance_eps = env_params.additional_params['stop_distance_eps']
        
        self.distribution = env_params.additional_params['distribution']
        print(self.distribution)
        super().__init__(env_params, sim_params, network, simulator)

        # Saving env variables for plotting
        self.steps = env_params.horizon
        self.obs_var_labels = {
            'edges': np.zeros((self.steps, self.k.vehicle.num_vehicles)),
            'velocities': np.zeros((self.steps, self.k.vehicle.num_vehicles)),
            'positions': np.zeros((self.steps, self.k.vehicle.num_vehicles))
        }

        self.edges = self.k.network.get_edge_list()
        self.num_taxi = network.vehicles.num_rl_vehicles
        self.taxis = [taxi for taxi in network.vehicles.ids if network.vehicles.get_type(taxi) == 'taxi']
        assert self.num_taxi == len(self.taxis)
        self.num_vehicles = network.vehicles.num_vehicles

        self.__dispatched_orders = []
        self.__pending_orders = []
        self.__reservations = []
        self.__need_reposition = None

        self.taxi_states = dict([[taxi, {"empty": True, "distance": 0, "pickup_distance": None}] for taxi in self.taxis])
        self.stop_time = [None] * len(self.taxis)

        self.hist_dist = [Queue(maxsize=int(self.env_params.additional_params['max_stop_time'] / self.sim_params.sim_step) + 1) for i in range(self.num_taxi)]
        self.action_mask = torch.zeros((self.num_taxi + 1, sum(self.action_space.nvec)), dtype=bool)

    def get_action_mask(self):
        if self.__need_reposition:
            taxi_id = self.taxis.index(self.__need_reposition)
            # print(self.action_mask[taxi_id].unsqueeze(0))
            return self.action_mask[taxi_id].unsqueeze(0)
        else:
            return self.action_mask[self.num_taxi].unsqueeze(0)

    @property
    def action_space(self):
        """See class definition."""
        return MultiDiscrete([self.num_taxi + 1, len(self.edges)]) # TODO: add routing

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
            low=-500,
            high=500,
            shape=( 1 + len(self.edges) + self.num_taxi * 9 + self.max_num_order * 5 + self.num_taxi + 2, )
        )
        return state_box

    def get_state(self):
        """See class definition."""

        time_feature = [self.time_counter / self.env_params.horizon]

        edges_feature = [
            self.k.kernel_api.edge.getLastStepVehicleNumber(edge) for edge in self.edges
        ]
        taxi_feature = []
        empty_taxi = self.k.vehicle.get_taxi_fleet(0)
        occupied_taxi = self.k.vehicle.get_taxi_fleet(1) + self.k.vehicle.get_taxi_fleet(2)
        
        for taxi in self.taxis:
            while taxi not in self.k.vehicle.get_rl_ids():
                raise KeyError
            x, y = self.k.vehicle.get_2d_position(taxi, error=(-1, -1))
            from_x, from_y = self.k.kernel_api.simulation.convert2D(self.k.kernel_api.vehicle.getRoute(taxi)[0], 0)
            to_x, to_y = self.k.kernel_api.simulation.convert2D(self.k.kernel_api.vehicle.getRoute(taxi)[-1], self.inner_length - 2)
            # cur_taxi_feature = [0, x, y, self.edges.index(self.k.kernel_api.vehicle.getRoute(taxi)[0]), self.edges.index(self.k.kernel_api.vehicle.getRoute(taxi)[-1])]
            cur_taxi_feature = [0, 0, 0, x, y, from_x, from_y, to_x, to_y]
            cur_taxi_feature[0 if taxi in empty_taxi else 1 if taxi in occupied_taxi else 2] = 1
            taxi_feature += cur_taxi_feature
        
        order_feature = self._get_order_state.tolist()
    
        index = [0] * len(self.taxis)
        if self.__reservations:
            need_reposition_taxi_feature = index + [-1, -1]
        else:
            empty_taxi_fleet = self.k.vehicle.get_taxi_fleet(0)
            self.__need_reposition = None
            for taxi in empty_taxi_fleet:
                if self.k.kernel_api.vehicle.isStopped(taxi):
                    self.__need_reposition = taxi
                    break
            
            if self.__need_reposition:
                # need_reposition_taxi_feature = [self.edges.index(self.k.kernel_api.vehicle.getRoadID(self.__need_reposition)), self.k.vehicle.get_position(self.__need_reposition)]
                x, y = self.k.vehicle.get_2d_position(self.__need_reposition, error=(-1, -1))
                index[self.taxis.index(self.__need_reposition)] = 1
                need_reposition_taxi_feature = index + [x, y]
            else:
                need_reposition_taxi_feature = index + [-1, -1]

        state = time_feature + edges_feature + taxi_feature + order_feature + need_reposition_taxi_feature
        return np.array(state)
    
    def _get_info(self):
        return {}

    def reset(self):
        observation = super().reset()
        self.__dispatched_orders = []
        self.__pending_orders = []
        self.__reservations = []
        self.__need_reposition = None
        self.taxi_states = dict([[taxi, {"empty": True, "distance": 0, "pickup_distance": None}] for taxi in self.taxis])
        self.hist_dist = [Queue(maxsize=int(self.env_params.additional_params['max_stop_time'] / self.sim_params.sim_step) + 1) for i in range(self.num_taxi)]
        self.action_mask = torch.zeros((self.num_taxi + 1, sum(self.action_space.nvec)), dtype=bool)
        self.num_complete_orders = 0
        self.total_valid_distance = 0
        self.total_valid_time = 0
        self.total_pickup_distance = 0
        self.total_pickup_time = 0
        self.total_wait_time = 0
        self.total_congestion_rate = 0
        self.stop_time = [None] * len(self.taxis)
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
        if self.__need_reposition:
            # taxi = self.__need_reposition
            # stop = self.k.kernel_api.vehicle.getStops(taxi, limit=1)[0]
            # print(self.k.vehicle.get_edge(taxi), stop.lane, self.k.vehicle.get_position(taxi), stop.startPos, stop.endPos)
            self.k.vehicle.reposition_taxi_by_road(self.__need_reposition, self.edges[rl_actions[1]])
            print('reposition {} to {}, cur_edge {}'.format(self.__need_reposition, self.edges[rl_actions[1]], self.k.vehicle.get_edge(self.__need_reposition)))
            self.__need_reposition = None
        elif self.__reservations:
            assert self.action_mask[self.num_taxi][rl_actions[0]] == False
            if rl_actions[0] == self.num_taxi: # do not dispatch when the special action is selected
                pass
            else:
                # check if the dispatch is valid
                cur_taxi = self.taxis[rl_actions[0]]
                cur_edge = self.k.vehicle.get_edge(cur_taxi)
                cur_pos = self.k.vehicle.get_position(cur_taxi)
                cur_res = self.__reservations[0]
                if not (cur_edge == cur_res.fromEdge and cur_pos > cur_res.departPos):
                    self.__pending_orders.append([self.__reservations[0], self.taxis[rl_actions[0]]]) # notice that we may dispach a order to a occupied_taxi
        else:
            pass    # nothing to do 
        self._dispatch_taxi()

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        reward = 0
        pickup_taxi = self.k.vehicle.get_taxi_fleet(1)
        occupied_taxi = self.k.vehicle.get_taxi_fleet(2)
        distances = [self.k.vehicle.get_distance(taxi) for taxi in self.taxis]
        cur_time = self.time_counter * self.sim_params.sim_step

        num_congestion = 0.0
        for edge in self.edges:
            if self.k.kernel_api.edge.getLastStepVehicleNumber(edge) > 0:
                if self.k.kernel_api.edge.getLastStepMeanSpeed(edge) < 3.0: # a threshold for congestion
                    num_congestion += 1
        self.total_congestion_rate += num_congestion / len(self.edges)

        for person in self.k.person.get_ids():
            if self.k.person.is_matched(person) or self.k.person.is_removed(person):
                continue
            reward -= self.wait_penalty * self.sim_params.sim_step
            self.total_wait_time += self.sim_params.sim_step

        # pickup price  
        for i, taxi in enumerate(self.taxis):
            if self.taxi_states[taxi]['empty'] and taxi in occupied_taxi and distances[i] > 0:
                assert self.taxi_states[taxi]['pickup_distance'] is None
                self.taxi_states[taxi]['pickup_distance'] = distances[i]
                self.taxi_states[taxi]['empty'] = False
                reward += self.pickup_price
                print('taxi {} pickup successfully'.format(taxi))

        # tle price
        for taxi in pickup_taxi:
            if cur_time - self.k.vehicle.reservation[taxi].reservationTime > self.max_pickup_time:
                reward -= self.tle_penalty * self.sim_params.sim_step

        # price about time 
        reward += len(occupied_taxi) * self.time_price * self.sim_params.sim_step
        
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
                    self.total_valid_time += self.sim_params.sim_step
                elif taxi in pickup_taxi:
                    self.total_pickup_distance += (distances[i] - self.taxi_states[taxi]['distance'])
                    self.total_pickup_time += self.sim_params.sim_step
                self.taxi_states[taxi]['distance'] = distances[i]

            # check empty
            if taxi not in occupied_taxi and not self.taxi_states[taxi]['empty']:
                self.taxi_states[taxi]['empty'] = True
                self.taxi_states[taxi]['pickup_distance'] = None
                self.num_complete_orders += 1
        normalizing_term = len(self.taxis) * \
            (self.pickup_price + self.sim_params.sim_step * self.time_price + 55.55 * self.sim_params.sim_step * self.distance_price) # default maxSpeed = 55.55 m/s
        # reward -= normalizing_term * 0.5
        # reward = reward / self.env_params.horizon
        return reward

    def additional_command(self):
        """See parent class."""
        self._check_route_valid()
        self._update_action_mask()
        self._add_request()
        self._remove_tle_request()
        self._check_arrived()

    def _check_route_valid(self):
        for veh_id in self.taxis:
            is_route_valid = self.k.kernel_api.vehicle.isRouteValid(veh_id)
            if not is_route_valid:
                # if the route is not valid, we need to reset the route
                self.k.kernel_api.vehicle.rerouteTraveltime(veh_id)
    
    def _check_arrived(self):
        
        def is_arrived(veh_id, edge, pos):
            cur_edge = self.k.vehicle.get_edge(veh_id)
            if cur_edge == edge:
                cur_pos = self.k.vehicle.get_position(veh_id)
                if abs(cur_pos - pos) < 10:
                    return True
            return False

        for i,  taxi in enumerate(self.taxis):
            if self.k.vehicle.is_pickup(taxi):
                tgt_edge, tgt_pos = self.k.vehicle.pickup_stop[taxi]
            elif self.k.vehicle.is_occupied(taxi):
                tgt_edge, tgt_pos = self.k.vehicle.dropoff_stop[taxi]
            elif self.k.vehicle.is_free(taxi):
                if len(self.k.kernel_api.vehicle.getStops(taxi)) == 0:
                    self.k.vehicle.stop(taxi)
                continue
            else:
                continue

            if is_arrived(taxi, tgt_edge, tgt_pos):
                if self.stop_time[i] is None:
                    self.k.kernel_api.vehicle.setSpeed(taxi, 0)
                    self.stop_time[i] = 0
                else:
                    self.stop_time[i] += 1
            
            assert self.stop_time[i] is None or self.stop_time[i] <= 3
            if self.stop_time[i] == 3:
                self.stop_time[i] = None
                if self.k.vehicle.is_pickup(taxi):
                    self.k.vehicle.pickup(taxi)
                    res = self.k.vehicle.reservation[taxi]
                    self.k.person.set_color(res.persons[0], (255, 255, 255)) # White
                elif self.k.vehicle.is_occupied(taxi):
                    self.k.vehicle.dropoff(taxi)
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
            if cur_edge in self.edges:
                edge_id = self.edges.index(cur_edge)
                self.action_mask[i][self.num_taxi + 1 + edge_id] = True

            # if taxi in unavailable:
            #     self.action_mask[self.num_taxi][i] = True

    def _add_request(self):
        if np.random.rand() > self.person_prob * self.sim_params.sim_step and self.distribution != "mode-4" and self.distribution != "mode-5":
            return 
        if self.distribution == 'random':
            idx = self.k.person.total
            edge_list = self.edges.copy()
            edge_id1 = np.random.choice(edge_list)
            edge_list.remove(edge_id1)
            edge_id2 = np.random.choice(edge_list)

            per_id = 'per_' + str(idx)
            pos = np.random.uniform(20, self.inner_length - 20)
            self.k.person.add_request(per_id, edge_id1, edge_id2, pos)
        elif self.distribution == 'mode-1': 
            # the request only appears at one edge
            idx = self.k.person.total
            edge_list = self.edges.copy()
            edge_id1 = 'bot3_1_0'
            edge_list.remove(edge_id1)
            edge_id2 = 'top2_2_0'

            per_id = 'per_' + str(idx)
            pos = np.random.uniform(20, self.inner_length - 20)
            self.k.person.add_request(per_id, edge_id1, edge_id2, pos)
        elif self.distribution == 'mode-2':
            # the request only appears at two different edges
            idx = self.k.person.total
            edge_list = self.edges.copy()
            rn =  np.random.rand()
            edge_id1 = 'bot3_1_0' if rn < 0.5 else 'top0_3_0'
            edge_list.remove(edge_id1)
            edge_id2 = 'top2_2_0' if rn < 0.5 else 'bot1_2_0'

            per_id = 'per_' + str(idx)
            pos = np.random.uniform(20, self.inner_length - 20)
            self.k.person.add_request(per_id, edge_id1, edge_id2, pos)
        elif self.distribution == 'mode-3':
            # the request appears at one edge before half of the time
            # the request appears at another edge after half of the time
            idx = self.k.person.total
            edge_list = self.edges.copy()
            time_ratio = self.time_counter / self.env_params.horizon
            edge_id1 = 'bot3_1_0' if time_ratio < 0.5 else 'bot0_3_0'
            edge_list.remove(edge_id1)
            edge_id2 = 'top2_1_0' if time_ratio < 0.5 else 'top1_3_0'

            per_id = 'per_' + str(idx)
            pos = np.random.uniform(self.inner_length)
            self.k.person.add_request(per_id, edge_id1, edge_id2, pos)
        elif self.distribution == 'mode-4':
            if self.time_counter % 20 != 1:
                return
            # the request appears at one edge before half of the time
            # the request appears at another edge after half of the time
            idx = self.k.person.total
            edge_list = self.edges.copy()
            time_ratio = self.time_counter / self.env_params.horizon
            edge_id1 = 'bot3_1_0' if time_ratio < 0.5 else 'bot0_3_0'
            edge_list.remove(edge_id1)
            edge_id2 = 'top2_1_0' if time_ratio < 0.5 else 'top1_3_0'

            per_id = 'per_' + str(idx)
            # pos = np.random.uniform(self.inner_length)
            pos = self.time_counter % self.grid_array['inner_length']
            self.k.person.add_request(per_id, edge_id1, edge_id2, pos)
        elif self.distribution == 'mode-5':
            if self.time_counter % 5 != 1:
                return
            # the request appears at one edge before half of the time
            # the request appears at another edge after half of the time
            idx = self.k.person.total
            edge_list = self.edges.copy()
            time_ratio = self.time_counter / self.env_params.horizon
            edge_id1 = 'bot3_1_0' if time_ratio < 0.5 else 'bot0_3_0'
            edge_list.remove(edge_id1)
            edge_id2 = 'top2_1_0' if time_ratio < 0.5 else 'top1_3_0'

            per_id = 'per_' + str(idx)
            # pos = np.random.uniform(self.inner_length)
            pos = self.time_counter % self.grid_array['inner_length']
            self.k.person.add_request(per_id, edge_id1, edge_id2, pos)
        else:
            raise NotImplementedError
        print('add request from', edge_id1, 'to', edge_id2, 'total', self.k.person.total)


    def _remove_tle_request(self):
        idlist = self.k.kernel_api.person.getIDList()
        for person in idlist:
            if self.k.kernel_api.person.getWaitingTime(person) > self.max_waiting_time:
                if not self.k.person.is_matched(person) and not self.k.person.is_removed(person):
                    self.k.person.remove(person)
                    print('tle request', person)

    
    def _dispatch_taxi(self):
        remain_pending_orders = []

        for res, veh_id in self.__pending_orders:
            # there would be some problems if the a taxi is on the road started with ":"
            # if the taxi is occupied now, we should dispatch this order later
            if self.k.kernel_api.vehicle.getRoadID(veh_id).startswith(':') or veh_id not in self.k.vehicle.get_taxi_fleet(0):
                remain_pending_orders.append([res, veh_id])
            elif self.k.kernel_api.person.getWaitingTime(res.persons[0]) <= self.max_waiting_time:
                self.__dispatched_orders.append((res, veh_id))
                self.k.vehicle.dispatch_taxi(veh_id, res)
                self.k.person.match(res.persons[0], veh_id)
                # print('dispatch {} to {}, remaining {} available taxis, cur_edge {}, cur_route {}'.format(res, veh_id, \
                #    len(self.k.vehicle.get_taxi_fleet(0)), self.k.vehicle.get_edge(veh_id), self.k.vehicle.get_route(veh_id)))
            else:
                print('order {} dispatch tle'.format(res))
        self.__pending_orders = remain_pending_orders
