"""Environments for networks with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m traffic light grid.
"""

import numpy as np
import re

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
    "time_price": 0.01,
    "distance_price": 0.03,
    "person_prob": 0.003
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
        self.starting_distance = env_params.additional_params['starting_distance']

        self.person_prob = env_params.additional_params['person_prob']
        
        self.max_num_order = env_params.additional_params['max_num_order']
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
        self.__reseravtions = []
        self.__need_reposition = None

        self.taxi_states = dict([[taxi, {"empty": True, "distance": 0, "pickup_distance": None}] for taxi in self.taxis])

    @property
    def action_space(self):
        """See class definition."""
        return MultiDiscrete([self.num_taxi + 1, len(self.edges)])

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
            low=-1,
            high=100,
            shape=( len(self.edges) + self.num_taxi * 3 + self.max_num_order * 3 + 1, )
        )
        return state_box

    def get_state(self):
        """See class definition."""

        edges_feature = [
            self.k.kernel_api.edge.getLastStepVehicleNumber(edge) for edge in self.edges
        ]
        taxi_feature = []
        empty_taxi = self.k.vehicle.get_taxi_fleet(0)
        occupied_taxi = self.k.vehicle.get_taxi_fleet(1) + self.k.vehicle.get_taxi_fleet(2)
        
        for taxi in self.taxis:
            while taxi not in self.k.vehicle.get_rl_ids():
                try:
                    self.k.vehicle.remove(taxi)
                    self.k.vehicle.add(
                        veh_id=taxi,
                        edge=np.random.choice(self.edges),
                        type_id='taxi',
                        lane=str(0),
                        pos=str(0),
                        speed=0)
                    
                except TraCIException as e:
                    print(e)
                    break
            cur_taxi_feature = [0, self.edges.index(self.k.kernel_api.vehicle.getRoute(taxi)[0]), self.edges.index(self.k.kernel_api.vehicle.getRoute(taxi)[-1])]
            cur_taxi_feature[0] = 0 if taxi in empty_taxi else 1 if taxi in occupied_taxi else 2
            taxi_feature += cur_taxi_feature
        
        order_feature = self._get_order_state.tolist()
    
        if self.__reseravtions:
            need_reposition_taxi_feature = [-1]
        else:
            empty_taxi_fleet = self.k.vehicle.get_taxi_fleet(0)
            self.__need_reposition = None
            for taxi in empty_taxi_fleet:
                if self.k.kernel_api.vehicle.isStopped(taxi):
                    self.__need_reposition = taxi
                    break
            if self.__need_reposition:
                need_reposition_taxi_feature = [self.edges.index(self.k.kernel_api.vehicle.getRoadID(self.__need_reposition))]
            else:
                need_reposition_taxi_feature = [-1]

        state = edges_feature + taxi_feature + order_feature + need_reposition_taxi_feature
        return np.array(state)
    
    def _get_info(self):
        return {}

    def reset(self):
        observation = super().reset()
        self.__dispatched_orders = []
        self.__reseravtions = []
        self.__need_reposition = None
        self.taxi_states = dict([[taxi, {"empty": True, "distance": 0, "pickup_distance": None}] for taxi in self.taxis])
        return observation

    @property
    def _get_order_state(self):
        if np.random.rand() < self.person_prob:
            person_ids = [int(per_id[4:]) for per_id in self.k.person.get_ids()]
            idx = max(person_ids) + 1 if len(person_ids) > 0 else 1
            edge_list = self.edges.copy()
            edge_id1 = np.random.choice(edge_list)
            edge_list.remove(edge_id1)
            edge_id2 = np.random.choice(edge_list)

            per_id = 'per_' + str(idx)
            pos = np.random.uniform(self.inner_length)
            self.k.kernel_api.person.add(per_id, edge_id1, pos)
            self.k.kernel_api.person.appendDrivingStage(per_id, edge_id2, 'taxi')
            self.k.kernel_api.person.setColor(per_id, (255, 0, 0))
        # print('add_request', per_id, 'from', str(edge_id1), 'to', str(edge_id2))
        orders = [[0, 0, 0]] * self.max_num_order
        reservations = self.k.person.get_reservations()
        self.__reseravtions = [res for res in reservations if res.id not in map(lambda x: x[0].id, self.__dispatched_orders)]
        for i, res in enumerate(self.__reseravtions):
            if i >= self.max_num_order:
                break
            
            waiting_time = self.k.kernel_api.person.getWaitingTime(res.persons[0])
            form_edge = res.fromEdge
            to_edge = res.toEdge
            orders[i] = [ waiting_time, self.edges.index(form_edge), self.edges.index(to_edge) ]
        
        return np.array(orders).reshape(-1)

    # def _apply_rl_actions(self, rl_actions):
        # pass

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        if self.__need_reposition:
            self.k.vehicle.reposition_taxi_by_road(self.__need_reposition, self.edges[rl_actions[1]])
            # print('reposition {} to {}'.format(self.__need_reposition, self.edges[rl_actions[1]]))
            self.__need_reposition = None
        elif self.__reseravtions:
            if rl_actions[0] == self.num_taxi: # do not dispatch when the specail action is selected
                pass
            else:
                self.__dispatched_orders.append([self.__reseravtions[0], self.taxis[rl_actions[0]]]) # notice that we may dispach a order to a occupied_taxi
        else:
            pass    # nothing to do 
        self._dispacth_taxi()

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        reward = 0
        occupied_taxi = self.k.vehicle.get_taxi_fleet(2)
        distances = [self.k.vehicle.get_distance(taxi) for taxi in self.taxis]

        # pickup price 
        for i, taxi in enumerate(self.taxis):
            if self.taxi_states[taxi]['empty'] and taxi in occupied_taxi and distances[i] > 0:
                assert self.taxi_states[taxi]['pickup_distance'] is None
                self.taxi_states[taxi]['pickup_distance'] = distances[i]
                self.taxi_states[taxi]['empty'] = False
                reward += self.pickup_price

        # price about time 
        reward += len(occupied_taxi) * self.time_price
        
        for i, taxi in enumerate(self.taxis):
            # price about distance
            if taxi in occupied_taxi and self.taxi_states[taxi]['pickup_distance'] and distances[i] - self.taxi_states[taxi]['pickup_distance'] > self.starting_distance:
                reward += (distances[i] - self.taxi_states[taxi]['distance']) * self.distance_price

            # update distance
            self.taxi_states[taxi]['distance'] = distances[i]
            
            # check empty
            if taxi not in occupied_taxi and not self.taxi_states[taxi]['empty']:
                self.taxi_states[taxi]['empty'] = True
                self.taxi_states[taxi]['pickup_distance'] = None
        
        return reward

    def additional_command(self):
        """See parent class."""
        self._check_route_valid()
        #TODO: remove TLE order

    def _check_route_valid(self):
        for veh_id in self.taxis:
            is_route_valid = self.k.kernel_api.vehicle.isRouteValid(veh_id)
            if not is_route_valid:
                # if the route is not valid, we need to reset the route
                self.k.kernel_api.vehicle.rerouteTraveltime(veh_id)
    
    def _dispacth_taxi(self):
        remain_dispatched_orders = []

        for res, veh_id in self.__dispatched_orders:
            # there would be some problems if the a taxi is on the road started with ":"
            # if the taxi is occupied now, we should dispatch this order later
            if self.k.kernel_api.vehicle.getRoadID(veh_id).startswith(':') or veh_id not in self.k.vehicle.get_taxi_fleet(0):
                remain_dispatched_orders.append([res, veh_id])
            else:
                print('dispatch {} to {}'.format(res, veh_id))
                self.k.vehicle.dispatch_taxi(veh_id, res)
        self.__dispatched_orders = remain_dispatched_orders
