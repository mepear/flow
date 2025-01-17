"""Contains a list of custom routing controllers."""
import random
import numpy as np

from flow.controllers.base_routing_controller import BaseRouter


class ContinuousRouter(BaseRouter):
    """A router used to continuously re-route of the vehicle in a closed ring.

    This class is useful if vehicles are expected to continuously follow the
    same route, and repeat said route once it reaches its end.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class.

        Adopt one of the current edge's routes if about to leave the network.
        """
        edge = env.k.vehicle.get_edge(self.veh_id)
        current_route = env.k.vehicle.get_route(self.veh_id)

        if len(current_route) == 0:
            # this occurs to inflowing vehicles, whose information is not added
            # to the subscriptions in the first step that they departed
            return None
        elif edge == current_route[-1]:
            # choose one of the available routes based on the fraction of times
            # the given route can be chosen
            num_routes = len(env.available_routes[edge])
            frac = [val[1] for val in env.available_routes[edge]]
            route_id = np.random.choice(
                [i for i in range(num_routes)], size=1, p=frac)[0]

            # pass the chosen route
            return env.available_routes[edge][route_id][0]
        else:
            return None


class MinicityRouter(BaseRouter):
    """A router used to continuously re-route vehicles in minicity network.

    This class allows the vehicle to pick a random route at junctions.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        vehicles = env.k.vehicle
        veh_id = self.veh_id
        veh_edge = vehicles.get_edge(veh_id)
        veh_route = vehicles.get_route(veh_id)
        veh_next_edge = env.k.network.next_edge(veh_edge,
                                                vehicles.get_lane(veh_id))
        not_an_edge = ":"
        no_next = 0

        if len(veh_next_edge) == no_next:
            next_route = None
        elif veh_route[-1] == veh_edge and veh_next_edge != []:
            random_route = random.randint(0, len(veh_next_edge) - 1)
            try:
                while veh_next_edge[0][0][0] == not_an_edge:
                    veh_next_edge = env.k.network.next_edge(
                        veh_next_edge[random_route][0],
                        veh_next_edge[random_route][1])
            except:
                print('veh_id', veh_id, 'veh_edge', veh_edge, 'veh_route', veh_route, 'veh_next_edge', veh_next_edge)
                raise KeyError
            next_route = [veh_edge, veh_next_edge[0][0]]
        else:
            next_route = None

        if veh_edge in ['e_37', 'e_51']:
            next_route = [veh_edge, 'e_29_u', 'e_21']

        return next_route

class OuterCycleRouter(BaseRouter):
    """A router used to continuously re-route vehicles in outer cycle.

    Vehicles are spreaded evenly in each lane of outer cycle. 

    Usage
    -----
    See base class for usage example.
    """

    CYCLES = [ \
        ['right3_0_0', 'bot3_1_0', 'bot3_2_0', 'bot3_3_0', 'left3_3_0', 'left2_3_0', \
            'left1_3_0', 'top0_3_0', 'top0_2_0', 'top0_1_0', 'right1_0_0', 'right2_0_0'], \
        ['left2_0_0', 'left1_0_0', 'bot0_1_0', 'bot0_2_0', 'bot0_3_0', 'right1_3_0', \
            'right2_3_0', 'right3_3_0', 'top3_3_0', 'top3_2_0', 'top3_1_0', 'left3_0_0']
        ]

    def _get_closest_edge(self, cur_edge, edges, env):
        min_dist = int(1e9)
        closest_edge = None
        for edge in edges:
            route = env.k.kernel_api.simulation.findRoute(cur_edge, edge).edges
            if len(route) < min_dist:
                min_dist = len(route)
                closest_edge = edge
        assert closest_edge is not None
        assert closest_edge in edges
        return closest_edge


    def choose_route(self, env):
        """See parent class."""
        veh_id = self.veh_id
        num_id = int(veh_id[4:])
        veh_edge = env.k.vehicle.get_edge(veh_id)
        veh_route = env.k.vehicle.get_route(veh_id)
        
        cycle = OuterCycleRouter.CYCLES[num_id & 1]
        if veh_edge in cycle:
            idx = cycle.index(veh_edge)
            next_edge = cycle[(idx + 1) % len(cycle)]
            next_route = [veh_edge, next_edge]
        elif veh_route[-1] in cycle:
            next_route = None
        else:
            closest_edge = self._get_closest_edge(veh_edge, cycle, env)
            closest_route = env.k.kernel_api.simulation.findRoute(veh_edge, closest_edge).edges
            next_route = list(closest_route)

        # print(veh_id, veh_edge, veh_route, next_route)

        return next_route


class InflowRouter(BaseRouter):
    """A router used to route vehicles from inflow edge to outflow edge.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        assert 'inflow' in self.router_params
        direction = self.router_params['inflow']
        if direction == 'top_left':
            src, dest = 'inflow_top_left', 'outflow_bot_right'
            src_id, dest_id = env.edges.index(src), env.edges.index(dest)
        

        vehicles = env.k.vehicle
        veh_id = self.veh_id
        veh_edge = vehicles.get_edge(veh_id)
        if veh_edge[0] == ':':
            return None
        veh_edge_id = env.edges.index(veh_edge)
        veh_route = vehicles.get_route(veh_id)
        veh_next_edge = env.k.network.next_edge(veh_edge, vehicles.get_lane(veh_id))

        cur_route_len = len(env.paired_routes[veh_edge_id][dest_id].edges)

        next_route = None
        if veh_route[-1] == veh_edge and veh_next_edge != []:
            feasible_next_edge = []
            if veh_next_edge[0][0][0] == ':':
                next_edge = []
                for edge in veh_next_edge:
                    next_edge.append(env.k.network.next_edge(edge[0], edge[1])[0])
                veh_next_edge = next_edge
            for edge, _ in veh_next_edge:
                next_route_len = len(env.paired_routes[env.edges.index(edge)][dest_id].edges)
                if next_route_len < cur_route_len:
                    feasible_next_edge.append(edge)
            next_edge = np.random.choice(feasible_next_edge)
            next_route = [veh_edge, next_edge]

        return next_route


class GridRouter(BaseRouter):
    """A router used to re-route a vehicle in a traffic light grid environment.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        if len(env.k.vehicle.get_route(self.veh_id)) == 0:
            # this occurs to inflowing vehicles, whose information is not added
            # to the subscriptions in the first step that they departed
            return None
        elif env.k.vehicle.get_edge(self.veh_id) == \
                env.k.vehicle.get_route(self.veh_id)[-1]:
            return [env.k.vehicle.get_edge(self.veh_id)]
        else:
            return None


class BayBridgeRouter(ContinuousRouter):
    """Assists in choosing routes in select cases for the Bay Bridge network.

    Extension to the Continuous Router.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        edge = env.k.vehicle.get_edge(self.veh_id)
        lane = env.k.vehicle.get_lane(self.veh_id)

        if edge == "183343422" and lane in [2] \
                or edge == "124952179" and lane in [1, 2]:
            new_route = env.available_routes[edge + "_1"][0][0]
        else:
            new_route = super().choose_route(env)

        return new_route


class I210Router(ContinuousRouter):
    """Assists in choosing routes in select cases for the I-210 sub-network.

    Extension to the Continuous Router.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        edge = env.k.vehicle.get_edge(self.veh_id)
        lane = env.k.vehicle.get_lane(self.veh_id)

        # vehicles on these edges in lanes 4 and 5 are not going to be able to
        # make it out in time
        if edge == "119257908#1-AddedOffRampEdge" and lane in [5, 4, 3]:
            new_route = env.available_routes[
                "119257908#1-AddedOffRampEdge"][0][0]
        else:
            new_route = super().choose_route(env)

        return new_route

class OpenRouter_Flow(BaseRouter):
    """
    A router used to continuously routes added inflow vehicles to exit of this network in OpenFlow Network.

    This class allows the vehicle to pick a random route(one of two road that can be selected) at junctions.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        src, dest = 'top3_4_0', 'top0_0_0'
        src_id, dest_id = env.edges.index(src), env.edges.index(dest)

        vehicles = env.k.vehicle
        veh_id = self.veh_id
        veh_edge = vehicles.get_edge(veh_id)
        if veh_edge[0] == ':':
            return None
        veh_edge_id = env.edges.index(veh_edge)
        veh_route = vehicles.get_route(veh_id)
        veh_next_edge = env.k.network.next_edge(veh_edge, vehicles.get_lane(veh_id))

        cur_route_len = len(env.paired_routes[veh_edge_id][dest_id].edges)

        next_route = None
        if veh_route[-1] == veh_edge and veh_next_edge != []:
            feasible_next_edge = []
            if veh_next_edge[0][0][0] == ':':
                next_edge = []
                for edge in veh_next_edge:
                    next_edge.append(env.k.network.next_edge(edge[0], edge[1])[0])
                veh_next_edge = next_edge
            for edge, _ in veh_next_edge:
                next_route_len = len(env.paired_routes[env.edges.index(edge)][dest_id].edges)
                if next_route_len < cur_route_len:
                    feasible_next_edge.append(edge)
            next_edge = np.random.choice(feasible_next_edge)
            next_route = [veh_edge, next_edge]

        return next_route


class OpenRouter_Inner(BaseRouter):
    """
    A router used to continuously routes inner background vehicles to in OpenFlow Network.

    This class allows the vehicle to pick a random route at junctions. However, they shouldn't enter any exit path.
    These vehicles can't be created on exit path.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        vehicles = env.k.vehicle
        veh_id = self.veh_id
        veh_edge = vehicles.get_edge(veh_id)
        veh_route = vehicles.get_route(veh_id)
        veh_next_edge = env.k.network.next_edge(veh_edge,
                                                vehicles.get_lane(veh_id))
        not_an_edge = ":"
        no_next = 0

        if len(veh_next_edge) == no_next:
            next_route = None
        elif veh_route[-1] == veh_edge and veh_next_edge != []:
            flag = True
            while(flag):
                veh_next_edge = env.k.network.next_edge(veh_edge, vehicles.get_lane(veh_id))
                random_route = random.randint(0, len(veh_next_edge) - 1)
                try:
                    while veh_next_edge[0][0][0] == not_an_edge:
                        veh_next_edge = env.k.network.next_edge(
                            veh_next_edge[random_route][0],
                            veh_next_edge[random_route][1])
                except:
                    print('veh_id', veh_id, 'veh_edge', veh_edge, 'veh_route', veh_route, 'veh_next_edge', veh_next_edge)
                    raise KeyError
                next_route = [veh_edge, veh_next_edge[0][0]]
                #print(veh_next_edge[0][0])
                if (veh_next_edge[0][0] != 'top0_0_0' and veh_next_edge[0][0] != 'bot3_4_0'):
                    flag = False
        else:
            next_route = None

        return next_route


class Flow_Select(BaseRouter):
    """
    A router used to continuously routes added inflow vehicles to exit of this network in Flow Network.

    This class allows the vehicle to pick one of two routes in the network (the outer two routes without tl).

    Usage
    -----
    See base class for usage example.
    """

    def __init__(self, veh_id, router_params, select_edge=None, src=None, dest=None):
        super().__init__(veh_id, router_params)
        self.src = src
        self.dest = dest
        self.select_edge = select_edge


    def choose_route(self, env):
        src, dest = self.src, self.dest
        src_id, dest_id = env.edges.index(src), env.edges.index(dest)

        vehicles = env.k.vehicle
        veh_id = self.veh_id
        veh_edge = vehicles.get_edge(veh_id)
        if veh_edge[0] == ':':
            return None
        veh_edge_id = env.edges.index(veh_edge)
        veh_route = vehicles.get_route(veh_id)
        veh_next_edge = env.k.network.next_edge(veh_edge, vehicles.get_lane(veh_id))

        cur_route_len = len(env.paired_routes[veh_edge_id][dest_id].edges)

        next_route = None
        if veh_route[-1] == veh_edge and veh_next_edge != []:
            feasible_next_edge = []
            if veh_next_edge[0][0][0] == ':':
                next_edge = []
                for edge in veh_next_edge:
                    next_edge.append(env.k.network.next_edge(edge[0], edge[1])[0])
                veh_next_edge = next_edge
            for edge, _ in veh_next_edge:
                next_route_len = len(env.paired_routes[env.edges.index(edge)][dest_id].edges)
                if (next_route_len < cur_route_len) and (edge in self.select_edge):
                    feasible_next_edge.append(edge)
            next_edge = np.random.choice(feasible_next_edge)
            next_route = [veh_edge, next_edge]

        return next_route

class FlowRouter_Inner(BaseRouter):
    """
    A router used to continuously routes inner background vehicles to in OpenFlow Network.

    This class allows the vehicle to pick a random route at junctions. However, they shouldn't enter any exit path.
    These vehicles can't be created on exit path.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        vehicles = env.k.vehicle
        veh_id = self.veh_id
        veh_edge = vehicles.get_edge(veh_id)
        veh_route = vehicles.get_route(veh_id)
        veh_next_edge = env.k.network.next_edge(veh_edge,
                                                vehicles.get_lane(veh_id))
        not_an_edge = ":"
        no_next = 0

        if len(veh_next_edge) == no_next:
            next_route = None
        elif veh_route[-1] == veh_edge and veh_next_edge != []:
            flag = True
            while(flag):
                veh_next_edge = env.k.network.next_edge(veh_edge, vehicles.get_lane(veh_id))
                random_route = random.randint(0, len(veh_next_edge) - 1)
                try:
                    while veh_next_edge[0][0][0] == not_an_edge:
                        veh_next_edge = env.k.network.next_edge(
                            veh_next_edge[random_route][0],
                            veh_next_edge[random_route][1])
                except:
                    print('veh_id', veh_id, 'veh_edge', veh_edge, 'veh_route', veh_route, 'veh_next_edge', veh_next_edge)
                    raise KeyError
                next_route = [veh_edge, veh_next_edge[0][0]]
                #print(veh_next_edge[0][0])
                if (('in' in veh_next_edge[0][0]) or 'out' in veh_next_edge[0][0]):
                    flag = True
                else:
                    flag = False
        else:
            next_route = None

        return next_route

class FluxBase_Router(BaseRouter):
    """
    A router used to continuously routes inner background vehicles to in OpenFlow Network.

    This class allows the vehicle to pick a route based on the number of cars on each road at junctions.
    However, they shouldn't enter any exit path. Also, they may choose more on edge with less cars.
    These vehicles can't be created on exit path.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        vehicles = env.k.vehicle
        veh_id = self.veh_id
        veh_edge = vehicles.get_edge(veh_id)
        veh_route = vehicles.get_route(veh_id)
        veh_next_edge = env.k.network.next_edge(veh_edge,
                                                vehicles.get_lane(veh_id))
        not_an_edge = ":"
        no_next = 0

        if len(veh_next_edge) == no_next:
            next_route = None
        elif veh_route[-1] == veh_edge and veh_next_edge != []:
            select = env.k.network.next_edge(veh_edge, vehicles.get_lane(veh_id))
            try:
                candidates = []
                flux = []
                # List all next edge
                for i in range(len(veh_next_edge)):
                    next_edge = env.k.network.next_edge(veh_next_edge[i][0], veh_next_edge[i][1])
                    candidates.append(next_edge[0][0])
                # Remove all flow edge
                for i in range(len(candidates) - 1, -1, -1):
                    if (('in' in candidates[i]) or ('out' in candidates[i])):
                        candidates.pop(i)
                # Get flow on each candidate edge
                if candidates[0][0] == not_an_edge:
                    select = random.choice(candidates)
                else:
                    for candidate in candidates:
                        flux.append(env.k.kernel_api.edge.getLastStepVehicleNumber(candidate))
                    min_val = min(flux)
                    minimum = []
                    other = []
                    flag = False
                    for i in range(len(candidates)):
                        if flux[i] == min_val:
                            minimum.append(i)
                        else:
                            other.append(i)
                    random_choice = random.randint(1, 10)
                    if random_choice <= 7:
                        # Choose one of road with minimal car
                        if len(minimum) != 0:
                            select = random.choice(minimum)
                            flag = True
                    else:
                        # Choose other cars
                        if len(other) != 0:
                            select = random.choice(other)
                            flag = True
                    if (flag):
                        select = candidates[select]
                    else:
                        select = random.choice(candidates)
            except:
                print('veh_id', veh_id, 'veh_edge', veh_edge, 'veh_route', veh_route, 'veh_next_edge', veh_next_edge)
                raise KeyError
            next_route = [veh_edge, select]
            #print(veh_next_edge[0][0])
        else:
            next_route = None

        return next_route

class IndexEnv_Router(BaseRouter):
    """
    A router used to continuously routes inner background vehicles to in OpenFlow Network.

    This class allows the vehicle to pick a random route at junctions. However, they shouldn't enter any exit path.
    These vehicles can't be created on exit path.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        edges =[]
        for idx in range(env.col_idx * env.cols):
            col = idx % env.cols
            ind = idx // env.cols
            edges.append("out_left{}_{}".format(col, ind))
            edges.append("out_right{}_{}".format(col, (env.row_idx - 1) * env.col_idx + ind))
            edges.append("in_right{}_{}".format(col, ind))
            edges.append("in_left{}_{}".format(col, (env.row_idx - 1) * env.col_idx + ind))
        for idx in range(env.rows * env.row_idx):
            row = idx % env.rows
            ind = idx // env.rows
            edges.append("out_top{}_{}".format(row, ind * env.col_idx))
            edges.append("out_bot{}_{}".format(row, ind * env.col_idx + env.col_idx - 1))
            edges.append("in_bot{}_{}".format(row, ind * env.col_idx))
            edges.append("in_top{}_{}".format(row, ind * env.col_idx + env.col_idx - 1))

        vehicles = env.k.vehicle
        veh_id = self.veh_id
        veh_edge = vehicles.get_edge(veh_id)
        veh_route = vehicles.get_route(veh_id)
        veh_next_edge = env.k.network.next_edge(veh_edge,
                                                vehicles.get_lane(veh_id))
        not_an_edge = ":"
        no_next = 0

        if len(veh_next_edge) == no_next:
            next_route = None
        elif veh_route[-1] == veh_edge and veh_next_edge != []:
            flag = True
            while(flag):
                veh_next_edge = env.k.network.next_edge(veh_edge, vehicles.get_lane(veh_id))
                random_route = random.randint(0, len(veh_next_edge) - 1)
                try:
                    while veh_next_edge[0][0][0] == not_an_edge:
                        veh_next_edge = env.k.network.next_edge(
                            veh_next_edge[random_route][0],
                            veh_next_edge[random_route][1])
                except:
                    print('veh_id', veh_id, 'veh_edge', veh_edge, 'veh_route', veh_route, 'veh_next_edge', veh_next_edge)
                    raise KeyError
                next_route = [veh_edge, veh_next_edge[0][0]]
                #print(veh_next_edge[0][0])
                if (veh_next_edge[0][0] not in edges):
                    flag = False
        else:
            next_route = None

        return next_route
