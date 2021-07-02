"""Contains the traffic light grid scenario class."""

from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams, PersonParams
from collections import defaultdict
import numpy as np

ADDITIONAL_NET_PARAMS = {
    # dictionary of traffic light grid array data
    "grid_array": {
        # number of horizontal rows of edges
        "row_num": 3,
        # number of vertical columns of edges
        "col_num": 3,
        # length of inner edges in the traffic light grid network
        "inner_length": None,
        # split a edge to several sub-edges, the number of sub-edges
        "outer_length": None,
        # length of outer edges in the traffic light grid network
        "sub_edge_num": 1,
    },
    # number of lanes in the horizontal edges
    "horizontal_lanes": 1,
    # number of lanes in the vertical edges
    "vertical_lanes": 1,
    # speed limit for all edges, may be represented as a float value, or a
    # dictionary with separate values for vertical and horizontal lanes
    "speed_limit": {
        "horizontal": 35,
        "vertical": 35
    }
}


class GridnxmNetwork(Network):
    """Traffic Light Grid network class.

    The traffic light grid network consists of m vertical lanes and n
    horizontal lanes, with a total of nxm intersections where the vertical
    and horizontal edges meet.

    Requires from net_params:

    * **grid_array** : dictionary of grid array data, with the following keys

      * **row_num** : number of horizontal rows of edges
      * **col_num** : number of vertical columns of edges
      * **inner_length** : length of inner edges in traffic light grid network

    * **horizontal_lanes** : number of lanes in the horizontal edges
    * **vertical_lanes** : number of lanes in the vertical edges
    * **speed_limit** : speed limit for all edges. This may be represented as a
      float value, or a dictionary with separate values for vertical and
      horizontal lanes.

    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from flow.networks import TrafficLightGridNetwork
    >>>
    >>> network = TrafficLightGridNetwork(
    >>>     name='grid',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams(
    >>>         additional_params={
    >>>             'grid_array': {
    >>>                 'row_num': 3,
    >>>                 'col_num': 2,
    >>>                 'inner_length': 500,
    >>>             },
    >>>             'horizontal_lanes': 1,
    >>>             'vertical_lanes': 1,
    >>>             'speed_limit': {
    >>>                 'vertical': 35,
    >>>                 'horizontal': 35
    >>>             }
    >>>         },
    >>>     )
    >>> )
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 persons=PersonParams(),
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize an n*m traffic light grid network."""
        optional = ["tl_logic", "outer_length"]
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params and p not in optional:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        for p in ADDITIONAL_NET_PARAMS["grid_array"].keys():
            if p not in net_params.additional_params["grid_array"] and p not in optional:
                raise KeyError(
                    'Grid array parameter "{}" not supplied'.format(p))

        # retrieve all additional parameters
        # refer to the ADDITIONAL_NET_PARAMS dict for more documentation
        self.vertical_lanes = net_params.additional_params["vertical_lanes"]
        self.horizontal_lanes = net_params.additional_params[
            "horizontal_lanes"]
        self.speed_limit = net_params.additional_params["speed_limit"]
        if not isinstance(self.speed_limit, dict):
            self.speed_limit = {
                "horizontal": self.speed_limit,
                "vertical": self.speed_limit
            }

        self.grid_array = net_params.additional_params["grid_array"]
        self.row_num = self.grid_array["row_num"]
        self.col_num = self.grid_array["col_num"]
        self.inner_length = self.grid_array["inner_length"]
        self.outer_length = self.grid_array.get("outer_length", None)
        self.sub_edge_num = self.grid_array["sub_edge_num"]

        # specifies whether or not there will be traffic lights at the
        # intersections (True by default)
        self.use_traffic_lights = net_params.additional_params.get(
            "traffic_lights", False)

        # radius of the inner nodes (ie of the intersections)
        # self.inner_nodes_radius = 2.9 + 3.3 * max(self.vertical_lanes,
        #                                           self.horizontal_lanes)
        self.inner_nodes_radius = 0

        # total number of edges in the network
        self.num_edges = 4 * ((self.col_num + 1) * self.row_num + self.col_num)

        # name of the network (DO NOT CHANGE)
        self.name = "BobLoblawsLawBlog"

        super().__init__(name, vehicles, net_params, persons, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):

        """Build out the inner nodes of the network.

        The inner nodes correspond to the intersections between the roads. They
        are numbered from bottom left, increasing first across the columns and
        then across the rows.

        For example, the nodes in a traffic light grid with 2 rows and 3 columns
        would be indexed as follows:

            |     |     |
        --- 3 --- 4 --- 5 ---
            |     |     |
        --- 0 --- 1 --- 2 ---
            |     |     |

        The id of a node is then "center{index}", for instance "center0" for
        node 0, "center1" for node 1 etc.

        Returns
        -------
        list <dict>
            List of inner nodes
        """
        node_type = "traffic_light" if self.use_traffic_lights else "priority"

        nodes = []
        inserted_nodes = []
        for row in range(self.row_num):
            for col in range(self.col_num):
                nodes.append({
                    "id": "center{}".format(row * self.col_num + col),
                    "x": col * self.inner_length,
                    "y": row * self.inner_length,
                    "type": node_type,
                    # "radius": self.inner_nodes_radius
                })

        inserted_nodes = []
        for row in range(self.row_num):
            for col in range(self.col_num):
                for n in range(self.sub_edge_num - 1):
                    cur_id = row * self.col_num + col
                    
                    if col < self.col_num - 1:
                        inserted_nodes.append({
                            "id": "{}-{}_{}".format(cur_id, cur_id + 1, n),
                            "x": col * self.inner_length + self.inner_length * (n + 1) / self.sub_edge_num,
                            "y": row * self.inner_length,
                            "type": "priority"
                        })
                    if row < self.row_num - 1:
                        inserted_nodes.append({
                            "id": "{}-{}_{}".format(cur_id, cur_id + self.col_num, n),
                            "x": col * self.inner_length,
                            "y": row * self.inner_length + self.inner_length * (n + 1) / self.sub_edge_num,
                            "type": "priority"
                        })


        return nodes + inserted_nodes

    def specify_edges(self, net_params):
        """Build out the inner edges of the network.

        The inner edges are the edges joining the inner nodes to each other.

        Consider the following network with n = 2 rows and m = 3 columns,
        where the rows are indexed from 0 to 1 and the columns from 0 to 2, and
        the inner nodes are marked by 'x':

                |     |     |
        (1) ----x-----x-----x----
                |     |     |
        (0) ----x-----x-(*)-x----
                |     |     |
               (0)   (1)   (2)

        There are n * (m - 1) = 4 horizontal inner edges and (n - 1) * m = 3
        vertical inner edges, all that multiplied by two because each edge
        consists of two roads going in opposite directions traffic-wise.

        On an horizontal edge, the id of the top road is "top{i}_{j}" and the
        id of the bottom road is "bot{i}_{j}", where i is the index of the row
        where the edge is and j is the index of the column to the right of it.

        On a vertical edge, the id of the right road is "right{i}_{j}" and the
        id of the left road is "left{i}_{j}", where i is the index of the row
        above the edge and j is the index of the column where the edge is.

        For example, on edge (*) on row (0): the id of the bottom road (traffic
        going from left to right) is "bot0_2" and the id of the top road
        (traffic going from right to left) is "top0_2".

        Returns
        -------
        list <dict>
            List of inner edges
        """
        edges = []

        def new_edge(index, from_node, to_node, orientation, lane):
            assert from_node != to_node
            if from_node < to_node:
                node_list = ["center{}".format(from_node)] + ["{}-{}_{}".format(from_node, to_node, n) for n in range(self.sub_edge_num - 1)] + ["center{}".format(to_node)]
            else: 
                node_list = ["center{}".format(from_node)] + ["{}-{}_{}".format(to_node, from_node, n) for n in range(self.sub_edge_num - 2, -1, -1)] + ["center{}".format(to_node)]

            new_edges = []
            for i, node in enumerate(node_list):
                if i + 1 < len(node_list):
                    new_edges.append({
                        "id": "{}{}_{}".format(lane, index, i),
                        "type": orientation,
                        "priority": 78,
                        "from": node,
                        "to": node_list[i + 1],
                        "length": self.inner_length / self.sub_edge_num
                    })
            return new_edges
        
        # Build the horizontal inner edges
        for i in range(self.row_num):
            for j in range(self.col_num - 1):
                node_index = i * self.col_num + j
                index = "{}_{}".format(i, j + 1)
                edges += new_edge(index, node_index + 1, node_index,
                                  "horizontal", "top")
                edges += new_edge(index, node_index, node_index + 1,
                                  "horizontal", "bot")

        # Build the vertical inner edges
        for i in range(self.row_num - 1):
            for j in range(self.col_num):
                node_index = i * self.col_num + j
                index = "{}_{}".format(i + 1, j)
                edges += new_edge(index, node_index, node_index + self.col_num,
                                  "vertical", "right")
                edges += new_edge(index, node_index + self.col_num, node_index,
                                  "vertical", "left")

        return edges

    def specify_routes(self, net_params):
        """See parent class."""
        routes = {}
        # conn = self.specify_connections(net_params)
        # for node_id in range(self.row_num * self.col_num):
        #     node_conn = conn['center{}'.format(node_id)]
        #     for c in node_conn:
        #         from_edge = c['from']
        #         target_edge = c['to']
        #         if from_edge not in routes:
        #             routes[from_edge] = []
        #         routes[from_edge].append([[from_edge, target_edge], 1])
        # for r in routes:
        #     l = len(routes[r])
        #     for i, route in enumerate(routes[r]):
        #         route[1] = 1. / l
        #         routes[r][i] = tuple(route)
                
        edges = self.specify_edges(net_params)
        for edge in edges:
            routes[edge['id']] = [edge['id']]
        return routes

    def specify_types(self, net_params):
        """See parent class."""
        types = [{
            "id": "horizontal",
            "numLanes": self.horizontal_lanes,
            "speed": self.speed_limit["horizontal"]
        }, {
            "id": "vertical",
            "numLanes": self.vertical_lanes,
            "speed": self.speed_limit["vertical"]
        }]

        return types

    # ===============================
    # ============ UTILS ============
    # ===============================

    def specify_connections(self, net_params):
        """Build out connections at each inner node.
        """
        con_dict = {}

        def new_con(side, from_id, to_id, signal_group, toside=None, from_sub_id=0, to_sub_id=0, inv=False):
            if toside is None:
                toside = side
            
            conn = []
            lane1s = range(self.vertical_lanes) if not inv else [self.vertical_lanes - 1]
            for lane1 in lane1s:
                for lane2 in range(self.vertical_lanes):
                    conn.append({
                    "from": side + from_id + "_{}".format(from_sub_id),
                    "to": toside + to_id + "_{}".format(to_sub_id),
                    "fromLane": str(lane1),
                    "toLane": str(lane2),                        
                    })
                
            # conn = [{
            #     "from": side + from_id + "_{}".format(from_sub_id),
            #     "to": toside + to_id + "_{}".format(to_sub_id)
            #     # "fromLane": str(lane),
            #     # "toLane": str(lane),
            #     # "signal_group": signal_group
            # }]
            # if conn[0]['signal_group'] is None:
            #     del conn[0]['signal_group']
            return conn

        # build connections at each inner node
        # node_ids = [edge['id'][-3:] for edge in self.edges]
        for node_id in range(self.row_num * self.col_num):
            conn = []
            i = node_id // self.col_num
            j = node_id % self.col_num
            top_edge_id = "{}_{}".format(i+1, j) if i + 1 < self.row_num else None
            bot_edge_id = "{}_{}".format(i, j) if i > 0 else None
            left_edge_id = "{}_{}".format(i, j) if j > 0 else None
            right_edge_id = "{}_{}".format(i, j+1) if j + 1 < self.col_num else None
            assert self.vertical_lanes == self.horizontal_lanes
            if right_edge_id is not None and top_edge_id is not None:
                conn += new_con('top', right_edge_id, top_edge_id, None, 'right', 0, 0)
                conn += new_con('left', top_edge_id, right_edge_id, None, 'bot', 0, 0)
            if top_edge_id is not None and left_edge_id is not None:
                conn += new_con('left', top_edge_id, left_edge_id, None, 'top', 0, self.sub_edge_num-1)
                conn += new_con('bot', left_edge_id, top_edge_id, None, 'right', self.sub_edge_num-1, 0)
            if bot_edge_id is not None and right_edge_id is not None:
                conn += new_con('right', bot_edge_id, right_edge_id, None, "bot", self.sub_edge_num-1, 0)
                conn += new_con('top', right_edge_id, bot_edge_id, None, "left", 0, self.sub_edge_num-1)
            if bot_edge_id is not None and left_edge_id is not None:
                conn += new_con('right', bot_edge_id, left_edge_id, None, "top", self.sub_edge_num-1, self.sub_edge_num-1)
                conn += new_con('bot', left_edge_id, bot_edge_id, None, "left", self.sub_edge_num-1, self.sub_edge_num-1)

            if top_edge_id is not None and bot_edge_id is not None:
                conn += new_con('right', bot_edge_id, top_edge_id, 2)
                conn += new_con('left', top_edge_id, bot_edge_id, 2)
            if left_edge_id is not None and right_edge_id is not None:
                conn += new_con('bot', left_edge_id, right_edge_id, 1)
                conn += new_con('top', right_edge_id, left_edge_id, 1)

            if top_edge_id is not None:
                conn += new_con('left', top_edge_id, top_edge_id, None, 'right', 0, 0, True)
                for n in range(self.sub_edge_num - 1):
                    conn += new_con('left', top_edge_id, top_edge_id, None, 'left', n+1, n)
            if bot_edge_id is not None:
                conn += new_con('right', bot_edge_id, bot_edge_id, None, 'left', self.sub_edge_num-1, self.sub_edge_num-1, True)
                for n in range(self.sub_edge_num-1):
                    conn += new_con('right', bot_edge_id, bot_edge_id, None, 'right', n, n+1)
            if left_edge_id is not None:
                conn += new_con('bot', left_edge_id, left_edge_id, None, 'top', self.sub_edge_num-1, self.sub_edge_num-1, True)
                for n in range(self.sub_edge_num-1):
                    conn += new_con('bot', left_edge_id, left_edge_id, None, 'bot', n, n+1)
            if right_edge_id is not None:
                conn += new_con('top', right_edge_id, right_edge_id, None, 'bot', 0, 0, True)
                for n in range(self.sub_edge_num-1):
                    conn += new_con('top', right_edge_id, right_edge_id, None, 'top', n+1, n)

            node_id = "center{}".format(node_id)
            con_dict[node_id] = conn

        return con_dict

    # TODO necessary?
    def specify_edge_starts(self):
        """See parent class."""
        length = 0
        edgestarts = []
        for edge in self.edges:
            # the current edge starts where the last edge ended
            edgestarts.append((edge['id'], length))
            # increment the total length of the network with the length of the
            # current edge
            length += float(edge['length'])

        return edgestarts

    @property
    def node_mapping(self):
        """Map nodes to edges.

        Returns a list of pairs (node, connected edges) of all inner nodes
        and for each of them, the 4 edges that leave this node.

        The nodes are listed in alphabetical order, and within that, edges are
        listed in order: [bot, right, top, left].
        """
        mapping = {}

        for node_id in range(self.row_num * self.col_num):
            conn = []
            i = node_id // self.col_num
            j = node_id % self.col_num
            top_edge_id = "left{}_{}".format(i+1, j) if i + 1 < self.row_num else None
            bot_edge_id = "right{}_{}".format(i, j) if i > 0 else None
            left_edge_id = "bot{}_{}".format(i, j) if j > 0 else None
            right_edge_id = "top{}_{}".format(i, j+1) if j + 1 < self.col_num else None
            node_id = "center{}".format(node_id)
            mapping[node_id] = [left_edge_id, bot_edge_id, right_edge_id, top_edge_id]

        return sorted(mapping.items(), key=lambda x: x[0])


class GridnxmNetworkInflow(GridnxmNetwork):
    def specify_nodes(self, net_params):
        nodes = super().specify_nodes(net_params)
        if 'inflow' in net_params.additional_params:
            inflow = net_params.additional_params['inflow']
            if 'top_left' in inflow:
                nodes.append({
                    "id": "center_top_left",
                    "x": -.5 * self.inner_length,
                    "y": (self.row_num - 1) * self.inner_length,
                    "type": 'priority',
                    # "radius": self.inner_nodes_radius
                })
                nodes.append({
                    "id": "center_bot_right",
                    "x": (self.col_num - 0.5) * self.inner_length,
                    "y": 0.,
                    "type": 'priority',
                    # "radius": self.inner_nodes_radius
                })
            if 'midtop_left' in inflow:
                nodes.append({
                    "id": "center_midtop_left",
                    "x": -.5 * self.inner_length,
                    "y": (self.row_num - 2) * self.inner_length,
                    "type": 'priority',
                    # "radius": self.inner_nodes_radius
                })
            if 'midbot_left' in inflow:
                nodes.append({
                    "id": "center_midbot_left",
                    "x": -.5 * self.inner_length,
                    "y": (self.row_num - 3) * self.inner_length,
                    "type": 'priority',
                    # "radius": self.inner_nodes_radius
                })
            if 'top_midleft' in inflow:
                nodes.append({
                    "id": "center_top_midleft",
                    "x": self.inner_length,
                    "y": (self.row_num - 0.5) * self.inner_length,
                    "type": 'priority',
                    # "radius": self.inner_nodes_radius
                })
            if 'top_midright' in inflow:
                nodes.append({
                    "id": "center_top_midright",
                    "x": 2. * self.inner_length,
                    "y": (self.row_num - 0.5) * self.inner_length,
                    "type": 'priority',
                    # "radius": self.inner_nodes_radius
                })
                
        return nodes

    def specify_edges(self, net_params):
        edges = super().specify_edges(net_params)
        if 'inflow' in net_params.additional_params:
            inflow = net_params.additional_params['inflow']
            if 'top_left' in inflow:
                edges.append({
                        "id": "inflow_top_left",
                        "type": 'horizontal',
                        "priority": 78,
                        "from": "center_top_left",
                        "to": "center{}".format((self.row_num - 1) * self.col_num),
                        "length": self.inner_length / self.sub_edge_num / 2
                    })
                edges.append({
                        "id": "outflow_bot_right",
                        "type": 'horizontal',
                        "priority": 78,
                        "from": "center{}".format(self.col_num - 1),
                        "to": "center_bot_right",
                        "length": self.inner_length / self.sub_edge_num / 2
                    })
            if 'midtop_left' in inflow:
                edges.append({
                    "id": "inflow_midtop_left",
                    "type": "horizontal",
                    "priority": 78,
                    "from": "center_midtop_left",
                    "to": "center{}".format((self.row_num - 2) * self.col_num),
                    "length": self.inner_length / self.sub_edge_num / 2
                })
            if 'midbot_left' in inflow:
                edges.append({
                    "id": "inflow_midbot_left",
                    "type": "horizontal",
                    "priority": 78,
                    "from": "center_midbot_left",
                    "to": "center{}".format((self.row_num - 3) * self.col_num),
                    "length": self.inner_length / self.sub_edge_num / 2
                })
            if 'top_midleft' in inflow:
                edges.append({
                    "id": "inflow_top_midleft",
                    "type": "vertical",
                    "priority": 78,
                    "from": "center_top_midleft",
                    "to": "center{}".format((self.row_num - 1) * self.col_num + 1),
                    "length": self.inner_length / self.sub_edge_num / 2
                })
            if 'top_midright' in inflow:
                edges.append({
                    "id": "inflow_top_midright",
                    "type": "vertical",
                    "priority": 78,
                    "from": "center_top_midright",
                    "to": "center{}".format((self.row_num - 1) * self.col_num + 2),
                    "length": self.inner_length / self.sub_edge_num / 2
                })
        return edges
    
    def specify_connections(self, net_params):
        con_dict = super().specify_connections(net_params)
        if 'inflow' in net_params.additional_params:
            inflow = net_params.additional_params['inflow']
            if 'top_left' in inflow:
                node_id = "center{}".format((self.row_num - 1) * self.col_num)
                con_dict[node_id].append({
                    "from": 'inflow_top_left',
                    "to": 'left{}_0_0'.format(self.row_num - 1),
                    "fromLane": str(0),
                    "toLane": str(0),                        
                    })
                con_dict[node_id].append({
                    "from": 'inflow_top_left',
                    "to": 'bot{}_1_0'.format(self.row_num - 1),
                    "fromLane": str(0),
                    "toLane": str(0),                        
                    })

                con_dict[node_id].append({
                    "from": 'bot0_{}_0'.format(self.col_num - 1),
                    "to": 'outflow_bot_right',
                    "fromLane": str(0),
                    "toLane": str(0),                        
                    }) 
                con_dict[node_id].append({
                    "from": 'left1_{}_0'.format(self.row_num - 1),
                    "to": 'outflow_bot_right',
                    "fromLane": str(0),
                    "toLane": str(0),                        
                    })
            if 'midtop_left' in inflow:
                node_id = "center{}".format((self.row_num - 2) * self.col_num)
                con_dict[node_id].append({
                    "from": 'inflow_midtop_left',
                    "to": 'left{}_0_0'.format(self.row_num - 2),
                    "fromLane": str(0),
                    "toLane": str(0),                        
                    })
                con_dict[node_id].append({
                    "from": 'inflow_midtop_left',
                    "to": 'bot{}_1_0'.format(self.row_num - 2),
                    "fromLane": str(0),
                    "toLane": str(0),                        
                    })
            if 'midbot_left' in inflow:
                node_id = "center{}".format((self.row_num - 3) * self.col_num)
                con_dict[node_id].append({
                    "from": 'inflow_midbot_left',
                    "to": 'left{}_0_0'.format(self.row_num - 3),
                    "fromLane": str(0),
                    "toLane": str(0),                        
                    })
                con_dict[node_id].append({
                    "from": 'inflow_midbot_left',
                    "to": 'bot{}_1_0'.format(self.row_num - 3),
                    "fromLane": str(0),
                    "toLane": str(0),                        
                    })
            if 'top_midleft' in inflow:
                node_id = "center{}".format((self.row_num - 1) * self.col_num + 1)
                con_dict[node_id].append({
                    "from": 'inflow_top_midleft',
                    "to": 'left{}_1_0'.format(self.row_num - 1),
                    "fromLane": str(0),
                    "toLane": str(0),                      
                    })
                con_dict[node_id].append({
                    "from": 'inflow_top_midleft',
                    "to": 'bot{}_2_0'.format(self.row_num - 1),
                    "fromLane": str(0),
                    "toLane": str(0),                        
                    })
            if 'top_midright' in inflow:
                node_id = "center{}".format((self.row_num - 1) * self.col_num + 2)
                con_dict[node_id].append({
                    "from": 'inflow_top_midright',
                    "to": 'left{}_2_0'.format(self.row_num - 1),
                    "fromLane": str(0),
                    "toLane": str(0),                      
                    })
                con_dict[node_id].append({
                    "from": 'inflow_top_midright',
                    "to": 'bot{}_3_0'.format(self.row_num - 1),
                    "fromLane": str(0),
                    "toLane": str(0),                        
                    })
        return con_dict

class GridnxmNetworkExpand(GridnxmNetwork):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.outer_length = self.grid_array['outer_length']

    @property
    def _outer_nodes(self):
        """Build out the outer nodes of the network.

        The outer nodes correspond to the extremities of the roads. There are
        two at each extremity, one where the vehicles enter the network
        (inflow) and one where the vehicles exit the network (outflow).

        Consider the following network with 2 rows and 3 columns, where the
        extremities are marked by 'x', the rows are labeled from 0 to 1 and the
        columns are labeled from 0 to 2:

                 x     x     x
                 |     |     |
        (1) x----|-----|-----|----x (*)
                 |     |     |
        (0) x----|-----|-----|----x
                 |     |     |
                 x     x     x
                (0)   (1)   (2)

        On row i, there are two nodes at the left extremity of the row, labeled
        "left_row_in{i}" and "left_row_out{i}", as well as two nodes at the
        right extremity labeled "right_row_in{i}" and "right_row_out{i}".

        On column j, there are two nodes at the bottom extremity of the column,
        labeled "bot_col_in{j}" and "bot_col_out{j}", as well as two nodes
        at the top extremity labeled "top_col_in{j}" and "top_col_out{j}".

        The "in" nodes correspond to where vehicles enter the network while
        the "out" nodes correspond to where vehicles exit the network.

        For example, at extremity (*) on row (1):
        - the id of the input node is "right_row_in1"
        - the id of the output node is "right_row_out1"

        Returns
        -------
        list <dict>
            List of outer nodes
        """
        nodes = []

        def new_node(x, y, name, i):
            return [{"id": name + str(i), "x": x, "y": y, "type": "priority"}]

        # build nodes at the extremities of columns
        for col in range(self.col_num):
            x = col * self.inner_length
            y = (self.row_num - 1) * self.inner_length
            nodes += new_node(x, - self.outer_length, "bot_col_in", col)
            nodes += new_node(x, - self.outer_length, "bot_col_out", col)
            nodes += new_node(x, y + self.outer_length, "top_col_in", col)
            nodes += new_node(x, y + self.outer_length, "top_col_out", col)

        # build nodes at the extremities of rows
        for row in range(self.row_num):
            x = (self.col_num - 1) * self.inner_length
            y = row * self.inner_length
            nodes += new_node(- self.outer_length, y, "left_row_in", row)
            nodes += new_node(- self.outer_length, y, "left_row_out", row)
            nodes += new_node(x + self.outer_length, y, "right_row_in", row)
            nodes += new_node(x + self.outer_length, y, "right_row_out", row)

        return nodes

    @property
    def _outer_edges(self):
        """Build out the outer edges of the network.

        The outer edges are the edges joining the inner nodes to the outer
        nodes.

        Consider the following network with n = 2 rows and m = 3 columns,
        where the rows are indexed from 0 to 1 and the columns from 0 to 2, the
        inner nodes are marked by 'x' and the outer nodes by 'o':

                o    o    o
                |    |    |
        (1) o---x----x----x-(*)-o
                |    |    |
        (0) o---x----x----x-----o
                |    |    |
                o    o    o
               (0)  (1)  (2)

        There are n * 2 = 4 horizontal outer edges and m * 2 = 6 vertical outer
        edges, all that multiplied by two because each edge consists of two
        roads going in opposite directions traffic-wise.

        On row i, there are four horizontal edges: the left ones labeled
        "in_bot{i}_0" (in) and "out_top{i}_0" (out) and the right ones labeled
        "out_bot{i}_{m}" (out) and "in_top{i}_{m}" (in).

        On column j, there are four vertical edges: the bottom ones labeled
        "out_left0_{j}" (out) and "in_right0_{j}" (in) and the top ones labeled
        "in_left{n}_{j}" (in) and "out_right{n}_{j}" (out).

        For example, on edge (*) on row (1): the id of the bottom road (out)
        is "out_bot1_3" and the id of the top road is "in_top1_3".

        Edges labeled by "in" are edges where vehicles enter the network while
        edges labeled by "out" are edges where vehicles exit the network.

        Returns
        -------
        list <dict>
            List of outer edges
        """
        edges = []

        def new_edge(index, from_node, to_node, orientation, length):
            return [{
                "id": index,
                "type": {"v": "vertical", "h": "horizontal"}[orientation],
                "priority": 78,
                "from": from_node,
                "to": to_node,
                "length": length
            }]

        for i in range(self.col_num):
            # bottom edges
            id1 = "in_right0_{}".format(i)
            id2 = "out_left0_{}".format(i)
            node1 = "bot_col_in{}".format(i)
            node2 = "center{}".format(i)
            node3 = "bot_col_out{}".format(i)
            edges += new_edge(id1, node1, node2, "v", self.outer_length)
            edges += new_edge(id2, node2, node3, "v", self.outer_length)

            # top edges
            id1 = "in_left{}_{}".format(self.row_num, i)
            id2 = "out_right{}_{}".format(self.row_num, i)
            node1 = "top_col_in{}".format(i)
            node2 = "center{}".format((self.row_num - 1) * self.col_num + i)
            node3 = "top_col_out{}".format(i)
            edges += new_edge(id1, node1, node2, "v", self.outer_length)
            edges += new_edge(id2, node2, node3, "v", self.outer_length)

        for j in range(self.row_num):
            # left edges
            id1 = "in_bot{}_0".format(j)
            id2 = "out_top{}_0".format(j)
            node1 = "left_row_in{}".format(j)
            node2 = "center{}".format(j * self.col_num)
            node3 = "left_row_out{}".format(j)
            edges += new_edge(id1, node1, node2, "h", self.outer_length)
            edges += new_edge(id2, node2, node3, "h", self.outer_length)

            # right edges
            id1 = "in_top{}_{}".format(j, self.col_num)
            id2 = "out_bot{}_{}".format(j, self.col_num)
            node1 = "right_row_in{}".format(j)
            node2 = "center{}".format((j + 1) * self.col_num - 1)
            node3 = "right_row_out{}".format(j)
            edges += new_edge(id1, node1, node2, "h", self.outer_length)
            edges += new_edge(id2, node2, node3, "h", self.outer_length)

        return edges

    def specify_nodes(self, net_params):
        inner_nodes = super().specify_nodes(net_params)
        return inner_nodes + self._outer_nodes
    
    def specify_edges(self, net_params):
        inner_edges = super().specify_edges(net_params)
        return inner_edges + self._outer_edges

    def specify_connections(self, net_params):
        """Build out connections at each inner node.
        """
        con_dict = super().specify_connections(net_params)
        
        def new_con(side, from_id, to_id, toside, from_sub_id=None, to_sub_id=None):
            assert "out" not in side
            assert "in" not in toside
            lane1s = range(self.vertical_lanes)
            lane2s = range(self.vertical_lanes)

            conn = []
            lane1s = range(self.vertical_lanes)
            for lane1 in lane1s:
                for lane2 in lane2s:
                    conn.append({
                    "from": side + from_id + "_{}".format(from_sub_id) if from_sub_id is not None else side + from_id,
                    "to": toside + to_id + "_{}".format(to_sub_id) if to_sub_id is not None else toside + to_id,
                    "fromLane": str(lane1),
                    "toLane": str(lane2),                        
                    })
            return conn
                

        # build connections at each inner node
        # node_ids = [edge['id'][-3:] for edge in self.edges]
        for node_id in range(self.row_num * self.col_num):
            conn = []
            i = node_id // self.col_num
            j = node_id % self.col_num
            top_edge_id = "{}_{}".format(i+1, j) # if i + 1 < self.row_num else None
            bot_edge_id = "{}_{}".format(i, j) # if i > 0 else None
            left_edge_id = "{}_{}".format(i, j) # if j > 0 else None
            right_edge_id = "{}_{}".format(i, j+1) # if j + 1 < self.col_num else None
            assert self.vertical_lanes == self.horizontal_lanes
            # bottom left
            if i  == 0 and j == 0:

                conn += new_con("in_right", bot_edge_id, top_edge_id, "right", to_sub_id=0)
                conn += new_con("left", top_edge_id, bot_edge_id, "out_left", from_sub_id=0)

                conn += new_con("in_bot", left_edge_id, right_edge_id, "bot", to_sub_id=0)
                conn += new_con("top", right_edge_id, left_edge_id, "out_top", from_sub_id=0)

                conn += new_con('in_bot', left_edge_id, top_edge_id, "right", to_sub_id=0)
                conn += new_con('left', top_edge_id, left_edge_id, 'out_top', from_sub_id=0)

                conn += new_con("in_right", bot_edge_id, right_edge_id, "bot", to_sub_id=0)
                conn += new_con("top", right_edge_id, bot_edge_id, "out_left", from_sub_id=0)

                conn += new_con("in_bot", left_edge_id, bot_edge_id, "out_left")
                conn += new_con("in_right", bot_edge_id, left_edge_id, "out_top")
            
            # bottom right
            elif i  == 0 and j + 1 == self.col_num:

                conn += new_con("in_right", bot_edge_id, top_edge_id, "right", to_sub_id=0)
                conn += new_con("left", top_edge_id, bot_edge_id, "out_left", from_sub_id=0)

                conn += new_con("bot", left_edge_id, right_edge_id, "out_bot", from_sub_id=0)
                conn += new_con("in_top", right_edge_id, left_edge_id, "top", to_sub_id=0)

                conn += new_con('in_top', right_edge_id, top_edge_id, "right", to_sub_id=0)
                conn += new_con('left', top_edge_id, right_edge_id, 'out_bot', from_sub_id=0)

                conn += new_con("bot", left_edge_id, bot_edge_id, "out_left", from_sub_id=self.sub_edge_num-1)
                conn += new_con("in_right", bot_edge_id, left_edge_id, "top", to_sub_id=self.sub_edge_num-1)

                conn += new_con("in_right", bot_edge_id, right_edge_id, "out_bot")
                conn += new_con("in_top", right_edge_id, bot_edge_id, "out_left")
            
            # top left
            elif i + 1 == self.row_num and j == 0:
                conn += new_con("right", bot_edge_id, top_edge_id, "out_right", from_sub_id=self.sub_edge_num-1)
                conn += new_con("in_left", top_edge_id, bot_edge_id, "left", to_sub_id=self.sub_edge_num-1)

                conn += new_con("in_bot", left_edge_id, right_edge_id, "bot", to_sub_id=0)
                conn += new_con("top", right_edge_id, left_edge_id, "out_top", from_sub_id=0)

                conn += new_con("in_left", top_edge_id, right_edge_id, "bot", to_sub_id=0)
                conn += new_con("top", right_edge_id, top_edge_id, "out_right", from_sub_id=0)

                conn += new_con("in_bot", left_edge_id, bot_edge_id, "left", to_sub_id=self.sub_edge_num-1)
                conn += new_con("right", bot_edge_id, left_edge_id, "out_top", from_sub_id=self.sub_edge_num-1)

                conn += new_con('in_bot', left_edge_id, top_edge_id, "out_right")
                conn += new_con('in_left', top_edge_id, left_edge_id, "out_top")

            # top right
            elif i + 1 == self.row_num and j + 1 == self.col_num:
                conn += new_con("right", bot_edge_id, top_edge_id, "out_right", from_sub_id=self.sub_edge_num-1)
                conn += new_con("in_left", top_edge_id, bot_edge_id, "left", to_sub_id=self.sub_edge_num-1)

                conn += new_con("bot", left_edge_id, right_edge_id, "out_bot", from_sub_id=self.sub_edge_num-1)
                conn += new_con("in_top", right_edge_id, left_edge_id, "top", to_sub_id=self.sub_edge_num-1)

                conn += new_con("in_top", right_edge_id, bot_edge_id, "left", to_sub_id=self.sub_edge_num-1)
                conn += new_con("right", bot_edge_id, right_edge_id, "out_bot", from_sub_id=self.sub_edge_num-1)

                conn += new_con('bot', left_edge_id, top_edge_id, "out_right", from_sub_id=self.sub_edge_num-1)
                conn += new_con('in_left', top_edge_id, left_edge_id, "top", to_sub_id=self.sub_edge_num-1)

                conn += new_con("in_left", top_edge_id, right_edge_id, "out_bot")
                conn += new_con("in_top", right_edge_id, top_edge_id, "out_right")

            # bot
            elif i == 0:
                conn += new_con("in_right", bot_edge_id, top_edge_id, "right", to_sub_id=0)
                conn += new_con("left", top_edge_id, bot_edge_id, "out_left", from_sub_id=0)

                conn += new_con("in_right", bot_edge_id, right_edge_id, "bot", to_sub_id=0)
                conn += new_con("top", right_edge_id, bot_edge_id, "out_left", from_sub_id=0)

                conn += new_con("bot", left_edge_id, bot_edge_id, "out_left", from_sub_id=self.sub_edge_num-1)
                conn += new_con("in_right", bot_edge_id, left_edge_id, "top", to_sub_id=self.sub_edge_num-1)
            # left
            elif j == 0:
                conn += new_con("in_bot", left_edge_id, right_edge_id, "bot", to_sub_id=0)
                conn += new_con("top", right_edge_id, left_edge_id, "out_top", from_sub_id=0)

                conn += new_con('in_bot', left_edge_id, top_edge_id, "right", to_sub_id=0)
                conn += new_con('left', top_edge_id, left_edge_id, 'out_top', from_sub_id=0)

                conn += new_con("in_bot", left_edge_id, bot_edge_id, "left", to_sub_id=self.sub_edge_num-1)
                conn += new_con("right", bot_edge_id, left_edge_id, "out_top", from_sub_id=self.sub_edge_num-1)
            # top
            elif i + 1 == self.row_num:
                conn += new_con("right", bot_edge_id, top_edge_id, "out_right", from_sub_id=self.sub_edge_num-1)
                conn += new_con("in_left", top_edge_id, bot_edge_id, "left", to_sub_id=self.sub_edge_num-1)

                conn += new_con("in_left", top_edge_id, right_edge_id, "bot", to_sub_id=0)
                conn += new_con("top", right_edge_id, top_edge_id, "out_right", from_sub_id=0)

                conn += new_con('bot', left_edge_id, top_edge_id, "out_right", from_sub_id=self.sub_edge_num-1)
                conn += new_con('in_left', top_edge_id, left_edge_id, "top", to_sub_id=self.sub_edge_num-1)
            # right
            elif j + 1 == self.col_num:
                conn += new_con("bot", left_edge_id, right_edge_id, "out_bot", from_sub_id=self.sub_edge_num-1)
                conn += new_con("in_top", right_edge_id, left_edge_id, "top", to_sub_id=self.sub_edge_num-1)

                conn += new_con("in_top", right_edge_id, bot_edge_id, "left", to_sub_id=self.sub_edge_num-1)
                conn += new_con("right", bot_edge_id, right_edge_id, "out_bot", from_sub_id=self.sub_edge_num-1)

                conn += new_con("left", top_edge_id, right_edge_id, "out_bot", from_sub_id=0)
                conn += new_con("in_top", right_edge_id, top_edge_id, "right", to_sub_id=0)
            
            node_id = "center{}".format(node_id)
            con_dict[node_id] += conn

        return con_dict
