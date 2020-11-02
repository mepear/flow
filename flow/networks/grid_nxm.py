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
        optional = ["tl_logic"]
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params and p not in optional:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        for p in ADDITIONAL_NET_PARAMS["grid_array"].keys():
            if p not in net_params.additional_params["grid_array"]:
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

        # specifies whether or not there will be traffic lights at the
        # intersections (True by default)
        self.use_traffic_lights = net_params.additional_params.get(
            "traffic_lights", True)

        # radius of the inner nodes (ie of the intersections)
        self.inner_nodes_radius = 2.9 + 3.3 * max(self.vertical_lanes,
                                                  self.horizontal_lanes)

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
        for row in range(self.row_num):
            for col in range(self.col_num):
                nodes.append({
                    "id": "center{}".format(row * self.col_num + col),
                    "x": col * self.inner_length,
                    "y": row * self.inner_length,
                    "type": node_type,
                    "radius": self.inner_nodes_radius
                })

        return nodes

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
            return [{
                "id": lane + index,
                "type": orientation,
                "priority": 78,
                "from": "center" + str(from_node),
                "to": "center" + str(to_node),
                "length": self.inner_length
            }]

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

        Connections describe what happens at the intersections. Here we link
        lanes in straight lines, which means vehicles cannot turn at
        intersections, they can only continue in a straight line.
        """
        con_dict = {}

        def new_con(side, from_id, to_id, lane, signal_group, toside=None):
            if toside is None:
                toside = side
            conn = [{
                "from": side + from_id,
                "to": toside + to_id,
                "fromLane": str(lane),
                "toLane": str(lane),
                "signal_group": signal_group
            }]
            if conn[0]['signal_group'] is None:
                del conn[0]['signal_group']
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
                for lane in range(self.vertical_lanes):
                    conn += new_con('top', right_edge_id, top_edge_id, lane, None, 'right')
                    conn += new_con('left', top_edge_id, right_edge_id, lane, None, 'bot')
            if top_edge_id is not None and left_edge_id is not None:
                for lane in range(self.vertical_lanes):
                    conn += new_con('left', top_edge_id, left_edge_id, lane, None, 'top')
                    conn += new_con('bot', left_edge_id, top_edge_id, lane, None, 'right')
            if bot_edge_id is not None and right_edge_id is not None:
                for lane in range(self.vertical_lanes):
                    conn += new_con('right', bot_edge_id, right_edge_id, lane, None, "bot")
                    conn += new_con('top', right_edge_id, bot_edge_id, lane, None, "left")
            if bot_edge_id is not None and left_edge_id is not None:
                for lane in range(self.vertical_lanes):
                    conn += new_con('right', bot_edge_id, left_edge_id, lane, None, "top")
                    conn += new_con('bot', left_edge_id, bot_edge_id, lane, None, "left")

            if top_edge_id is not None and bot_edge_id is not None:
                for lane in range(self.vertical_lanes):
                    conn += new_con('right', bot_edge_id, top_edge_id, lane, 2)
                    conn += new_con('left', top_edge_id, bot_edge_id, lane, 2)
            if left_edge_id is not None and right_edge_id is not None:
                for lane in range(self.horizontal_lanes):
                    conn += new_con('bot', left_edge_id, right_edge_id, lane, 1)
                    conn += new_con('top', right_edge_id, left_edge_id, lane, 1)

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
