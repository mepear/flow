"""
Contains the traffic light grid scenario class.
This network also allows a flow from right-up corner to left-down corner.
"""

from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams, PersonParams

ADDITIONAL_NET_PARAMS = {
    # dictionary of traffic light grid array data
    "grid_array": {
        # number of horizontal rows of edges
        "row_num": 3,
        # number of vertical columns of edges
        "col_num": 3,
        # length of inner edges in the traffic light grid network
        "inner_length": None,
        # length of outer edges in the traffic light grid network
        "outer_length": None,
        # split a edge to several sub-edges, the number of sub-edges
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

class GridnxmNetworkOpen(Network):
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
        self.sub_edge_num = self.grid_array.get("sub_edge_num", 1)

        # specifies whether or not there will be traffic lights at the
        # intersections (False by default)
        self.use_traffic_lights = net_params.additional_params.get(
            "traffic_lights", False)

        # radius of the inner nodes (ie of the intersections)
        # self.inner_nodes_radius = 2.9 + 3.3 * max(self.vertical_lanes,
        #                                           self.horizontal_lanes)
        self.inner_nodes_radius = 0

        # total number of edges in the network
        self.num_edges = 4 * self.col_num * self.row_num - 2 * self.col_num - 2 * self.row_num + 4

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

                   3 --- 4 --- 5 --- 6(7)
                   |     |     |
        -1(-2) --- 0 --- 1 --- 2

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
                    "x": col * self.inner_length + self.inner_length,
                    "y": row * self.inner_length,
                    "type": node_type,
                    # "radius": self.inner_nodes_radius
                })

        # Add left-down corner node
        nodes.append({
            "id": "center-1",
            "x": 0,
            "y": 0,
            "type": node_type,
        })

        nodes.append({
            "id": "center-2",
            "x": 0,
            "y": 0,
            "type": node_type,
        })

        # Add right-up coner node
        nodes.append({
            "id": "center{}".format(self.row_num * self.col_num),
            "x": self.col_num * self.inner_length + self.inner_length,
            "y": (self.row_num - 1) * self.inner_length,
            "type": node_type,
        })

        nodes.append({
            "id": "center{}".format(self.row_num * self.col_num + 1),
            "x": self.col_num * self.inner_length + self.inner_length,
            "y": (self.row_num - 1) * self.inner_length,
            "type": node_type,
        })

        inserted_nodes = []

        for row in range(self.row_num):
            for col in range(self.col_num):
                for n in range(self.sub_edge_num - 1):
                    cur_id = row * self.col_num + col

                    if col < self.col_num - 1:
                        inserted_nodes.append({
                            "id": "{}-{}_{}".format(cur_id, cur_id + 1, n),
                            "x": col * self.inner_length + self.inner_length * (n + 1) / self.sub_edge_num + self.inner_length,
                            "y": row * self.inner_length,
                            "type": "priority"
                        })
                    if row < self.row_num - 1:
                        inserted_nodes.append({
                            "id": "{}-{}_{}".format(cur_id, cur_id + self.col_num, n),
                            "x": col * self.inner_length + self.inner_length,
                            "y": row * self.inner_length + self.inner_length * (n + 1) / self.sub_edge_num,
                            "type": "priority"
                        })

        # Add left-down and right-up part sub-nodes
        for n in range(self.sub_edge_num - 1):
            inserted_nodes.append({
                "id": "-1-0_{}".format(n),
                "x": self.inner_length * (n + 1) / self.sub_edge_num,
                "y": 0,
                "type": "priority"
            })
            inserted_nodes.append({
                "id": "-2-0_{}".format(n),
                "x": self.inner_length * (n + 1) / self.sub_edge_num,
                "y": 0,
                "type": "priority"
            })
            inserted_nodes.append({
                "id": "{}-{}_{}".format(self.row_num * self.col_num - 1, self.row_num * self.col_num, n),
                "x": self.inner_length * self.col_num + self.inner_length * (n + 1) / self.sub_edge_num,
                "y": (self.row_num - 1) * self.inner_length,
                "type": "priority"
            })
            inserted_nodes.append({
                "id": "{}-{}_{}".format(self.row_num * self.col_num - 1, self.row_num * self.col_num + 1, n),
                "x": self.inner_length * self.col_num + self.inner_length * (n + 1) / self.sub_edge_num,
                "y": (self.row_num - 1) * self.inner_length,
                "type": "priority"
            })

        return nodes + inserted_nodes

    def specify_edges(self, net_params):
        """Build out the inner edges of the network.

        The inner edges are the edges joining the inner nodes to each other.

        Consider the following network with n = 2 rows and m = 3 columns,
        where the rows are indexed from 0 to 1 and the columns from 0 to 2, and
        the inner nodes are marked by 'x':

        (1)     x-----x-----x----x
                |     |     |    |
        (0) ----x-----x-(*)-x    |
                |     |     |    |
        (-1)   (0)   (1)   (2)  (3)

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
                node_list = ["center{}".format(from_node)] + ["{}-{}_{}".format(from_node, to_node, n) for n in
                                                              range(self.sub_edge_num - 1)] + [
                                "center{}".format(to_node)]
            else:
                node_list = ["center{}".format(from_node)] + ["{}-{}_{}".format(to_node, from_node, n) for n in
                                                              range(self.sub_edge_num - 2, -1, -1)] + [
                                "center{}".format(to_node)]

            new_edges = []
            for i, node in enumerate(node_list):
                if i + 1 < len(node_list):
                    new_edges.append({
                        "id": "{}{}_{}".format(lane, index, i),
                        "type": orientation,
                        "priority": 78,  # Why 78?
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

        # Build the two added horizontal inner edges

        index = "{}_{}".format(0, 0)
        edges += new_edge(index, 0, -1, "horizontal", "top")
        edges += new_edge(index, -2, 0, "horizontal", "bot")

        index = "{}_{}".format(self.row_num - 1, self.col_num)
        edges += new_edge(index, self.col_num * self.row_num, self.col_num * self.row_num - 1, "horizontal", "top")
        edges += new_edge(index, self.col_num * self.row_num - 1, self.col_num * self.row_num + 1, "horizontal", "bot")

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

        # specify certain connections for given edge pair
        # In form of {"from" : edge id,
        #             "to" : edge id, \
        #             "fromLane": (sub-edge of a certain edge and it is by default 0) number between 0 to self.lanes - 1,
        #             "toLane": number between 0 to self.lanes - 1,
        #             }

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
        for node_id in range(-2, self.row_num * self.col_num + 2):
            conn = []
            i = node_id // self.col_num
            j = node_id % self.col_num
            top_edge_id = "{}_{}".format(i + 1, j) if i + 1 < self.row_num else None
            bot_edge_id = "{}_{}".format(i, j) if i > 0 else None
            left_edge_id = "{}_{}".format(i, j) if j > 0 else None
            right_edge_id = "{}_{}".format(i, j + 1) if j + 1 < self.col_num else None

            if (node_id == -1 or node_id == -2):
                top_edge_id = None
                bot_edge_id = None
                left_edge_id = None
                right_edge_id = None
            elif (node_id == 0):
                top_edge_id = "1_0"
                bot_edge_id = None
                left_edge_id = "0_0"
                right_edge_id = "0_1"
            elif (node_id == self.row_num * self.col_num - 1):
                top_edge_id = None
                right_edge_id = "{}_{}".format(self.row_num - 1, self.col_num)
            elif (node_id == self.row_num * self.col_num or node_id == (self.col_num * self.row_num + 1)):
                top_edge_id = None
                bot_edge_id = None
                left_edge_id = None
                right_edge_id = None

            assert self.vertical_lanes == self.horizontal_lanes
            if right_edge_id is not None and top_edge_id is not None:
                conn += new_con('top', right_edge_id, top_edge_id, None, 'right', 0, 0)
                conn += new_con('left', top_edge_id, right_edge_id, None, 'bot', 0, 0)
            if top_edge_id is not None and left_edge_id is not None:
                conn += new_con('left', top_edge_id, left_edge_id, None, 'top', 0, self.sub_edge_num - 1)
                conn += new_con('bot', left_edge_id, top_edge_id, None, 'right', self.sub_edge_num - 1, 0)
            if bot_edge_id is not None and right_edge_id is not None:
                conn += new_con('right', bot_edge_id, right_edge_id, None, "bot", self.sub_edge_num - 1, 0)
                conn += new_con('top', right_edge_id, bot_edge_id, None, "left", 0, self.sub_edge_num - 1)
            if bot_edge_id is not None and left_edge_id is not None:
                conn += new_con('right', bot_edge_id, left_edge_id, None, "top", self.sub_edge_num - 1,
                                self.sub_edge_num - 1)
                conn += new_con('bot', left_edge_id, bot_edge_id, None, "left", self.sub_edge_num - 1,
                                self.sub_edge_num - 1)

            if top_edge_id is not None and bot_edge_id is not None:
                conn += new_con('right', bot_edge_id, top_edge_id, 2)
                conn += new_con('left', top_edge_id, bot_edge_id, 2)
            if left_edge_id is not None and right_edge_id is not None:
                conn += new_con('bot', left_edge_id, right_edge_id, 1)
                conn += new_con('top', right_edge_id, left_edge_id, 1)

            if top_edge_id is not None:
                conn += new_con('left', top_edge_id, top_edge_id, None, 'right', 0, 0, True)
                for n in range(self.sub_edge_num - 1):
                    conn += new_con('left', top_edge_id, top_edge_id, None, 'left', n + 1, n)
            if bot_edge_id is not None:
                conn += new_con('right', bot_edge_id, bot_edge_id, None, 'left', self.sub_edge_num - 1,
                                self.sub_edge_num - 1, True)
                for n in range(self.sub_edge_num - 1):
                    conn += new_con('right', bot_edge_id, bot_edge_id, None, 'right', n, n + 1)
            if left_edge_id is not None:
                conn += new_con('bot', left_edge_id, left_edge_id, None, 'top', self.sub_edge_num - 1,
                                self.sub_edge_num - 1, True)
                for n in range(self.sub_edge_num - 1):
                    conn += new_con('bot', left_edge_id, left_edge_id, None, 'bot', n, n + 1)
            if right_edge_id is not None:
                conn += new_con('top', right_edge_id, right_edge_id, None, 'bot', 0, 0, True)
                for n in range(self.sub_edge_num - 1):
                    conn += new_con('top', right_edge_id, right_edge_id, None, 'top', n + 1, n)

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

        for node_id in range(-2, self.row_num * self.col_num + 2):
            conn = []
            if (node_id == -1 or node_id == (self.row_num * self.col_num + 1)):
                mapping[node_id] = []
            elif (node_id == -2):
                right_edge_id = "bot0_0"
                node_id = "center{}".format(node_id)
                mapping[node_id] = [right_edge_id]
            elif (node_id == 0):
                right_edge_id = "bot0_1"
                left_edge_id = "top0_0"
                top_edge_id = "right1_0"
                node_id = "center{}".format(node_id)
                mapping[node_id] = [left_edge_id, right_edge_id, top_edge_id]
            elif (node_id == self.row_num * self.col_num):
                left_edge_id = "top{}_{}".format(self.row_num - 1, self.col_num)
                node_id = "center{}".format(node_id)
                mapping[node_id] = [left_edge_id]
            elif (node_id == self.col_num * self.row_num - 1):
                left_edge_id = "top{}_{}".format(self.row_num - 1, self.col_num - 1)
                right_edge_id = "bot{}_{}".format(self.row_num - 1, self.col_num)
                bot_edge_id = "left{}_{}".format(self.row_num - 1, self.col_num - 1)
                node_id = "center{}".format(node_id)
                mapping[node_id] = [left_edge_id, bot_edge_id, right_edge_id]
            else:
                i = node_id // self.col_num
                j = node_id % self.col_num
                top_edge_id = "right{}_{}".format(i + 1, j) if i + 1 < self.row_num else None
                bot_edge_id = "left{}_{}".format(i, j) if i > 0 else None
                left_edge_id = "top{}_{}".format(i, j) if j > 0 else None
                right_edge_id = "bot{}_{}".format(i, j + 1) if j + 1 < self.col_num else None
                node_id = "center{}".format(node_id)
                mapping[node_id] = [left_edge_id, bot_edge_id, right_edge_id, top_edge_id]

        return sorted(mapping.items(), key=lambda x: x[0])