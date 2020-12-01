"""Grid example."""
from flow.controllers import GridRouter, IDMController, RLController
from flow.controllers.routing_controllers import MinicityRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, PersonParams
from flow.core.params import TrafficLightParams
from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.params import InFlows
# from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.envs.dispatch_and_reposition import DispatchAndRepositionEnv, ADDITIONAL_ENV_PARAMS
from flow.networks import GridnxmNetwork

USE_INFLOWS = False

v_enter = 10
inner_length = 100
n_rows = 4
n_columns = 4

grid_array = {
    "inner_length": inner_length,
    "row_num": n_rows,
    "col_num": n_columns,
    "sub_edge_num": 1
}


def gen_edges(col_num, row_num):
    """Generate the names of the outer edges in the grid network.

    Parameters
    ----------
    col_num : int
        number of columns in the grid
    row_num : int
        number of rows in the grid

    Returns
    -------
    list of str
        names of all the outer edges
    """
    edges = []

    # build the left and then the right edges
    for i in range(col_num):
        edges += ['left' + str(row_num) + '_' + str(i)]
        edges += ['right' + '0' + '_' + str(i)]

    # build the bottom and then top edges
    for i in range(row_num):
        edges += ['bot' + str(i) + '_' + '0']
        edges += ['top' + str(i) + '_' + str(col_num)]

    return edges


def get_flow_params(col_num, row_num, additional_net_params):
    """Define the network and initial params in the presence of inflows.

    Parameters
    ----------
    col_num : int
        number of columns in the grid
    row_num : int
        number of rows in the grid
    additional_net_params : dict
        network-specific parameters that are unique to the grid

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the network
    """
    initial = InitialConfig(spacing="random", min_gap=5)

    inflow = InFlows()
    outer_edges = gen_edges(col_num, row_num)
    for i in range(len(outer_edges)):
        inflow.add(
            veh_type='human',
            edge=outer_edges[i],
            probability=0.25,
            departLane='free',
            departSpeed=20)

    net = NetParams(
        inflows=inflow,
        additional_params=additional_net_params)

    return initial, net


def get_non_flow_params(enter_speed, add_net_params):
    """Define the network and initial params in the absence of inflows.

    Note that when a vehicle leaves a network in this case, it is immediately
    returns to the start of the row/column it was traversing, and in the same
    direction as it was before.

    Parameters
    ----------
    enter_speed : float
        initial speed of vehicles as they enter the network.
    add_net_params: dict
        additional network-specific parameters (unique to the grid)

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the network
    """
    additional_init_params = {'enter_speed': enter_speed}
    initial = InitialConfig(
        spacing='random', min_gap=10, additional_params=additional_init_params) # gap needs to be large enough
    net = NetParams(additional_params=add_net_params)

    return initial, net

persons = PersonParams()
vehicles = VehicleParams()

vehicles.add(
    veh_id="idm",
    acceleration_controller=(IDMController, {}),
    routing_controller=(MinicityRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='all_checks',
        min_gap=10.0,
        decel=10.0,  # avoid collisions at emergency stops
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="sumo_default",
    ),
    initial_speed=0,
    num_vehicles=20)
# vehicles.add(
#     veh_id="rl",
#     acceleration_controller=(RLController, {}),
#     routing_controller=(MinicityRouter, {}),
#     car_following_params=SumoCarFollowingParams(
#         speed_mode="obey_safe_speed",
#     ),
#     initial_speed=0,
#     num_vehicles=5)
vehicles.add(
    veh_id="taxi",
    initial_speed=1,
    acceleration_controller=(RLController, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='all_checks',
        min_gap=10.0,
        decel=10.0,  # avoid collisions at emergency stops
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="sumo_default",
    ),
    num_vehicles=10,
    is_taxi=True)

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

tl_logic = TrafficLightParams(baseline=False)
phases = [{
    "duration": "31",
    "minDur": "8",
    "maxDur": "45",
    "state": "GrGrGrGrGrGr"
}, {
    "duration": "6",
    "minDur": "3",
    "maxDur": "6",
    "state": "yryryryryryr"
}, {
    "duration": "31",
    "minDur": "8",
    "maxDur": "45",
    "state": "rGrGrGrGrGrG"
}, {
    "duration": "6",
    "minDur": "3",
    "maxDur": "6",
    "state": "ryryryryryry"
}]
tl_logic.add("center0", phases=phases, programID=1)
tl_logic.add("center1", phases=phases, programID=1)
tl_logic.add("center2", phases=phases, programID=1, tls_type="actuated")

additional_net_params = {
    "grid_array": grid_array,
    "speed_limit": 35,
    "horizontal_lanes": 2,
    "vertical_lanes": 2,
    "print_warnings": False, # warnings in building net
}

if USE_INFLOWS:
    initial_config, net_params = get_flow_params(
        col_num=n_columns,
        row_num=n_rows,
        additional_net_params=additional_net_params)
else:
    initial_config, net_params = get_non_flow_params(
        enter_speed=v_enter,
        add_net_params=additional_net_params)

additional_params = ADDITIONAL_ENV_PARAMS.copy()
additional_params['time_price'] = 1
additional_params['max_waiting_time'] = 1000
additional_params["person_prob"] = 0.01
flow_params = dict(
    # name of the experiment
    exp_tag='grid-intersection',

    # name of the flow environment the experiment is running on
    env_name=DispatchAndRepositionEnv,

    # name of the network class the experiment is running on
    network=GridnxmNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=False,
        print_warnings=False
        # taxi_dispatch_alg="greedy"
    ),

    # environment related parameters (see flow.core.params.EnvParams)

    env=EnvParams(
        horizon=1000,
        additional_params=additional_params,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=net_params,

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,
    per=persons,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=initial_config,

    # traffic lights to be introduced to specific nodes (see
    # flow.core.params.TrafficLightParams)
    # tls=tl_logic,
)
