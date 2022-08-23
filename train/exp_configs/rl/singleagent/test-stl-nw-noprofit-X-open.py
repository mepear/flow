"""Grid example."""
from flow.controllers import IDMController, RLController
from flow.controllers.routing_controllers import MinicityRouter, OpenRouter_Flow, OpenRouter_Inner
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, PersonParams
from flow.core.params import TrafficLightParams
from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.params import InFlows
from flow.envs.dispatch_and_reposition import DispatchAndRepositionEnv, ADDITIONAL_ENV_PARAMS
from flow.networks import GridnxmNetworkOpen

v_enter = 10
inner_length = 50
n_rows = 4
n_columns = 4

grid_array = {
    "inner_length": inner_length,
    "row_num": n_rows,
    "col_num": n_columns,
    "sub_edge_num": 1
}


def get_non_flow_params(enter_speed, inflows, add_net_params):
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

    edges_distribution = []

    for i in range(1, 4):
        for j in range(4):
            edges_distribution.append("bot{}_{}_0".format(j, i))
            edges_distribution.append("top{}_{}_0".format(j, i))
            edges_distribution.append("left{}_{}_0".format(i, j))
            edges_distribution.append("right{}_{}_0".format(i, j))

    initial = InitialConfig(
        x0=2.5, spacing='uniform', min_gap=10, additional_params=additional_init_params, edges_distribution=edges_distribution) # gap needs to be large enough
    net = NetParams(inflows=inflows, additional_params=add_net_params)

    return initial, net

persons = PersonParams()
vehicles = VehicleParams()

# These vehicles is added from entrance of this network to the exit of this network
vehicles.add(
    veh_id="idm",
    acceleration_controller=(IDMController, {}),
    routing_controller=(OpenRouter_Flow, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='all_checks',
        min_gap=5,
        decel=10.0,  # avoid collisions at emergency stops
        max_speed=10,
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="no_lc_safe",
    ),
    initial_speed=0,
    num_vehicles=0,
    color="blue")

# Theses vehicles can only stay in this network thus they shouldn't be created in some out paths and need special router
vehicles.add(
    veh_id="inner",
    acceleration_controller=(IDMController, {}),
    routing_controller=(OpenRouter_Inner, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='all_checks',
        min_gap=5,
        decel=10.0,  # avoid collisions at emergency stops
        max_speed=10,
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="no_lc_safe",
    ),
    initial_speed=0,
    num_vehicles=25)

# These vehicles are taxis
vehicles.add(
    veh_id="taxi",
    initial_speed=0,
    acceleration_controller=(RLController, {}),
    # routing_controller=(MinicityRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='all_checks',
        min_gap=5,
        decel=10.0,  # avoid collisions at emergency stops
        max_speed=10,
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="sumo_default",
    ),
    num_vehicles=20,
    is_taxi=False)

tl_logic = TrafficLightParams(baseline=False)
phases = [{
    "duration": "10",
    "minDur": "10",
    "maxDur": "10",
    "state": "GGggrrrrGGggrrrr"
}, {
    "duration": "1",
    "minDur": "1",
    "maxDur": "1",
    "state": "yyyyrrrryyyyrrrr"
}, {
    "duration": "10",
    "minDur": "10",
    "maxDur": "10",
    "state": "rrrrGGggrrrrGGgg"
}, {
    "duration": "1",
    "minDur": "1",
    "maxDur": "1",
    "state": "rrrryyyyrrrryyyy"
}]
tl_logic.add("center9", phases=phases)
tl_logic.add("center10", phases=phases)
tl_logic.add("center5", phases=phases)
tl_logic.add("center6", phases=phases)

additional_net_params = {
    "grid_array": grid_array,
    "speed_limit": 35,
    "horizontal_lanes": 1,
    "vertical_lanes": 1,
    "print_warnings": False, # warnings in building net
}

inflows = InFlows()
inflows.add('top3_4_0', 'idm', probability=0.1, depart_speed='random', name='idm')

initial_config, net_params = get_non_flow_params(
    enter_speed=v_enter,
    add_net_params=additional_net_params,
    inflows=inflows)

additional_params = ADDITIONAL_ENV_PARAMS.copy()
additional_params["time_price"] = -0.0075
additional_params["distance_price"] = 0
additional_params["pickup_price"] = 1
additional_params["wait_penalty"] = 0.000
additional_params["tle_penalty"] = 0.0075
additional_params["person_prob"] = 0.06
additional_params["max_waiting_time"] = 30
additional_params["free_pickup_time"] = 0.0
additional_params["distribution"] = 'mode-6'
additional_params["n_mid_edge"] = 1
flow_params = dict(
    # name of the experiment
    exp_tag='grid-intersection',

    # name of the flow environment the experiment is running on
    env_name=DispatchAndRepositionEnv,

    # name of the network class the experiment is running on
    network=GridnxmNetworkOpen,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=1,
        render=False,
        print_warnings=False,
        restart_instance=True
        # taxi_dispatch_alg="greedy"
    ),

    # environment related parameters (see flow.core.params.EnvParams)

    env=EnvParams(
        horizon=500,
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
    tls=tl_logic,
)