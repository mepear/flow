"""Grid example."""
from flow.controllers import IDMController, RLController
from flow.controllers.routing_controllers import FlowRouter_Inner, IndexEnv_Router
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, PersonParams
from flow.core.params import TrafficLightParams
from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams
from flow.envs.dispatch_and_reposition_open import DispatchAndRepositionEnv_with_index, ADDITIONAL_ENV_PARAMS
from flow.networks import GridnxmNetworkExpand_with_Index

v_enter = 10
inner_length = 50
n_rows = 4
n_columns = 4

grid_array = {
    "inner_length": inner_length,
    "outer_length": inner_length // 2,
    "row_num": n_rows,
    "col_num": n_columns,
    "sub_edge_num": 1,
    "row_idx": 2,
    "col_idx": 2
}


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
        x0=2.5, spacing='uniform', min_gap=10, additional_params=additional_init_params, edges_distribution='inner') # gap needs to be large enough
    net = NetParams(additional_params=add_net_params)

    return initial, net

persons = PersonParams()
vehicles = VehicleParams()

# Theses vehicles can only stay in this network thus they shouldn't be created in some out paths and need special router
for idx in range(grid_array['row_idx'] * grid_array['col_idx']):
    vehicles.add(
        veh_id="inner_{}".format(idx),
        acceleration_controller=(IDMController, {}),
        routing_controller=(IndexEnv_Router, {}),
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
for idx in range(grid_array['row_idx'] * grid_array['col_idx']):
    vehicles.add(
        veh_id="taxi_{}".format(idx),
        initial_speed=0,
        acceleration_controller=(RLController, {}),
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

for i in range(4):
    tl_logic.add("center9_{}".format(i), phases=phases)
    tl_logic.add("center10_{}".format(i), phases=phases)
    tl_logic.add("center5_{}".format(i), phases=phases)
    tl_logic.add("center6_{}".format(i), phases=phases)

additional_net_params = {
    "grid_array": grid_array,
    "speed_limit": 35,
    "horizontal_lanes": 1,
    "vertical_lanes": 1,
    "print_warnings": False, # warnings in building net
}

initial_config, net_params = get_non_flow_params(
    enter_speed=v_enter,
    add_net_params=additional_net_params)

additional_params = ADDITIONAL_ENV_PARAMS.copy()
additional_params["time_price"] = 0.02
additional_params["distance_price"] = 0.02
additional_params["pickup_price"] = 1
additional_params["wait_penalty"] = 0.000
additional_params["tle_penalty"] = 0.02
additional_params["person_prob"] = 0.06
additional_params["max_waiting_time"] = 30
additional_params["free_pickup_time"] = 0.0
additional_params["distribution"] = 'mode-X1-open'
additional_params["n_mid_edge"] = 1
flow_params = dict(
    # name of the experiment
    exp_tag='grid-intersection',

    # name of the flow environment the experiment is running on
    env_name=DispatchAndRepositionEnv_with_index,

    # name of the network class the experiment is running on
    network=GridnxmNetworkExpand_with_Index,

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