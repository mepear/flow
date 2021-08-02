import numpy as np

def gen_request(env):
    tp = 0
    if env.distribution == 'random':
        idx = env.k.person.total
        in_edge_list = [edge for edge in env.edges.copy() if 'out' not in edge and 'in' not in edge]
        edge_id1 = np.random.choice(in_edge_list)
        # out_edge_list = [edge for edge in env.edges.copy() if 'in' not in edge]
        # for debug 
        # out_edge_list = [edge for edge in env.edges.copy() if 'out' in edge]
        out_edge_list = [edge for edge in env.edges.copy() if 'out' not in edge and 'in' not in edge]
        
        if edge_id1 in out_edge_list:
            out_edge_list.remove(edge_id1)
        edge_id2 = np.random.choice(out_edge_list)

        per_id = 'per_' + str(idx)
        if 'in' not in edge_id1:
            pos = np.random.uniform(20, env.inner_length - 20)
        else:
            pos = env.outer_length - 1
    elif env.distribution == 'mode-1': 
        # the request only appears at one edge
        idx = env.k.person.total
        edge_list = env.edges.copy()
        edge_id1 = 'bot3_1_0'
        edge_list.remove(edge_id1)
        edge_id2 = 'top2_2_0'

        per_id = 'per_' + str(idx)
        pos = np.random.uniform(20, env.inner_length - 20)
    elif env.distribution == 'mode-11': 
        # the request only appears at one edge
        idx = env.k.person.total
        edge_list = env.edges.copy()
        edge_id1 = 'bot3_1_0'
        edge_list.remove(edge_id1)
        edge_id2 = 'top1_2_0'

        per_id = 'per_' + str(idx)
        pos = np.random.uniform(20, env.inner_length - 20)
    elif env.distribution == 'mode-12': 
        # the request only appears at one edge
        idx = env.k.person.total
        edge_list = env.edges.copy()
        edge_id1 = 'bot3_1_0'
        edge_list.remove(edge_id1)
        edge_id2 = 'top2_3_0'

        per_id = 'per_' + str(idx)
        pos = np.random.uniform(20, env.inner_length - 20)
    elif env.distribution == 'mode-13': 
        idx = env.k.person.total
        rn =  np.random.rand()
        edge_id1 = 'bot3_1_0'
        edge_id2 = 'left1_3_0' if rn < 0.5 else 'bot0_3_0'
        per_id = 'per_' + str(idx)
        pos = np.random.uniform(20, env.inner_length - 20)
    elif env.distribution == 'mode-14':
        idx = env.k.person.total
        rn =  np.random.rand()
        edge_id1 = np.random.choice(['bot3_1_0', 'top3_1_0', 'left3_0_0', 'right3_0_0', \
            'left3_1_0', 'right3_1_0', 'top2_1_0', 'bot2_1_0'])
        edge_id2 = 'left1_3_0' if rn < 0.5 else 'bot0_3_0'
        per_id = 'per_' + str(idx)
        pos = np.random.uniform(20, env.inner_length - 20)
    elif env.distribution == 'mode-15':
        idx = env.k.person.total
        edge_id1 = 'bot3_1_0'
        edge_id2 = 'top0_2_0'
        per_id = 'per_' + str(idx)
        pos = np.random.uniform(20, env.inner_length - 20)
    elif env.distribution == 'mode-X':
        # the request only appears at one edge
        idx = env.k.person.total
        rn =  np.random.rand()
        edge_id1 = 'bot3_1_0' if rn < 0.5 else 'top3_3_0'
        edge_id2 = 'top0_3_0' if rn < 0.5 else 'bot0_1_0'
        tp = 0 if rn < 0.5 else 1

        per_id = 'per_' + str(idx)
        pos = np.random.uniform(20, env.inner_length - 20)
    elif env.distribution == 'mode-X1':
        idx = env.k.person.total
        rn, rn2 =  np.random.rand(), np.random.rand()
        if rn < 0.5:
            edge_id1 = 'bot3_1_0'
            edge_id2 = 'left1_3_0' if rn2 < 0.5 else 'bot0_3_0'
            tp = 0
        else:
            edge_id1 = 'top3_3_0'
            edge_id2 = 'left1_0_0' if rn2 < 0.5 else 'top0_1_0'
            tp = 1
        per_id = 'per_' + str(idx)
        pos = np.random.uniform(20, env.inner_length - 20)
    elif env.distribution == 'mode-X1-1':
        idx = env.k.person.total
        rn, rn2 =  np.random.rand(), np.random.rand()
        if rn < 0.5:
            edge_id1 = 'bot3_1_0'
            edge_id2 = 'left1_3_0' if rn2 < 0.5 else 'bot0_3_0'
            tp = 0
        else:
            edge_id1 = 'bot2_1_0'
            edge_id2 = 'bot1_3_0' if rn2 < 0.5 else 'left2_3_0'
            tp = 1
        per_id = 'per_' + str(idx)
        pos = np.random.uniform(20, env.inner_length - 20)
    elif env.distribution == 'mode-X2':
        idx = env.k.person.total
        t =  env.time_counter / env.env_params.sims_per_step / env.env_params.horizon
        rn = np.random.rand()
        if t < 0.5:
            edge_id1 = 'bot3_1_0'
            edge_id2 = 'left1_3_0' if rn < 0.5 else 'bot0_3_0'
            tp = 0
        else:
            edge_id1 = 'top3_3_0'
            edge_id2 = 'left1_0_0' if rn < 0.5 else 'top0_1_0'
            tp = 1
        per_id = 'per_' + str(idx)
        pos = np.random.uniform(20, env.inner_length - 20)
    elif env.distribution == 'mode-X2-mid':
        idx = env.k.person.total
        t =  env.time_counter / env.env_params.sims_per_step / env.env_params.horizon
        rn = np.random.rand()
        if t < 0.5:
            edge_id1 = 'bot3_1_0'
            edge_id2 = 'top0_2_0'
            tp = 0
        else:
            edge_id1 = 'top3_3_0'
            edge_id2 = 'bot0_2_0'
            tp = 1
        per_id = 'per_' + str(idx)
        pos = np.random.uniform(20, env.inner_length - 20)
    elif env.distribution == 'mode-X3':
        idx = env.k.person.total
        rn, rn2 =  np.random.rand(), np.random.rand()
        if rn < 1 / 3:
            edge_id1 = 'bot3_1_0'
            edge_id2 = 'left1_3_0' if rn2 < 0.5 else 'bot0_3_0'
            tp = 0
        elif rn < 2 / 3:
            edge_id1 = 'bot2_1_0'
            edge_id2 = 'left1_3_0' if rn2 < 0.5 else 'bot0_3_0'
            tp = 1
        else:
            edge_id1 = 'bot1_1_0'
            edge_id2 = 'left1_3_0' if rn2 < 0.5 else 'bot0_3_0'
            tp = 2
        per_id = 'per_' + str(idx)
        pos = np.random.uniform(20, env.inner_length - 20)
    elif env.distribution == 'mode-2':
        # the request only appears at two different edges
        idx = env.k.person.total
        edge_list = env.edges.copy()
        rn =  np.random.rand()
        edge_id1 = 'bot3_1_0' if rn < 0.5 else 'top0_3_0'
        edge_list.remove(edge_id1)
        edge_id2 = 'top2_3_0' if rn < 0.5 else 'bot1_1_0'

        per_id = 'per_' + str(idx)
        pos = np.random.uniform(20, env.inner_length - 20)
    elif env.distribution == 'mode-3':
        # the request appears at one edge before half of the time
        # the request appears at another edge after half of the time
        idx = env.k.person.total
        edge_list = env.edges.copy()
        time_ratio = env.time_counter / env.env_params.horizon
        edge_id1 = 'bot3_1_0' if time_ratio < 0.5 else 'bot0_3_0'
        edge_list.remove(edge_id1)
        edge_id2 = 'top2_1_0' if time_ratio < 0.5 else 'top1_3_0'

        per_id = 'per_' + str(idx)
        pos = np.random.uniform(env.inner_length)
    elif env.distribution == 'mode-4':
        if env.time_counter % 20 != 1:
            return
        # the request appears at one edge before half of the time
        # the request appears at another edge after half of the time
        idx = env.k.person.total
        edge_list = env.edges.copy()
        time_ratio = env.time_counter / env.env_params.horizon
        edge_id1 = 'bot3_1_0' if time_ratio < 0.5 else 'bot0_3_0'
        edge_list.remove(edge_id1)
        edge_id2 = 'top2_1_0' if time_ratio < 0.5 else 'top1_3_0'

        per_id = 'per_' + str(idx)
        # pos = np.random.uniform(env.inner_length)
        pos = env.time_counter % env.grid_array['inner_length']
    elif env.distribution == 'mode-5':
        if env.time_counter % 5 != 1:
            return
        # the request appears at one edge before half of the time
        # the request appears at another edge after half of the time
        idx = env.k.person.total
        edge_list = env.edges.copy()
        time_ratio = env.time_counter / env.env_params.horizon
        edge_id1 = 'bot3_1_0' if time_ratio < 0.5 else 'bot0_3_0'
        edge_list.remove(edge_id1)
        edge_id2 = 'top2_1_0' if time_ratio < 0.5 else 'top1_3_0'

        per_id = 'per_' + str(idx)
        # pos = np.random.uniform(env.inner_length)
        pos = env.time_counter % env.grid_array['inner_length']
    elif env.distribution == "random+mode-X2":
        if np.random.rand() < env.distribution_random_ratio:
            idx = env.k.person.total
            in_edge_list = [edge for edge in env.edges.copy() if 'out' not in edge and 'in' not in edge]
            edge_id1 = np.random.choice(in_edge_list)
            out_edge_list = [edge for edge in env.edges.copy() if 'out' not in edge and 'in' not in edge]
            
            if edge_id1 in out_edge_list:
                out_edge_list.remove(edge_id1)
            edge_id2 = np.random.choice(out_edge_list)

            per_id = 'per_' + str(idx)
            if 'in' not in edge_id1:
                pos = np.random.uniform(20, env.inner_length - 20)
            else:
                pos = env.outer_length - 1
            tp = 2
        else:
            idx = env.k.person.total
            t =  env.time_counter / env.env_params.sims_per_step / env.env_params.horizon
            rn = np.random.rand()
            if t < 0.5:
                edge_id1 = 'bot3_1_0'
                edge_id2 = 'left1_3_0' if rn < 0.5 else 'bot0_3_0'
                tp = 0
            else:
                edge_id1 = 'top3_3_0'
                edge_id2 = 'left1_0_0' if rn < 0.5 else 'top0_1_0'
                tp = 1
            per_id = 'per_' + str(idx)
            pos = np.random.uniform(20, env.inner_length - 20)
    else:
        raise NotImplementedError
    return per_id, edge_id1, edge_id2, pos, tp