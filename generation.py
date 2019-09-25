from Q1 import *
import scipy


def init_graph():
    degrees = np.zeros(node_count)
    A_new = np.zeros((node_count, node_count))
    sum_degrees = 0
    for i in range(m0):
        degrees[i] = 2
        sum_degrees += 2
        A_new[i, i] += 1
    return degrees, A_new, sum_degrees


def add_node(t):
    global sum_degrees
    cur_m = m + np.random.choice([0,1], p=[p0, p1])


    distribution = np.append(degrees[:t], cur_m)/(sum_degrees + cur_m)
    sum_degrees += 2 * cur_m
    target_nodes = np.random.choice(range(t+1), cur_m, replace=False, p=distribution)
    degrees[t] += cur_m
    for target in target_nodes:
        degrees[target] += 1
        A_new[t, target] += 1
        if not target == t:
            A_new[target, t] += 1


def add_nodes():
    for t in range(m0, node_count):
        add_node(t)

def bonus_add_node(t):
    global sum_degrees
    e = 0.6
    cur_m = m + np.random.choice([0, 1], p=[p0, p1])
    switch = np.random.choice([0,1], p=[e, 1-e])
    if switch == 0:

        unnormalized = 1 / np.append(degrees[:t], cur_m)
        distribution = unnormalized/unnormalized.sum()
        # distribution = 1 - np.append(degrees[:t], cur_m) / (sum_degrees + cur_m)
    else:
        distribution = np.append(degrees[:t], cur_m) / (sum_degrees + cur_m)
    sum_degrees += 2 * cur_m
    target_nodes = np.random.choice(range(t + 1), cur_m, replace=False, p=distribution)
    degrees[t] += cur_m
    for target in target_nodes:
        degrees[target] += 1
        A_new[t, target] += 1
        if not target == t:
            A_new[target, t] += 1

def bonus_add_nodes():
    for t in range(m0, node_count):
        bonus_add_node(t)


if __name__ == '__main__':
    args = get_args()
    fname = os.path.join('networks', args.file)
    out_fname = "{}_bonus.npz".format(args.file.split('.')[0])

    directed = args.file in directed_fnames
    A, A_sparse, node_count, edge_count = load_data(fname, directed)
    del A, A_sparse
    m0 = m = int(edge_count/node_count)
    offset = int(edge_count % node_count)
    p1 = offset/node_count
    p0 = 1 - p1
    print("m: {}, m0: {}".format(m, m0))
    degrees, A_new, sum_degrees = init_graph()
    # add_nodes()
    bonus_add_nodes()

    if not os.path.exists('generated'):
        os.makedirs('generated')
    print("Number of edges: {}".format(sum(degrees)/2))
    # pdb.set_trace()
    scipy.sparse.save_npz(os.path.join('generated', out_fname), csr_matrix(A_new))
