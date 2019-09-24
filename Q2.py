from Q1 import *


def init_graph():
    degrees = np.zeros(node_count)
    A_new = np.zeros((node_count, node_count))
    degrees[0] = 2
    A_new[0][0] += 1
    return degrees, A_new


def add_node(t):
    sum_degrees = 2 + (2*t-1)*m
    distribution = np.append(degrees[:t]/sum_degrees, m/sum_degrees)
    target_nodes = np.random.choice(range(t+1), m, replace=False, p=distribution)
    degrees[t] += m
    for target in target_nodes:
        degrees[target] += 1
        A_new[t][target] += 1
        if not target == t:
            A_new[target][t] += 1


def add_nodes():
    for t in range(1, node_count):
        add_node(t)


if __name__ == '__main__':
    args = get_args()
    fname = os.path.join('networks', args.file)
    out_fname = args.file.split('.')[0]

    directed = args.file in directed_fnames
    A, A_sparse, node_count, edge_count = load_data(fname, directed)
    m = int(edge_count/node_count)
    print(m)
    degrees, A_new = init_graph()
    add_nodes()

    if not os.path.exists('generated'):
        os.makedirs('generated')
    print("Number of edges: {}".format(sum(degrees)/2))
    np.save(os.path.join('generated', out_fname), A_new)