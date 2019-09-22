import numpy as np
import os
import pdb
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigs
import seaborn as sns
directed_fnames = ['metabolic.edgelist.txt', 'citation.edgelist.txt', 'email.edgelist.txt', 'www.edgelist.txt']
from collections import Counter


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--all', action="store_true")
    parser.add_argument('-a', action="store_true")
    parser.add_argument('-b', action="store_true")
    parser.add_argument('-c', action="store_true")
    parser.add_argument('-d', action="store_true")
    parser.add_argument('-e', action="store_true")
    parser.add_argument('-f', action="store_true")
    parser.add_argument('--file', type=str, default='protein.edgelist.txt')
    args = parser.parse_args()
    return args


def load_data(fname, directed):
    verticies = set()
    edges = set()

    with open(fname, "r") as f:
        lines = f.readlines()
        for line in lines:
            source = int(line.split('\t')[0])
            target = int(line.split('\t')[1])
            verticies.add(source)
            verticies.add(target)
            edges.add((source, target))

    A = np.zeros((len(verticies), len(verticies)))

    print("Number of verticies: {}".format(len(verticies)))
    print("Number of edges: {}".format(len(edges)))

    if directed:
        for i, j in edges:
            A[i][j] = 1
    else:
        for i, j in edges:
            A[j][i] = A[i][j] = 1

    return A, csr_matrix(A), len(verticies), len(edges)


def calc_degree(degrees, title=""):
    max_degree = np.log(np.max(degrees))
    bins = np.logspace(0, max_degree, 30, base=np.e)

    counts, values = np.histogram(degrees, bins=bins)
    values = values[:-1]

    non_zero_idx = np.where(counts != 0)[0]
    non_zero_counts = counts[non_zero_idx]
    non_zero_values = values[non_zero_idx]
    log_x = np.log(non_zero_values); log_y = np.log(non_zero_counts)
    z = np.polyfit(log_x, log_y, deg=1)
    print(z)
    f = np.poly1d(z)
    plt.title('{}degree distribution'.format(title))
    plt.scatter(log_x, log_y)
    plt.plot(log_x, f(log_x), 'tab:orange')

    plt.xlabel('log of Degrees (d), base e')
    plt.ylabel('log of Frequency, base e')
    plt.savefig(os.path.join(out_folder, '{}degree distribution'.format(title)))
    plt.clf()

    with open(os.path.join(out_folder, '{}slope.txt'.format(title)), "w") as f:
        f.write("slope: {}".format(z[0]))


def plot_degree_distribution():
    if not directed:
        degrees = np.sum(A, axis=1).astype(int)
        calc_degree(degrees)
    else:
        out_degrees = np.sum(A, axis=1).astype(int)
        in_degrees = np.sum(A, axis=0).astype(int)

        calc_degree(out_degrees, "out ")
        calc_degree(in_degrees, "in ")


def plot_clustering_coeff():
    if not directed:
        A_3 = A_sparse.multiply(A_sparse.multiply(A_sparse)).toarray()
        degrees = A.sum(axis=1)
    else:
        A_3 = A_undirected_sparse.multiply(A_undirected_sparse.multiply(A_undirected_sparse)).toarray()
        degrees = A_undirected.sum(axis=1)

    A_diag = A_3.diagonal()
    cs = 2 * A_diag / (degrees * (degrees - 1))
    mask = np.isfinite(cs)
    cs = cs[mask]
    degrees = degrees[mask]
    plt.title('Clustering coefficient distribution')
    plt.hist(cs, log=True, bins=50)

    plt.xlabel('Clustering coefficient')
    plt.ylabel('Frequency(log scale)')
    plt.savefig(os.path.join(out_folder, 'Clustering coefficient distribution'))
    plt.clf()

    with open(os.path.join(out_folder, 'avg_clustering_coeff.txt'), "w") as f:
        f.write("Average clustering coefficient: {}".format(np.mean(cs)))
    return cs, degrees


def plot_shortest_path_dist():
    D = csgraph.shortest_path(A_sparse)
    D = D[np.isfinite(D)]

    plt.title('Shortest path lengths distribution')
    unique, counts = np.unique(D.flatten(), return_counts=True)
    plt.scatter(unique[1:], counts[1:])
    plt.xlabel('Lengths of shorted paths')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(out_folder, 'Shortest path lengths distribution'))
    plt.clf()
    average = np.mean(np.mean(D))

    with open(os.path.join(out_folder, 'avg_shortest_path_distance.txt'), "w") as f:
        f.write("Average shortest path distance: {}".format(average))
    return D


def plot_connected_components():
    n_cc, ccs = csgraph.connected_components(A_sparse)

    values, counts = np.unique(ccs, return_counts=True)
    max_count = np.max(counts)

    plt.title('Connected components distribution')
    plt.scatter(range(1, len(counts) + 1), counts/node_count, s=2)

    plt.xlabel("Connected components")
    plt.ylabel("Proportions of nodes in each CC")
    plt.savefig(os.path.join(out_folder, 'connected components distribution'))
    plt.clf()

    with open(os.path.join(out_folder, 'connected_components.txt'), "w") as f:
        f.write("Number of CC's: {}. Proportion of nodes in GCC: {}.".format(n_cc, max_count / node_count))


def plot_spectral_gap():
    D = np.diag(A.sum(axis=1))
    L = D - A
    # eigenvalues and eigenvectors
    vals, vecs = eigs(csr_matrix(L))
    vals = vals[np.argsort(vals)]

    plt.title('Eigenvalue distribution distribution')
    plt.scatter(range(1, len(vals) + 1), vals, s=5)

    plt.ylabel('eigenvalue')
    plt.savefig(os.path.join(out_folder, 'eigenvalue distribution distribution'))
    plt.clf()

    spectral_gap = vals[vals != 0][0]
    with open(os.path.join(out_folder, 'spectral_gap.txt'), "w") as f:
        f.write("Spectral gap: {}".format(spectral_gap))
    return spectral_gap


def plot_degree_correlation():
    edge_count = A.sum()
    if directed:
        degrees = A_undirected.sum(axis=0).astype(int)
    else:
        degrees = A.sum(axis=0).astype(int)

    values = np.unique(degrees)
    max_degree = int(np.max(values))
    DC = np.zeros((max_degree+1, max_degree+1))

    for i in range(node_count):
        N_i = A[i].nonzero()[0]
        ki = degrees[i]
        for j in N_i:
            kj = degrees[j]
            DC[ki][kj] += 1
    DC /= edge_count

    plt.title('Degree correlations distribution')
    with sns.axes_style("white"):
        ax = sns.heatmap(DC, cmap=sns.cm.rocket_r)
        ax.invert_yaxis()
    plt.savefig(os.path.join(out_folder, 'eigenvalue distribution distribution'))
    plt.clf()


def plot_degree_clustering(cc, degrees):
    max_degree = int(np.max(np.unique(degrees)))
    max_cc = int(np.max(np.unique(cc)))
    DCC = np.zeros((max_degree+1, max_cc+1))

    for i in range(len(degrees)):
        ki = int(degrees[i])
        cci = int(cc[i])
        DCC[ki, cci] += 1

    zipped = list(zip(cc, degrees))

    cnt = Counter(zipped)
    values = cnt.keys()  # equals to list(set(words))
    counts = list(cnt.values())
    x = list(map(lambda x: x[0], values))
    y = list(map(lambda x: x[1], values))

    plt.title('Degree-clustering coefficient distribution')
    plt.scatter(x, y, s=counts)

    plt.xlabel("Degree")
    plt.ylabel("Clustering coefficients")
    plt.savefig(os.path.join(out_folder, 'degree clustering coeff distribution'))
    plt.clf()


if __name__ == '__main__':
    args = get_args()
    fname = os.path.join('networks', args.file)
    directed = args.file in directed_fnames
    A, A_sparse, node_count, edge_count = load_data(fname, directed)
    if directed:
        A_undirected = np.maximum(A, A.transpose())
        A_undirected_sparse = csr_matrix(A_undirected)

    out_folder = os.path.join("results", args.file.split(".")[0])
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if args.all:
        args.a = args.b = args.c = args.d = args.e = args.f = True
    if args.a:
        plot_degree_distribution()
    if args.b:
        cc, degrees = plot_clustering_coeff()
        plot_degree_clustering(cc, degrees)
    if args.c:
        plot_shortest_path_dist()
    if args.d:
        plot_connected_components()
    if args.e:
        plot_spectral_gap()
    if args.f:
        plot_degree_correlation()