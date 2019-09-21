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

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--file', '-f', type=str, default='protein.edgelist.txt')
    args = parser.parse_args()
    return args


def load_data(fname):
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
        for i,j in edges:
            A[i][j] = 1
    else:
        for i,j in edges:
            A[j][i] = A[i][j] = 1

    return A, csr_matrix(A)


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
    f = np.poly1d(z)
    plt.title('{} degree distribution'.format(title))
    plt.scatter(log_x, log_y)
    plt.plot(log_x, f(log_x), 'tab:orange')

    plt.xlabel('log of Degrees (d), base e')
    plt.ylabel('log of Frequency, base e')
    plt.show()


def plot_degree_distribution():
    if not directed:
        degrees = np.sum(A, axis=1).astype(int)

        calc_degree(degrees)
    else:
        out_degrees = np.sum(A, axis=1).astype(int)
        in_degrees = np.sum(A, axis=0).astype(int)

        calc_degree(out_degrees, "out")
        calc_degree(in_degrees, "in")


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
    cc, count = np.unique(cs, return_counts=True)

    plt.title('Clustering coefficient distribution')
    plt.scatter(cc, count)

    plt.xlabel('Clustering coefficient')
    plt.ylabel('Frequency')
    plt.show()

    print("Average clustering coefficient: {}".format(np.mean(cs)))
    return cs, degrees


def plot_shortest_path_dist():

    D = csgraph.shortest_path(A_sparse)
    D = D[np.isfinite(D)]

    unique, counts = np.unique(D.flatten(), return_counts=True)
    plt.scatter(unique[1:], counts[1:])
    plt.xlabel('Shorted path length')
    plt.ylabel('Frequency')
    plt.show()
    average = np.mean(np.mean(D))
    print("Average distance: {}".format(average))
    return D


def plot_connected_components():
    n_cc, ccs = csgraph.connected_components(A_sparse)
    _, counts = np.unique(ccs, return_counts=True)
    pdb.set_trace()
    print("Number of CC's: {}. Proportion of nodes in GCC: {}.".format(n_cc, np.max(counts)/node_size))


def plot_spectral_gap():
    D = np.diag(A.sum(axis=1))
    L = D - A
    # eigenvalues and eigenvectors
    vals, vecs = eigs(csr_matrix(L))
    pdb.set_trace()
    vals = vals[np.argsort(vals)]
    plt.scatter(range(len(vals)), vals, s=2)

    plt.ylabel('eigenvalue')
    plt.show()

    spectral_gap = vals[vals != 0][0]
    print("Spectral gap: {}".format(spectral_gap))
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

    for i in range(node_size):
        N_i = A[i].nonzero()[0]
        ki = degrees[i]
        for j in N_i:
            kj = degrees[j]
            DC[ki][kj] += 1
    DC /= edge_count


    with sns.axes_style("white"):
        ax = sns.heatmap(DC, cmap=sns.cm.rocket_r)
        ax.invert_yaxis()
        plt.show()
    return degrees

def plot_degree_clustering(cc, degrees):
    cc = 100*cc
    max_degree = int(np.max(np.unique(degrees)))
    max_cc = int(np.max(np.unique(cc)))
    # DCC = np.zeros((max_degree+1, max_cc+1))
    # for i in range(len(degrees)):
    #     ki = int(degrees[i])
    #     cci = int(cc[i])
    #     DCC[ki, cci] += 1

    heatmap, xedges, yedges = np.histogram2d(cc, degrees, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()

    # with sns.axes_style("white"):
    #     ax = sns.heatmap(DCC, cmap=sns.cm.rocket_r)
    #     ax.invert_yaxis()
    #     plt.show()

if __name__ == '__main__':
    args = get_args()
    fname = os.path.join('networks', args.file)
    directed = args.file in directed_fnames
    A, A_sparse = load_data(fname)
    if directed:
        A_undirected = np.maximum(A, A.transpose())
        A_undirected_sparse = csr_matrix(A_undirected)

    node_size = A.shape[0]
    # plot_degree_distribution()
    cc, degrees = plot_clustering_coeff()
    # plot_shortest_path_dist()
    # plot_connected_components()
    # plot_spectral_gap()
    # plot_degree_correlation()
    plot_degree_clustering(cc, degrees)
    
