import os
from igraph import *
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import svds

sys.path.append(os.path.abspath("../../"))
from constants import PATH_DIR, GRAPH_FILE

#methods = ['random', 'xnb_iterative', 'ens', 'degree_iterative']
#methods_str = ['RandN', 'NB', 'ENS', 'Degree']
methods = ['random', 'ens', 'degree_iterative']
methods_str = ['RandN', 'ENS', 'Degree']

markers = ['v','^', '|','o']
budgets = [1, 3, 5, 10, 20, 50]


def tick_function(n, X):
    V = X / n
    return ["%.4f" % z for z in V]


def plot_drop(first_lambda, gname, i):

    lambda_drop = [0] # 0 for no nodes removed
    method = methods[i]

    for budget in budgets:
        input_filename = os.path.join(PATH_DIR, 'rm_nodes/{}/graph-{}_b{}.pkl'.format(method, gname, budget))
        print("reading data from: ", input_filename)

        g = Graph.Read_Pickle(input_filename)
        
        A = g.get_adjacency_sparse()
        A = A.astype(float)
        U, S, Vt = svds(A, k = 1, which='LM')
        drop = 100 * (first_lambda[0] - S[0])/first_lambda[0]
        print("% eigen drop", drop)
        lambda_drop.append(drop)

    print([0] + budgets, lambda_drop, methods_str[i])

    plt.plot([0] + budgets, lambda_drop, label=methods_str[i], marker=markers[i], linewidth=2, markersize=6)


def plot_fragm(lcc_size_orig, gname, i):

    lcc_size_drop = []
    method = methods[i]

    for budget in budgets:
        input_filename = os.path.join(PATH_DIR, 'rm_nodes/{}/graph-{}_b{}.pkl'.format(method, gname, budget))
        print("reading data from: ", input_filename)

        g = Graph.Read_Pickle(input_filename)
        cluster_sizes = [len(cl.vs) for cl in g.clusters().subgraphs()]
        # print("budget, cluster  sizes: ", budget,  sorted(cluster_sizes, reverse=True)[:51])

        max_size = max(cluster_sizes)
        drop = max_size / lcc_size_orig
        print("frac lcc drop", drop)
        lcc_size_drop.append(drop)

    print(budgets, lcc_size_drop)

    plt.plot([0] + budgets, [1] + lcc_size_drop, marker = markers[i], label=methods_str[i], linewidth=2, markersize=6)


def set_axis(ax, lcc_size_orig, metric, fsize1, fsize2):
    plt.legend(methods_str, fontsize=fsize1)
    plt.xlabel("Nodes hardened (#)", fontsize=fsize1)
    if metric == 'lambda':
        plt.ylabel(r'$\Delta \lambda \%$', fontsize=fsize1)
        plt.ylim((-5, 105)) 
    else:
        plt.ylabel(r'$\sigma$', fontsize=fsize1)
    plt.xticks(fontsize=fsize2)
    plt.yticks(fontsize=fsize2)

    # x-axis is number of nodes hardened, and top secondary x-axis is fraction of nodes
    ax2 = ax.twiny()
    ax2Ticks = ax.get_xticks()
    ax2.set_xticklabels(tick_function(lcc_size_orig, ax2Ticks))

    ax2.set_xticks(ax.get_xticks())
    ax2.tick_params(labelsize=13)
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xlabel("Nodes hardened (frac.)", fontsize=14)


def save_plot(gname, metric, fsize1):
    plt.tight_layout()

    basedir = os.path.join(PATH_DIR, 'plots/rm_nodes/')
    plot_filename = os.path.join(basedir, '{}_{}_drop.png'.format(gname, metric))

    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename, format='png')
    print('Printed plot:', plot_filename)
    plt.clf()


if __name__ == '__main__':

    #metric = 'lambda'
    metric = 'lcc'

    gname = os.path.splitext(GRAPH_FILE)[0]
    input_filename = os.path.join(PATH_DIR, gname + ".pkl")

    print("reading data from: ", input_filename)
    g = Graph.Read_Pickle(input_filename)

    A = g.get_adjacency_sparse()
    A = A.astype(float)
    U, first_lambda, Vt = svds(A, k = 1, which='LM')
    print("original lambda: ", first_lambda)
    lcc_size_orig = len(g.vs)
    print("original lcc size: ", lcc_size_orig)

    fig, ax = plt.subplots()

    for j in range(len(methods)):
        if metric == 'lambda':
            plot_drop(first_lambda, gname, j)
        else:
            plot_fragm(lcc_size_orig, gname, j)
        
    set_axis(ax, lcc_size_orig, metric, 16, 14)
    save_plot(gname, metric, 16)

