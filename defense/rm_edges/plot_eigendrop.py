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

# linestyles = ['-', '--', '-.', ':', '-', '--']
markers = ['v', '^', '+', 'x', 'o', '*']

def plot_drop(first_lambda, gname, budgets, splits, i):
        
    lambda_drop = [0] # 0 for no edges removed
    for budget in budgets:
        
        if splits == '0r':
            input_filename = os.path.join(PATH_DIR, 'rm_edges/random/graph-{}_b{}.pkl'.format(gname, budget))
        elif splits == '0m':
            input_filename = os.path.join(PATH_DIR, 'rm_edges/met/graph-{}_b{}.pkl'.format(gname, budget))
        else:
            input_filename = os.path.join(PATH_DIR, 'rm_edges/met/after_nodesplit/{}/splits{}/graph_b{}.pkl'.format(gname, splits, budget))
        print("reading data from: ", input_filename)

        g = Graph.Read_Pickle(input_filename)

        A = g.get_adjacency_sparse()
        A = A.astype(float)
        U, S, Vt = svds(A, k = 1, which='LM')
        drop = 100 * (first_lambda[0] - S[0])/first_lambda[0]
        print("% drop", drop)
        lambda_drop.append(drop)

    print([0] + budgets, lambda_drop)

    if splits == '0r':
        labelstr = 'RandE'
    elif splits == '0m':
        labelstr = "MET"
    else:
        labelstr = "NodeSplit-{}+MET".format(splits)
    budgets_frac = [b/100 for b in budgets]

    plt.plot([0] + budgets_frac, lambda_drop, label=labelstr, marker=markers[i], linewidth=3, markersize=8)

    plt.xlabel("Edges hardened (frac.)", fontsize=20)
    plt.ylabel(r'$\Delta \lambda \%$', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    


def save_plot(gname):
    
    ax = plt.gca()

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, fontsize=20)

    plt.tight_layout()

    basedir = os.path.join(PATH_DIR, 'plots')
    plot_filename = os.path.join(basedir, 'rm_edges', '{}_eigendrop.png'.format(gname))

    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename, format='png')
    print('Printed plot:', plot_filename)
    plt.clf()


if __name__ == '__main__':

    splits = ['0r', '0m', 10, 50]
    budgets = [5, 10, 20, 50]

    fig, ax = plt.subplots()
    
    gname = os.path.splitext(GRAPH_FILE)[0]
    input_filename = os.path.join(PATH_DIR, gname + ".pkl")
    
    print("reading data from: ", input_filename)
    g = Graph.Read_Pickle(input_filename)

    A = g.get_adjacency_sparse()
    A = A.astype(float)
    U, first_lambda, Vt = svds(A, k = 1, which='LM')
    print("original lambda: ", first_lambda)

    for j in range(len(splits)):
        s = splits[j]
        plot_drop(first_lambda, gname, budgets, s, j)

    save_plot(gname)


