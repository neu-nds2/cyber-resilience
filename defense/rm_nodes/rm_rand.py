import random
import pandas as pd
import numpy as np
import os
from igraph import *

sys.path.append(os.path.abspath("../../"))
from constants import PATH_DIR, GRAPH_FILE

from scipy.sparse.linalg import svds

def get_rand_nodes(g, budget):

    print("Removing {} random nodes: ".format(budget))
    b = budget
    rm_nodes = []
    node_names = g.vs['name']

    while b > 0:
        i = random.randint(0, len(g.vs) - 1)
        n = node_names[i]
        if n in rm_nodes: continue

        rm_nodes.append(n)
        b -= 1

    print("rm nodes: ", rm_nodes)
    return rm_nodes


def compute_singular_value(g):

    # A is a csr_sparse matrix
    A = g.get_adjacency_sparse()
    A = A.astype(float)

    _, singular_val, _ = svds(A, k=3, which='LM')
    singular_val = max(singular_val) # it is in a list format

    print("singular_val: ", singular_val)

    return singular_val


if __name__ == '__main__':

    rm_nodes_method = "random"
    budgets = [1, 3, 5, 10, 20, 50]

    gname = os.path.splitext(GRAPH_FILE)[0]
    g_filename = os.path.join(PATH_DIR, gname + ".pkl")

    print("g_filename: ", g_filename)
    g = Graph.Read_Pickle(g_filename)
   
    # randomly get a number of nodes equal to budget
    rm_nodes = get_rand_nodes(g, max(budgets))

    results_dir_info = os.path.join(PATH_DIR, "rm_nodes/{}/info/".format(rm_nodes_method))
    results_dir = os.path.join(PATH_DIR, "rm_nodes/{}/".format(rm_nodes_method))
    os.makedirs(results_dir_info, exist_ok=True)

    info_dict = defaultdict()
    lambda_init = compute_singular_value(g)

    # compute eigen drop and save the perturbed graphs
    for b in budgets:
        print("Number nodes before deletion, budget: ", len(g.vs), b)
        g1 = g.copy()
        g1.delete_vertices(rm_nodes[:b])
        lambda_final = compute_singular_value(g1)
        lambda_drop = 100.0 * abs(lambda_final - lambda_init) / lambda_init

        info_dict[b] = lambda_drop

        print("Number nodes after deletion: ", len(g1.vs))
        results_filename = os.path.join(results_dir, "graph-{}_b{}.pkl".format(gname, b))
        g1.write_pickle(results_filename)
        print("Graph written to file: ", results_filename)

    print('info', info_dict)
    col_headers = ["budget (%)", "percent_drop"]
    info_df = pd.DataFrame.from_dict(info_dict,  orient='index')
    results_filename_info = os.path.join(results_dir_info, "graph-{}_eigendrop.csv".format(gname))
    info_df.to_csv(results_filename_info, index = False)

    print("Finished")


