import pandas as pd
import numpy as np
import os
from igraph import *
import pickle

from scipy.sparse.linalg import svds

sys.path.append(os.path.abspath("../../"))
from constants import PATH_DIR, GRAPH_FILE


# reads a pickle file, returns contents
def read_pickle(filename):
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data
    except IOError:
        return {}

def get_top_ens_nodes(g, ens_file, budget):

    print("Reading ens metrics from: ", ens_file)
    ens_dict = read_pickle(ens_file)
    sorted_ens_dict = dict(sorted(ens_dict.items(), key = lambda x: x[1], reverse = True))
    top_ens_nodes = list(sorted_ens_dict.keys())[:budget]

    for n in top_ens_nodes[:10]:
        print(n, sorted_ens_dict[n])

    return top_ens_nodes

def compute_singular_value(g):

    # A is a csr_sparse matrix
    A = g.get_adjacency_sparse()
    A = A.astype(float)

    _, singular_val, _ = svds(A, k=3, which='LM')
    singular_val = max(singular_val) # it is in a list format

    print("singular_val: ", singular_val)

    return singular_val

if __name__ == '__main__':

    rm_nodes_method = "ens"
    budgets = [1, 3, 5, 10, 20, 50]

    gname = os.path.splitext(GRAPH_FILE)[0]
    g_filename = os.path.join(PATH_DIR, gname + ".pkl")
    ens_filename = os.path.join(PATH_DIR, "graph_metrics/ens_values_{}.pkl".format(gname))
    print("ens_filename: ", ens_filename)
    print("g_filename: ", g_filename)
    g = Graph.Read_Pickle(g_filename)
   
    rm_nodes = get_top_ens_nodes(g, ens_filename,  max(budgets))

    results_dir_lambda = os.path.join(PATH_DIR, "rm_nodes/{}/info/".format(rm_nodes_method))
    results_dir = os.path.join(PATH_DIR, "rm_nodes/{}/".format(rm_nodes_method))

    os.makedirs(results_dir_lambda, exist_ok=True)

    info_dict = defaultdict()
    lambda_init = compute_singular_value(g)

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
    info_df = pd.DataFrame.from_dict(info_dict, orient='index')
    results_filename_lambda = os.path.join(results_dir_lambda, "graph-{}_eigendrop.csv".format(gname))
    info_df.to_csv(results_filename_lambda, index = False)

    print("Finished")

