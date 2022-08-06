import pandas as pd
import numpy as np
import os
from igraph import *
from  scipy import sparse
from scipy.sparse.linalg import svds

sys.path.append(os.path.abspath("../../"))
from constants import PATH_DIR, GRAPH_FILE

def get_top_degree_nodes(g, budget):

    g_tmp = g.copy()
    top_degree_nodes = []

    for i in range(budget):

        #get degrees for all nodes
        degree_dict = dict()
        vertices = g_tmp.vs()["name"]
        for n in vertices:
            degree_dict[n] = g_tmp.degree(n, mode = "ALL")
        #sort
        sorted_degree_dict = dict(sorted(degree_dict.items(), key = lambda x: x[1], reverse = True))

        #get top degree node
        top_degree_node = list(sorted_degree_dict.keys())[:1][0]
        top_degree_nodes.append(top_degree_node)
        #print('nodes count', len(g_tmp.vs))
        g_tmp.delete_vertices(top_degree_node)
        #print('nodes count after', len(g_tmp.vs))

    return top_degree_nodes


# singular value and eigen value are the same for undirected graphs
def compute_singular_value(g):
    if len(g.es()) < 1: return 0.0
    # A is a csr_sparse matrix
    A = g.get_adjacency_sparse()
    A = A.astype(float)
    _, singular_val, _ = svds(A, k = 3, which = 'LM')
    singular_val = max(singular_val) # it is in a list format
    print("singular_val: ", singular_val)
    return singular_val


if __name__ == '__main__':

    rm_nodes_method = "degree_iterative"
    budgets = [1, 3, 5, 10, 20, 50]

    gname = os.path.splitext(GRAPH_FILE)[0]
    g_filename = os.path.join(PATH_DIR, gname + ".pkl")

    print("g_filename: ", g_filename)
    g = Graph.Read_Pickle(g_filename)

    results_dir_info = os.path.join(PATH_DIR, "rm_nodes/{}/info/".format(rm_nodes_method))
    results_dir = os.path.join(PATH_DIR, "rm_nodes/{}/".format(rm_nodes_method))
    os.makedirs(results_dir_info, exist_ok = True)

    info_dict = defaultdict()
    lambda_init = compute_singular_value(g)

    for b in budgets:
        g1 = g.copy()
        print("Number nodes before deletion, budget: ", len(g1.vs), b)
        rm_nodes = get_top_degree_nodes(g1, b)
        print('budget, rm_nodes', b, rm_nodes)
        g1.delete_vertices(rm_nodes)
        print("new graph: ", len(g1.vs), len(g1.es))

        lambda_final = compute_singular_value(g1)
        lambda_drop = 100.0 * abs(lambda_final - lambda_init) / lambda_init
        info_dict[b] = lambda_drop

        print("Number nodes after deletion: ", len(g1.vs))
        results_filename = os.path.join(results_dir,  "graph-{}_b{}.pkl".format(gname, b))
        g1.write_pickle(results_filename)
        print("Graph written to file: ", results_filename)

    print('info', info_dict)
    col_headers = ["budget (%)", "percent_drop"]
    info_df = pd.DataFrame.from_dict(info_dict, orient = 'index')
    results_filename_info = os.path.join(results_dir_info, "graph-{}_eigendrop.csv".format(gname))
    info_df.to_csv(results_filename_info, index = False)

    print("Finished")

