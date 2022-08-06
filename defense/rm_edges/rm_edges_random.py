import pandas as pd
import sys
import os
import random

from igraph import *
from scipy.sparse.linalg import svds

sys.path.append(os.path.abspath("../../"))
from constants import PATH_DIR, GRAPH_FILE

def rm_edges_rand_graph(g, gname, budget, budget_perc, outdir):

    print("budget: ", budget)
    edges = random.sample(g.get_edgelist(), budget)

    g1 = g.copy()
    g1.delete_edges(edges)
    print("Remaining edges: ", len(g1.es()))
    outfile = os.path.join(outdir, "graph-{}_b{}.pkl".format(gname, budget_perc))
    os.makedirs(os.path.dirname(outfile), exist_ok = True)
    print("Saving new graph to ", outfile)
    g1.write_pickle(outfile)

    singular_val_new = compute_singular_value(g1)
    return singular_val_new


def rm_rand(g, gname, budgets, budgets_perc, outdir):

    # A is a csr_sparse matrix
    A = g.get_adjacency_sparse()
    A = A.astype(float)

    _, singular_val_orig, _ = svds(A, k=3, which='LM')
    singular_val_orig = max(singular_val_orig)  # it is in a list format
    print("singular_val: ", singular_val_orig)

    (a, _) = A.nonzero()
    edges_num = len(a)
    print("edges_num orig: ", edges_num)

    info_list = []
    info_list.append([0, 0, round(singular_val_orig, 2), 0])

    for i in range(len(budgets)):
        budget = budgets[i]
        budget_perc = budgets_perc[i]

        if budget >= edges_num:
            info_list.append([budget, 100, 0, 100])
            continue

        singular_val_new = rm_edges_rand_graph(g, gname, budget, budget_perc, outdir)

        percent_drop = 100.0 * abs(singular_val_new - singular_val_orig) / singular_val_orig
        
        info = [budget, budget_perc, round(singular_val_new, 2), round(percent_drop, 2)]
        print(info)
        info_list.append(info)

    return info_list



def compute_singular_value(g):
    
    # A is a csr_sparse matrix
    A = g.get_adjacency_sparse()
    A = A.astype(float)

    _, singular_val, _ = svds(A, k=3, which='LM')
    singular_val = max(singular_val) # it is in a list format

    print("singular_val: ", singular_val)

    return singular_val


if __name__ == '__main__':

    outdir_info = os.path.join(PATH_DIR, "rm_edges/random/info/")
    outdir_graph = os.path.join(PATH_DIR, "rm_edges/random/")

    gname = os.path.splitext(GRAPH_FILE)[0]
    filename = os.path.join(PATH_DIR, gname + ".pkl")
    
    print("Reading full graph from: ", filename)
    g = Graph.Read_Pickle(filename)

    nEdges_orig = len(g.es())
    print("edges orig graph: ", nEdges_orig)

    budgets_perc = [1, 3, 5, 10, 20, 50]
    print("budgets_perc: ", budgets_perc)
    budgets = [int(nEdges_orig * b / 100.0) for b in budgets_perc]
    print("budgets: ", budgets)
   
    info_list = rm_rand(g, gname, budgets, budgets_perc, outdir_graph)

    results_file = os.path.join(outdir_info, "graph-{}_eigendrop.csv".format(gname))
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    col_headers = ["budget (#)", "budget (%)", "lambda1", "percent_drop"]
    info_df = pd.DataFrame(info_list, columns=col_headers)
    info_df.to_csv(results_file, index = False)

    print(info_df)
    print("rand lambda1 drop results in file: ", results_file)
    print("Finished")


