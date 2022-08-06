import pandas as pd
import numpy as np
import sys
import os
from igraph import *

from scipy.sparse.linalg import eigs

sys.path.append(os.path.abspath("../../"))
from constants import PATH_DIR, GRAPH_FILE

def compute_eigen_value(g):
    # A is a csr_sparse matrix
    A = g.get_adjacency_sparse()
    A = A.astype(float)

    eigen_val, _ = eigs(A, k=1, which='LM')
    eigen_val = abs(eigen_val[0])  # it is in a list format

    return eigen_val


def get_top_degree_nodes(g, budget):

    # keep a dictionary with degree per each node in the graph
    degree_dict = dict()
    vertices = g.vs()["name"]
    for n in vertices:
        degree_dict[n] = g.degree(n, mode = "ALL")

    # sort the list of nodes by degree
    sorted_degree_dict = dict(sorted(degree_dict.items(), key = lambda x: x[1], reverse = True))

    # get top nodes, as many as the budget
    top_degree_nodes = list(sorted_degree_dict.keys())[:budget]

    return top_degree_nodes


# split across all iterations
def split_nodes(g1, nodes_list, dup_names_dict):
    g = g1.copy()
    for n in nodes_list:
        g, dup_names_dict = split_node(g, n, dup_names_dict)
    return g, dup_names_dict


# split one node
def split_node(g, n, dup_names_dict):

    # get neighbors of node n
    n_neigh = g.neighbors(n)

    # split node n in half
    # get nodes that will belong to each peer
    neigh_num = len(n_neigh)
    neigh_num_half = neigh_num // 2
    neigh_1 = set(n_neigh[:neigh_num_half])
    neigh_2 = set(n_neigh[neigh_num_half:])
    neigh_2 = g.vs(neigh_2)["name"]

    # construct name of the newly created node
    # the second substring represents how many times this node has been split
    # find name that starts with dup and ends with n, get max splits
    n_strs = n.split("_")
    base_hostname = n_strs[-1]

    if base_hostname in dup_names_dict:
        n_iter = 1 + dup_names_dict[base_hostname] 
    else:
        n_iter = 0
    dup_names_dict[base_hostname] = n_iter

    n_2 = "dup_" + str(n_iter) + '_' + base_hostname # name of second node
    # print("new node: ", n_2)

    # update edges 
    edges_to_add = []
    edges_to_rm = []
    for i in neigh_2:
        tup_rm = (i, n) # remove these edges from node n
        tup_add = (i, n_2)  # add these edges to new node n_2
        edges_to_add.append(tup_add)
        edges_to_rm.append(tup_rm)

    # print(edges_to_add)
    # print(edges_to_rm)
   
    # delete half of edges from node n
    g.delete_edges(edges_to_rm)

    # add new node and its edges
    g.add_vertices(n_2) 
    g.add_edges(edges_to_add)  

    # just some checks
    # comp =  g.clusters().subgraphs()
    # if len(comp) > 1:
    #     print(n, len(g.neighbors(n)), n_2, len(g.neighbors(n_2)))
    #     for c in comp:
    #         if len(c.vs) < 10:
    #             print(c.vs['name'])
    
    return g, dup_names_dict


def get_info(gnew):

    info = []

    info.append(compute_eigen_value(gnew)) # eigenvalue
    comp = gnew.clusters()
    info.append(len(comp)) # number of connected components

    # frac of largest connected component relative to graph size
    info.append(len(comp.giant().vs) / len(gnew.vs)) 
    return info


def save_info_to_file(info_list, column_names, info_file):
    info_df = pd.DataFrame(info_list, columns=column_names)
    # info_df = info_df.round(3)
    info_df.to_csv(info_file, index=False)


def algo_iter_split(g, budgets, info_list, column_names, info_file, outdir_graph):

    # dup_names_dict helps us make sure there are no name collisions for added nodes
    dup_names_dict = dict()
    g1 = g.copy()

    print("\ninitial, num nodes, num edges: ", len(g1.vs()), len(g1.es()))
    
    for i in budgets: 
        
        # split top node at each step
        top_degree_node = get_top_degree_nodes(g1, 1)
        g1, dup_names_dict = split_nodes(g1, top_degree_node, dup_names_dict)
       
        # save info about eigendrop and fragmentation after each nodesplit operation
        info = [i] + get_info(g1)
        info_list.append(info)
        save_info_to_file(info_list, column_names, info_file)
       
        # save the perturbed graph for some iterations
        #  if i < 101 or i in [200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]:
        if i <  101:
            graph_file = os.path.join(outdir_graph, "graph_splits{}.pkl".format(i))
            os.makedirs(os.path.dirname(graph_file), exist_ok=True)
            g1.write_pickle(graph_file)
        
    print("\nfinal, num nodes, num edges: ", len(g1.vs()), len(g1.es()))


if __name__ == '__main__':

    method = 'nodesplit'
    column_names = ['split_num', 'eigen_val', 'lcc_num', 'max_lcc_nodes_frac'] 
    budgets = np.arange(1, 101)  # splits, i.e. nodes to add

    gname = os.path.splitext(GRAPH_FILE)[0]
    filename = os.path.join(PATH_DIR, gname + ".pkl")

    outdir_info = os.path.join(PATH_DIR, "{}/{}/info/".format(method, gname))
    outdir_graph = os.path.join(PATH_DIR, "{}/{}/".format(method, gname))

    info_file = os.path.join(outdir_info, "info_{}_{}.csv".format(gname, method))
    os.makedirs(os.path.dirname(info_file), exist_ok=True)

    print("Reading graph from: ", filename)
    g = Graph.Read_Pickle(filename)

    print("\nnum nodes, num edges: ", len(g.vs()), len(g.es()))
    eigen_val_orig = compute_eigen_value(g)
    print("eigen val: ", eigen_val_orig)

    info_list = []
    info = [0, eigen_val_orig, 1, 1]
    info_list.append(info)

    algo_iter_split(g, budgets, info_list, column_names, info_file, outdir_graph)
    
    print("Finished")


