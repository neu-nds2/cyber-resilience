import pandas as pd
import numpy as np
import sys
import os
from igraph import *
import pickle

from scipy.sparse.linalg import eigs

sys.path.append(os.path.abspath("../../"))
from constants import PATH_DIR, GRAPH_FILE


# writes a pickle file
def write_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def compute_eigen_value(g):
    # A is a csr_sparse matrix
    A = g.get_adjacency_sparse()
    A = A.astype(float)

    eigen_val, _ = eigs(A, k=1, which='LM')
    eigen_val = abs(eigen_val[0])  # it is in a list format

    return eigen_val


# test 3 methods for extracting communities
def compute_communities(g, method):
    g1 = g.copy()
    try:
        if method == 'leiden':
            clusters = g1.community_leiden(objective_function='modularity', n_iterations=-1)
        elif method == 'infomap':
            clusters = g1.community_infomap()
        elif method == 'leading_eigenvector':
            clusters = g1.community_leading_eigenvector()
       
        # get the edges crossing communities and the  modularity metric of extracted communities
        edges_crossing_bool = clusters.crossing()
        edges_crossing = [i for i in range(len(edges_crossing_bool)) if edges_crossing_bool[i] == True]
        print("number of edges crossing communities: ", len(edges_crossing))
        modularity = g1.modularity(clusters)
        return edges_crossing, modularity
    except:
        print("didn't converge")
        return None, None


# remove inter-community edges
def remove_edges(g, edges_crossing):
    edges_removed = edges_crossing
    gnew = g.copy()
    gnew.delete_edges(edges_removed)
    #print("nodes, edges after communities, diff edges: ", len(gnew.vs), len(gnew.es), (len(g.es) - len(gnew.es)))

    return gnew


# iterate through the list of boundary nodes, in decreasing order of their degree
# recompute degree after each step
def eval_degree_lcc(g, boundary_nodes, topk_num):
    gnew1 = g.copy()
    nodes_to_rm = boundary_nodes.copy()

    fragm = dict()
        
    # what is fragm if we remove that node?
    selected_lcc = []

    i = 0
    while i < min(topk_num + 1, len(boundary_nodes)):
        # print(len(nodes_to_rm))
        # boundary nodes sorted by degree
        nodes_degrees = dict(zip(nodes_to_rm, gnew1.degree(nodes_to_rm)))
        sorted_nodes_degrees = dict(sorted(nodes_degrees.items(), key = lambda x: x[1], reverse = True))
        nodes_to_rm = list(sorted_nodes_degrees.keys())

        # find the first one in lcc
        for j in range(len(nodes_to_rm)):
            if nodes_to_rm[j] not in gnew1.clusters().giant().vs['name']: continue
            break # found one to remove

        # if none is part of the lcc, just get the first node
        if j >= len(nodes_to_rm):
            ind = 0
        else:
            ind = j
        
        n = nodes_to_rm.pop(ind)
        selected_lcc.append(n)
        gnew1.delete_vertices(n)
        fragmn = len(gnew1.clusters().giant().vs) / len(gnew1.vs)
        fragm[i] = fragmn
        i = i + 1

    return fragm, selected_lcc


# iterate through the list of boundary nodes, in decreasing order of their coverage
# do not recompute coverage after each step, just once in the beginning
# i.e., number of intercommunity edges that they connect to
def eval_coverage_lcc(g, topk_num, coverage_dict):
    gnew1 = g.copy()
    boundary_nodes= list(coverage_dict.keys())
    num_edges_covered = [len(x) for x in coverage_dict.values()]
    nodes_coverage = dict(zip(boundary_nodes, num_edges_covered))
    nodes_to_rm = boundary_nodes.copy()

    fragm = dict()

    # what is fragm if we remove that node?
    selected_lcc = []

    i = 0
    while i < min(topk_num + 1, len(boundary_nodes)):

        # boundary nodes sorted by coverage
        sorted_nodes_coverage = dict(sorted(nodes_coverage.items(), key = lambda x: x[1], reverse = True))
        nodes_to_rm = list(sorted_nodes_coverage.keys())

        # find the first one in lcc
        for j in range(len(nodes_to_rm)):
            if nodes_to_rm[j] not in gnew1.clusters().giant().vs['name']: continue
            break # found one to remove

        # if none is part of the lcc, just get the first node
        if j >= len(nodes_to_rm):
            ind = 0
        else:
            ind = j

        n = nodes_to_rm.pop(ind)
        nodes_coverage.pop(n)

        selected_lcc.append(n)
        gnew1.delete_vertices(n)
        fragmn = len(gnew1.clusters().giant().vs) / len(gnew1.vs)
        fragm[i] = fragmn
        i = i + 1

    return fragm, selected_lcc


# save info  about the resulting connected components, after intercommunity edges were removed
def get_info(g, gnew):

    subgraphs_comm = gnew.clusters().subgraphs()

    info = []
    clusters_nodes_num = []
    clusters_edges_num = []
    eigen_vals = []

    print("\nnum_clusters: ", len(subgraphs_comm))
    info.append(len(subgraphs_comm))

    i = 0
    for c in subgraphs_comm:
        clusters_nodes_num.append(len(c.vs()))
        clusters_edges_num.append(len(c.es()))
        eigen_vals.append(compute_eigen_value(c))
        # print(clusters_nodes_num[i], clusters_edges_num[i], eigen_vals[i])
        i = i + 1

    #print("\nclusters_nodes_num: ", sorted(clusters_nodes_num))
    #print("\nclusters_edges_num: ", sorted(clusters_edges_num))
    #print("\neigen vals: ", sorted(eigen_vals))

    info.append(max(clusters_nodes_num))
    info.append(max(clusters_edges_num))
    info.append(max(clusters_nodes_num) / len(gnew.vs()))
    info.append(max(clusters_edges_num) / len(gnew.es()))
    edges_between = len(g.es()) - len(gnew.es())

    info.append(edges_between)
    info.append(edges_between / len(g.es()))
    info.append('n/a')
    info.append(max(eigen_vals))
    info.append(compute_eigen_value(gnew))

    return info


def save_info_to_file(info_list, column_names, info_file):

    info_df = pd.DataFrame(info_list, columns=column_names)
    # info_df = info_df.round(3)
    print("\n", info_df)

    info_df.to_csv(info_file, index=False)
    print("results in file: ", info_file)


# also tested community isolation by removing intercommunity nodes instead of edges
# next function obtains intercommunity nodes 
def get_boundary_nodes(g, edges_crossing):
    nodes_dict = dict()

    for edge in edges_crossing:
        source = g.es[edge].source
        target = g.es[edge].target
        if source not in nodes_dict:
            nodes_dict[source] = [edge]
        else:
            nodes_dict[source].append(edge)

        if target not in nodes_dict:
            nodes_dict[target] = [edge]
        else:
            nodes_dict[target].append(edge)

    print("\nboundary nodes num: ", len(nodes_dict))
    nodes = g.vs(list(nodes_dict.keys()))['name']
    newdict = dict(zip(nodes, nodes_dict.values()))
    return newdict


# run the CI-Node method, where nodes on the boundary are removed to isolate community
def run_isolation(g, edges_crossing, topk_num, ce_algo, splits, outdir_info_boundary, lcc=True):
    methods = ['degree', 'coverage']
    nodes_dict = get_boundary_nodes(g, edges_crossing)

    for method in methods:
        m = method + "_lcc"
        if method == 'degree':
            fragm_dict, boundary_nodes_selected = eval_degree_lcc(g, list(nodes_dict.keys()), topk_num)
        elif method == 'coverage':
            fragm_dict, boundary_nodes_selected = eval_coverage_lcc(g, topk_num, nodes_dict)

        # save fragmentation and name of sorted boundary nodes
        fragm_file = os.path.join(outdir_info_boundary, "fragm_{}_{}_splits{}.pkl".format(m, ce_algo, splits))
        write_pickle(fragm_file, fragm_dict)
    
        nodes_file = os.path.join(outdir_info_boundary, "boundary_nodes_{}_{}_splits{}.pkl".format(m, ce_algo, splits))
        write_pickle(nodes_file, boundary_nodes_selected)

        print("\nmethod, fragm_dict: ", m, fragm_dict)
        print("\nboundary_nodes_selected: ", boundary_nodes_selected)


def algo_iter_split(g, gname, splits, topk_num, ce_algo, info_list, column_names, info_file, outdir_info_boundary, outdir_graph):

    if splits == 0:
        g1 = g.copy()
    else:
        file_splits = os.path.join(PATH_DIR, 'nodesplit/{}/graph_splits{}.pkl'.format(gname, splits))
        print("reading nodesplit graph from:", file_splits)
        if not os.path.exists(file_splits):
            print("file not exists: ", file_splits)
            exit()
        g1 = Graph.Read_Pickle(file_splits)

    print("\nCompute communities")
    g2 = g1.copy()
    edges_crossing, modularity  = compute_communities(g2, ce_algo)
    if edges_crossing == None: return   # probably hasn't converged
   
    # run community isolation with boundary node removal, i.e., CI-Node
    # run_isolation(g1, edges_crossing, topk_num, ce_algo, splits, outdir_info_boundary)

    # run community isolation with edge removal, i.e., CI-Edge
    # save info for the new graph after isolation by removal of all boundary edges
    gnew = remove_edges(g1, edges_crossing)
    info = [splits] + get_info(g, gnew) + [modularity]
    info_list.append(info)
    save_info_to_file(info_list, column_names, info_file)
   
    # save the new graph after all boundary edges were removed
    graph_file = os.path.join(outdir_graph, "graph_{}_splits{}.pkl".format(ce_algo, splits))
    os.makedirs(os.path.dirname(graph_file), exist_ok=True)
    gnew.write_pickle(graph_file)
    

if __name__ == '__main__':

    # ce_algo = 'leading_eigenvector'
    # ce_algo = 'infomap'
    ce_algo = 'leiden'

    column_names = ['split_num', 'communities', 'max_community_nodes', 'max_community_edges', 'max_community_nodes_frac', 'max_community_edges_frac', 'edges_between_communities', 'edges_between_communities_frac', 'monitored_nodes', 'max_community_eigen_val', 'new_graph_eigen_val', 'modularity'] 

    topk_num = 200 # for CI-Node only (removing intercommunity nodes, instead of edges)
    splits = np.arange(0, 101) # for hybrid NodeSplit + CI
    
    gname = os.path.splitext(GRAPH_FILE)[0]
    filename = os.path.join(PATH_DIR, gname + ".pkl")
    
    outdir_info = os.path.join(PATH_DIR, "isolation/info/{}/".format(gname))
    outdir_graph = os.path.join(PATH_DIR, "isolation/graphs/{}/".format(gname))

    info_file = os.path.join(outdir_info, "info_{}_{}.csv".format(gname, ce_algo))
    outdir_info_boundary = os.path.join(outdir_info, "boundary")
    os.makedirs(outdir_info_boundary, exist_ok=True)

    print("Reading graph from: ", filename)
    g = Graph.Read_Pickle(filename)

    print("\nundirected, num nodes, num edges: ", len(g.vs()), len(g.es()))
    eigen_val_orig = compute_eigen_value(g)
    print("eigen val: ", eigen_val_orig) 

    # info for whole LCC
    info_list = []
    info = [0, 'orig_graph', len(g.vs), len(g.es), 1, 1, 0, 0, 0, eigen_val_orig, eigen_val_orig, 'n/a']
    info_list.append(info)

    print("\nCompute communities: ", ce_algo)

    for k in splits:
        if k == 0:
            print("first, no splitting:")
        algo_iter_split(g, gname, k, topk_num, ce_algo, info_list, column_names, info_file, outdir_info_boundary, outdir_graph)
    
    print("Finished")


