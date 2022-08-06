import pandas as pd
import numpy as np
import sys
import os
import igraph as ig
from igraph import *
import time
import pickle

sys.path.append(os.path.abspath("../"))
from constants import PATH_DIR, GRAPH_FILE

# writes a pickle file
def write_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


# the  egonet is the  "neighborhood" of a node plus the node itself
def build_1_hop_egonet(g, node):

    vertices = set(g.neighbors(node))
    vertices.add(node)
    egonet = g.subgraph(vertices)
    # print("egonet vertices: ", egonet.vs["name"], len(egonet.vs))

    return egonet

def ens_calculate_node(g, i):

    egonet = build_1_hop_egonet(g, i)
    egonet.simplify()

    # t is the number of the total ties in the egocentric network (excluding those ties to the ego) and
    # n is the number of total nodes in the egocentric network (excluding the ego)

    # single node, no links
    ens_node = 0
    n = 0
    t = 0
    links_num = 0

    if len(egonet.vs) > 1:
        t = len(egonet.es) - len(g.neighbors(i))
        n = len(egonet.vs) - 1
        ens_node = n - t * 2 / n
        links_num = len(egonet.es)

    return ens_node


def ens_seq_loop(g):
    ens_dict = dict()
    node_names = g.vs()["name"]

    # just for printing
    n = 100000
    crt = 1
    start_time = time.time()

    # for each node, calculate ens
    for i in node_names:
        ens_dict[i] = ens_calculate_node(g, i)
        if crt % n == 0:
            print("Processed {} nodes in {} sec ".format(n, time.time() - start_time))
            start_time = time.time()
        crt += 1

    print("Processed last {} nodes in {} sec ".format((crt-1) % n, time.time() - start_time))

    return ens_dict


# --- Main ---

if __name__ == '__main__':

    gname = os.path.splitext(GRAPH_FILE)[0]
    filename = os.path.join(PATH_DIR, gname + ".pkl")

    print("Reading full graph from: ", filename)
    g = Graph.Read_Pickle(filename)

    ens_dict = ens_seq_loop(g)

    outdir = os.path.join(PATH_DIR, "graph_metrics/")
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "ens_values_{}.pkl".format(gname))

    write_pickle(outfile, ens_dict)

    print("Finished writting ens metrics per node to file: ", outfile)
    print("records: ", len(ens_dict.keys()))
    # print(ens_dict)

    # making sure there is no error
    for n in list(sorted(ens_dict.keys())):
        if ens_dict[n] <= 0:
            print(n, ens_dict[n])
            

