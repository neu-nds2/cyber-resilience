import os
import sys
import pandas as pd
import igraph as ig
from igraph import *

sys.path.append(os.path.abspath("../"))
from constants import PATH_DIR, GRAPH_FILE

# construct the connections graph using the source/target nodes
# connections parameter is a dataframe
def construct_graph(connections, col_names):
    
    FromNodeId = col_names[0]
    ToNodeId = col_names[1]

    connections = connections.astype(str)

    g = ig.Graph(directed=True)

    # Adding vertices
    g_vertices = list(set(connections[[FromNodeId, ToNodeId]].values.reshape(-1)))
    g.add_vertices(g_vertices)
    
    # Adding edges
    g_edges = connections[[FromNodeId, ToNodeId]].values
    g.add_edges(g_edges)
    print("\nGraph edges, head and size, subset: \n",  g_edges[:10], len(g_edges))
 
    return g


if __name__ == '__main__':

    # construct graph
    print("Getting connections")

    input_file = os.path.join(PATH_DIR, GRAPH_FILE)
    col_names = ["FromNodeId", "ToNodeId"]

    connections = pd.read_csv(input_file, header=None, names=col_names, comment='#', delimiter="\t")
    print(connections)

    print("Constructing graph")
    g = construct_graph(connections, col_names)
    g.simplify()  # no loops or multi-links
    g.to_undirected()  # undirected
    print("nodes, edges graph before  3-core: ", len(g.vs), len(g.es))
    g = g.k_core(3)  # only nodes with three or more neighbors, i.e., the 3-core graph
        
    if len(g.vs) == 0: exit()

    g = g.clusters().giant()  # only the largest connected component
    print("nodes, edges graph, final: ", len(g.vs), len(g.es))

    graph_file = os.path.splitext(input_file)[0] + ".pkl"
    g.write_pickle(graph_file)


