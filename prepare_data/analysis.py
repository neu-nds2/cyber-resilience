
import pandas as pd
import os
import igraph as ig
from igraph import *

sys.path.append(os.path.abspath("../"))
from constants import PATH_DIR, GRAPH_FILE

# --- Main ---

def g_metrics(g, mode):

    # collect metrics for graph g
    results = []
    
    # number of connected components
    cc_num = len(g.components(mode = mode))
    results.append(cc_num)
    print("cc_num = {}, mode = {}".format(cc_num, mode))
    
    # number of nodes
    g_nodes = len(g.vs())
    results.append(g_nodes)

    # number of edges
    g_edges = len(g.es())
    results.append(g_edges)
    print("Size of g: nodes # = {}, edges # = {}".format(g_nodes, g_edges))
   
    # average degree
    g_degree = mean(g.degree())
    results.append(g_degree)
    print("Degree: ", g_degree)

    # density
    g_density = g_edges / (g_nodes * (g_nodes - 1))
    if mode == 'weak':
        g_density = 2 * g_density

    results.append(g_density)
    print("Density: ", g_density)
    
    # graph diameter
    g_diameter = g.diameter()
    results.append(g_diameter)
    print("Diameter: ", g_diameter)
    
    # average path length
    g_dist = g.average_path_length()
    results.append(g_dist)
    print("Average path length: ", g_dist)

    # transitivity, global and local
    g_trans = g.transitivity_undirected()
    results.append(g_trans)
    print("Transitivity: ", g_trans)

    g_trans_avg = g.transitivity_avglocal_undirected()
    results.append(g_trans_avg)
    print("Avg. transitivity: ", g_trans_avg)

    col_headers = ['cc_num', 'g_nodes', 'g_edges', 'g_degree', 'g_density', 'g_diameter', 'g_avg_path', 'g_transitivity', 'g_transitivity_avglocal']
    info_df = pd.DataFrame(results).T
    info_df.columns = col_headers
    print(info_df)

    return info_df


if __name__ == '__main__':

    mode = 'weak' # undirected

    gname = os.path.splitext(GRAPH_FILE)[0]
    filename = os.path.join(PATH_DIR, gname + ".pkl")
    
    outdir = os.path.join(PATH_DIR, "graph_metrics/")
    os.makedirs(outdir, exist_ok=True)
    results_file = os.path.join(outdir, "metrics_{}.csv".format(gname))

    print("Reading full graph from: ", filename)
    g = Graph.Read_Pickle(filename)

    info_df = g_metrics(g, mode)
    info_df.to_csv(results_file, index=False)
    print("results in file: ", results_file)

