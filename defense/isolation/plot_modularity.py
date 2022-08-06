import os
from igraph import *
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath("../../"))
from constants import PATH_DIR, GRAPH_FILE

if __name__ == '__main__':

    markers = ['x', '<', '|', 'o', 'v', '^']
    col_headers = ['split_num', 'modularity']
    inbasedir = os.path.join(PATH_DIR, "isolation")
        
    # just an example, with one sample file
    gname = os.path.splitext(GRAPH_FILE)[0]
    gnames = [gname]

    fig,ax = plt.subplots()
 
    i = 0
    for gname in gnames:
        input_filename = os.path.join(inbasedir, 'info/{}/info_{}_leiden.csv'.format(gname, gname))
        print("reading data from: ", input_filename)
        df_isolation = pd.read_csv(input_filename, usecols = col_headers)
        df_isolation = df_isolation.drop([0])
        df_isolation = df_isolation[df_isolation['split_num'].isin([0, 10, 50, 100])]
        print(df_isolation)

        # ------------------ plot modularity --------------------
        plt.plot(df_isolation['split_num'], df_isolation['modularity'], marker=markers[i], label=gname)
        i = i + 1

    plt.xlabel("Number of split nodes", fontsize=20)
    plt.ylabel(r'Modularity', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=20)
    ax.tick_params(labelsize=18)
    plt.tight_layout()

    plot_filename = os.path.join(PATH_DIR, 'plots/isolation/modularity_isolation.png')

    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename, format='png')
    print('Printed plot:', plot_filename)
    plt.clf()


