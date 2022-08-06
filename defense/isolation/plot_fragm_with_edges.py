import os
from igraph import *
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit

sys.path.append(os.path.abspath("../../"))
from constants import PATH_DIR, GRAPH_FILE

# define the true objective function
def func1(x, a, b, c):
    return a * np.exp(-b * x) + c

def func(x, a, b, c, d, e, f, g):
	return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + (f * x**6) + g


def tick_function(n, X):
    V = X / n
    return ["%.5f" % z for z in V]


if __name__ == '__main__':

    gname = os.path.splitext(GRAPH_FILE)[0]
    inbasedir = os.path.join(PATH_DIR, "isolation")
    input_filename = os.path.join(inbasedir, 'info/{}/info_{}_leiden.csv'.format(gname, gname))

    # split_num,communities,max_community_nodes,max_community_edges,max_community_nodes_frac,max_community_edges_frac,edges_between_communities,edges_between_communities_frac,max_community_singular_val,new_graph_singular_val

    print("reading data from: ", input_filename)
    col_headers = ['split_num', 'max_community_nodes_frac', 'edges_between_communities_frac']

    df_fragm = pd.read_csv(input_filename, usecols = col_headers)
    df_fragm = df_fragm.drop([0])
    print(df_fragm)

    # ------------------ plot fragmentation, interpolated, fitted --------------------

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()

    x = df_fragm['split_num']
    y = df_fragm['max_community_nodes_frac']

    xnew = np.linspace(x.min(), x.max(), 300)
    gfg = make_interp_spline(x, y, k=3)
    y_new = gfg(xnew)

    lns1 = ax.plot(xnew, y_new, label=r'$\sigma$ (y1-axis)')

    popt, pcov = curve_fit(func, x, y)

    # summarize the parameter values
    a, b, c, d, e, f, g = popt
    print(popt)
    # define a sequence of inputs between the smallest and largest known inputs
    x_line = np.arange(min(x), max(x), 1)
    print(x_line)
    # calculate the output for the range
    y_line = func(x_line, a, b, c, d, e, f, g)
    # create a line plot for the mapping function
    lns2 = ax.plot(x_line, y_line, '--', color='red', label=r'$\sigma$ (y1-axis, interpolated)')

    # --------------------plot edges removed vs splits --------------------------------------

    x = df_fragm['split_num']
    y = df_fragm['edges_between_communities_frac']

    xnew = np.linspace(x.min(), x.max(), 300)
    gfg = make_interp_spline(x, y, k=3)
    y_new = gfg(xnew)

    popt, pcov = curve_fit(func, x, y)

    # summarize the parameter values
    a, b, c, d, e, f, g = popt
    print(popt)
    # define a sequence of inputs between the smallest and largest known inputs
    x_line = np.arange(min(x), max(x), 1)
    print(x_line)
    # calculate the output for the range
    y_line = func(x_line, a, b, c, d, e, f, g)
    # create a line plot for the mapping function
    lns3 = ax2.plot(x_line, y_line, '--', color='k', label='edges (y2-axis, interpolated)')

    ax.set_ylim([0, 1])
    #ax2.set_ylim([0, 0.1])
    ax.set_xlabel("Number of split nodes", fontsize=20)
    ax.set_ylabel(r'$\sigma$', fontsize=20)
    ax2.set_ylabel("Boundary edges", fontsize=20)

    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize=16)
    ax.tick_params(axis='both', labelsize=18)
    ax2.tick_params(axis='both', labelsize=18)
    
    plt.tight_layout()
    plot_filename = os.path.join(PATH_DIR, 'plots/isolation/fragm_with_edges_{}_isolation.png'.format(gname))

    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename, format='png')
    print('Printed plot:', plot_filename)
    plt.clf()


