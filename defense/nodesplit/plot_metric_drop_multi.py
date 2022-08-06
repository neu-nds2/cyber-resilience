import os
from igraph import *
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

sys.path.append(os.path.abspath("../../"))
from constants import PATH_DIR, GRAPH_FILE

def tick_function(n, X):
    V = X / (n + X)
    return ["%.2f" % z for z in V]


# plotting eigendrop (y-axis)  by number of splits (x-axis)
def plot_drop(ax, gnames):

    inbasedir = os.path.join(PATH_DIR, 'nodesplit')
    col_headers = ['split_num', 'eigen_val']

    for gname in gnames:

        input_filename = os.path.join(inbasedir, '{}/info/info_{}_nodesplit.csv'.format(gname, gname))
        print("reading data from: ", input_filename)

        df_info = pd.read_csv(input_filename, usecols = col_headers)

        # plot a few datapoints
        # df_filtered = df_info[df_info['split_num'].isin([0, 30, 60, 100])]
        df_filtered = df_info[df_info['split_num'].isin([0, 25, 50, 75, 100])]
        df_info = df_filtered
        print(df_info)

        # get the eigenvalue (same as singular value for undirected graphs), compute eigendrop
        eigen_orig = df_info['eigen_val'][0]
        eigendrop = 100 * (eigen_orig - df_info['eigen_val']) / eigen_orig

        ax.plot(df_info['split_num'], eigendrop, marker='.', linewidth=3, markersize=8, label=gname)

    ax.set_title(r'$\Delta \lambda \%$ by $k$', fontsize=20)
    ax.set_xlabel(r'Number of splits ($k$)', fontsize=20)
    ax.set_ylabel(r'$\Delta \lambda \%$', fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xticks([0, 25, 50, 75, 100])


# plotting eigendrop (x-axis) by fraction of splits (x-axis)
def plot_drop_fk(ax, gnames):

    inbasedir = os.path.join(PATH_DIR, 'nodesplit')
    col_headers = ['split_num', 'eigen_val']
    data = dict()
    min_xd = 100000

    for gname in gnames:

        # get number of nodes "n" in original graph, to compute the fraction of new nodes added during splits
        filename = os.path.join(PATH_DIR, gname + ".pkl")
        print("\nReading full graph from: ", filename)
        g = Graph.Read_Pickle(filename)
        n = len(g.vs)

        input_filename = os.path.join(inbasedir, '{}/info/info_{}_nodesplit.csv'.format(gname, gname))
        print("reading data from: ", input_filename)

        df_info = pd.read_csv(input_filename, usecols = col_headers)

        # compute eigendrop
        eigen_orig = df_info['eigen_val'][0]
        eigendrop = 100 * (eigen_orig - df_info['eigen_val']) / eigen_orig

        ylabel_str = r'$\Delta \lambda \%$'

        xd = df_info['split_num'] / (n + df_info['split_num'])
        data[gname] = dict()
        data[gname]['x'] = xd
        data[gname]['y'] = eigendrop
        data[gname]['n'] = n
        min_xd = min(min_xd, max(xd))
        print("min xd: ", min_xd)

    # plot eigendrop 
    # the min_xd is used to adjust number of splits for various graph sizes,
    # in order to keep the same fraction on the x-axis for all graphs
    # adding the annotation with the equivalent number of splits corresponding to the fraction of new nodes
    for gname in gnames:
        xd = data[gname]['x'][data[gname]['x'] <= min_xd]
        yd = data[gname]['y'][:len(xd)]

        ax.plot(xd, yd, label='port {}'.format(gname), linewidth=3)

        xs = int(data[gname]['n'] * min_xd / (1 - min_xd))
        print("xs: ", xs)
        print("xd: ", xd)
        print("yd: ", yd)

        x = xd[xs]
        y = yd[xs]
        print("x, y: ", x, y)
        plt.scatter(x, y, marker='x')

        yadd = 0
        xadd = 26

        label = f"k={xs}"

        plt.annotate(label, # this is the text
                 (x, y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(xadd, yadd), # distance from text to points (x,y)
                 ha='left', fontsize=16) # horizontal alignment can be left, right or center

    ax.set_title(r'$\Delta \lambda \%$ by $f_k$', fontsize=20)
    ax.set_xlabel(r'Fraction of new nodes ($f_k$)', fontsize=20)
    ax.set_ylabel(ylabel_str, fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))


def save_plot(gnames, plot_filename):
    
    plt.legend(gnames, fontsize=18)
    plt.tight_layout()


    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename, format='png')
    print('Printed plot:', plot_filename)
    plt.clf()


if __name__ == '__main__':

    splits_on_xaxis = False

    # just an example, with one sample file
    gname = os.path.splitext(GRAPH_FILE)[0]
    gnames = [gname]

    gname = os.path.splitext(GRAPH_FILE)[0]
    g_filename = os.path.join(PATH_DIR, gname + ".pkl")

    outdir = os.path.join(PATH_DIR, 'plots/nodesplit/')
    
    fig, ax = plt.subplots()

    if splits_on_xaxis:
        # plotting eigendrop by number of splits
        plot_filename  = os.path.join(outdir, 'eigendrop_splits.png')
        plot_drop(ax, gnames)
    else:
        # plotting eigendrop by fraction of splits
        plot_filename  = os.path.join(outdir, 'eigendrop_frac.png')
        plot_drop_fk(ax, gnames)
    
    save_plot(gnames, plot_filename)


