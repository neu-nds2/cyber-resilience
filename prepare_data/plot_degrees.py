
import os
import sys
import pandas as pd
import numpy as np
from igraph import *

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import InsetPosition

import powerlaw

sys.path.append(os.path.abspath("../"))
from constants import PATH_DIR, GRAPH_FILE

# check how close is our degree distribution to power law

# count the number of nodes with each  degree
def get_degree_distrib(g):
    vertices = g.vs()["name"]
    degree_distrib = dict()
    names = dict() # also saving hostnames

    for n in vertices:
        degree = g.degree(n)
        if degree not in degree_distrib:
            degree_distrib[degree] = 1
            names[degree] = [n]
        else:
            degree_distrib[degree] += 1
            names[degree].append(n)
   
    for d in degree_distrib:
        degree_distrib[d] = degree_distrib[d] / len(vertices)

    return degree_distrib, names 


def plot_degrees(g):

    data = g.degree()
   
    # using the powerlaw package from Alstott et. al

    fit = powerlaw.Fit(np.array(data), xmin=1, discrete=True)
    # fit = powerlaw.Fit(np.array(data), discrete=True)
    print(fit.power_law.xmin, fit.power_law.alpha, fit.power_law.sigma, fit.power_law.D)
    #R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio = True)
    #print(R, p)
    
    fig2 = fit.plot_ccdf(color='b', linewidth=2, label='Data')
    fit.power_law.plot_ccdf(data, color='r', linestyle='--', ax=fig2, label='Power law fit')
    fit.exponential.plot_ccdf(data, color='g', linestyle='--', ax=fig2, label='Exponential fit')

    return fit.power_law.alpha, fit.power_law.xmin, fit.power_law.sigma, fit.power_law.D



def save_plot(ax1, degree_distrib, plot_file):

    # setting up the main figure
    ax1.legend(fontsize=18, loc='lower right')
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.tick_params(axis='both', which='minor', labelsize=16)
    ax1.set_xlabel('Degree', fontsize=18)
    ax1.set_ylabel(r'ccdf(X)', fontsize=18)
    ax1.loglog()

    # plotting the inset
    # Create a set of inset Axes
    ax2 = plt.axes([0,0,1,1])

    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax1, [0.05,0.05,0.35,0.35])
    ax2.set_axes_locator(ip)

    x, y = zip(*degree_distrib.items())
    x = list(x)
    y = list(y)
    ax2.scatter(x, y, marker='.')
    ax2.loglog()

    #  no ticks for the inset, because it would look too crowded
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xticks([], minor=True)
    ax2.set_yticks([], minor=True)
    
    plt.tight_layout()

    plt.savefig(plot_file, format='png')
    plt.clf()
    print("Plot saved to file: ", plot_file)


if __name__ == '__main__':

    gname = os.path.splitext(GRAPH_FILE)[0]
    filename = os.path.join(PATH_DIR, gname + ".pkl")
    
    print("\nReading full graph from: ", filename)
    g = Graph.Read_Pickle(filename)
    print("edges, nodes: ", len(g.es), len(g.vs))
    
    # plotting degrees
    fig, ax1 = plt.subplots()

    alpha, xmin, sigma, D = plot_degrees(g)
    
    # saving plots
    outdir_plots = os.path.join(PATH_DIR, "plots/degrees_distribution/")
    os.makedirs(outdir_plots, exist_ok=True)
   
    plot_file = os.path.join(outdir_plots, "degrees_{}.png".format(gname))

    degree_distrib, names = get_degree_distrib(g) # need this for inset
    
    save_plot(ax1, degree_distrib, plot_file)

    print("Finished")

