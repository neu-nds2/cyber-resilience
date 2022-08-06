import os
from igraph import *
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

from scipy.sparse.linalg import eigs, svds, eigsh

sys.path.append(os.path.abspath("../"))
from constants import PATH_DIR, GRAPH_FILE

# wannacry traces and their attack parameters
# we will plot resulting infection for each trace

siidr_params = {
        'wc_1_500ms': (0.10, 0.06, 0.76, 0.04),
        'wc_1_1s': (0.11, 0.07, 0.71, 0.07),
        'wc_1_5s': (0.37, 0.52, 0.27, 0.44),
        'wc_1_10s': (0.12, 0.06, 0.75, 0.05),
        'wc_4_1s': (0.14, 0.07, 0.75, 0.08),
        'wc_4_5s': (0.12, 0.07, 0.76, 0.07),
        'wc_8_20s': (0.13, 0.09, 0.74, 0.08)
}


plotFootprint = False

# main function, reads infection statistics and constructs the plots
def plot_all_data(gname, model, num_steps, defenses, defenses_labels, defense_indexes, lambda_1, trace_name, beta, mu, gamma1, gamma2):

    footprint = dict()
    s_vals = dict() 

    fig, ax = plt.subplots()

    if plotFootprint: # attack mitigation, footprint is I+ID+R
        compartments = ['infected', 'infected_dormant', 'recovered']
    else: # attack eradication, plot infection I+ID
        compartments = ['infected', 'infected_dormant']

    comp = len(compartments)

    col_headers = ['step']
    for c in compartments:
        col_headers.extend(['{}_count'.format(c), '{}_frac'.format(c)])

    #  loop  through defenses
    for i in defense_indexes:
        d = defenses[i]
        d_label = defenses_labels[i]

        input_filename = os.path.join(PATH_DIR, 'attack/{}/{}'.format(model, gname), '{}/{}/{}_{}steps_beta_{}_mu_{}_gamma1_{}_gamma2_{}.csv'.format(d, trace_name, model, num_steps, beta, mu, gamma1, gamma2))
        
        if not os.path.exists(input_filename):
            print("file not found: ", input_filename)
            return None, None
        
        print("reading spm data from: ", input_filename)

        # read and plot infection statistics 
        try:
            df_avg = pd.read_csv(input_filename, usecols = col_headers)
            print(df_avg)

            # sum over the 3 compartments (basically everything that is not still susceptible, has been infected at some point)
            if comp == 3: # footprint I+ID+R
                df_avg['infected_nodes_total']= df_avg.iloc[:, [1,3,5]].sum(axis=1)
                df_avg['infected_frac_total']= df_avg.iloc[:, [2,4,6]].sum(axis=1)
            elif comp == 2:  # just infection I+ID
                df_avg['infected_nodes_total']= df_avg.iloc[:, [1,3]].sum(axis=1)
                df_avg['infected_frac_total']= df_avg.iloc[:, [2,4]].sum(axis=1)
            elif comp == 1:  # just active infection I
                df_avg['infected_frac_total']= df_avg.iloc[:, [2]].sum(axis=1)
                df_avg['infected_nodes_total']= df_avg.iloc[:, [1]].sum(axis=1)
        
            df_avg['step'] += 1 # make the step starts at 1, not zero
            print("df_avg:\n", df_avg.tail())
       
            # effective attack strength "s" from the epidemic threshold theorem,  s=lambda x (beta/mu)
            scrt = round(lambda_1[d_label] * (beta/mu),1)

            if trace_name in ['wc_1_5s']:
                df_avg.drop(df_avg.index[100:], axis=0, inplace=True)

            if plotFootprint: # check mitigation
                df_avg.plot('step', 'infected_frac_total', linestyle = '-', linewidth=2, label = "{}".format(d_label), ax = ax)
            else:   # check eradication
                df_avg.plot('step', 'infected_nodes_total', linestyle = '-', linewidth=2, label = "{} (s={})".format(d_label, scrt), ax = ax)

            footprint[d_label] = round(df_avg['infected_frac_total'].iloc[-1], 2)
            s_vals[d_label] = scrt
        except:
            print("file not available")
            exit()

    # set up and save the plot to file
    plt.xlabel("Time Step", fontsize=20)
    if plotFootprint:
        plt.ylabel(r'Footprint ($I + I_D + R$)', fontsize=20)
        dirname = "mitigate"
        plt.legend(fontsize=18)
    else:
        plt.ylabel(r'Infected nodes ($I + I_D$)', fontsize=20)
        plt.yscale('log')
        plt.legend(fontsize=16)
        dirname="eradicate"

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.tick_params(labelsize=18)

    plt.tight_layout()

    plots_dir = os.path.join(PATH_DIR, 'plots/attack/{}'.format(dirname))
    plot_filename = os.path.join(plots_dir, '{}/{}.png'.format(gname, trace_name))
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename, format='png')
    print('Printed plot:', plot_filename)

    plt.clf()

    return footprint, s_vals


def get_filename(method, method1, gname, splits, budget_edges, budget_nodes):
    if method == 'nodefense':
        filename = os.path.join(PATH_DIR, gname + ".pkl")
    elif method == 'nodesplit':
        filename = os.path.join(PATH_DIR, '{}/{}/graph_splits{}.pkl'.format(method, gname, splits))
    elif method == 'rm_edges' and method1 == 'random':
        filename = os.path.join(PATH_DIR, '{}/{}/graph-{}_b{}.pkl'.format(method, method1, gname, budget_edges))
    elif method == 'met' and method1 == '':
        filename = os.path.join(PATH_DIR, 'rm_edges/{}/graph-{}_b{}.pkl'.format(method, gname, budget_edges))
    elif method == 'met' and method1 == 'after_nodesplit':
        filename = os.path.join(PATH_DIR, 'rm_edges/{}/{}/{}/splits{}/graph_b{}.pkl'.format(method, method1, gname, splits, budget_edges))
    elif method == 'rm_nodes':
        filename = os.path.join(PATH_DIR, '{}/{}/graph-{}_b{}.pkl'.format(method, method1, gname, budget_nodes))
    elif method == 'isolation' and method1 == '':
        filename = os.path.join(PATH_DIR, '{}/graphs/{}/graph_leiden_splits0.pkl'.format(method, gname))
    elif method == 'isolation' and method1 == 'after_nodesplit':
        filename = os.path.join(PATH_DIR, '{}/graphs/{}/graph_leiden_splits{}.pkl'.format(method, gname, splits))
    else:
        print("Incorrect method! ")
        print(method, method1)
        return None

    if not os.path.exists(filename):
        print("file does not exist!")
        print(filename)
        return None

    return filename


def compute_eigen_value(g):
    # A is a csr_sparse matrix
    A = g.get_adjacency_sparse()
    A = A.astype(float)

    eigen_val, _ = eigs(A, k=1, which='LM')
    eigen_val = abs(eigen_val[0])  # it is in a list format

    return eigen_val


def get_eigenvalue(method, method1, gname, splits, budget_edges, budget_nodes):
    g_filename = get_filename(method, method1, gname, splits, budget_edges, budget_nodes)
    if not g_filename:  exit()

    #read graph from file
    print("\nReading graph from: ", g_filename)
    g = Graph.Read_Pickle(g_filename)
    eigen_val = compute_eigen_value(g)

    return eigen_val


if __name__ == '__main__':

    model = 'SIIDR'
    # num_steps =1000
    num_steps = 100

    # defenses plotted
    defenses_eigen = [('nodefense', ''), ('met', 'after_nodesplit')]
    defenses_fragm = [('rm_nodes', 'degree_iterative'), ('rm_nodes', 'ens'), ('isolation', ''), ('isolation', 'after_nodesplit')]
    defenses_tuples = defenses_eigen + defenses_fragm

    defenses_labels = ['no defense', 'NodeSplit+MET', 'Degree', 'ENS', 'CI-Edge', 'NodeSplit+CI-Edge']
    
    defenses= [os.path.join(defense_t[0], defense_t[1]) for defense_t in defenses_tuples]
    print(defenses)

    # filtering out some lines
    if plotFootprint:
        defense_indexes = [2, 5, 4, 0]
    else:
        defense_indexes = [2, 3, 1, 5, 0]

    # budgets
    budget_edges = 10
    budget_nodes = 50
    splits = 50

    lambda_1 = dict()
    gname = os.path.splitext(GRAPH_FILE)[0]

    # lambda 1 of the graphs for each method, used to compute effective attack strength "s"
    for i in defense_indexes:
        d_label = defenses_labels[i]
        d = defenses_tuples[i]
        method = d[0]
        method1 = d[1]
        lambda_1[d_label] = get_eigenvalue(method, method1, gname, splits, budget_edges, budget_nodes)

    print("lambda_1: ", lambda_1)

    footprint = dict()
    s_vals = dict()

    #  for each wannacry trace, get the attack parameters
    for param_index in range(len(siidr_params)):
        trace_name = list(siidr_params.keys())[param_index]
        params = list(siidr_params.values())[param_index]
        #print(params)
        beta = params[0]
        mu = params[1]
        gamma1 = params[2]
        gamma2 = params[3]
   
        print("plotting: ", trace_name, beta, mu, gamma1, gamma2)

        #  get the plots
        footprint[trace_name], s_vals[trace_name] = plot_all_data(gname, model, num_steps, defenses, defenses_labels, defense_indexes, lambda_1, trace_name, beta, mu, gamma1, gamma2)
    
    # get infection footprint values, in case we want to use it in a table
    print(pd.DataFrame.from_dict(footprint).transpose())
    #print(pd.DataFrame.from_dict(s_vals).transpose())

