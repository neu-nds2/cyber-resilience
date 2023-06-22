import random
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import igraph as ig
from igraph import *

import argparse
import math
import time

sys.path.append(os.path.abspath("../"))
from constants import PATH_DIR

# the parameters are, in order: 
# name of attack trace: (beta (infection probability),  mu (recovery rate), gamma1, gamma2)

siidr_params = {
        'wc_1_500ms': (0.10, 0.06, 0.76, 0.04),
        'wc_1_1s': (0.11, 0.07, 0.71, 0.07),
        'wc_1_5s': (0.37, 0.52, 0.27, 0.44),
        'wc_1_10s': (0.12, 0.06, 0.75, 0.05),
        'wc_4_1s': (0.14, 0.07, 0.75, 0.08),
        'wc_4_5s': (0.12, 0.07, 0.76, 0.07),
        'wc_8_20s': (0.13, 0.09, 0.74, 0.08)
}


class SPM_Simulation:
    
    def __init__(self, graph, spm_model, runs, steps, beta, mu, gamma1, gamma2, alpha, num_experiments, results_dir, **kwargs):
        
        self.graph = graph
        self.graph_og = graph.copy()
        
        self.spm_model = spm_model

        self.results_dir = results_dir
        
        self.runs = runs
        self.steps = steps
        
        self.num_experiments = num_experiments
        
        #S -> I (SI, SIS, SIR, SIIDR) probability
        self.beta = beta
        
        #I -> R (SIR, SIIDR) or I -> S (SIS)probability
        self.mu = mu
        
        #I -> ID probability
        self.gamma1 = gamma1
        
        #ID -> I probability
        self.gamma2 = gamma2
        
        self.alpha = alpha

        self.susceptible = set()
        self.infected = set()
        self.recovered = set()
        self.infected_dormant = set()
        
    def run_single_sim(self):
    
        for step in range(self.steps):

            self.track_simulation(step)
            if len(self.infected) == 0 and len(self.infected_dormant) == 0:
                continue

            print("susceptible={}, infected={}, infected_dormant={}, recovered={}, graph={}".format(len(self.susceptible), len(self.infected), len(self.infected_dormant), len(self.recovered), len(self.graph.vs)))
            
            infected_new = set()
            
            #Susceptible - Infected model
            if self.spm_model == 'SI':
                #get infected nodes, get susceptible neighbors (exclude infected), infect the with p = beta
                for node in self.infected:
                    nbrs = self.graph.neighbors(node, mode = 'out')
                    nbrs = set(nbrs).difference(self.infected)

                    nbrs_infected = set([n for n in nbrs if random.random() <= self.beta])
                    
                    infected_new |= nbrs_infected
                    
                self.infected |= infected_new
                self.susceptible = self.susceptible.difference(infected_new)
                
            #Susceptible - Infected - Susceptible model
            elif self.spm_model == 'SIS':

                #get infected nodes, get susceptible neighbors (exclude infected), infect the with p = beta
                for node in self.infected:
                    nbrs = set(self.graph.neighbors(node, mode = 'out'))
                    nbrs &= self.susceptible
                    nbrs_infected = set([n for n in nbrs if random.random() <= self.beta])
                    infected_new |= nbrs_infected

                #get infected nodes, cure (move to susceptible) them with p = mu
                cured = set([n for n in self.infected if random.random() <= self.mu])
                self.infected -= cured
                self.infected |= infected_new
                self.susceptible -= infected_new
                self.susceptible |= cured

            #Susceptible - Infected - Recovered model
            elif self.spm_model == 'SIR':
                #get infected nodes, get susceptible neighbors (exclude infected and recovered),
                #infect the with p = beta
                for node in self.infected:
                    nbrs = self.graph.neighbors(node, mode = 'out')
                    nbrs = set(nbrs).difference(self.infected).difference(self.recovered)
                    nbrs_infected = set([n for n in nbrs if random.random() <= self.beta])
                    infected_new = infected_new.union(nbrs_infected)
                
                #get infected nodes, exclude newly infected nodes, cure (move to recovered) them with p = mu
                cured = set([n for n in self.infected if random.random() <= self.mu])
                self.infected = self.infected.union(infected_new)
                self.infected = self.infected.difference(cured)
                self.susceptible = self.susceptible.difference(infected_new)
                self.recovered = self.recovered.union(cured)
                
                
            #Susceptible - Infected - Infected Dormant - Recovered model
            else:
                #new infections with p = beta
                for node in self.infected:
                    if len(self.susceptible) != 0:
                        nbrs = set(self.graph.neighbors(node, mode = 'out'))
                        nbrs &= self.susceptible
                        nbrs_infected = set([n for n in nbrs if random.random() <= self.beta])
                        infected_new |= nbrs_infected
        
                #moving from infected_dormant to infected with p = gamma2
                dormant_to_infected = set([n for n in self.infected_dormant if random.random() <= self.gamma2])
    
                #leaving infected to infected dormant and recovered
                if len(self.susceptible) != 0:
                    infected_leaving = set([n for n in self.infected if random.random() <= self.gamma1 + self.mu])
                    recovered_prob = self.mu     / (self.mu + self.gamma1)
                    dormant_prob = self.gamma1 / (self.mu + self.gamma1)
                    recovered_new  = set([n for n in infected_leaving if random.random() <= recovered_prob])
                    dormant_new  = infected_leaving - recovered_new

                #updating sets
                self.infected -= dormant_new
                self.infected -= recovered_new
                self.susceptible -= infected_new
                self.infected |= (infected_new | dormant_to_infected)
                self.infected_dormant -= dormant_to_infected
                self.infected_dormant |= dormant_new
                self.recovered |= recovered_new

                if len(self.susceptible) + len(self.infected) + len(self.infected_dormant) + len(self.recovered) != len(self.graph.vs):
                    print("infected_new={}, dormant_new={}, dormant_to_infected={}, recovered_new={}".format(len(infected_new), len(dormant_new), len(dormant_to_infected), len(recovered_new)))
                    print("Not correct!! susceptible={}, infected={}, infected_dormant={}, recovered={}, graph={}".format(len(self.susceptible), len(self.infected), len(self.infected_dormant), len(self.recovered), len(self.graph.vs)))
                    exit()

        history = dict()
        history["susceptible"] = [v['susceptible_num'] for k, v in self.sim_info.items()]
        history["infected"] = [v['infected_num'] for k, v in self.sim_info.items()]
        history["infected_dormant"] = [v['infected_dormant_num'] for k, v in self.sim_info.items()]
        history["recovered"] = [v['recovered_num'] for k, v in self.sim_info.items()]

        return history
    
    
    def track_simulation(self, step):

        self.sim_info[step] = {
            'infected_num': len(self.infected),
            'susceptible_num': len(self.susceptible),
            'recovered_num': len(self.recovered),
            'infected_dormant_num': len(self.infected_dormant)
        }

    def reset_simulation(self, num_experiment, starting_points):

        self.graph = self.graph_og.copy()
        nodes_set = set([elem.index for elem in self.graph.vs])
  
        # initially, a fraction of the nodes are infected
        self.infected = set(starting_points)
        self.susceptible = nodes_set - self.infected

        self.recovered = set()
        self.infected_dormant = set()

        self.sim_info = defaultdict()
        # print("num_experiment, starting point: ", num_experiment, self.starting_points[num_experiment])
        print("num_experiment", num_experiment)
        print("num_infected, num nodes_set: ", len(self.infected), len(nodes_set))


    def run_simulation(self, num_experiment, starting_points):
        """
        Averages the simulation over the number of 'runs'.
        :return: a list containing the average value at each 'step' of the simulation.
        """
        
        compartments = ["susceptible", "infected", "infected_dormant", "recovered"]
        sim_results = list(range(self.runs))
              
        for r in range(self.runs):
            # print('simulation number', r)
            self.reset_simulation(num_experiment, starting_points)
            sim_results[r] = self.run_single_sim()

        # average over runs, which keep the same starting point and only vary the attack params beta, mu, gammas
        # num runs is usually set to 1 in our tests; 
        # instead we'll average over the number of experiments, which also vary the starting point in addition to params
        avg_results_dict = dict()
        for c in compartments:
            avg_results = []
            for t in range(self.steps):
                avg_results.append(np.mean([sim_results[r][c][t] for r in range(self.runs)]))
            avg_results_dict[c] = avg_results
          
        return avg_results_dict
    

    def run_simulation_over_experiments(self, frac_infected=None, lcc_only=False):
        
        compartments = ["susceptible", "infected", "infected_dormant", "recovered"]

        sim_exp_results = list(range(self.num_experiments))
        
        lcc_vs = self.graph.clusters().giant().vs

        for r in range(self.num_experiments):
            print('experiment number {}, num_runs = {}'.format(r, self.runs))
            
            if lcc_only: # run attack within the largest connected component
                nodes_set = set([elem.index for elem in lcc_vs])
            else:  # run attack within the whole graph
                nodes_set = set([elem.index for elem in self.graph.vs])

            # randomly choose a starting point
            # initial infection for this experiment
            if frac_infected == None:  # just one node
                starting_points = set(np.random.choice(list(nodes_set), size = 1, replace = False))
            else: # a fraction of nodes
                starting_points = set(np.random.choice(list(nodes_set), size = int(frac_infected * len(nodes_set)), replace = False))

            print("new exp, starting points:", r, starting_points)
 
            sim_exp_results[r] = self.run_simulation(r, starting_points)
            self.save_results(sim_exp_results, r)


    def save_results(self, sim_results, num_exp):
        
        compartments = ["susceptible", "infected", "infected_dormant", "recovered"]
        results_file = os.path.join(self.results_dir, "{}_{}steps_beta_{}_mu_{}_gamma1_{}_gamma2_{}.csv".format(self.spm_model, self.steps, self.beta, self.mu, self.gamma1, self.gamma2))
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        g_num_nodes = len(self.graph.vs)

        info = []

        for t in range(self.steps):

            # form one row in the info df; contains all the compartments info
            crt_step = [t]
            
            # average number of nodes within each compartment, over all experiments
            for c in compartments:
                if num_exp == 0:
                    avg_count = sim_results[0][c][t]
                    stdev_count = 0
                else:
                    avg_count = np.mean([sim_results[r][c][t] for r in range(num_exp)])
                    stdev_count = np.std([sim_results[r][c][t] for r in range(num_exp)])

                avg_frac = avg_count / g_num_nodes
                stdev_frac =  stdev_count / g_num_nodes
                
                crt_step.extend([avg_count, avg_frac, stdev_frac])

            info.append(crt_step)
        
        col_headers = ['step']
        for c in compartments:
            col_headers.extend(['{}_count'.format(c), '{}_frac'.format(c), '{}_stdev'.format(c)])

        info_df = pd.DataFrame(info, columns=col_headers)
        # info_df = info_df.round(2)
        print(info_df)

        info_df.to_csv(results_file, index=False)
        print("propagation results in file: ", results_file)


# this is the main function, given a graph and a model, it runs all the simulations for different parameters
def run_spm(g, results_dir, frac_infected=None, lcc_only=False, m='SI', num_experiments=10, num_runs=10, num_steps=10, beta=0.5, mu=0.8, gamma1=0.25, gamma2=0.25, alpha=0):

    ds = SPM_Simulation(graph = g, spm_model = m, runs = num_runs, steps = num_steps,
                        beta = beta, mu = mu, gamma1 = gamma1, gamma2 = gamma2, alpha=alpha,
                        num_experiments = num_experiments, results_dir = results_dir)

    ds.run_simulation_over_experiments(frac_infected, lcc_only)


def get_filename(method, method1, port, splits, budget_edges, budget_nodes):
    day = "2021_11_08"
    if method == 'nodefense':
        filename = os.path.join(PATH_DIR, 'graph_port{}_{}.pkl'.format(port, day))
    elif method == 'nodesplit':
        filename = os.path.join(PATH_DIR, '{}/port{}/graph_p{}_{}_splits{}.pkl'.format(method, port, port, method, splits))
    elif method == 'rm_edges' and method1 == 'random':
        filename = os.path.join(PATH_DIR, '{}/{}/graph-port{}_b{}.pkl'.format(method, method1, port, budget_edges))
    elif method == 'met' and method1 == '':
        filename = os.path.join(PATH_DIR, '{}/graph-port{}_b{}.pkl'.format(method, port, budget_edges))
    elif method == 'met' and method1 == 'after_nodesplit':
        filename = os.path.join(PATH_DIR, '{}/{}/port{}/splits{}/graph-port{}_b{}.pkl'.format(method, method1, port, splits, port, budget_edges))
    elif method == 'rm_nodes':
        filename = os.path.join(PATH_DIR, '{}/{}/graph-port{}_b{}.pkl'.format(method, method1, port, budget_nodes))
    elif method == 'isolation' and method1 == '':
        filename = os.path.join(PATH_DIR, '{}/graphs/port{}/graph_p{}_leiden_splits0.pkl'.format(method, port, port))
    elif method == 'isolation' and method1 == 'after_nodesplit':
        filename = os.path.join(PATH_DIR, '{}/graphs/port{}/graph_p{}_leiden_splits{}.pkl'.format(method, port, port, splits))
    else:
        print("Incorrect method! ")
        print(method, method1)
        return None

    if not os.path.exists(filename):
        print("file does not exist!")
        print(filename)
        return None

    return filename


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("PORT", help="The infected port.")
    parser.add_argument("INDEX", help="Defense method, by index.")
    parser.add_argument("PARAM_INDEX", help="Parameters set, by index.")
    args = parser.parse_args()
    print(args)

    port = int(args.PORT)
    index = int(args.INDEX) 
    param_index = int(args.PARAM_INDEX) 
    print(port, index, param_index)

    seed = 1
    random.seed(seed)
    np.random.seed(seed)

    # all defenses
    # defenses_eigen = [('nodefense', ''), ('nodesplit', ''), ('rm_edges', 'random'),  ('met', ''), ('met', 'after_nodesplit')]
    # defenses_fragm = [('rm_nodes', 'random'), ('rm_nodes', 'degree_iterative'), ('rm_nodes', 'ens'), ('rm_nodes', 'xnb_iterative'), ('isolation', ''), ('isolation', 'after_nodesplit')]

    defenses_eigen = [('nodefense', ''), ('nodesplit', '')]
    defenses_fragm = [('rm_nodes', 'degree_iterative'), ('isolation', ''), ('isolation', 'after_nodesplit')]

    defenses = defenses_eigen + defenses_fragm

    method = defenses[index][0]
    method1 = defenses[index][1]
    defense = os.path.join(method, method1)

    model = 'SIIDR'

    #  budgets
    budget_nodes = 50
    splits = 50
    budget_edges = 10
    
    num_experiments = 500 # number of experiments, each having a different starting point
    num_runs = 1 # we do num_runs for each starting point
    num_steps = 100  # number of simulation steps
    # num_steps = 1000  # number of simulation steps; make sure footprint  converges
    frac_infected = None  # a single starting point (default), not a fraction
    lcc_only = False

    print("port = {}, defense = {}".format(port, defense))
    print("num_experiments = {}, num_runs = {}, num_steps = {}, frac_infected = {}".format(num_experiments, num_runs, num_steps, frac_infected))
    print("budget_edges = {}, budget_nodes = {}, splits = {}".format(budget_edges, budget_nodes, splits))

    g_filename = get_filename(method, method1, port, splits, budget_edges, budget_nodes)
    if not g_filename:  exit()

    #read graph from file
    print("\nReading graph from: ", g_filename)
    g = Graph.Read_Pickle(g_filename)
   
    if param_index >= 0:
        lparams = [list(siidr_params.values())[param_index]]
    else:
        lparams = siidr_params.values()

    for params in lparams:
        beta = params[0]
        mu = params[1]
        gamma1 = params[2]
        gamma2 = params[3]
        results_dir = os.path.join(PATH_DIR, "siidr_attack/port{}/{}/{}/".format(port, defense, list(siidr_params.keys())[param_index]))
        run_spm(g, results_dir, frac_infected, lcc_only, model, num_experiments, num_runs, num_steps, beta, mu, gamma1, gamma2, alpha=0)


