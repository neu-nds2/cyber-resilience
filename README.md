## Cyber Network Resilience against Self-Propagating Malware Attacks

This is the code for the paper: 

A. Chernikova, N. Gozzi, S. Boboila, P. Angadi, J. Loughner, M. Wilden, N. Perra, T. Eliassi-Rad, and A. Oprea, "Cyber Network Resilience against Self-Propagating Malware Attacks", In Proceedings of the 27th European Symposium on Research in Computer Security (ESORICS) 2022.

Project goal: Improve network resilience by designing defense strategies which leverage the network topology. The topology is  derived from the network communication traffic.

#### Sample Data and Results

Sample data and results can be found in the "sample_data_and_results" directory. We exemplify the code use on the public dataset Oregon-1 (file oregon1_010526.txt) from "http://snap.stanford.edu/data/index.html".


#### Code Structure

Auxiliary Files:

* constants.py - The path directory.

Directories:

* prepare_data
* defense
* attack

##### Preparing Data (prepare_data directory)

* `construct_graph.py`: Constructs the communication graph using the igraph API and saves it in a Pickle format (.pkl)

* `analysis.py`: Computes some topological metrics for a graph, such as number of nodes/edges, average path length, average degree, diameter, density, etc. See graph properties for the Oregon dataset in "sample_data_and_results/graph_metrics"

* `plot_degrees.py`: Plots the degree distribution of a graph, and fits the distribution to a power law and an exponential distribution. See the degree distribution of the Oregon dataset in "sample_data_and_results/plots/degrees_distribution"

* `compute_ens.py`: Computes the effective network size metric (ENS) for each node in the communication graph. 

##### Defenses (defense directory)

The budget and  port needs to be set within each script itself. Default values are already there.
Plotting scripts are available for each defense, within each method's directory.

* NodeSplit (nodesplit directory)

  * `nodesplit.py`: Iteratively split the top node in two, move half of its edges to a newly created node. 

* Edge Hardening (rm_edges directory)

  * `rm_edges_random.py`: Randomly remove a number of edges given by the budget.

  * `met_undirected.py` : Runs the MET algorithm by Le et. al described in "MET: A Fast Algorithm for Minimizing Propagation in Large Graphs with Small Eigen-Gaps".

* Node Hardening (rm_nodes directory)

  * `rm_rand.py`: Randomly remove a number of nodes given by the budget. 

  * `rm_top_degree_iterative.py`: Progressively remove the node with highest degree, until the budget is reached.

  * `rm_top_ens.py`: Remove top nodes with highest effective network size. 
  
* Isolation (isolation directory)

  * `isolate_communities.py`: Extract communities using either Leiden, Infomap or Leading Eigenvector algorithms, and then harden their borders, e.g., harden all inter-community edges in order to separate the communities.


##### Attack (attack directory)

`SPM_with_real_params.py`: 
  * Implements the four SPM models -- SI, SIS, SIR, SIIDR -- within the Simulation class.
 
  * It provides the functionality to run SPM attacks, e.g. SIIDR using the attack parameters given in the same file. The parameters represent infection probability (beta), recovery rate (mu), gamma1, gamma2. 

  * The start point of the attack is chosen randomly. 

  * Several other parameters are set here, including the number of experiments (default 500), the number of time steps during simulation, the budgets used to build the resilent graphs, etc. 

`run_attack.sh`: It is used to run an attack by setting up some command  line parameters:  
  * defense index: graph to run the attack on, for example index 0 is  the graph with 'no defense', and index 1 is the resulting graph after nodesplit was applied, etc. This is the index in the list called 'defenses' from the main script, i.e., `SPM_with_real_params.py`
  
  * param index: attack propagation parameters, e.g., param_index 2 is the set of parameters corresponding to wannacry trace wc_1_5s, etc. The correspondence is set up in the siidr_params data structure from `SPM_with_real_params.py`.


#### Running the Code

We run the code as regular python scripts. Here is an example:

cd defenses/nodesplit

python -u nodesplit.py &> out.txt 

To run the attack, we run the bash script as follows:

cd attack

sh run_attack.sh


#### Sample Results

* Several plots are found within the sample_data_and_results/plots directory, comparing the performance of various defenses on the sample Oregon data.


