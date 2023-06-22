import sys
import os

# add the parent directory to the path
sys.path.insert(0, os.path.abspath("../"))

dir_path = os.path.dirname(os.path.realpath(__file__))

PATH_DIR = os.path.join(dir_path, "sample_data_and_results")

# name of the file containing  the input graph 
GRAPH_FILE = "oregon1_010526.txt"


