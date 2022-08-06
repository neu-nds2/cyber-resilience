from  scipy import sparse
from scipy.sparse import csc_matrix, triu
from scipy.sparse.linalg import svds
import numpy as np

from igraph import *

sys.path.append(os.path.abspath("../../"))
from constants import PATH_DIR, GRAPH_FILE

budget = 10
splits = 0

gname = os.path.splitext(GRAPH_FILE)[0]

if splits == 0:
    filename = os.path.join(PATH_DIR, gname + ".pkl")
else:
    filename = os.path.join(PATH_DIR, "nodesplit/{}/graph_splits{}.pkl".format(gname, splits))

print("Reading full graph from: ", filename)

g = Graph.Read_Pickle(filename)

A = g.get_adjacency_sparse()
A = A.astype(float)

#Find the list of edges that we want to remove
E = []
#Take statics how many times we recompute scores
RC = 0 
#Record when we recompute scores
pointRC = [] 
Error = 0

#Begin with two singular values
nSV = 2
trackNSV = [nSV]
#Tracking all matrices
U = []
S = []
V = []
#At beginning, this gap is updated in subsequent recompuations
previousGap = 1000000 

#budget on edge removal
nEdges_port = len(g.es())
k = np.round(nEdges_port * budget/100)
print('k', k)

epsilon = 0.5
first_lambda = 0
maxSV = 0

while len(E) < k:
    
    print("Eigenvalue round", RC)

    Atr = triu(A, 1)
    (a, b) = Atr.nonzero()
    nEdges = len(a)   # no need to divide by 2, because we are working on triu not A

    print("nEdges: ", nEdges)

    score = np.zeros((nEdges, nSV))
    U, S, Vt = svds(A, k = nSV + 1, which='LM')
    
    print('S',S)

    p = S.argsort()
    
    S = S[p]
    U = U[:,p]

    RC = RC + 1
    
    SVs= S

    if (len(E) == 0):
        first_lambda = max(SVs)
           
    pivot = min(SVs)
    pivot_idx = np.argmin(SVs)
    
    SVs = SVs[SVs != pivot] 
    
    U = np.delete(U, pivot_idx, 1)  
 
    ## undirected graph, using abs  ----------
    for i in range(nSV):
        Ui = abs(U[:, i])  
        score[:, i] = 2 * Ui[a] * Ui[b]
 
    print("score: ", score)

    currentGap = max(SVs) - pivot
    
    if currentGap > previousGap:
        newnSV = nSV - 1
    
    elif currentGap < previousGap:
        newnSV = nSV + 1;
    
    if newnSV < 1:
        newnSV = 1
            
    previousGap = currentGap
        
    maxSV = max(SVs)
    maxSVIdx = np.argmax(SVs)

    print("SVs: ", SVs)
    print("pivot, maxSVIdx: ", pivot, maxSVIdx)

    while (maxSV > (pivot - epsilon)) and (len(E) < k):
        
        # print("maxSV, pivot-epsilon: ", maxSV, (pivot-epsilon))

        edgeIdx = np.argmax(score[:, maxSVIdx])

        for j in range(nSV):
            pickScore = score[edgeIdx, j] 

            SVs[j] = SVs[j] - pickScore
            score[edgeIdx, j] = 0 
        
        # mark edge as removed in adj matrix
        ## changed here ----------
        A[a[edgeIdx], b[edgeIdx]] = 0
        A[b[edgeIdx], a[edgeIdx]] = 0
        E.append((a[edgeIdx], b[edgeIdx]))        
        
        maxSV = max(SVs)
        maxSVIdx = np.argmax(SVs)
        # print("maxSV: ", maxSV)

    nSV = newnSV

E_names = []  
for edge in E:
    source_name = g.vs(int(edge[0]))['name'][0]
    target_name = g.vs(int(edge[1]))['name'][0]    
    E_names.append((source_name, target_name))

print("will remove edges: ", len(E_names))

g.delete_edges(E_names)
A = g.get_adjacency_sparse()
A = A.astype(float)
U, S, Vt = svds(A, k = nSV+1, which='LM')
SVs= S
maxSV = max(SVs)
drop = 100 * (first_lambda - maxSV)/first_lambda
print("% drop: ", drop, "lambda orig: ", first_lambda, "lambda final: ", maxSV)

if splits == 0:
    basedir = os.path.join(PATH_DIR, 'rm_edges/met')
else:
    basedir = os.path.join(PATH_DIR, 'rm_edges/met/after_nodesplit/{}/splits{}'.format(gname, splits))

filename = os.path.join(basedir, 'graph_b{}.pkl'.format(budget))
os.makedirs(os.path.dirname(filename), exist_ok=True)

g.save(filename, format = 'pickle')
print("final graph saved to " + filename) 


