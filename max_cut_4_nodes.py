import qaoa_methods
import operators
import networkx as nx
import numpy as np

n = 4 # Number of Nodes
p = 6 # Number of layer

g = nx.Graph()
g.add_nodes_from(np.arange(0, n, 1))
elist=[(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
g.add_weighted_edges_from(elist)

qaoa_methods.qaoa(n, g, layer=p, pool=operators.qaoa())
