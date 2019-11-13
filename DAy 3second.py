import networkx as nx
# Undirected Graph
G = nx.Graph()
G.add_edge('A','B')
G.add_edge('B','C')
nx.draw_networkx(G)



# Directed Graph
D = nx.DiGraph()
D.add_edge('B','A')
D.add_edge('B','C')
# Check
D.is_directed()
# True
nx.draw_networkx(D)



# Multi-Graph
M = nx.MultiGraph()
M.add_edge('B','A')
M.add_edge('B','C')
# Check
M.is_multigraph()
# True
nx.draw_networkx(M)




# Directed Multi-Graph
DM = nx.MultiDiGraph()
DM.add_edge('B','A')
DM.add_edge('B','C')
# Check
DM.is_multigraph()
# True
nx.draw_networkx(DM)



#To create a bipartite graph:
from networkx.algorithms import bipartite
B = nx.Graph()
# Add nodes
B.add_nodes_from(['A','B','C','D','E'], bipartite=0)
B.add_nodes_from([1,2,3,4], bipartite=1)
# Add edges
B.add_edges_from([('A',1),('B',1),('C',1),('C',3),('D',2),('E',3),('E',4)])
#CHeck
bipartite.is_bipartite(B)
# True
#Projected Bipartite Graph
B = nx.Graph()
B.add_edges_from([('A',1), ('B',1), ('C',1),('D',1),('H',1), \
                    ('B', 2), ('C', 2), ('D', 2),('E', 2), ('G', 2), ('E', 3), \
                    ('F', 3), ('H', 3), ('J', 3), ('E', 4), ('I', 4), ('J', 4) ])

# set of nodes to generate a projected graph from a partition
X = set(['A','B','C','D', 'E', 'F','G', 'H', 'I','J'])
P = bipartite.projected_graph(B, X)
nx.draw_networkx(P)





#EDGE TYPES
# Weighted Edges
W = nx.Graph()
W.add_edge('A','B', weight=5)
W.add_edge('B','C', weight=6)
nx.draw_networkx(W)
# Signed Edges
S = nx.Graph()
S.add_edge('A','B', sign='+')
S.add_edge('B','C', sign='-')
nx.draw_networkx(S)
# Edge Attributes
R = nx.Graph()
R.add_edge('A','B', relation='friend')
R.add_edge('B','C', relation='coworker')
R.add_edge('B','D', relation='family')
R.add_edge('A','B', relation='friend', weight=5)
nx.draw_networkx(R)



#Node Atrributes
G=nx.MultiGraph()
G.add_node('A',role='manager')
G.add_node('B',role='employee')
G.node['A']['role'] = 'team member'
G.node['B']['role'] = 'engineer'
nx.draw_networkx(G)


