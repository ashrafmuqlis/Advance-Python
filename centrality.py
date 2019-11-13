import matplotlib.pyplot as plt
import networkx as nx
G = nx.karate_club_graph()
G = nx.convert_node_labels_to_integers(G,first_label=1)
#Display the degree
print("Node Degree")
for v in G:
    print('%s %s' % (v, G.degree(v)))

nx.draw_circular(G, with_labels=True)
plt.show()
## #nodes: 34 and #edges: 78
print('Total no of nodes', len(G.nodes()), 'and', 'Total no of edges:', len(G.edges()))

## calculate degree centrality,
degree_centrality = nx.degree_centrality(G)
degree_centrality
#Closeness centrality of a node is just the average of all of those shortest paths. 

closeness_centrality = nx.closeness_centrality(G)
closeness_centrality
#between centrality
nx.betweenness_centrality(G,normalized = True, endpoints = False)
nx.betweenness_centrality(G, normalized = True, endpoints = False, k = 10)
nx.betweenness_centrality_subset(G,[34, 33, 21, 30,16, 27, 15, 23, 10], [1, 4, 13, 11, 6, 12, 17, 7],normalized=True)
nx.edge_betweenness_centrality(G, normalized=True)
nx.edge_betweenness_centrality_subset(G, [34, 33,21, 30, 16, 27, 15, 23, 10], [1, 4, 13, 11, 6, 12, 17,7], normalized=True)
     

#HITS Algorithm
G1=nx.path_graph(4)
nx.draw_networkx(G1)
h=nx.hits(G1)
h
