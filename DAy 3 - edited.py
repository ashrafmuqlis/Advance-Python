#pip install networkx
import networkx as nx
G = nx.Graph() 
G.add_node('A') 
G.add_node('B')
G.add_node('C')
G.add_node('D')
G.add_edge('A','B',relation='friend') 
G.add_edge('B','C',relation='neighbor') 
G.add_edge('B','D',relation='friend')
G.add_edge('A','D',relation='friend')  
G.add_node('A',role="Trader")
G.add_node('B',role="Trader")
G.add_node('C',role="Manager")
G.add_node('D',role="Manager")
nx.draw_networkx(G)

#ADD COLORS to the node
nodes=nx.draw_networkx_nodes(G,pos=nx.spring_layout(G),nodelist=[1, 2, 3, 4],node_color=["#A0CBE2", "#A0CBE2", "#FF0000", "#FFFF00"])
nx.draw_networkx_nodes(G,node_color=["#A0CBE2", "#A0CBE2", "#FF0000", "#FFFF00"])
#nx.draw_spring(X, font_size=20, width=2,node_color=["#A0CBE2", "#A0CBE2", "#FF0000", "#FFFF00"])
        
                                     
#DRAW WEIGHTED GRAPH                                     
import networkx as nx
import matplotlib.pyplot as plt
G=nx.Graph()
i=1
G.add_node(i,pos=(i,i))
G.add_node(2,pos=(2,2))
G.add_node(3,pos=(1,0))
G.add_edge(1,2,weight=0.5)
G.add_edge(1,3,weight=9.8)
pos=nx.get_node_attributes(G,'pos')
nx.draw_networkx(G,pos)
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
plt.show()                                                 


#Exercise 1
import networkx as nx
import matplotlib.pyplot as plt
Weight_G = nx.Graph() 
Weight_G.add_node('A',pos=(2,7))             
Weight_G.add_node('B',pos=(2.5,6))             
Weight_G.add_node('C',pos=(2,5))             
Weight_G.add_node('D',pos=(1.5,4))             
Weight_G.add_node('E',pos=(2.5,3))             
Weight_G.add_node('F',pos=(3,4))             
Weight_G.add_node('G',pos=(3,7.5))             
Weight_G.add_node('H',pos=(3,2.333))             
Weight_G.add_node('I',pos=(2,3))             
Weight_G.add_node('J',pos=(2.5,1))             
Weight_G.add_edge('A','B', weight = 6)
Weight_G.add_edge('B','C', weight = 13)
Weight_G.add_edge('C','F', weight = 21)
Weight_G.add_edge('F','G', weight = 9)
Weight_G.add_edge('C','E', weight = 25)
Weight_G.add_edge('D','E', weight = 2)
Weight_G.add_edge('E','H', weight = 9)
Weight_G.add_edge('E','J', weight = 15)
Weight_G.add_edge('E','I', weight = 10)
Weight_G.add_edge('I','J', weight = 3)
pos = nx.get_node_attributes(Weight_G,'pos')
nx.draw_networkx(Weight_G,pos)
labels = nx.get_edge_attributes(Weight_G,'weight')
nx.draw_networkx_edge_labels(Weight_G,pos,edge_labels=labels)
plt.show()


#Exercise 2
import networkx as nx
from networkx.algorithms import bipartite
B = nx.Graph()
B.add_nodes_from(['Keri','Gabriel','Susan','Nate','Abrianna'], bipartite=0) # Add the node attribute "bipartite"
B.add_nodes_from(['Lucy','Dan','George','Marilyn'], bipartite=1)
B.add_edges_from([('Keri','Lucy'), ('Keri','Marilyn'), ('Gabriel','Lucy'), ('Gabriel','George'), ('Susan','Marilyn'), ('Nate','Dan'), ('Nate','George'), ('Abrianna','Lucy'), ('Abrianna','George')])
nx.is_connected(B)
bipartite.is_bipartite(B)
bottom_nodes, top_nodes = bipartite.sets(B)
G = bipartite.projected_graph(B, top_nodes)
nx.draw_networkx(G)

#Exercise 3
import networkx as nx
from networkx.algorithms import bipartite
B = nx.Graph()
B.add_nodes_from(['A','B','C','D','E'], bipartite=0) # Add the node attribute "bipartite"
B.add_nodes_from(['1','2','3','4','5'], bipartite=1)
B.add_edges_from([('A','1'), ('A','2'), ('A','3'), ('A','4'), ('A','5'), ('B','2'), ('C','3'), ('C','4'), ('C','5'), ('D','4'), ('E','5')])
nx.is_connected(B)
bipartite.is_bipartite(B)
bottom_nodes, top_nodes = bipartite.sets(B)
G = bipartite.projected_graph(B, top_nodes)
nx.draw_networkx(G)



#MULTIGRAPH
G = nx.MultiGraph()
G.add_edge('A','B',relation='friend')
G.add_edge('A','B',relation='neighbor')
G.add_edge('B','C',relation='coworker')
G.add_edge('C','F',relation='coworker')
G.add_edge('C','F',relation='friend')
nx.draw_networkx(G)

#SIGNED GRAPH
G = nx.MultiGraph()
G.add_edge('A','B',sign='+')
G.add_edge('B','C',sign='-')
G.add_edge('C','F',sign='-')
nx.draw_networkx(G)

#Bipartite Graph
from networkx.algorithms import bipartite
B = nx.Graph()
B.add_nodes_from([1,2,3,4], bipartite=0) # Add the node attribute "bipartite"
B.add_nodes_from(['a','b','c'], bipartite=1)
B.add_edges_from([(1,'a'), (1,'b'), (2,'b'), (2,'c'), (3,'c'), (4,'a')])
nx.is_connected(B)
bottom_nodes, top_nodes = bipartite.sets(B)
G = bipartite.projected_graph(B, top_nodes)
nx.draw_networkx(G)
#USING panda and matplotlib

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
 
# Build a dataframe with your connections
df = pd.DataFrame({ 'from':['A', 'B', 'C','A'], 'to':['D', 'A', 'E','C']})
 
# And a data frame with characteristics for your nodes
carac = pd.DataFrame({ 'ID':['A', 'B', 'C','D','E'], 'myvalue':['group1','group1','group2','group3','group3'] })
 
# Build your graph
G=nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.Graph() )
 
# The order of the node for networkX is the following order:
G.nodes()
# Thus, we cannot give directly the 'myvalue' column to netowrkX, we need to arrange the order!
 
# Here is the tricky part: I need to reorder carac to assign the good color to each node
carac= carac.set_index('ID')
carac=carac.reindex(G.nodes())
 
# And I need to transform my categorical column in a numerical value: group1->1, group2->2...
carac['myvalue']=pd.Categorical(carac['myvalue'])
carac['myvalue'].cat.codes
 
# Custom the nodes:
nx.draw(G, with_labels=True, node_color=carac['myvalue'].cat.codes, cmap=plt.cm.Set1, node_size=1500)


#Coefficient

import networkx as nx
G = nx.DiGraph()
G.add_edge('A','K')
G.add_edge('A','B')
G.add_edge('A','C')
G.add_edge('B','K')
G.add_edge('B','A')
G.add_edge('B','C')
G.add_edge('C','A')
G.add_edge('C','B')
G.add_edge('C','F')
G.add_edge('C','E')
G.add_edge('D','E')
G.add_edge('E','C')
G.add_edge('E','F')
G.add_edge('E','H')
G.add_edge('F','C')
G.add_edge('F','G')
#G.add_edge('I','J')
nx.draw_networkx(G)

#Clustering Coefficient
#G is a graph, A is a node
nx.clustering(G,'F')

#Global clustering
nx.average_clustering(G)

#Global clustering - Transitivity
nx.transitivity(G)

#Distance
nx.shortest_path(G,'A','H')
nx.shortest_path_length(G,'A','H')
nx.average_shortest_path_length(G)
nx.diameter(G)
nx.radius(G)
nx.periphery(G)
nx.center(G)

#Eccentricity
nx.eccentricity(G)

#Breadth-First Search
T=nx.bfs_tree(G,'E')
T.edges()
nx.draw_networkx(T)

#Karate Club 
G = nx.karate_club_graph()
G = nx.convert_node_labels_to_integers(G, first_label = 1)
nx.is_connected(G)
nx.number_connected_components(G)
nx.draw_networkx(G)
nx.diameter(G)
nx.radius(G)
nx.periphery(G)
nx.center(G)
nx.average_shortest_path_length(G)
nx.is_strongly_connected(G)
nx.is_weakly_connected(G)
nx.strongly_connected_components(G)
nx.node_connectivity(G)
nx.minimum_node_cut(G)
nx.edge_connectivity(G)
nx.minimum_edge_cut(G)

#Exercise
import networkx as nx
G = nx.Graph()
G.add_edge('E','F')
G.add_edge('E','C')
G.add_edge('E','D')
G.add_edge('E','B')
G.add_edge('E','G')
G.add_edge('E','H')
G.add_edge('F','A')
G.add_edge('F','C')
G.add_edge('F','E')
G.add_edge('C','F')
G.add_edge('C','A')
G.add_edge('C','E')
G.add_edge('C','B')
G.add_edge('B','A')
G.add_edge('B','C')
G.add_edge('B','E')
G.add_edge('B','D')
G.add_edge('D','B')
G.add_edge('D','E')
G.add_edge('G','E')
G.add_edge('G','H')
G.add_edge('G','M')
G.add_edge('G','K')
G.add_edge('K','G')
G.add_edge('K','M')
G.add_edge('K','N')
G.add_edge('N','H')
G.add_edge('N','K')
G.add_edge('H','E')
G.add_edge('H','G')
G.add_edge('H','M')
G.add_edge('H','N')
nx.draw_networkx(G)
nx.is_strongly_connected(G)
nx.is_weakly_connected(G)
nx.strongly_connected_components(G)
nx.node_connectivity(G)
nx.minimum_node_cut(G)
nx.edge_connectivity(G)
nx.minimum_edge_cut(G)


#Exercise
import networkx as nx
import matplotlib.pyplot as plt
G = nx.Graph()
G.add_edge('A','H')
G.add_edge('A','G')
G.add_edge('A','B')
G.add_edge('B','A')
G.add_edge('B','C')
G.add_edge('C','B')
G.add_edge('C','D')
G.add_edge('C','E')
G.add_edge('D','F')
G.add_edge('D','C')
G.add_edge('F','D')
G.add_edge('F','G')
G.add_edge('F','I')
G.add_edge('G','A')
G.add_edge('G','H')
G.add_edge('G','I')
G.add_edge('G','F')
G.add_edge('H','A')
G.add_edge('H','G')
degrees = G.degree()
degree_values = sorted(set(dict(degrees).values()))
histogram = [list(dict(degrees).values()).count(i) / 
            float(nx.number_of_nodes(G))
            for i in degree_values]
plt.hist(histogram)
plt.show()

#Exercise
import networkx as nx
import matplotlib.pyplot as plt
G = nx.DiGraph()
G.add_edge('E','C')
G.add_edge('C','B')
G.add_edge('B','A')
G.add_edge('H','A')
G.add_edge('H','G')
G.add_edge('G','A')
G.add_edge('G','I')
G.add_edge('G','F')
G.add_edge('F','D')
G.add_edge('D','C')
degrees = G.in_degree()
degree_values = sorted(set(dict(degrees).values()))
histogram = [list(dict(degrees).values()).count(i) / 
            float(nx.number_of_nodes(G))
            for i in degree_values]
plt.hist(histogram)
plt.show()