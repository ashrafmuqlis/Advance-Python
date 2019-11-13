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
