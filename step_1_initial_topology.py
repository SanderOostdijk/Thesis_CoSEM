"""
Procedures for to define the initial spanning tree in the ONLT

Version 2.0 (2025)

@author: pheijnen
"""

import networkx as nx
import pyvisgraph as vg
from scipy.spatial.distance import euclidean
from math import inf
#Local imports
import general_procedures as gp

"""
=============================================================================
PROCEDURE to define, save and draw initial network based on minimal spanning tree,
calculate and print cost
=============================================================================
"""
def initial_network(deqs,input_dict,beta,spc,upc,cpc,routing,obstacles,
                    existing,output_path,node_size=100, allowed_edge_transitions=None, 
                    transport_parameters=None):
# Get existing connections if any
    (existing_edges, coord) = input_dict['existing connections']
# If no routing network
    if not routing or not input_dict['routing']:
# Find the minimum length spanning tree around obstacles and with existing
# connections if any
        T = min_length_spanning_tree(input_dict,obstacles,existing_edges,coord, allowed_edge_transitions)
# If routing network
    else:
# Find minimum length steiner tree in routing network with existing
# connections if any
        T = min_length_steiner_tree(deqs,input_dict,
                                    existing_edges,coord,beta,
                                    spc,upc,cpc,allowed_edge_transitions)
# Add capacity to tree
    T = gp.final_capacity_graph(input_dict,deqs,T)

# Calculate and print total graph cost
    T.graph['cost'] = gp.total_cost(T,beta,spc,upc,cpc,transport_parameters=transport_parameters, allowed_edge_transitions=allowed_edge_transitions)
    print('\nTotal cost minimum-spanning tree:',T.graph['cost'])
    print()

# Delete not connected nodes if not terminal or storage nodes
    MST = T.copy()
    MST.remove_nodes_from([node for node in T.nodes()
                          if T.degree(node)==0 and
                          not T.nodes[node]['stamp'] in ['terminal','storage']])

# Draw minimal spanning tree
    gp.draw_network(MST,'Minimum spanning tree: '+str(MST.graph['cost']),
                    input_dict,routing,existing,node_size)

# Add edge cost and node cost as attributes to edges and nodes
    MST = gp.edge_cost(MST,beta,upc,cpc,transport_parameters=transport_parameters, allowed_edge_transitions=allowed_edge_transitions)
    MST = gp.node_cost(MST,spc)
# Save network to excel
    gp.save_network_to_excel(MST,output_path,title='MST')

# Return network
    return MST

"""
=============================================================================
PROCEDURE to find minimum length spanning tree
=============================================================================
"""
def min_length_spanning_tree(input_dict,obstacles,existing_edges,coord, allowed_edge_transitions):
# Define inital tree with existing connections if any
    initial_tree = tree_with_existing_connections(input_dict,existing_edges,coord)
# Define complete graph with all potential connections
    complete_graph = all_connections(input_dict,obstacles,coord)
# Find min length spanning tree
    MLST = min_spanning_tree_with_existing(initial_tree,complete_graph, allowed_edge_transitions)
# Redirect edges in case of obstacles
    MLST = gp.redirected_graph(MLST,input_dict,obstacles)
# Return graph
    return MLST

"""
=============================================================================
PROCEDURE to find minimal length Steiner tree in a graph
=============================================================================
"""
def min_length_steiner_tree(deqs,input_dict,existing_edges,coord,beta,
                            spc,upc,cpc, allowed_edge_transitions):
# Define initial tree with existing connections if any
    initial_tree = tree_with_existing_connections(input_dict,existing_edges,coord)
# Define routing network with all potential connections
    routing_network = input_dict["routing"]
# Find min length Steiner tree
    MLST = min_length_steiner_tree_with_existing(initial_tree,routing_network,input_dict)
# Remove cycles if any
    MLST = gp.best_forest(deqs,MLST,input_dict,
                          routing_network,beta,spc,upc,cpc, allowed_edge_transitions)
# Return graph
    return MLST

"""
=============================================================================
PROCEDURE to find minimum Steiner tree in graph
with shortest-distance-first-heuristic
=============================================================================
"""
def min_length_steiner_tree_with_existing(initial_tree,routing_network,input_dict):
# Copy initial tree with existing connections
    T = initial_tree.copy()
# Define dictionary of terminal coordinates
    coord = {i:routing_network.nodes[i]['coord'] for i in routing_network.nodes()
             if routing_network.nodes[i]['stamp'] in ['terminal','storage']}
# Define complete graph with edge length equal to the routing path
    complete_graph = complete_routing_length_graph(input_dict,coord)
# Define minimum spanning tree in this complete graph
    MST = nx.minimum_spanning_tree(complete_graph)
# For each edge in minimum spanning tree
    for node1,node2 in MST.edges():
# Find the shortest path through routing network connecting the end nodes
        short_path = nx.dijkstra_path(routing_network,node1,node2)
# Find all edges on this shortest path and their weights
        edges_on_sp = [(i,j,routing_network[i][j]['weight'])
                       for i,j in zip(short_path[:-1],short_path[1:])]
# Add these weighted edges to Steiner tree
        T.add_weighted_edges_from(edges_on_sp)
# Add stamp and coordinates to steiner nodes
    for i in T.nodes():
        T.nodes[i]['stamp'] = routing_network.nodes[i]['stamp']
        T.nodes[i]['coord'] = routing_network.nodes[i]['coord']
# Return the final tree with cycles removed
    return gp.removed_cycles(T)

"""
=============================================================================
PROCEDURE to find minimum length spanning tree
including existing connections with coordinates of end points in coord1
=============================================================================
"""
def tree_with_existing_connections(input_dict,existing_edges,coord1):
# Define empty graph
    G = nx.Graph()
# Get coordinates of terminals
    coord_terminals = input_dict['coordinates']
# Get coordinates of storage
    coord_storage = input_dict['coordinates storage']
# Update coordinates
    coord = coord_terminals.copy()
    coord.update(coord_storage)
# Add all terminals and storage nodes to graph
    G.add_nodes_from(coord.keys())
# For all edges in existing connections
    for (node1,node2), attr in existing_edges.items():
# Add edge with given capacity, given current capacity and given weight and category
        capacity = attr['capacity']
        category = attr.get('category', 'Onbekend')
        G.add_edge(node1,node2,capacity=capacity,current=capacity, category=category,
                   weight=euclidean(coord1[node1],coord1[node2]))
# Add coordinates as attribute to nodes
    nx.set_node_attributes(G,coord1,'coord')
# For each node in the graph
    for node in G.nodes():
# Set node stamp as attribute to node
        if node in coord_terminals.keys():
            G.nodes[node]['stamp'] = 'terminal'
        elif node in coord_storage.keys():
            G.nodes[node]['stamp'] = 'storage'
        else: G.nodes[node]['stamp'] = 'existing'

# Add extra points from routing network if any
    for node in [node for node in coord1.keys() if not node in G.nodes()]:
        G.add_node(node,coord = coord1[node],stamp='steiner')
# Return graph with existing connections
    return G

"""
=============================================================================
PROCEDURE to find all potential connections between nodes in graph IT
=============================================================================
"""
def all_connections(input_dict,obstacles,coord):
# Find obstacles if any
# If obstacles
    if obstacles:
# Define all connections between nodes with real length around obstacles
        complete_graph = complete_obstacle_length_graph(input_dict,coord, obstacles)
    else:
# Complete graph with euclidean distance as edge weights
        complete_graph = complete_length_graph(coord)
# Return complete graph H
    return complete_graph

"""
=============================================================================
PROCEDURE to define complete weighted graph
with edge length (based on euclidean distance between coordinates) as weights
=============================================================================
"""
def complete_length_graph(coord):
# Define a complete graph with all keys from coordinates dictionary as nodes
    complete_graph = nx.complete_graph(coord.keys())
# Add node coordinates as node attribute
    nx.set_node_attributes(complete_graph,coord,'coord')
# Loop through all edges in the complete graph
    for node1,node2 in complete_graph.edges():
# Calculate the euclidean distance between the connected nodes
        complete_graph[node1][node2]['weight'] = euclidean(coord[node1],coord[node2])
# Return the complete weighted graph
    return complete_graph

"""
=============================================================================
PROCEDURE to define complete weighted graph within routing network
with edge length (based on euclidean length of path) as weights
=============================================================================
"""
def complete_routing_length_graph(input_dict,coord):
# Define a complete graph with all keys from coordinates dictionary as nodes
    complete_graph = nx.complete_graph(coord.keys())
# Add node coordinates as node attribute
    nx.set_node_attributes(complete_graph,coord,'coord')
# Get routing network
    routing_network = input_dict['routing']
# Loop through all edges in the complete graph
    for node1,node2 in complete_graph.edges():
# Find shortest path in routing network
        short_path = nx.dijkstra_path(routing_network,node1,node2)
# Define node coordinates on path
        coord_list = [routing_network.nodes[node]['coord'] for node in short_path]
# Calculate total length of path between node i and j
        complete_graph[node1][node2]['weight'] = sum([euclidean(coord1,coord2)
                                                      for coord1,coord2 in
                                                      zip(coord_list[:-1],coord_list[1:])])
# Return the complete weighted graph
    return complete_graph

"""
=============================================================================
PROCEDURE to define complete weighted graph around obstacles
with edge length (based on euclidean distance) as weights
=============================================================================
"""
def complete_obstacle_length_graph(input_dict,coord, obstacles):
# Define a complete graph with all keys from coordinates dictionary as nodes
    complete_graph = nx.complete_graph(coord.keys())
# Add node coordinates as node attributes
    nx.set_node_attributes(complete_graph,coord,'coord')
# Get obstacles
    if obstacles:
        obstacles_dict = input_dict['obstacles']
    visg = gp.visibility_graph(obstacles_dict)
# Loop through all edges in H
    for node1,node2 in complete_graph.edges():
# Find shortest path around obstacles from i to j
        short_path = visg.shortest_path(vg.Point(*coord[node1]),vg.Point(*coord[node2]))
# Define node coordinates on path
        coord_list = [(node.x,node.y) for node in short_path]
# Calculate total length of path between node i and j
        complete_graph[node1][node2]['weight'] = sum([euclidean(coord1,coord2)
                                                      for coord1,coord2 in
                                                      zip(coord_list[:-1],coord_list[1:])])
# Return complete graph with real length as edge weights
    return complete_graph


"""
=============================================================================
PROCEDURE to find minimum (length) spanning tree with existing edges and
all potential edges
=============================================================================
"""
def min_spanning_tree_with_existing(initial_tree,complete_graph, allowed_edge_transitions):
# Only use a tree of existing edges
    G1 = nx.minimum_spanning_tree(initial_tree)
# Copy nodes and attributes from complete_graph
    G1.add_nodes_from(complete_graph.nodes(data=True))
# Define dictionary for all potential edges with their weight
    weight_dict = {(node1,node2): complete_graph[node1][node2]['weight']
                   for (node1,node2) in complete_graph.edges()}
# Sort edges on their weight from large to small
    weight_sort = sorted(weight_dict.items(), key=lambda x: x[1], reverse=True)
# While still edges left and G1 not connected
    while weight_sort and not nx.is_connected(G1):
# Take next shortest edge
        (node1,node2),w = weight_sort.pop()
# add category
        cat = gp.inherit_category(allowed_edge_transitions)
# Add weighed edge to G1
        G1.add_edge(node1,node2,weight=w, category=cat)
# If G1 is not a forest anymore
        if not nx.is_forest(G1):
# Remove edge
            G1.remove_edge(node1,node2)
# Add all the existing edges
    G1.add_edges_from(initial_tree.edges(data=True))
# Return final G1
    return G1