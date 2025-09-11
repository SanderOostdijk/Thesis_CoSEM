"""
Procedures for minimum cost spanning tree in the ONLT

Version 2.0 (2025)

@author: pheijnen
"""

import networkx as nx
from scipy.spatial.distance import euclidean
from math import inf
# Local imports
import general_procedures as gp


"""
=============================================================================
PROCEDURE to determine and save the minimal cost spanning tree
=============================================================================
"""
def min_cost_spanning_tree(deqs,input_dict,beta,spc,upc,cpc,
                           routing,obstacles,existing,
                           output_path,
                           output=False,node_size=100,
                           extreme_length=inf,
                           cost_deviation=0,
                           transport_parameters=None,
                           allowed_edge_transitions=None):
# Read initial network
    MST = gp.network_from_excel(output_path,'MST')
# Find nodes in neighborhood
    circle = neighboring_nodes(MST,input_dict,routing,extreme_length)
# Determine the min-cost-spanning tree based on the k demand-supply profiles
    FT = edge_turn_tree(deqs,MST,beta,spc,upc,cpc,input_dict,output,
                        routing,existing,circle,cost_deviation,transport_parameters=transport_parameters, allowed_edge_transitions=allowed_edge_transitions)

# Redirect graph if obstacles
    FT = gp.redirected_graph(FT,input_dict,obstacles)
# Assign the final capacity based on all demand-supply patterns
    FT = gp.final_capacity_graph(input_dict,deqs,FT)
# Calculate the cost of the new network
    FT.graph['cost'] = gp.total_cost(FT,beta,spc,upc,cpc, transport_parameters=transport_parameters, allowed_edge_transitions=allowed_edge_transitions)
# If network better than initial network
    if FT.graph['cost']<(1-cost_deviation)*MST.graph['cost']:
# Print the total cost of the new network
        print('\nTotal cost minimum-cost spanning tree:',round(FT.graph['cost'],2))
        print()
# If network not better than initial network, keep initial network
    else:
        print('\nNo improvement on Minimal Spanning Tree found')
        FT = MST.copy()
        print()
# Draw the best network found
    gp.draw_network(FT,'Minimum-cost spanning tree: '+str(round(FT.graph['cost'],2)),
                 input_dict,routing,existing,node_size)

# Set cost attributes to edges and nodes
    FT = gp.edge_cost(FT,beta,upc,cpc, transport_parameters=transport_parameters, allowed_edge_transitions=allowed_edge_transitions)
    FT = gp.node_cost(FT,spc)
# Save network to file
    gp.save_network_to_excel(FT,output_path,title='FT')
# Return network
    return FT

"""
=============================================================================
PROCEDURE to define nodes in neighborhood of max extreme_length
=============================================================================
"""
def neighboring_nodes(G,input_dict,routing,extreme_length):
    if extreme_length == inf:
        if routing and input_dict['routing']:
# Every node contains every node of the routing network in the circle
            circle = {node: [neighbor for neighbor in input_dict['routing'].nodes()]
                      for node in input_dict['routing'].nodes()}
        else:
# Every node contains every node in the circle
            circle = {node: [neighbor for neighbor in G.nodes()] for node in G.nodes()}
# If routing network need to be used
    elif routing:
# Get routing network
        routing_network = input_dict['routing']
# Calculate all shortest path length between each pair of nodes within radius extreme_length
        path_lengths = dict(nx.all_pairs_dijkstra_path_length(routing_network,
                                                              cutoff=extreme_length))
# For each node determine nodes in neighborhood
        circle = {node: [neighbor for neighbor in path_lengths[node].keys()
                          if node!=neighbor] for node in path_lengths.keys()}
# If no routing network
    else:
# Determine coordinates from original Graph
        coord = nx.get_node_attributes(G, 'coord')
# For each node determine nodes in neighborhood
        circle = {node: [neighbor for neighbor in coord.keys() if
                         euclidean(coord[node], coord[neighbor]) < extreme_length
                         and node!=neighbor]
                  for node in coord.keys()}
# Return dictionary with all neighboring nodes
    return circle

"""
=============================================================================
PROCEDURE to calculate the length of an edge (i,j) and return (length,i,j)
=============================================================================
"""
def edge_length(T,i,j):
    return (round(euclidean(T.nodes[i]['coord'],T.nodes[j]['coord'])),i,j)

"""
=============================================================================
PROCEDURE to find cheaper network by edge turn heuristic
=============================================================================
"""
def next_edge_turn_tree(deqs,T,beta,spc,upc,cpc,input_dict,routing,circle,cost_deviation,transport_parameters,allowed_edge_transitions):
# Make a copy of the network
    best_T = T.copy()
# Break search if first better network is found
    found = False
# Find all potential edges to be removed
    edges_to_remove = [(edge_length(T,k,l),k,l) for k,l in T.edges()
                       if k in circle[l]
                       and not 'current' in T[k][l]]
# Sort potential edges on weight from small to large
    edges_to_remove.sort()
# Go through all edges (not too long) in the network
    while edges_to_remove and not found:
# Make copy of current network
        T_copy = T.copy()
# Take the longest edge left
        length,node1,node2 = edges_to_remove.pop()
# Remove edge
        T_copy.remove_edge(node1,node2)
# For each potential new edges
        for length,new_node1,new_node2 in candidate_edges(T_copy,node1,node2,circle,
                                                          routing,input_dict):
# Copy the graph with removed edge
            T_copy_2=T_copy.copy()
# If routing network applies
            if routing and input_dict['routing']:
# Add weighted shortest path to replace edge e to network
                T_copy_2 = add_routing_path(input_dict,T_copy_2,(new_node1,new_node2))
# If no routing network
            else:
# Add new edge to network
                T_copy_2.add_edge(new_node1,new_node2,weight=length)
# Add optimal capacity to the new network
            T_copy_2 = gp.final_capacity_graph(input_dict,deqs,T_copy_2)
# Calculate total network cost, for 
            T_copy_2.graph['cost'] = gp.total_cost(T_copy_2,beta,spc,upc,cpc,transport_parameters=transport_parameters, allowed_edge_transitions=allowed_edge_transitions)
# If cost reduction
            if T_copy_2.graph['cost']<(1-cost_deviation)*best_T.graph['cost']:
# Keep the best graph
                best_T = T_copy_2.copy()
# Better network found, stop search
                found = True
                break
# Return best network
    return best_T

"""
=============================================================================
PROCEDURE to add shortest path through routing network to replace edge e
=============================================================================
"""
def add_routing_path(input_dict,G,edge_to_replace):
# Copy current network
    G_copy = G.copy()
# Read routing network from path
    routing_network = input_dict['routing']
# Determine shortest path through routing network to replace edge e
    short_path = nx.dijkstra_path(routing_network,edge_to_replace[0],edge_to_replace[1])
# List of edges to be added
    new_edges = [[node1,node2,routing_network[node1][node2]['weight']]
                  for node1, node2 in zip(short_path[:-1],short_path[1:])]
# Add weighted edges to network
    G_copy.add_weighted_edges_from(new_edges)
# For each node on shortest path not in network before
    for node in [node for node in short_path if not 'stamp' in G_copy.nodes[node]]:
# Add stamp and coordinates as attributes
        G_copy.nodes[node]['stamp'] = 'steiner'
        G_copy.nodes[node]['coord'] = routing_network.nodes[node]['coord']
# Remove cycles from network
    G_copy = gp.removed_cycles(G_copy)
# Return resulting network
    return G_copy

"""
=============================================================================
PROCEDURE to find all candidate new edges
=============================================================================
"""
def candidate_edges(T,node1,node2,circle,routing,input_dict):
# Find 2 connected components
    components = list(nx.connected_components(T))
# Find component that contains node1
    component_1 = [c for c in components if node1 in c][0]
# Find component that contains node2
    component_2 = [c for c in components if node2 in c][0]
    if routing and input_dict['routing']:
# Get routing network if routing
        routing_network = input_dict['routing']
        candidates =[]
# Find all candidate paths connecting components
        for node,alt_node,component in [(node1,node2,component_2),
                               (node2,node1,component_1)]:
            for new_node in [i for i in component if i in circle[alt_node]
                             and i != alt_node]:
                P = nx.dijkstra_path(routing_network,node,new_node)
# Check if path is not partly in network yet
                if not any([(i,j) in T.edges() for i,j in zip(P,P[1:])]):
                    L = round(nx.dijkstra_path_length(routing_network,node,new_node))
                    candidates += [(L,node,new_node)]
    else:
# Find all candidate new edges (not too long and connecting the two components)
        candidates = ([edge_length(T,node1,new_node)
                       for new_node in component_2
                       if new_node!=node2 and new_node in circle[node1]]+
                      [edge_length(T,node2,new_node)
                       for new_node in component_1
                       if new_node!=node1 and new_node in circle[node2]])
# Sort candidate edges on their weight from small to large
    candidates.sort()
# Return all candidate edges
    return candidates

"""
=============================================================================
PROCEDURE to find cheaper network by edge turn iterations
=============================================================================
"""
def edge_turn_tree(deqs,T,beta,spc,upc,cpc,input_dict,
                   output,routing,existing,circle,cost_deviation,
                   transport_parameters,allowed_edge_transitions):
# Assign final capacity based on k demand-supply profiles
    best_G = gp.final_capacity_graph(input_dict, deqs,T)
# Calculate total cost of the network based on k demand-supply profiles
    best_G.graph['cost']=gp.total_cost(best_G,beta,spc,upc,cpc, transport_parameters=transport_parameters, 
                                       allowed_edge_transitions=allowed_edge_transitions)
# If intermediate output required, print starting cost
    if output: print('Starting cost:',round(best_G.graph['cost'],2))
# Initial cost to start off improvement round
    min_cost = inf
# While improvements can be found
    while best_G.graph['cost'] < min_cost:
# Best cost found so far
        min_cost = best_G.graph['cost']
# Find next tree by edge swap
        best_G = next_edge_turn_tree(deqs,best_G,beta,
                                     spc,upc,cpc,input_dict,
                                     routing,circle,cost_deviation,
                                     transport_parameters=transport_parameters,
                                     allowed_edge_transitions=allowed_edge_transitions)
# If intermediate output need to be shown
        if output:
# Print current cost
            print('Current cost:',round(best_G.graph['cost']))
# Plot current network
            gp.draw_network(best_G,'Intermediate tree:'+
                         str(best_G.graph['cost']),
                         input_dict,routing,existing)
# Return best network found
    return best_G
