"""
Procedures for min cost Steiner tree in the ONLT

Version 2.0 (2025)

@author: pheijnen
"""

import networkx as nx
from itertools import combinations
from scipy.spatial.distance import euclidean
from math import inf,cos
from shapely.geometry import Polygon, LineString, LinearRing,MultiPoint
# Local imports
import general_procedures as gp
import geometry_steiner_procedures as gsp
"""
=============================================================================
PROCEDURE to determine minimum-cost-steiner tree
=============================================================================
"""
def min_cost_steiner_tree(deqs,input_dict,beta,obstacles,existing,
                          output_path,
                          spc=0,upc=0,cpc=1,
                          output=True,node_size=100,extreme_length=inf,
                          cost_deviation=0, transport_parameters=None, 
                          allowed_edge_transitions=None):
# Read minimum-cost-spanning-tree from previous round
    FT = gp.network_from_excel(output_path,'FT')
# Find minimum-cost-steiner-tree on k demand-supply profiles by adding Steiner nodes
    ST = steiner_tree_with_new_nodes(deqs,FT,beta,input_dict,
                                     obstacles,existing,
                                     spc,upc,cpc,output,
                                     extreme_length=extreme_length,
                                     cost_deviation=cost_deviation,
                                     transport_parameters=transport_parameters,
                                     allowed_edge_transitions=allowed_edge_transitions)

# If lower cost network is found
    if ST.graph['cost']<(1-cost_deviation)*FT.graph['cost']:
# Print new cost
        print('\nTotal cost Steiner tree: ',round(ST.graph['cost'],2),'\n')
# Otherwise
    else:
# Print that no improved network has been found
        print('\nNo improvement on minimum cost spanning tree found')
# Copy the original network
        ST = FT.copy()
# Draw network
    gp.draw_network(ST,'Final Steiner network: '+
                    str(round(ST.graph['cost'],2)),
                     input_dict,False,existing,node_size)
# Add cost to edges and nodes
    ST = gp.edge_cost(ST,beta,upc,cpc, 
                      transport_parameters=transport_parameters, 
                      allowed_edge_transitions=allowed_edge_transitions)
    ST = gp.node_cost(ST,spc)
# Save the best network to file
    gp.save_network_to_excel(ST,output_path,title='ST')
# Return the best network
    return ST

"""
=============================================================================
PROCEDURE to find minimum cost steiner tree by adding steiner nodes to small
angles in G if cost are reduced
=============================================================================
"""
def steiner_tree_with_new_nodes(deqs,G,beta,input_dict,
                                obstacles,existing,
                                spc,upc,cpc,
                                output,extreme_length,cost_deviation,
                                transport_parameters,
                                allowed_edge_transitions):
# Copy graph G
    G_copy = G.copy()
# If output required
    if output:
# Print starting cost
        G_copy = gp.final_capacity_graph(input_dict,deqs,G_copy)
        G_copy.graph['cost']=gp.total_cost(G_copy,beta,spc,upc,cpc,
                                           transport_parameters=transport_parameters,
                                           allowed_edge_transitions=allowed_edge_transitions)
        print('Starting cost: ',round(G_copy.graph['cost'],2))

# Find profitable direct links between nodes and existing connections if any
    best_G = direct_links(deqs, G_copy,input_dict,
                          beta,spc,upc,cpc,obstacles,existing,
                          extreme_length,cost_deviation,output,
                          transport_parameters=transport_parameters,
                          allowed_edge_transitions=allowed_edge_transitions)

# Initialize set of angles that are already scanned
    tabu_angles = []
# Find smallest angle and its size in graph not already scanned
    angle = smallest_angle(best_G,tabu_angles,beta)
# While not all angles are scanned
    while angle:
# Add this angle to tabu list to avoid repetitive selection of the same angle
        tabu_angles.append(angle)
# Add a Steiner node to this smallest angle
        G_copy = graph_with_added_steiner_node(deqs,best_G,angle,beta,spc,upc,cpc,
                                               input_dict,cost_deviation, transport_parameters=transport_parameters, 
                                               allowed_edge_transitions=allowed_edge_transitions)
# Find common neighbor
        common = list(nx.common_neighbors(G_copy,angle[0],angle[1]))
# Put all new angles in tabu angles
        if common:
            tabu_angles.extend(([angle[0],common[0],angle[1]],
                                [angle[1],common[0],angle[0]],
                                [angle[1],common[0],angle[2]],
                                [angle[2],common[0],angle[1]],
                                [angle[0],common[0],angle[2]],
                                [angle[2],common[0],angle[0]]))

# Redirect edges around obstacles if any
        G_copy = gp.redirected_graph(G_copy,input_dict,obstacles)
# Add sufficient capacity for k-demand-supply profiles to graph
        G_copy = gp.final_capacity_graph(input_dict,deqs,G_copy)
# Calculate the total cost of this new network
        G_copy.graph['cost'] = gp.total_cost(G_copy,beta,spc,upc,cpc,
                                           transport_parameters=transport_parameters,
                                           allowed_edge_transitions=allowed_edge_transitions)
# If cheaper network found
        if G_copy.graph['cost'] < (1-cost_deviation)*best_G.graph['cost']:
# Copy the new network
            best_G = G_copy.copy()
# If intermediate output is required
            if output:
# Print the current cost
                print('current cost: ',round(best_G.graph['cost'],2))
# Draw the current network
                gp.draw_network(best_G,'Intermediate Steiner: '
                             +str(best_G.graph['cost']),
                             input_dict,False,existing)
# Find next smallest angle in the new graph
        angle = smallest_angle(best_G,tabu_angles,beta)
# Add final capacity for all demand-supply profiles
    best_G = gp.final_capacity_graph(input_dict,deqs,best_G)
# Add total cost as attribute to the network
    best_G.graph['cost']=gp.total_cost(best_G,beta,spc,upc,cpc,
                                       transport_parameters=transport_parameters,
                                       allowed_edge_transitions=allowed_edge_transitions)
# Return best network
    return best_G

"""
=============================================================================
PROCEDURE to add profitable direct links to existing connections if any
=============================================================================
"""
def direct_links(deqs, G,input_dict,beta,spc,upc,cpc,
                 obstacles,existing,extreme_length,cost_deviation,output,transport_parameters,allowed_edge_transitions):
# Copy given graph
    best_G = G.copy()
# Get obstacles if any
    obstacle_dict = input_dict['obstacles']
# Find potential splits in existing connections
    pos_split = potential_splits(G,obstacle_dict,extreme_length)
# While potential splits left
    while pos_split:
# Copy best graph so far
        G_copy = best_G.copy()
# Take next potential split
        length,new_point,node,connection = pos_split.pop()
# Split line-segment on given point
        G_copy = split_line(G_copy,connection,new_point,node)
# Find lowest-cost forest by removing one edge of new cycle made
        G_copy = gp.best_forest(deqs, G_copy,input_dict,False,beta,spc,upc,cpc,
                                transport_parameters=transport_parameters,
                                allowed_edge_transitions=allowed_edge_transitions)
# If cheaper network found
        if G_copy.graph['cost'] < (1-cost_deviation)*best_G.graph['cost']:
# Keep best network
            best_G = G_copy.copy()
# Determine new potential splits in new network
            pos_split = potential_splits(best_G,obstacle_dict,extreme_length)
# If output is required and new network is found
    if output and not nx.is_isomorphic(best_G,G):
# Draw network with direct connections
        gp.draw_network(best_G,'Direct connections: '+str(best_G.graph['cost']),
                        input_dict,False,existing)
# Print current cost
        print('current cost: ',round(best_G.graph['cost'],2))
# Return the best network
    return best_G

"""
=============================================================================
PROCEDURE to find potential direct links to existing connections
=============================================================================
"""
def potential_splits(G,obstacle_dict,extreme_length):
# Define empty list for all possible splits
    pos_split = []
# For each existing connection in network
    for node1,node2 in [(node1,node2) for node1,node2 in G.edges()
                        if 'current' in G[node1][node2]]:
# Get coordinates of end points of existing connection
        coord_1,coord_2 = G.nodes[node1]['coord'],G.nodes[node2]['coord']
# If existing connection crosses one or more obstacles
        if gp.crosses_obstacle(coord_1, coord_2, obstacle_dict):
# Determine new lines between network nodes and existing connection outside obstacles
            new_lines = [(nearest_crossing_edge_line(G.nodes[p]['coord'],coord_1,
                                                    coord_2,obstacle_dict),p)
                         for p in G.nodes() if not p in (node1, node2)]
# If existing connection does not cross any obstacle
        else:
# Determine new lines between network nodes and projection point on existing connection
            new_lines = [(best_split(G.nodes[p]['coord'], coord_1, coord_2, obstacle_dict),p)
                         for p in G.nodes() if not p in (node1, node2)]
# Remove None solution from list
        new_lines = [(x,p) for x,p in new_lines if x is not None]
# Define potential splits by its length, new point on existing connection,
# network node and existing connection if new line is not longer than extreme length
        pos_split += [(euclidean(new_point, G.nodes[p]['coord']), new_point, p, (node1, node2))
                      for new_point,p in new_lines if new_point and
                      euclidean(new_point, G.nodes[p]['coord']) < extreme_length]
# Sort potential split on length between new node and network node from large to small
    pos_split.sort(reverse=True)
# Return list of potential splits
    return pos_split

"""
=============================================================================
Procedure to find coordinates of point p on line (a,b) nearest to p
just outside obstacles
=============================================================================
"""
def nearest_crossing_edge_line(p,a,b,obstacle_dict):
# For each obstacle
    for obstacle in obstacle_dict.values():
# Define the ring of obstacle borders
        lring = LinearRing(obstacle)
# Define polygon of obstacle
        polygon = Polygon(obstacle)
# Define line segment between coordinate points a and b
        line = LineString([a,b])
# If line segment crosses polygon
        if polygon.crosses(line):
# Find intersection points with borders of polygon
            IS = lring.intersection(line)
            if type(IS)==MultiPoint:
                P = list(IS.geoms)
# Shift nearest point in P towards a or b
                q1 = shift_point(nearest_to(a,P),a)
                q2 = shift_point(nearest_to(b,P),b)
            else:
                q1 = shift_point(IS,a)
                q2 = shift_point(IS,b)
# Define line segment between point p and new point q1
            line1 = LineString([p,q1])
# If line segment crosses polygon new point is q2 else q1
            np = q2 if polygon.crosses(line1) else q1

# If new line does not cross any obstacle
            if not gp.crosses_obstacle(np,p,obstacle_dict):
# Return the new point
                return np

"""
=============================================================================
Procedure to find nearest point in P to given point a
=============================================================================
"""
def nearest_to(a,P):
# Calculate distance between a and all points in P
    L = [(euclidean(a,(p.x,p.y)),p) for p in P]
# Sort list on distance
    L.sort()
# Return nearest point
    return L[0][1]

"""
=============================================================================
Procedure to slightly shift point p towards point a
=============================================================================
"""
def shift_point(p,a):
# Define shift size
    eps = 0.01
# Return shifted point
    return (p.x+eps*(a[0]-p.x),p.y+eps*(a[1]-p.y))

"""
=============================================================================
PROCEDURE to find the projection point on an existing connection
=============================================================================
"""
def best_split(p,a,b,obstacle_dict):
# Transform the angle [p,a,b] to [p-a,0,b-a], with A=p-a and B=b-a
    A,B = gsp.translate_to_zero(p,a,b)
# Calculate the size of the angle at point a between vector p-a and b-a
    size = gsp.angle_size(A,B)
# Calculate the length of the projection from p on the line (a,b)
    d = euclidean(p,a)*cos(size)
# Calculate the coordinates of the projection point
    np = (a[0]+d*(b[0]-a[0])/euclidean(a,b),
          a[1]+d*(b[1]-a[1])/euclidean(a,b))

# If the projection point lies between a and b and
# the project line does not cross an obstacle
    if (min(a[0],b[0])<=np[0]<=max(a[0],b[0]) and
        min(a[1],b[1])<=np[1]<=max(a[1],b[1]) and
        not gp.crosses_obstacle(np,p,obstacle_dict) and
        np != a and np != b):
# Return projection point
        return np

"""
=============================================================================
PROCEDURE to find a set of all angles in a graph not in tabu list
=============================================================================
"""
def all_angles(G,tabu):
# Define list of all angles in G
    angle_list = []
# For each combination of 3 nodes
    for nodes in combinations(G,3):
# Define sub graph with these 3 nodes
        G_sub = G.subgraph(nodes)
# Check if sub graph is connected
        if nx.is_connected(G_sub) and nx.is_tree(G_sub):
# Collect the 3 nodes (middle node occurs twice)
            angle = [node for edge in G_sub.edges() for node in edge]
# Sort nodes in angle on frequency
            angle = sorted(angle,key=angle.count)
# If none of the edges in the angle are existing connections
            if not (('current' in G[angle[0]][angle[2]]) or
                    ('current' in G[angle[2]][angle[1]])):
# Put last node in the middle to quarantee middle node is angle node
                angle_list.append([angle[0],angle[2],angle[1]])
# Return list of all angles without angles in tabu list
    return [a for a in angle_list if not a in tabu]

"""
=============================================================================
PROCEDURE to find smallest angle in graph not in list of tabu-angles
=============================================================================
"""
def smallest_angle(G,tabu,beta):
# Find list of all angles in G given als list of 3 nodes
# Remove tabu angles
    angle_list = all_angles(G,tabu)
# If angles found
    if angle_list:
# Initial size of smallest angle
        min_angle_size = inf
# Find coordinates of nodes in G
        coord = nx.get_node_attributes(G,'coord')
# For all angles
        for angle in angle_list:
# Transform [A,B,C] to [A-B,0,C-B]
            A,B = gsp.translate_to_zero(*[coord[node] for node in angle])
# Calculate the size of the angle between vector A and B
            angle_size = gsp.angle_size(A,B)
# If angle smaller than minimum so far
            if angle_size < min_angle_size:
# Keep smallest angle and its size
                min_angle_size = angle_size
                name_angle = angle
# Only continue if smallest angle is sufficiently small
        if min_angle_size<(2.09+0.46*beta):
# Return smallest angle
            return name_angle

"""
=============================================================================
PROCEDURE to find Steiner en not-Steiner nodes in full Steiner tree
(subgraph of larger graph G) containing given Steiner node s
=============================================================================
"""
def terminals_in_full_steiner_tree(G,s):
# Find all non-steiner nodes in G
    not_steiner = set([i for i in G.nodes() if G.nodes[i]['stamp'] in
                   ['terminal','storage','corner','split','existing']])
# Find all Steiner nodes in G
    steiner = set([i for i in G.nodes() if G.nodes[i]['stamp'] == 'steiner'])
# Start with initial Steiner node given
    check = set([s])
# Start with empty set of not-steiner nodes in full Steiner tree
    notsteiner_in_FST = set()
# Start with empty Steiner node set
    steiner_in_FST = set()
# While new Steiner nodes
    while check:
# Take the last one to check
        i = check.pop()
# Add to set of Steiner nodes
        steiner_in_FST.add(i)
# Find neighbors of this Steiner node
        nb = set(G.neighbors(i))
# Add terminal neigbors to Terminals set
        notsteiner_in_FST.update(nb.intersection(not_steiner))
# Add Steiner node neighbors to Check set
        check.update(nb.intersection(steiner))
# Remove Steiner nodes that were already checked
        check.difference_update(steiner_in_FST)
# Return all terminals and steiner nodes in full Steiner tree
    return notsteiner_in_FST,steiner_in_FST

"""
=============================================================================
PROCEDURE to check if rewiring edge in angle gives a cheaper network
=============================================================================
"""
def best_tree_in_angle(deqs,G,angle,input_dict,beta,spc,upc,cpc,transport_parameters, allowed_edge_transitions):
# Make a copy of the graph
    G_closed, G_best = G.copy(),G.copy()
# Close the angle by adding the missing edge
    G_closed.add_edge(angle[0],angle[2],
                      weight = euclidean(G.nodes[angle[0]]['coord'],
                                         G.nodes[angle[2]]['coord']))
# For each of the original edges in the angle
    for edge in [(angle[0],angle[1]),(angle[1],angle[2])]:
# Make a copy of the graph
        G_copy = G_closed.copy()
# Remove the edge
        G_copy.remove_edge(*edge)
# Add sufficient capacity to the remaining graph for k demand-supply profiles
        G_copy = gp.final_capacity_graph(input_dict,deqs,G_copy)
# Calculate totale cost of new graph
        G_copy.graph['cost'] = gp.total_cost(G_copy,beta,spc,upc,cpc,
                                           transport_parameters=transport_parameters,
                                           allowed_edge_transitions=allowed_edge_transitions)
# If cheaper graph found
        if G_copy.graph['cost']<G_best.graph['cost']:
# Keep the best network so far
            G_best = G_copy.copy()
# Return the graph
    return G_best

"""
=============================================================================
PROCEDURE to add a Steiner node to graph G in angle an = [A,B,C] based on
capacity-cost-exponent beta and for demand-supply patterns in folder
=============================================================================
"""
def graph_with_added_steiner_node(deqs,G,angle,beta,spc,upc,cpc,input_dict,cost_deviation, 
                                  transport_parameters, allowed_edge_transitions):
# Find the best tree in the angle by rewiring the edges
    G_best = best_tree_in_angle(deqs,G,angle,input_dict,
                                beta,spc,upc,cpc,transport_parameters=transport_parameters,
                                allowed_edge_transitions=allowed_edge_transitions)
# Find steiner nodes in original graph G
    steiner = [node for node in G.nodes() if G.nodes[node]['stamp'] == 'steiner']
# Make copy of the original graph
    G_copy = G.copy()
# Remove edges in given angle
    G_copy.remove_edges_from({(angle[0],angle[1]),(angle[1],angle[2])})
# Add extra steiner node s
    new_steiner = G_copy.number_of_nodes()
    G_copy.add_node(new_steiner,stamp='steiner')
# Connect new Steiner node with all 3 angle points
# (NB.edge weights are temporary equal to 1)
    G_copy.add_weighted_edges_from({(angle[0],new_steiner,1),
                                    (angle[1],new_steiner,1),
                                    (angle[2],new_steiner,1)})
# Assign capacity to the new edges
    G_copy = gp.final_capacity_graph(input_dict,deqs,G_copy)
# If all steiner nodes are now connected to exactly 3 other nodes
    if all(G_copy.degree(node) == 3 for node in steiner):
# Find the other steiner nodes and the not steiner nodes
# in the full steiner tree containing the new steiner point
        fst_not_steiner,fst_steiner = terminals_in_full_steiner_tree(G_copy,new_steiner)
# Find the optimal location of Steiner nodes in the full Steiner tree
        coord_steiner = gsp.coordinates_steiner_nodes(G_copy,fst_not_steiner,
                                                      fst_steiner,beta)
# If for all Steiner nodes an optimal location has been found
        if all(node in coord_steiner.keys() for node in fst_steiner):
# Add coordinates as attribute to the nodes
            nx.set_node_attributes(G_copy,coord_steiner,'coord')
# Add euclidean length as attribute to the edges
            G_copy = gp.euclidean_weighted_graph(G_copy)
# Calculate total cost
            G_copy.graph['cost'] = gp.total_cost(G_copy,beta,
                                                 spc,upc,cpc,
                                                 transport_parameters=transport_parameters,
                                                 allowed_edge_transitions=allowed_edge_transitions)
# If better network found
            if G_copy.graph['cost'] < (1-cost_deviation)*G_best.graph['cost']:
# Return new graph with extra Steiner node
                return G_copy
# If new network is not better
            else:
# Return the best network so far
                return G_best
        else:
            return G_best
    else:
        return G_best

"""
=============================================================================
PROCEDURE to split existing connection e at location in graph G
and connect new node to node
=============================================================================
"""
def split_line(G,edge,location,node):
# Copy current network
    G_copy = G.copy()
# Determine number for the new node
    new_node = G_copy.number_of_nodes()
# Add the new node as splitting point on given location
    G_copy.add_node(new_node,stamp='split',coord = location)
# Add edges between end nodes of existing connection and new node and node and new node
    G_copy.add_edges_from([(edge[0],new_node),(new_node,edge[1]),(new_node,node)])
# For the new splitted segments of existing connection
    for i,j in [(edge[0],new_node),(new_node,edge[1])]:
# Copy the current capacity and category of existing connection
        G_copy[i][j]['current'] = G[edge[0]][edge[1]]['current']
        G_copy[i][j]['category'] = G[edge[0]][edge[1]]['category']
# Add the edge lenght as weight to the connections
        G_copy[i][j]['weight'] = euclidean(G_copy.nodes[i]['coord'],
                                           G_copy.nodes[j]['coord'])
# Add the edge length as weight and category to the new connection between p and c
    G_copy[new_node][node]['weight']= euclidean(location,G_copy.nodes[node]['coord'])
# Remove the old existing connection
    G_copy.remove_edge(*edge)
# Return the network with splitted existing connection
    return G_copy