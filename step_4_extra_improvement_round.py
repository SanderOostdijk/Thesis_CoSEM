"""
Procedures for final round in the ONLT

Version 2.0 (2025)

@author: pheijnen
"""

from math import inf
# Local imports
import general_procedures as gp
import step_2_minimum_cost_spanning_tree as step2
import step_3_minimum_cost_steiner_tree as step3

"""
=============================================================================
PROCEDURE to find extra improvements by edge-swaps or additional Steiner points
=============================================================================
"""
def extra_improvement_round(deqs,input_dict,beta,obstacles,existing,
                            spc,upc,cpc,output_path,
                            output=False,node_size=100,
                            extreme_length=inf,
                            cost_deviation=0, 
                            allowed_edge_transitions=None, 
                            transport_parameters=None):
# Read minimum-cost-Steiner-tree from file
    ST_1 = gp.network_from_excel(output_path,'ST')
# Read minimum-cost-spanning-tree from file
    FT_1 = gp.network_from_excel(output_path,'FT')

# Keep best network
    best_G = ST_1.copy()
# Improvement round
    run = 0
# Repeat improvement round as long as better network is found
    while ST_1.graph['cost']< (1-cost_deviation)*FT_1.graph['cost']:
# Print run number
        print('Improvement round',run)
# Improvement round
        run += 1

# Find nodes in neighborhood
        circle = step2.neighboring_nodes(ST_1, input_dict, False, extreme_length)

# Find better network by edge swaps
        FT_1 = step2.edge_turn_tree(deqs, ST_1, beta,
                                    spc, upc, cpc, input_dict, output,
                                    routing=False, existing=False,
                                    circle=circle,
                                    cost_deviation=cost_deviation,
                                    transport_parameters=transport_parameters,
                                    allowed_edge_transitions=allowed_edge_transitions)
# Redirect edges around obstacles if any
        FT_1 = gp.redirected_graph(FT_1,input_dict,obstacles)
# Add final capacity to network
        FT_1 = gp.final_capacity_graph(input_dict,deqs,FT_1)
        #FT1 = cleaned_final_network(FT1)
# Compute new cost
        FT_1.graph['cost'] = gp.total_cost(FT_1,beta,spc,upc,cpc,
                                           transport_parameters=transport_parameters,
                                           allowed_edge_transitions=allowed_edge_transitions)
# If better network found
        if FT_1.graph['cost'] < (1-cost_deviation)*ST_1.graph['cost']:
# Print new cost
            print('\nTotal cost edge-improved tree:',
                  round(FT_1.graph['cost']),'\n')
# Draw best network
            gp.draw_network(FT_1,'Edge-improved tree: '+
                            str(round(FT_1.graph['cost'],2)),
                            input_dict,False,existing,node_size)
# Keep best network
            best_G = FT_1.copy()
# Add extra Steiner points if useful
            ST_1 = step3.steiner_tree_with_new_nodes(deqs, FT_1,
                                                     beta, input_dict,
                                                     obstacles,existing,
                                                     spc, upc, cpc,
                                                     output,
                                                     extreme_length=
                                                     extreme_length,
                                                     cost_deviation=
                                                     cost_deviation,
                                                     transport_parameters=transport_parameters,
                                                     allowed_edge_transitions=allowed_edge_transitions)

# If better network found
            if ST_1.graph['cost'] < (1-cost_deviation)*FT_1.graph['cost']:
# Print new cost
                print('\nTotal cost Steiner-improved tree: ',
                      round(ST_1.graph['cost']),'\n')
# Keep best network
                best_G = ST_1.copy()
# Draw best network so far
                gp.draw_network(ST_1,'Steiner improved tree: '
                                 +str(round(ST_1.graph['cost'],2)),
                                 input_dict,False,existing,node_size)
            else:
                print('No improvement found')
        else:
            print('No improvement found')
            break
# Remove zero-capacity edges and obsolete Steiner and splitting nodes
    best_G =  cleaned_final_network(best_G)
# Calculate final cost
    best_G.graph['cost'] = gp.total_cost(best_G,beta,spc,upc,cpc,
                                         transport_parameters=transport_parameters,
                                         allowed_edge_transitions=allowed_edge_transitions)
# Print final best network
    gp.draw_network(best_G,'Final best network: '+str(best_G.graph['cost']),
                    input_dict,False,existing,node_size)
    print('\nTotal cost Final best network: ',
                      round(best_G.graph['cost'],2),'\n')
# Add edge cost and node cost to edges and nodes in graph
    best_G = gp.edge_cost(best_G,beta,upc,cpc,
                          transport_parameters=transport_parameters,
                          allowed_edge_transitions=allowed_edge_transitions)
    best_G = gp.node_cost(best_G,spc)
# Save the best network to file
    gp.save_network_to_excel(best_G,output_path,title='bestG')
# Return the best network
    return best_G

"""
=============================================================================
PROCEDURE to clean final network
=============================================================================
"""
def cleaned_final_network(G):
# Make copy of graph
    G_copy = G.copy()
# Find zero-capacity edges
    zero_cap_edges = [(i,j) for (i,j) in G_copy.edges()
                      if G_copy[i][j]['capacity'] == 0]
# Remove zero-capacity edges
    G_copy.remove_edges_from(zero_cap_edges)
# Find nodes, not terminal, storage or existing, with degree 0 or 1
    obsolete_nodes = [i for i in G_copy.nodes()
                   if ((not (G_copy.nodes[i]['stamp']
                             in ['terminal','storage','existing']))
                   and G_copy.degree[i]<=1)]
# Remove these obsolete nodes
    G_copy.remove_nodes_from(obsolete_nodes)
# Find Steiner and splitting nodes of degree 2
    obsolete_splitting = [i for i in G_copy.nodes()
                          if ((G_copy.nodes[i]['stamp']
                               in ['steiner','split'])
                              and G_copy.degree[i] == 2)]
# For each of these obsolete splitting nodes
    for node in obsolete_splitting:
# Find the two neighbors
        nb = tuple(G_copy.neighbors(node))

#If original edge was existing connection
        if ('current' in G_copy[nb[0]][node] and
            'current' in G_copy[nb[1]][node]):
            #cat = gp.inherit_category(G_copy, , new_node2)
# Add direct edge between neighbors with attributes weight and capacity derived from original edges
            G_copy.add_edge(*nb,weight = (G_copy[nb[0]][node]['weight']
                                      +G_copy[nb[1]][node]['weight']),
                            capacity = G_copy[nb[0]][node]['capacity'],
                            category = G_copy[nb[0]][node]['category'])


# Add existing capacity
            G_copy[nb[0]][nb[1]]['current'] = G_copy[nb[0]][node]['current']
# Remove the obsolete node and its incident edges
            G_copy.remove_node(node)
#Return relabeled graph
    return gp.relabeled_graph(G_copy)