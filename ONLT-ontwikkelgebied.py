# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:07:10 2025

@author: SanderOostdijk
"""
import os
os.chdir(r"C:\CoSEM\24-25\Master Thesis\ONLT\SQ-4 modelbouw\ONLT model")
import networkx as nx
from math import inf
import time
from datetime import timedelta
# Local imports
import step_0_plot_demands as step0
import step_1_initial_topology as step1
import step_2_minimum_cost_spanning_tree as step2
import step_3_minimum_cost_steiner_tree as step3
import step_4_extra_improvement_round as step4
import general_procedures as gp

"""
=============================================================================
Choose parameter values
=============================================================================
"""
#start volledige timer
t0 = time.perf_counter()

# Folder with input file
folder = 'Input-output model/'
# Folder with output files
output_folder = 'Input-output model/Output_ontwikkelmodel/'
# Name of Excel file with input data
inputfile = 'input_data (ontwikkeling_model).xlsx'
# Name of Excel to which output data is written
outputfile = 'output_data_ontwikkelmodel.xlsx'

# Path of the output file
output_path = output_folder+outputfile

# Capacity-cost-exponent (range [0,1])
beta = 0.6

# If there are routing restrictions set routing to True
routing = False
# If there are obstacles set obstacles to True
obstacles = False
# If there are existing connections set existing to True
existing = True

# Absolute cost for extra splitting point (also for extra nodes in routing network)
spc = 0
# Relative cost for using existing capacity compared to building new connection
upc = 0
# Relative cost for increasing existing capacity compared to building new connections
cpc = 1


transport_parameters = {
    'Rekenperiode':40,
    'E_GH2':0.000121,
    'E_LH2':0.000142,
    'Pipeline':{'Diameter':0.508, 
          'Behoud energie':0.9, 
          'Opex_fix':0.04},
    'Road':{'Cap_truck':4324.9,
               'Cost_var':0.0009973,
               'Cost_var_snelweg':0.0008029,
               'Lifetime_truck':11,
               'Invest_truck':860000,
               'Opex_fix':0.02,
               'max_trucks_per_day':25},
    'Waterway':{'Cap_boot':354500,
               'Cost_var':0.01599, 
               'Lifetime_boot':25, 
               'Invest_boot':4100000,
               'Opex_fix':0.02}
    }

allowed_edge_transitions = ['Pipeline', 'Road']


# Node size in plots
node_size = 100
# Intermediate plots shown?
show_output = False

# In order to save computation time:
# New edges are only included in the search if smaller than extreme_length
# (Suggested value: (Xrange + Yrange) divided by 3 or 4
extreme_length = inf
# Search will be finished if new cost differ less than required cost deviation
# (number between 0 and 1)
cost_deviation = 0

#%%
"""
=============================================================================
STEP 0: Read input data and plot demand and supply
=============================================================================
"""
# Start timer
start = time.perf_counter()

# Store all input data in a dictionary
input_dict = gp.input_dict(folder+inputfile, obstacles, existing, routing)


# Check if there is storage in network
storage = gp.storage_in_network(folder+inputfile)
# If there is storage or existing==True set demand_equals supply to false, otherwise check if demand equals supply in all timesteps
if storage or existing:
    deqs = False
else:
    deqs = gp.demand_equals_supply(input_dict["demand"])

# Plot total demand/supply of nodes
step0.demand_over_nodes(input_dict)

# Plot total demand/supply over time
step0.demand_over_time(input_dict)

# Plot demand/suppy per node for each time step
step0.plot_demand(input_dict['demand'])

# stop steptimer

end = time.perf_counter()
cel1_duur = end - start
print(f"Step 0 duurde: {str(timedelta(seconds=int(cel1_duur)))}")
#%%
"""
=============================================================================
STEP 1: DETERMINE MINIMAL LENGTH SPANNING TREE
with sufficient capacity for all T demand-supply patterns
=============================================================================
"""
# Start timer
start = time.perf_counter()

MST = step1.initial_network(deqs, input_dict, beta, spc, upc, cpc,
                            routing, obstacles, existing,
                            output_path, node_size, 
                            allowed_edge_transitions=allowed_edge_transitions,
                            transport_parameters=transport_parameters)

# stop steptimer

end = time.perf_counter()
cel1_duur = end - start
print(f"Step 1 duurde: {str(timedelta(seconds=int(cel1_duur)))}")
#%%
"""
=============================================================================
STEP 2: DETERMINE (SUB)MINIMAL COST SPANNING TREE
with sufficient capacity for all T demand-supply patterns
(to suppress intermediate output, use output=False)
=============================================================================
"""
# Start timer
start = time.perf_counter()

FT = step2.min_cost_spanning_tree(deqs, input_dict, beta,
                                  spc, upc, cpc,
                                  routing, obstacles,existing,
                                  output_path,
                                  output=show_output,
                                  node_size=node_size,
                                  extreme_length=extreme_length,
                                  cost_deviation=cost_deviation,
                                  transport_parameters=transport_parameters,
                                  allowed_edge_transitions=allowed_edge_transitions)
# stop steptimer
end = time.perf_counter()
cel1_duur = end - start
print(f"Step 2 duurde: {str(timedelta(seconds=int(cel1_duur)))}")
#%%
"""
=============================================================================
Only for no routing network

STEP 3: DETERMINE (SUB)MINIMAL COST STEINER TREE
with sufficient capacity for all T demand-supply patterns
(to suppress intermediate output, use output=False)
=============================================================================
"""
# Start timer
start = time.perf_counter()

if not routing:
    ST = step3.min_cost_steiner_tree(deqs, input_dict, beta,
                                     obstacles, existing, output_path,
                                     spc, upc, cpc,
                                     output=show_output,
                                     node_size = node_size,
                                     extreme_length=extreme_length,
                                     cost_deviation=cost_deviation,
                                     transport_parameters=transport_parameters,
                                     allowed_edge_transitions=allowed_edge_transitions)
# stop steptimer

end = time.perf_counter()
cel1_duur = end - start
print(f"Step 3 duurde: {str(timedelta(seconds=int(cel1_duur)))}")
#%%
"""
=============================================================================
STEP 4: EXTRA IMPROVEMENT ROUND
with sufficient capacity for all T demand-supply patterns
(to suppress intermediate output, use output=False)
=============================================================================
"""
# Start timer
start = time.perf_counter()

if not routing:
    NT = step4.extra_improvement_round(deqs, input_dict, beta,
                                       obstacles,existing,
                                       spc, upc, cpc, output_path,
                                       output=show_output,
                                       node_size=node_size,
                                       extreme_length=extreme_length,
                                       cost_deviation=cost_deviation,
                                       transport_parameters=transport_parameters,
                                       allowed_edge_transitions=allowed_edge_transitions)

# stop steptimer
end = time.perf_counter()
cel1_duur = end - start
print(f"Step 4 duurde: {str(timedelta(seconds=int(cel1_duur)))}")

t1 = time.perf_counter()
totaal_duur = t1 - t0
print(f"Totale looptijd script: {str(timedelta(seconds=int(totaal_duur)))}")