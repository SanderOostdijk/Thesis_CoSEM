"""
Procedures to plot demand and supply in the ONLT

Version 2.0 (2025)

@author: pheijnen
"""
import numpy as np
import matplotlib.pyplot as plt

"""
=============================================================================
PROCEDURE to find the total demand and supply for all nodes over all time steps
=============================================================================
"""
def demand_over_nodes(input_dict):
# Read all demand-supply profiles
    data_file = input_dict['demand']
# Define dictionaries for total demand and supply per node (start value = 0)
    demand = {node:0 for node in data_file[0].keys()}
    supply = {node:0 for node in data_file[0].keys()}
# For each demand supply profile
    for timestep in data_file.keys():
# For each node
        for node in data_file[timestep].keys():
# Define demand for node
            node_demand = data_file[timestep][node]
# If node is consumer
            if node_demand > 0:
# Add to total demand of node
                demand[node] += node_demand
# If node is supplier
            else:
                supply[node] += abs(node_demand)
# Increase total number of time steps by 1
    total_time = len(data_file.keys())
# Print total number of time steps
    print('Total number of time steps:',total_time)
# Plot bar chart with total supply and total demand per node
    plot_demand_and_supply(supply,demand,
                           title = 'total supply-demand for each node',
                           xlabel = 'supply and demand node')
# Return total number of time steps
    return total_time

"""
=============================================================================
PROCEDURE to plot the total demand and supply for all nodes over all time steps
=============================================================================
"""
def plot_demand_and_supply(supply,demand,title,xlabel):
# Define labels for bars on horizontal axis
    x_bars = np.arange(len(supply))
# Define correction value to show two bars in one plot
    corr = 0.4
# Plot supply over all nodes in red
    plt.bar(2*x_bars-corr,list(supply.values()), align='center',
            color='red',label = 'supply')
# Plot demand over all nodes in blue
    plt.bar(2*x_bars+corr, list(demand.values()), align='center',
            color='blue',label = 'demand')
# Plot tick on horinzonal axis
    plt.xticks(2*x_bars, list(supply.keys()))
# Plot labels, legend and title
    plt.xlabel(xlabel)
    plt.ylabel('total supply-demand')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()

"""
=============================================================================
PROCEDURE to find the total demand and supply for all time steps over all nodes
=============================================================================
"""
def demand_over_time(input_dict):
# Read all demand-supply profiles
    data_file = input_dict['demand']
# Define dictionaries for total demand and supply per time step (start value = 0)
    demand = {timestep:0 for timestep in data_file.keys()}
    supply = {timestep:0 for timestep in data_file.keys()}

# For each time step
    for timestep in data_file.keys():
# For each node
        for node in data_file[timestep].keys():
# Find demand or supply for node i
            node_demand = data_file[timestep][node]
# If node is consumer
            if node_demand > 0:
# Add to total demand of timestep
                demand[timestep] += node_demand
# If node is producer
            else:
# Add to total supply of timestep
                supply[timestep] += abs(node_demand)
# Plot bar chart with total supply and total demand per timestep
    plot_demand_and_supply(supply,demand,
                           title = 'total supply-demand for each time step',
                           xlabel = 'time step')

"""
=============================================================================
PROCEDURE to plot k (or all) representative demand-supply patterns for all nodes
=============================================================================
"""
def plot_demand(demand_dict):
    demand_over_time = [[] for node in demand_dict[0].keys()]
# For each k demand-supply pattern
    for timestep in demand_dict.keys():
# Make list of all demand and supply of nodes over all time steps
        for node,demand_supply in enumerate(demand_dict[timestep].values()):
            demand_over_time[node].append(float(demand_supply))
# For each node and its demand-supply over time
    for node,demand in enumerate(demand_over_time):
# Plot a line plot of the demand
        plt.plot(demand,label='terminal'+str(node),marker = 'o')
# Define labels and legend
    plt.xlabel('time step')
    plt.ylabel('demand or supply')
    plt.legend(loc='upper right')
    plt.xticks(range(len(demand_over_time[0])))
    plt.show()