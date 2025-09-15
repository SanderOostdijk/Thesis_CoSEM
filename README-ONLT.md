# Optimal Network Layout Tool

## Optimal Network Layout Tool
The tool finds a minimal cost network that connects multiple sources with multiple sinks taking into account the demand and supply patterns over various time steps. Depending on the time, nodes can be suppliers at some times and consumers at other times. Routing can (if needed) be limited to specific connections or around obstacles in the area. Part of the network can already exist, and new connections can use existing ones if sufficient capacity remains and the reuse is cost-effective.  

## Software
The software is written in Python 3. To use the model, you should install Anaconda. You can find Anaconda on https://www.anaconda.com/download/. Choose the latest Python 3 version available. We will use the Python-environment Spyder that comes with Anaconda.

## Input
As input, the model uses an Excel file with a specific format. There is one sub-directory 'case_study'  in the main folder. This directory contains the Excel files ‘input_data_simpel.xlsx’ and  ‘input_data.xlsx’ for a working example. These input files can be adapted or copied.
The Excel file might contain several worksheets that are clarified in the ONLT tutorial.


## Description
The ONLT works from the user_interface file in which 5 main steps are executed. 

## Step 0: Preparation
This is a preparation step. All data is read from the input-file and stored in a dictionary. Besides the ONLT tests whether the demand and supply sum up to zero in each time step. If so, this will speed up the search for minimal cost network. Also some plots on the demand and supply are generated. 

## Step 1: Minimum Spanning Tree
The starting network will be the minimum spanning tree. It is the minimum-length network that connects all nodes. Just enough capacity is assigned to the connections in the network to satisfy the most demand by supply or storage at every time step and to put the remaining supply in the nearest possible storage. A plot of the final network and the corresponding costs are given as an output. The results are also saved to an output Excel file.

## Step 2: Minimum Cost Spanning Tree
The minimum-spanning-tree solution did not consider the capacity cost for building a specific connection. By rewiring the connections in the minimum-spanning-tree, a better solution might be found. The rewiring process is a heuristic process and does not guarantee the optimal solution to be found since earlier decisions may restrict later choices.  

## Step 3: Minimum Cost Steiner Tree
In some cases, shorter networks exist when allowing extra splitting points in the connections. Since length generally contributes significantly to the building costs, this might reduce the network costs. Adding extra splitting points in the network might also introduce new costs for building these splitting points. So a good balance has to be found.
This model step adds splitting points to the network if they give profitable improvements. 

## Step 4: Improvement Rounds
In the last round, step 2 and 3 are repeated as long as better results are found.

## Results
All results on the intermediate networks and the final one can be found in the Excel output file. 

## Tutorial
The extensive tutorial gives all the details of working with the ONLT.
