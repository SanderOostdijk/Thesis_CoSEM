"""
Extra geometric procedures for min cost Steiner tree in the ONLT

Version 2.0 (2025)

@author: pheijnen
"""

import sympy.geometry as sg
import networkx as nx
import numpy as np
from math import pi,acos

"""
=============================================================================
PROCEDURE to determine extra point to find optimal location of steiner node
given the three nodes A,B,C the steiner node is connected to and the forces
on the related edges
=============================================================================
"""
def third_point(A,B,C,force_A,force_B,force_C):
    if force_C != 0:
# Distance between point A and point B
        dist_AB = A.distance(B)
# Circle with midpoint A and radius d*CB/cC
        circle_A = sg.Circle(A,dist_AB*force_B/force_C)
# Circle with midpoint B and radius d*cA/cC
        circle_B = sg.Circle(B,dist_AB*force_A/force_C)
# Intersection point gives top of triangle on AB
        point_inter = circle_A.intersection(circle_B)
# If not 2 intersections points found
        if len(point_inter)<2: return []
# Given that P on the other side of AB than C
        elif same_side(A,B,C,point_inter[0]):
            return [round(float(point_inter[1][0]),5),
                    round(float(point_inter[1][1]),5)]
        else:
# Return third point of the circle with correct angles
            return [round(float(point_inter[0][0]),5),
                    round(float(point_inter[0][1]),5)]
    else: return []

"""
=============================================================================
PROCEDURE to find optimal location of Steiner node connected to A,B,C
given the third point PS
=============================================================================
"""
def location_steiner_point(A,B,C,PS):
# Circle through AB and 3d point with angle alpha_C on one circle arch and
# pi-alpha_C on the other circle_arch
    circle_ABP = sg.Circle(A,B,PS)
# If point C lies not within the circle
    if C.distance(circle_ABP.center) > circle_ABP.radius:
# Line through C and 3d point
        line_C = sg.Line(PS,C)
# Intersection of line with circle with correct angles
        S = circle_ABP.intersection(line_C)
# Intersection point is Steiner point
        if [int(S[0][0]),int(S[0][1])] == [int(PS[0]),int(PS[1])]:
            SP = S[1]
        else:
            SP = S[0]
# Check if Steiner point lies on arc AB
        if same_side(A,B,C,SP):
# Return optimal location of Steiner point
            return (float(SP[0]),float(SP[1]))
        else:
            return None
    else:
        return None


"""
=============================================================================
PROCEDURE To find the angle ABC in which point B is transformed to (0,0)
=============================================================================
"""
def translate_to_zero(A,B,C):
    return [A[0]-B[0],A[1]-B[1]],[C[0]-B[0],C[1]-B[1]]

"""
=============================================================================
PROCEDURE to find the size of the inner angle between vectors v and w
=============================================================================
"""
def inner_angle(v,w):
    np.seterr(divide='ignore', invalid='ignore')
    cosx=round(np.dot(v,w)/(np.linalg.norm(v)*np.linalg.norm(w)),5)
    return acos(cosx) # in radians

"""
=============================================================================
PROCEDURE to find the inner angle size between vectors A and B
=============================================================================
"""
def angle_size(A, B):
    inner=inner_angle(A,B)
    det = np.linalg.det([A,B])
    if det>0:
        inner = 2*pi-inner
    if inner > pi:
        return 2*pi-inner
    else:
        return inner

"""
=============================================================================
PROCEDURE to determine if point P and point C are on same side of line AB
=============================================================================
"""
def same_side(A,B,C,P):
    cp1 = ((A[1]-B[1])*(C[0]-A[0])+(B[0]-A[0])*(C[1]-A[1]))
    cp2 = ((A[1]-B[1])*(P[0]-A[0])+(B[0]-A[0])*(P[1]-A[1]))
    if cp1*cp2>=0:
        return True
    else:
        return False

"""
=============================================================================
PROCEDURE to find in full Steiner tree G
the Steiner node with the most neighbors with known locations
=============================================================================
"""
def highest_known_neighbors(G,coord,tabu):
    best = []
    best_nb = []
# Initial value for minimum number of known neighbors
    min_nb = 0
# For each steiner node with unknown location
    for i in [i for i in G.nodes() if not i in coord.keys() and not i in tabu]:
# Find neighbors of the steiner node in the graph
        nb = set(nx.neighbors(G,i))
# Find number of neighbors with known location
        known_nb = list(set(nb).intersection(set(coord.keys())))
# Remember node and number of known neighbors if better than before
        if len(known_nb) > min_nb:
            min_nb = len(known_nb)
            best_nb = known_nb
            best = i
# Return steiner node with most known neighbors
    return best,best_nb

"""
=============================================================================
PROCEDURE to give the full steiner tree from graph G
with given terminals and steiner nodes
=============================================================================
"""
def full_steiner_tree(G,terminals,steiner):
    return G.subgraph(list(terminals)+list(steiner))

"""
=============================================================================
PROCEDURE to find optimal position of Steiner node minimizing cost
=============================================================================
"""
def coordinates_steiner_nodes(G,terminals,steiner,beta):
# All known coordinates for the full Steiner tree
    coord = {i:G.nodes[i]['coord'] for i in terminals}
    coord1 = coord.copy()
# Full Steiner subgraph of G given terminals and steiner nodes
    FST = full_steiner_tree(G,terminals,steiner)
# Steiner nodes already scanned
    tabu = []
# While not all coordinates of Steiner nodes are known
    while not all([j in coord1.keys() for j in steiner]):
# Determine Steiner node with highest number of located neighbors
        s,kn_nb = highest_known_neighbors(FST,coord1,tabu)
# Add Steiner node to tabu list
        tabu += [s]
# If all 3 neighbors of Steiner nodes are known
        if len(kn_nb) == 3:
            A = sg.Point(coord1[kn_nb[0]]) #First neighbor
            B = sg.Point(coord1[kn_nb[1]]) #Second neighbor
            C = sg.Point(coord1[kn_nb[2]]) #Third neighbor
# Define the forces (costs/length) on the edges adjacent to the Steiner node
            cA = FST[kn_nb[0]][s]['capacity']**beta
            cB = FST[kn_nb[1]][s]['capacity']**beta
            cC = FST[kn_nb[2]][s]['capacity']**beta
# Find optimal location of Steiner node
            PS = third_point(A,B,C,cA,cB,cC)
            if PS:
                sp = location_steiner_point(A,B,C,PS)
                if sp:
                    coord1[s] = sp
                    tabu = []
                else: break
            else: break
# If only 2 neighbors are known
        elif len(kn_nb) == 2:
            A = sg.Point(coord1[kn_nb[0]]) #First neighbor
            B = sg.Point(coord1[kn_nb[1]]) #Second neighbor
# Determine unknown neigbors
            nb = set(nx.neighbors(FST,s)).difference(set(kn_nb))
# Take the highest-value steiner node
            i = max(nb.difference(set(tabu)))
# Take a (random) location for steiner node s
            if s in coord.keys():
                C = sg.Point(coord[s])
            else:
                C = sg.Point([0,0])
# Define the forces (costs/length) on the edges adjacent to Steiner node s
            cA = FST[kn_nb[0]][s]['capacity']**beta
            cB = FST[kn_nb[1]][s]['capacity']**beta
            cC = FST[i][s]['capacity']**beta
# Find optimal location of Steiner node i
            PS = third_point(A,B,C,cA,cB,cC)
# If optimal location found, update full steiner tree
            if PS:
                FST1 = FST.copy()
                x = 100+FST1.number_of_nodes()
                #FST.add_node(x)
                FST1.add_edge(i,x)
                FST1[i][x]['capacity'] = FST1[s][i]['capacity']
                FST = FST1.copy()
                coord1[x] = (PS[0],PS[1])
            else:
                break
# If all coordinates found return these
    if not all([j in coord1.keys() for j in steiner]):
        return {}
    else: return coord1



