#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 11:17:59 2017

@author: Aaron Renfroe
"""

from lib.workingWith3dDataSets import GeneratedDataSet, DataSet3d
import copy

'''
Class: ConnectingTissue

The class receives:
    a 2d list of size 3xn (x vals, y vals and z vals) of the coordinates
        where the length of the x_vals, y_vals, and z_vals must be equal
    connections (dict): a dictionary that contains node connectons.
        The order of these connections is correct but the direction is not correct for every fiber
        example: 0 -> 1, 1 -> 2, 2 -> 3, 3->4, 9->8, 8 -> 7, 7->6, 6 -> 5
    *con_from_1_to_2: can take multiple values as the connecting tissue connects two surfaces together
        connections to external surfaces with the this fibers node as the key and the other surfaces node as the value
        
Attributes:
    nodes (list(list(float),list(float),list(float))): The provided 2d Array of 3d coordinates
    graph (dict{int : int}): a dictionary of nodal connections: # think chains
        example: 0 -> 1, 1 -> 2, 2 -> 3, 3->4, 5->6, 6 -> 7 
        If you look, 0 is the start and 4 is the end as 4 does not appear as a 'key' in the dictionary and is not linked to another node
        Likewise 0 is not a connection to another node as it is never a 'value' in the dictionary
        So therefore 0 is the starting node and 4 is the ending node and 1,2,3
    fiber_keys (list(list(int))): a list of the indexes of the fibers, for example the first item in the list will contain the indexs for all the nodes that are in that fiber
    starting_nodes (list(int)): the indexes of the nodes that connect to surface1
    ending_nodes (list(int)): the indexes of the nodes that connect to surface2
'''
class ConnectingTissue(GeneratedDataSet):

    """
    Constructor: __init__
    """
    def __init__(self, nodes, connections, con_from_1_to_2):
        self.avw_connections = con_from_1_to_2

        xyz = [[],[],[]]
        for i,v in enumerate(nodes):
            
            xyz[0].append(v[0])
            xyz[1].append(v[1])
            xyz[2].append(v[2])
            
        GeneratedDataSet.__init__(self,xyz[0], xyz[1], xyz[2])
        
        self.starting_nodes, self.ending_nodes, self.fibers_keys = self._init_start_end_graph_for_para(connections, con_from_1_to_2)
#        starting_nodes2, ending_nodes2, fibers2 = self._init_start_end_graph_for_para(connections2, con_from_1_to_2)
							 
        
#        self.fibers_keys = []
#        
#        for index, item in enumerate(self.ending_nodes):
#            
#            j = item
#            fiber = [j]
#            
#            while j in self.graph:
#                fiber.append(self.graph[j])
#                j = self.graph[j]
#                
#            self.fibers_keys.append(fiber)
            

										   
								 
				
										  
    
    
    """
    Function: update_node
    
    given an index and a point
    updates the node coordinates at that index
    """
    def update_node(self, i, point):
        self.modify_point(point,i)
        #self.nodes[i][0] = point.x
        #self.nodes[i][1] = point.y
        #self.nodes[i][2] = point.z
        
    """
    Function: nodes_as_xyz_list
    
    returns: a 2d array with the first column being all the x values of the nodes and the z values are the 3rd column
    can be used to create a DataSet3d
    """
    def nodes_as_xyz_list(self):
        xs = copy.deepcopy(self.xAxis)
        ys = copy.deepcopy(self.yAxis)
        zs = copy.deepcopy(self.zAxis)
        return [xs,ys,zs]


    """
    Function: asDataSet3d
    
    to keep legacy code working, might not be needed, comment it out and see if the code breaks :)
    """
    def asDataSet3d(self):
        return self
    
    """
    Function: fibers
    
    returns: a list(list(Points))
        example: the first Node of the first fiber
            x = value[0][0][0]
            y = value[0][0][1]
            z = value[0][0][2]
         a 3d array of the fibers with coordinate values 
         as opposed to fiber_keys which is a 2d array that contians lists of indexes
         
    """
    def fibers(self):
        fiber_nodes = []
        for f_indexs in self.fibers_keys:
            run = []
            for i in f_indexs:
                self.node(i)
                run.append(self.node(i))
            fiber_nodes.append(run)
            
        return fiber_nodes
    
    
    """
    Function: get_ends_of_fibers
    
    returns: the fibers nodes connected to the body
    """
    def get_ends_of_fibers(self):
        
        xs = []
        ys = []
        zs = []
        for node in self.ending_nodes:
            point = self.node(node)
            xs.append(point.x)
            ys.append(point.y)
            zs.append(point.z)

        return DataSet3d(xs, ys, zs)

    """
    Function: get_starts_of_fibers
    
    returns: the fibers nodes connected to the surface or tissue
    """
    def get_starts_of_fibers(self):
        
        xs = []
        ys = []
        zs = []
        for node in self.starting_nodes:
            point = self.node(node)
            xs.append(point.x)
            ys.append(point.y)
            zs.append(point.z)

        return DataSet3d(xs, ys, zs)
        
    """
    Function: _init_start_end_graph
    
    This Method fixes the direction the nodes are in because the connections given are not all in the same direction
        meaning sometime the starting node will be on surface one and the fiber goes to surface two.
        and vice versa
    Parameters: 
        connections (dict): a dictionary that contains node connectons. half of which are in the wrong order
        ext_connections:  connections to external surfaces with the this node indexs as the key and the other surfaces node as the value
            used to establish a graph whose verticies are traveling in the right direction
    Returns: list of the  indexes of the starting nodes, list of the  indexes of the ending nodes
            
    """
    def _init_start_end_graph(self, connections, ext_connections):
        
        if len(list(connections.values())[0]) > 1:
             raise ValueError("The current code is only written for a 1d line. You pasted a node with multiple connections indicating a surface")
        
        
        reverse_connections = {}
        new_connections = {}
        for k in connections:
            reverse_connections[connections[k][0] - 1] = k - 1
            new_connections[k-1] = connections[k][0] -1
        
        first = [k for k in new_connections]
        second = [v for v in new_connections.values()]
        start_test = list(ext_connections.keys())
        graph = {}
        
        for fiber_start in start_test:
            link = fiber_start
             
            if link in new_connections:
                while link in new_connections:
                    graph[link] = new_connections[link]
                    link = new_connections[link]
            elif link in reverse_connections:
                 while link in reverse_connections:
                    graph[link] = reverse_connections[link]
                    link = reverse_connections[link]
            else:
                #print("This should never happen")
                raise ValueError("node was not found")
                        
        
        first = [k for k in graph]
        second = [v for v in graph.values()]
        starting_nodes = list(set(second) - set(first))
        ending_nodes = list(set(first) - set(second))
        #ending_nodes = list(set(second) - set(first))
        
        return starting_nodes, ending_nodes, graph
        
    '''
    Function: _init_start_end_graph_for_para
    '''
    def _init_start_end_graph_for_para(self, connections, ext_connections):
        
        
        if len(connections[0]) > 2:
             raise ValueError("The current code is only written for a 1d line. You pasted a node with multiple connections indicating a surface")
        
        
        fibers = []
        current_run = []
        nodes_on_surface = list(ext_connections.keys())
        prev = None
        for i in range(0, len(connections)):
            
            con = connections[i]
            if con == [708, 12]:
                print("stop")
                
            con[0] = con[0] - 1 
            con[1] = con[1] - 1 
            
            if 0 < i and len(current_run) > 0 and con[0] != prev:
                current_run.append(prev)
                if current_run[0] in nodes_on_surface:
        
                    current_run.reverse()
                
                fibers.append(current_run)
                current_run = []
            
            current_run.append(con[0])
                
            prev = con[1]
        
        
        current_run.append(prev)
        if current_run[0] in nodes_on_surface:
            current_run.reverse()
        fibers.append(current_run)
        
        starting_nodes = [item[0] for item in fibers]
        ending_nodes = [item[-1] for item in fibers]

        return starting_nodes, ending_nodes, fibers
        
        
        
        
        
