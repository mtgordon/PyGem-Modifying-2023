#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 12:20:07 2017

@author: Aaron Renfroe @ California Baptist University 
"""
import copy

class DataSet3d(object):
        
    def __init__(self,x, y, z): #[x1,x2, x3, x4,...] ,[y1,y2, y3, y4,...] ,[z1,z2, z3, z4,...] 
        
        lenx = len(x)
        if lenx != len(y) != len(z):
            raise ValueError("Axis Arrays must be same length")
        else:
            self.number_of_points = lenx
            self.xAxis = x
            self.yAxis = y
            self.zAxis = z
        
    def generatedCopy(self):
        return GeneratedDataSet(copy.deepcopy(self.xAxis), copy.deepcopy(self.yAxis), copy.deepcopy(self.zAxis[:]))
    
    def get_list_of_points(self):
        new_list = []
        for index in range(0,len(self.xAxis)):
            new_list.append(Point(self.xAxis[index], self.yAxis[index], self.zAxis[index]))
        return new_list
    
    def zipped(self):
        return zip(self.xAxis, self.yAxis, self.zAxis)
    
    def node(self, i):
        return Point(self.xAxis[i], self.yAxis[i], self.zAxis[i])
    
    def __str__(self):
        return str("DataSet3d: " + str(self.number_of_points) + " Points\n" + "x: "+ str(self.xAxis) +"\n" + "y: "+ str(self.yAxis) +"\n" + "z: "+ str(self.zAxis))
    
    def to_inp_string():
        print("Todo: inplement this")
        

   
class GeneratedDataSet(DataSet3d):
    # This functionality can probably be moved from here in to DataSet3d since python doesnt have private members or functions
    """
    Given a specific index returns the node at that index as Point
    Returns (Point)
    """
    
    def modify_point(self, point, i):
        try:
            self.xAxis[i] = point.x
            self.yAxis[i] = point.y
            self.zAxis[i] = point.z
        except IndexError:
            raise IndexError("DataSet3d does not have a point at the given index: " + str(i))
    
    def trim(self):
        if len(self.xAxis) > self.number_of_points:
            self.xAxis = self.xAxis[:self.number_of_points]
            self.yAxis = self.yAxis[:self.number_of_points]
            self.zAxis = self.zAxis[:self.number_of_points]
                
        return self

class Point:
    
    def __init__(self,x, y, z=None):
        
        self.x = x
        self.y = y
        self.z = z
        
        
        
        
    # This is just a helper function that defines what print(myPoint) looks like  Point x: 1.001, y: 1.0004, z: 23.4444
    def __str__(self):
        return "Point,  x: " + str(self.x) + " y: " + str(self.y)+" Z: " + str(self.z)
    
    def __repr__(self):
        return "x: " + str(self.x) + ", y: " + str(self.y)+", Z: " + str(self.z)
    
    '''
    p1 = Point(1,2,3)
    p2 = p1
    p2.x = 4
    print(p1) -> 4, 2, 3
    print(p2) -> 4, 2, 3
    p1 = Point(1,2,3)
    p2 = p1.copy()
    p2.x = 4
    print(p1) -> 1, 2, 3
    print(p2) -> 4, 2, 3
    
    '''
    def copy(self):
        return Point(float(self.x), float(self.y), float(self.z))
    
    def __hash__(self):
        return hash(self.x, self.y, self.z)

    def __eq__(self, other):
        return (self.x, self.y, self.z) == (other.x. other.y, other.z)

    def distance(self, other):
        return ((self.x-other.x)**2+(self.y-other.y)**2+(self.z-other.z)**2)**0.5