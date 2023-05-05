# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:51:35 2019

@author: mgordon
"""

import math
import numpy as np
import configparser
import json
import subprocess
import os
import time
from numpy import genfromtxt, zeros
from lib.Surface_Tools import getBnodes, numOfNodes, pythag, find_starting_ending_points_for_inside, findInnerNodes, findClosestNodeNumber
from lib.workingWith3dDataSets import Point
from lib.Node_Distances import getXClosestNodes
import lib.IOfunctions as io
from scipy import interpolate #, UnivariateSpline # InterpolatedUnivariateSpline, 
import csv
from sympy import Point3D
from sympy.geometry import Line3D
from lib.workingWith3dDataSets import DataSet3d
from lib.IOfunctions import extractPointsForPartFrom
from lib.odb_io import get_odb_data
import matplotlib.pyplot as plt
from math import hypot

'''
Function: Calc_Reaction_Forces
'''
def Calc_Reaction_Forces(path_base_file_name, output_base_filencurveame, GenericINPFile, INI_file, frame):
    
    config = configparser.ConfigParser()
    config.sections()
    config.read(INI_file)
    
    AbaqusBatLocation= config["SYSTEM_INFO"]["AbaqusBatLocation"]
    
################ Starting New Code
    headers = []
#    forces = []
    
    PartName1 = "OPAL325_CL_V6-1"
    PartNameCS = "OPAL325_CL_v6"
    headers.append(PartNameCS + ' Total Resultant Force')
    headers.append(PartNameCS + ' Total Component Force')
    
    frames = frame
    Bnodes = getBnodes(GenericINPFile, PartNameCS)
    PassingNodes1 = ','.join(str(i) for i in Bnodes)
    
    PartName2 = "OPAL325_US_V6-1"
    PartNameCS = "OPAL325_US_v6"
    headers.append(PartNameCS + ' Total Resultant Force')
    headers.append(PartNameCS + ' Total Component Force')
    
    
    Bnodes = getBnodes(GenericINPFile, PartNameCS)
    PassingNodes2 = ','.join(str(i) for i in Bnodes)
  
   
#   COORD is the tag for the coordinate for the ODB file
    node_property = "RF"
    
#    Use the last time step to get final position
    frames = frame
    

    nodes = str(PassingNodes1) + ';' + str(PassingNodes2)
    
    odb_filename = path_base_file_name

    material_names = PartName1 + ';' + PartName2
    
    forces = get_odb_data(material_names, nodes, node_property, frames, AbaqusBatLocation, odb_filename)

    print(forces)
    
                 
    print(headers, forces)

    return(headers, forces)


'''
Function: getFEADataCoordinates

Taken from ExposedVaginalWallMeasurements.py
Cannot be imported from there because then it runs the file
Need to move the function to a file for the library
'''
def getFEADataCoordinates(FileName):
    csv_data = np.genfromtxt (FileName, delimiter=",")

# If there are multiple rows, get the last one
    try:
        data = csv_data[-1,:]
    except IndexError:
        data = csv_data
#    print(FileName)
#    print(len(data))
    nodes = math.floor((len(data)/3))
    x=np.zeros(nodes)
    y=np.zeros(nodes)
    z=np.zeros(nodes)
    for i in range(0,nodes):
        x[i]=data[i*3]
        y[i]=data[i*3+1]
        z[i]=data[i*3+2]
    return [x,y,z]


# from https://www.geeksforgeeks.org/program-to-find-equation-of-a-plane-passing-through-3-points/
# Python program to find equation of a plane  
# passing through given 3 points. 
  
'''
Function: equation_plane

Function to find equation of plane. 
'''
def equation_plane(x1, y1, z1, x2, y2, z2, x3, y3, z3):
      
    a1 = x2 - x1 
    b1 = y2 - y1 
    c1 = z2 - z1 
    a2 = x3 - x1 
    b2 = y3 - y1 
    c2 = z3 - z1 
    a = b1 * c2 - b2 * c1 
    b = a2 * c1 - a1 * c2 
    c = a1 * b2 - b1 * a2 
    d = (- a * x1 - b * y1 - c * z1)
    return(a,b,c,d)


# from https://www.geeksforgeeks.org/distance-between-a-point-and-a-plane-in-3-d/
# Python program to find the Perpendicular(shortest) 
# distance between a point and a Plane in 3 D. 
  
  
'''
Function: shortest_distance

Function to find distance 
'''
def shortest_distance(x1, y1, z1, a, b, c, d):
      
    d = (a * x1 + b * y1 + c * z1 + d) 
#    print("d = ", d)
    e = (math.sqrt(a * a + b * b + c * c))
#    print("e = ", e)
#    Might need to add a negative side if we want the other direction
    perpendicular_distance = -1*d/e
    return(perpendicular_distance)

'''
Function: calc_prolapse_size

###########################################################
# Read in the parameters from the ini file
###########################################################
'''
def calc_prolapse_size(GenericINPFile, ODBFile, INP_File, INI_file, frame, AVW_csv_file):
#"Redo soft tissue prolapse measurement to use a point at the top of the PM-Mid and top of PM-Body with same x coordinate:

    print('Generic INP: ', GenericINPFile)
    print('File for prolapse measuring: ', ODBFile)
    print('Specific INP: ', INP_File)
    print('INI: ', INI_file)
    
    config = configparser.ConfigParser()
    config.sections()
    
    config.read(INI_file)

    ##### Change to choose 2 points from PM-Mid and one from PBody
    ##### for MP-Mid I should choose the point with negative X that is about
    ##### 4mm up (Y coord) and the point with a positive X that is about 4mm up (Y coord)
    ##### for PBody, choose the point that is the closest to 0,0,0
#1) get the PM Body inner arc nodes
    PM_MID      = "OPAL325_PM_mid"
    connections = io.get_interconnections(INP_File, PM_MID)
    PM_Mid = io.get_dataset_from_file(INP_File, PM_MID)
    starting_index, ending_index = find_starting_ending_points_for_inside(PM_Mid)
    innerNodes = findInnerNodes(PM_Mid, connections, starting_index, ending_index)    

#2) find the point with the highest Y value = PM Mid top
    MaxY = -999999
    for i in innerNodes:
        if PM_Mid.node(i).y > MaxY:
            MaxY = PM_Mid.node(i).y
            midsaggital_x = PM_Mid.node(i).x
            PM_Mid_top_node = i
    PM_Mid_top_original = Point3D(PM_Mid.node(PM_Mid_top_node).x,PM_Mid.node(PM_Mid_top_node).y,PM_Mid.node(PM_Mid_top_node).z)
    
#3) collect all PM Body points who's x coords are PM Mid top.x +/- 3            
#4) Choose the top point = PM Body top
    PBODY       = "OPAL325_PBody"
    pbody_points = np.array(io.extractPointsForPartFrom(INP_File, PBODY))
#    print(pbody_points)
#    pbody_surface = DataSet3d(list(pbody_points[:, 0]), list(pbody_points[:, 1]), list(pbody_points[:, 2]))
    MaxY = -999999
#    print(pbody_surface)
    for i in range(0,len(pbody_points)):
        if abs(pbody_points[i][0] - midsaggital_x) < 5:
            if pbody_points[i][1] > MaxY:
                MaxY = pbody_points[i][1]
                pbody_top_middle_node = i

#5) set PM Body top.x = PM MId top.x
#    set the x coordinate to that of PM Top s (midsaggital_x) o that we get the midsaggital line
    pbody_top_original = Point3D(midsaggital_x, pbody_points[pbody_top_middle_node][1], pbody_points[pbody_top_middle_node][2])


#6) Create a line with PBody top and PM Mid top
    prolapse_measurement_line_absolute = Line3D(PM_Mid_top_original, pbody_top_original)
    

    
##########################################################################################################
## I need to get the coordinates of the points for the plane after the run
## ...for this I need the output file
    
#####**************************TOP OF NEW CODE*************************************************
    AbaqusBatLocation= config["SYSTEM_INFO"]["AbaqusBatLocation"]    
    #   COORD is the tag for the coordinate for the ODB file
    node_property = "COORD"

#    Use the last time step to get final position
    frames = frame
    nodes = str(PM_Mid_top_node) + ';' + str(pbody_top_middle_node)
    odb_filename = ODBFile[:-4]
    material_names = PM_MID.upper() + '-1' + ';' + PBODY.upper() + '-1'
    points = get_odb_data(material_names, nodes, node_property, frames, AbaqusBatLocation, odb_filename)
    PM_Mid_top_deformed = Point3D(midsaggital_x, points[0][1], points[0][2])
    pbody_top_deformed = Point3D(midsaggital_x, points[1][1], points[1][2])
       
    plane_nodes = [PM_Mid_top_node + 1, pbody_top_middle_node + 1]
    
    
    
    
    
    
    
    
    
######**************************TOP OF OLD CODE*************************************************
#    
##    Generate a temporary file to store the coordinates from the ODB file
##    that will be used to calculate the plane
#    DataFileName = "Temp_Coords" # Not done yet
#    
#    #   COORD is the tag for the coordinate for the ODB file
#    Variable1 = "COORD"
##   We don't want a header in our temp file
#    Headerflag = 'N'
##    Do create a new file for the temp file
#    NewFileFlag = 'Y'
##    Use the last time step to get final position
#    Frames = frame
#    
#        
#    MaterialName = PM_MID.upper() + '-1'
#    PassingNodes = str(PM_Mid_top_node)
#    
#    upgraded_inp_file = ODBFile[:-4]
#    CallString = AbaqusBatLocation + '  CAE noGUI=".\lib\Get_Data_From_ODB"  -- -odbfilename "' + upgraded_inp_file.replace('\\','\\\\') + '" -partname ' + MaterialName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile ' + DataFileName + ' -headerflag ' + Headerflag + " -newfileflag " + NewFileFlag + " -frames " + Frames
#    print('_________________________________________')
#    print(CallString)
#    print('_________________________________________')
#    try:
#        os.remove(DataFileName+'.csv')
#    except:
#        pass
#
#    subprocess.call(CallString)
#    time.sleep(3)
#    
##        print(DataFileName)
#    Coordinates = genfromtxt(DataFileName + '.csv', delimiter=',')
#    PM_Mid_top_deformed = Point3D(midsaggital_x, Coordinates[1], Coordinates[2])
#    
#    
#    
#    MaterialName = PBODY.upper() + '-1'
#    PassingNodes = str(pbody_top_middle_node)
#    plane_nodes = [PM_Mid_top_node + 1, pbody_top_middle_node + 1]
#    upgraded_inp_file = ODBFile[:-4]
#    CallString = AbaqusBatLocation + '  CAE noGUI=".\lib\Get_Data_From_ODB"  -- -odbfilename "' + upgraded_inp_file.replace('\\','\\\\') + '" -partname ' + MaterialName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile ' + DataFileName + ' -headerflag ' + Headerflag + " -newfileflag " + NewFileFlag + " -frames " + Frames
#    print('_________________________________________')
#    print(CallString)
#    print('_________________________________________')
#    try:
#        os.remove(DataFileName+'.csv')
#    except:
#        pass
#
#    subprocess.call(CallString)
#    time.sleep(3)
#    
##        print(DataFileName)
#    Coordinates = genfromtxt(DataFileName + '.csv', delimiter=',')
#    pbody_top_deformed = Point3D(midsaggital_x, Coordinates[1], Coordinates[2])
#    
#    
######**************************BOTTOM OF OLD CODE*************************************************
    
    
    prolapse_measurement_line_deformed = Line3D(PM_Mid_top_deformed, pbody_top_deformed)
    
#    desired_distance = json.loads(config["PROLAPSE_SIZE"]["Vaginal_Wall_Distance"])


#7) Calculate distance between AVW points whose x val is PM Mid top.x +/- 3
#8) the largest distance of those points (hopefully correctly signed) is the prolapse size"    
#    ###################################################################
#    # Next get the points for the vaginal wall
##    PrintMaterialHeader = 'N'
##    NewOutputFile = 'Y'
#    ML = ["OPAL325_AVW_v6"]
#    MaterialList = ['OPAL325_AVW_V6-1']
#    MaterialSizeList = [numOfNodes(GenericINPFile, ML[0])]
#    nodes = list(range(1,MaterialSizeList[0]+1))
#    PassingNodes = ','.join(str(i) for i in nodes)
#    Variable1 = "COORD"
#    Headerflag = 'N'
#    NewFileFlag = 'Y'
#    Frames = frame
#    MaterialName = MaterialList[0]
#    DataFileName = 'AVW_nodal_coordinates'
#    CallString = AbaqusBatLocation + '  CAE noGUI=".\lib\Get_Data_From_ODB"  -- -odbfilename "' + upgraded_inp_file.replace('\\','\\\\') + '" -partname ' + MaterialName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile ' + DataFileName + ' -headerflag ' + Headerflag + " -newfileflag " + NewFileFlag + " -frames " + Frames
##        print(CallString)
#    subprocess.call(CallString)
#    
#    #May be needed for when I run it on a virtual machine because the files have to be uploaded
#    time.sleep(3)
    
#        print(DataFileName)
    (xs, ys, zs) = getFEADataCoordinates(AVW_csv_file)
#########################################


    THRESHOLD = 3 # Max Distance point can be from X axis

    
#    Getting the middle nodes and finding the one with the largest z coordiante
    middle_nodes = []
    max_z = -999999
    for index, xval in enumerate(xs):
        if (abs(0 - xval) < THRESHOLD):
            middle_nodes.append((xs[index], ys[index], zs[index]))
            if zs[index] > max_z:
                max_z = zs[index]
                start = (ys[index], zs[index])
   
    
    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################
    spline_ordered = [start]
    count = len(middle_nodes)
    
    distance_array=[0]
    used = []
    ## Sorting the spline 
    for i in range(0,count):
        least_dist = float("inf")
        least_indx = 0
        for j , point in enumerate(middle_nodes):
            if j not in used:
#                Calculate the distance between the last point added and the point that is being investigated
                dist = pythag(spline_ordered[-1][0] - point[1], spline_ordered[-1][1] - point[2])
                if dist < least_dist and j not in used:
                    least_dist = dist
                    least_indx = j
#       Keeping a log of the running total distance
        distance_array.append(least_dist+distance_array[-1])
        spline_ordered.append((middle_nodes[least_indx][1], middle_nodes[least_indx][2]))
        used.append(least_indx)
    
#    print('First Attempt')
#    print(spline_ordered)
    spline_ordered.pop(0)
    distance_array.pop(0)

#   If the last one has a higher y coordinate than the first one, the last one
#    is the distal end of the AVW and therefore the array will be flipped to put it first    
    if spline_ordered[0][0] < spline_ordered[-1][0]:
        spline_ordered = list(reversed(spline_ordered))
        
        
        
    X = np.array(spline_ordered).T # -> transpose
    

    curve_y = interpolate.UnivariateSpline(distance_array, list(X[0]), k = 5)
    curve_z = interpolate.UnivariateSpline(distance_array, list(X[1]), k = 5)
    
#    print(X)
    ## Set up for getting the length
#    new_vals = [[],[]]
#   Create an array that is small distance steps from one end of the AVW to the other
    steps = np.arange(0,distance_array[-1], 1)
    dist = 0
#    last_val = None



#    print(beststep)

#    dist2_array = [0]
    ## summing the length
    
#    last_val2 = [curve_x(steps[0]), curve_y(steps[0])]
    new_ys = []
    new_zs = []
#    for i, step in enumerate(steps[1:bestindex]):

### Find the step that is the closest to the GI Point
#    min_dist = 999999999    
#    this_dist = []
    for i, step in enumerate(steps):

#        val2 = [curve_x(step), curve_y(step)]
#        print(step)
#        value = curve_x(step)[()]
#        print('xs :', value)
        new_ys.append(curve_y(step)[()])
        new_zs.append(curve_z(step)[()])
#        new_xs.append(3)
#        new_xs.append(value)
#        print('xval :', curve_x(step))
#        print('step :', step)
        

#        last_val2 = val2
        
#    print('xs :', new_xs)
#    print('ys :', new_ys)
    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################    
    
    
    
    ## Setting up for Sorting the spline 
#    print('Unordered')
#    print(L)


#    print(middle_nodes)
#########################################
    # Check the distance for each AVW node to the deformed or undeformed line
    max_prolapse = -9999999999
    max_prolapse_absolute = -9999999999
#        print("Number of AVW points = ", len(xs))
    for j in range (len(middle_nodes)):
#                whole_matrix = np.array([[midsaggital_x, PM_Mid_top_original.y, PM_Mid_top_original.z], [midsaggital_x, pbody_top_original.y, pbody_top_original.z], [midsaggital_x, new_xs[j], new_ys[j]]], dtype='float')
        whole_matrix = np.array([[midsaggital_x, PM_Mid_top_original.y, PM_Mid_top_original.z], [midsaggital_x, pbody_top_original.y, pbody_top_original.z], [midsaggital_x, middle_nodes[j][1], middle_nodes[j][2]]], dtype='float')

        side_of_line = np.sign(np.linalg.det(whole_matrix.astype(float)))
        side_of_line2 = np.sign((pbody_top_original.y -PM_Mid_top_original.y)*(middle_nodes[j][2] - pbody_top_original.z) - (middle_nodes[j][1] - pbody_top_original.y)*(pbody_top_original.z - PM_Mid_top_original.z))

        distance_relative = prolapse_measurement_line_deformed.distance(Point3D(middle_nodes[j])) * side_of_line * -1
        distance_absolute = prolapse_measurement_line_absolute.distance(Point3D(middle_nodes[j])) * side_of_line * -1
#        print(middle_nodes[j], float(distance_absolute), side_of_line, side_of_line2)
        #            print(distance)
        if distance_relative > max_prolapse:
            max_prolapse = float(distance_relative)
            max_prolapse_node = j
            
        if distance_absolute > max_prolapse_absolute:
            max_prolapse_absolute = float(distance_absolute)
            max_prolapse_absolute_node = j

    print('original code :', max_prolapse, max_prolapse_absolute)

#    max_prolapse = -9999999999
#    max_prolapse_absolute = -9999999999
#
#    for j in range (len(new_xs)):
#        whole_matrix = np.array([[midsaggital_x, PM_Mid_top_original.y, PM_Mid_top_original.z], [midsaggital_x, pbody_top_original.y, pbody_top_original.z], [midsaggital_x, new_xs[j], new_ys[j]]], dtype='float')
#
#        side_of_line = np.sign(np.linalg.det(whole_matrix.astype(float)))
##        side_of_line2 = np.sign((pbody_top_original.y -PM_Mid_top_original.y)*(new_ys[j] - pbody_top_original.z) - (new_xs[j] - pbody_top_original.y)*(pbody_top_original.z - PM_Mid_top_original.z))
##        print(middle_nodes[j])
##        print((midsaggital_x, new_xs[j,0], new_ys[j,0]))
##        distance_relative = prolapse_measurement_line_deformed.distance(Point3D(middle_nodes[j])) * side_of_line * -1
#        distance_relative = prolapse_measurement_line_deformed.distance(Point3D([midsaggital_x, new_xs[j], new_ys[j]])) * side_of_line * -1
#        distance_absolute = prolapse_measurement_line_absolute.distance(Point3D([midsaggital_x, new_xs[j], new_ys[j]])) * side_of_line * -1
##        print(midsaggital_x, new_xs[j], new_ys[j], float(distance_absolute), side_of_line, side_of_line2)
#        if distance_relative > max_prolapse:
#            max_prolapse = float(distance_relative)
#            new_max_prolapse_node = (midsaggital_x, new_xs[j], new_ys[j])
#            
#        if distance_absolute > max_prolapse_absolute:
#            max_prolapse_absolute = float(distance_absolute)
#            new_max_prolapse_absolute_node = (midsaggital_x, new_xs[j], new_ys[j])
#
#    print('new code :', max_prolapse,max_prolapse_absolute)

#    print(middle_nodes[max_prolapse_node])
#    print(new_xs[new_max_prolapse_node], new_ys[new_max_prolapse_node])
#    print('graphing this: ', middle_nodes[1][:])
#    plt.plot(new_xs, new_ys)
##    plt.plot(middle_nodes[1][:], middle_nodes[2][:])
#    plt.plot([float(PM_Mid_top_deformed.y), float(pbody_top_deformed.y)], [float(PM_Mid_top_deformed.z), float(pbody_top_deformed.z)])
#    plt.plot([float(PM_Mid_top_original.y), float(pbody_top_original.y)], [float(PM_Mid_top_original.z), float(pbody_top_original.z)])
#    , float(PM_Mid_top_original.z)], [midsaggital_x, float(pbody_top_original.y), float(pbody_top.z)])
######################################################
##                New Negative Prolapse Code (not currently needed)
#                
#    spline_ordered = [start]
#    count = len(middle_nodes)
#    
#    distance_array=[0]
#    used = []
#    ## Sorting the spline 
#    for i in range(0,count):
#        least_dist = float("inf")
#        least_indx = 0
#        for j , point in enumerate(middle_nodes):
#            if j not in used:
##                Calculate the distance between the last point added and the point that is being investigated
#                dist = pythag(spline_ordered[-1][0] - point[0], spline_ordered[-1][1] - point[1])
#                if dist < least_dist and j not in used:
#                    least_dist = dist
#                    least_indx = j
##       Keeping a log of the running total distance
#        distance_array.append(least_dist+distance_array[-1])
#        spline_ordered.append(middle_nodes[least_indx])
#        used.append(least_indx)
#    
##    print('First Attempt')
##    print(spline_ordered)
#
##   If the last one has a higher y coordinate than the first one, the last one
##    is the distal end of the AVW and therefore the array will be flipped to put it first    
#    if spline_ordered[0][0] < spline_ordered[-1][0]:
#        distal_AVW = spline_ordered[-1]
#    else:
#        distal_AVW = spline_ordered[0]
#                
#                
#                
############################                
##                    Negative prolapse code
##   If the prolapse is negative, then a different clinical measurement is used
#    if max_prolapse < 5:
#
#
## Create the interpolation
#        interpolator = interpolate.Rbf(xs, zs, ys, function='linear', smooth=1)
#
## Grab the highest Z value for midsaggital node
##            May need to go back and change this if there are small bulges with
##           negative prolapse values or the midsaggital isn't the highest Z
#        endZ = max(zs)
#
##            find distance between that point (X=0, Z=max, y = point/interpolation) and the next (X = 0, Z = max - 1, y = interpolated)
#        distance = 0.0
#        StepSize = 0.02
##            Distance along the vaginal wall that clinicians use to determine
##            prolapse size
##            desired_distance = 25
#        oldZ = endZ
#        oldY = interpolator(0, endZ)
#        while distance < desired_distance:
#            newZ = oldZ - StepSize
#            newY= interpolator(0, newZ)
#
## Add it to the total distance
#        
#            distance += pythag(newZ - oldZ, oldY - newY)
#            oldY = newY
#            oldZ = newZ
#
## Check to see if it is greater than the target
## If not loop
## If so, interpolate the new point
## Find the distance from that point to the planes that were found above            
##                zRange = np.arange(fromValue, toValue, INCREMENT * (1 if fromValue < toValue else -1))
#
#        distance_relative = shortest_distance(0,newY,newZ,a,b,c,d)
#        distance_absolute = shortest_distance(0,newY,newZ,aBefore,bBefore,cBefore,dBefore)
#        max_prolapse = distance_relative
#        max_prolapse_absolute = distance_absolute
##            print('Coordinates are: 0, ', newY, newZ)
#        
#        
#        
#######################################################
#        Continuing after adding negative prolapse code
#    print("The prolapse size is: ", float(max_prolapse))
#    print("It occurred at : ", new_max_prolapse_node)
#    
#    print("The absolute prolapse size is: ", max_prolapse_absolute)
#    print("It occurred at :", new_max_prolapse_absolute_node)
#    print([midsaggital_x, float(PM_Mid_top_original.y), float(PM_Mid_top_original.z)], [midsaggital_x, float(pbody_top_original.y), float(pbody_top.z)])
    return(max_prolapse, max_prolapse_absolute, max_prolapse_node, plane_nodes)
#    return(float(max_prolapse), float(max_prolapse_absolute), new_max_prolapse_node, plane_nodes)
    

'''
Function: calc_prolapse_size_plane
'''
def calc_prolapse_size_plane(GenericINPFile, ODBFile, INP_File, INI_file, frame):
    # read from input file
#    ODBFile = base_file_name + '.odb'
#    INP_File = base_file_name + '.inp'
#    print('&&&&&&&&&&&&&&&& Inside the loop I thought I was in')
    print('Generic INP: ', GenericINPFile)
    print('File for prolapse measuring: ', ODBFile)
    print('Specific INP: ', INP_File)
    print('INI: ', INI_file)
    
    config = configparser.ConfigParser()
    config.sections()
    
    config.read(INI_file)

    ##### Change to choose 2 points from PM-Mid and one from PBody
    ##### for MP-Mid I should choose the point with negative X that is about
    ##### 25% of the way down the inner arch and the corresponding positive X node
    ##### for PBody, choose the point that is the closest to 0,0,0

    PM_MID      = "OPAL325_PM_mid"
    connections = io.get_interconnections(GenericINPFile, PM_MID)
    PM_Mid = io.get_dataset_from_file(GenericINPFile, PM_MID)
    starting_index, ending_index = find_starting_ending_points_for_inside(PM_Mid)
    innerNodes = findInnerNodes(PM_Mid, connections, starting_index, ending_index)    
    
    MaxY = -999999
    MinY = 99999
    for i in innerNodes:
        if PM_Mid.node(i).y < MinY:
            MinY = PM_Mid.node(i).y
        if PM_Mid.node(i).y > MaxY:
            MaxY = PM_Mid.node(i).y
    
    Range = MaxY - MinY
    RangeFromTop = 0.25
    TargetY = MaxY - RangeFromTop * Range

    PosClosestToY = 99999
    NegClosestToY = 99999
    for i in innerNodes:

        DistToTargetY = abs(PM_Mid.node(i).y - TargetY)
        if PM_Mid.node(i).x < 0 and DistToTargetY < NegClosestToY:
            NegNode = i + 1 #i is the index, not the node number
            NegClosestToY = DistToTargetY
        elif PM_Mid.node(i).x > 0 and DistToTargetY < PosClosestToY:
            PosNode = i + 1 #i is the index, not the node number
            PosClosestToY = DistToTargetY

    # The points to form the plane are input coordinates so we need to find the
    # node (number) that is closest to these coordinates
    plane_tissues  = config["PROLAPSE_SIZE"]["plane_tissues"]
    tissue_list = plane_tissues.split(',')
        

    Center = Point(0,0,0)
    plane_node_3, point_3 = getXClosestNodes(Center, 1, tissue_list[2], GenericINPFile)
    
    plane_nodes = [PosNode, NegNode, int(plane_node_3[0]) + 1]

    AbaqusBatLocation= config["SYSTEM_INFO"]["AbaqusBatLocation"]
    
    
##########################################################################################################
## I need to get the coordinates of the points for the plane after the run
## ...for this I need the output file
    
#    Generate a temporary file to store the coordinates from the ODB file
#    that will be used to calculate the plane
    DataFileName = "Temp_Coords" # Not done yet

    upgraded_inp_file = ODBFile[:-4]

#   COORD is the tag for the coordinate for the ODB file
    Variable1 = "COORD"
#   We don't want a header in our temp file
    Headerflag = 'N'
#    Do create a new file for the temp file
    NewFileFlag = 'Y'
#    Use the last time step to get final position
    Frames = frame
    x = zeros(3)
    y = zeros(3)
    z = zeros(3)
    xBefore = zeros(3)
    yBefore = zeros(3)
    zBefore = zeros(3)
    
    for i in range (3):
#        print(i)
        MaterialName = tissue_list[i].upper() + '-1'
        PassingNodes = str(plane_nodes[i])



        CallString = AbaqusBatLocation + '  CAE noGUI=".\lib\Get_Data_From_ODB"  -- -odbfilename "' + upgraded_inp_file.replace('\\','\\\\') + '" -partname ' + MaterialName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile ' + DataFileName + ' -headerflag ' + Headerflag + " -newfileflag " + NewFileFlag + " -frames " + Frames
        print('_________________________________________')
        print(CallString)
        print('_________________________________________')
        try:
            os.remove(DataFileName+'.csv')
        except:
            pass

        subprocess.call(CallString)

        time.sleep(3)

#        try:
#            os.environ.pop('PYTHONIOENCODING')
#        except KeyError:
#            pass
#        
#        JobPath = "I:\\Testing New Code"
#        
#        cmd = subprocess.Popen(CallString, cwd = JobPath, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
#                                           stderr=subprocess.PIPE, shell=True).communicate()[0]
##        
#        print(cmd)
#        print("error" in str(cmd))
#        
#        if ("error" in str(cmd)):
#            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#            print('There may have been a conversion error. Trying again')
#            path = os.path.dirname(os.path.abspath(upgraded_inp_file))
#            file_name = os.path.basename(upgraded_inp_file)
#            original_file_name = file_name.replace('Working_','')
#            file_to_analyze = upgraded_inp_file
#            file = path + "\\" + original_file_name
#            try:
#                os.remove(upgraded_inp_file.encode('unicode_escape'))
#            except:
#                pass
#            CallString = AbaqusBatLocation + ' -upgrade -job "' + file_to_analyze.replace('\\','\\\\') + '" -odb "' + file.replace('\\','\\\\') + '"'
#            print(CallString)
#            subprocess.call(CallString)
#
#
#    #        subprocess.call(CallString)
#            time.sleep(3)
#            
#            
#            CallString = AbaqusBatLocation + '  CAE noGUI=".\lib\Get_Data_From_ODB"  -- -odbfilename "' + upgraded_inp_file.replace('\\','\\\\') + '" -partname ' + MaterialName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile ' + DataFileName + ' -headerflag ' + Headerflag + " -newfileflag " + NewFileFlag + " -frames " + Frames
#            print('_________________________________________')
#            print(CallString)
#            print('_________________________________________')
#            try:
#                os.remove(DataFileName+'.csv')
#            except:
#                pass
#    
#            time.sleep(3)
#    
#            try:
#                os.environ.pop('PYTHONIOENCODING')
#            except KeyError:
#                pass
#            
#            JobPath = "I:\\Testing New Code"
#            
#            cmd = subprocess.Popen(CallString, cwd = JobPath, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
#                                               stderr=subprocess.PIPE, shell=True).communicate()[0]
#            print(cmd)
#        
#        time.sleep(3)
#        
#        if ("error" in str(cmd)):
#            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#            print('There may have been a conversion error. Trying again')
#            path = os.path.dirname(os.path.abspath(upgraded_inp_file))
#            file_name = os.path.basename(upgraded_inp_file)
#            original_file_name = file_name.replace('Working_','')
#            file_to_analyze = upgraded_inp_file
#            file = path + "\\" + original_file_name
#            try:
#                os.remove(upgraded_inp_file.encode('unicode_escape'))
#            except:
#                pass
#            CallString = AbaqusBatLocation + ' -upgrade -job "' + file_to_analyze.replace('\\','\\\\') + '" -odb "' + file.replace('\\','\\\\') + '"'
#            print(CallString)
#            subprocess.call(CallString)
#
#
#    #        subprocess.call(CallString)
#            time.sleep(3)
#            
#            
#            CallString = AbaqusBatLocation + '  CAE noGUI=".\lib\Get_Data_From_ODB"  -- -odbfilename "' + upgraded_inp_file.replace('\\','\\\\') + '" -partname ' + MaterialName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile ' + DataFileName + ' -headerflag ' + Headerflag + " -newfileflag " + NewFileFlag + " -frames " + Frames
#            print('_________________________________________')
#            print(CallString)
#            print('_________________________________________')
#            try:
#                os.remove(DataFileName+'.csv')
#            except:
#                pass
#    
#            time.sleep(3)
#    
#            try:
#                os.environ.pop('PYTHONIOENCODING')
#            except KeyError:
#                pass
#            
#            JobPath = "I:\\Testing New Code"
#            
#            cmd = subprocess.Popen(CallString, cwd = JobPath, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
#                                               stderr=subprocess.PIPE, shell=True).communicate()[0]
#            print(cmd)
        
        time.sleep(3)
#        print(DataFileName)
        Coordinates = genfromtxt(DataFileName + '.csv', delimiter=',')
        x[i] = Coordinates[0]
        y[i] = Coordinates[1]
        z[i] = Coordinates[2]

#        Also find the coordinates of the points before displacement for comparison
#        print(tissue_list[i],plane_nodes[i])
#        print(tissue_list[i])
#        print("OPAL325_PM_mid")
#            print(INP_File)
        NodeCoordinate = io.extractNodesFromINP(INP_File + '.inp', tissue_list[i], [plane_nodes[i]])
#        print(NodeCoordinate[0][1])
        xBefore[i] = NodeCoordinate[0][0]
        yBefore[i] = NodeCoordinate[0][1]
        zBefore[i] = NodeCoordinate[0][2]
    
    (a,b,c,d) = equation_plane(x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2])
#        print('Deformed Plane', a,b,c,d)
    (aBefore,bBefore,cBefore,dBefore) = equation_plane(xBefore[0], yBefore[0], zBefore[0], xBefore[1], yBefore[1], zBefore[1], xBefore[2], yBefore[2], zBefore[2])
#        print('Undeformed Plane: ', aBefore,bBefore,cBefore,dBefore)
    ###################################################################
    # Next get the points for the vaginal wall
    PrintMaterialHeader = 'N'
    NewOutputFile = 'Y'
    ML = ["OPAL325_AVW_v6"]
    MaterialList = ['OPAL325_AVW_V6-1']
    MaterialSizeList = [numOfNodes(GenericINPFile, ML[0])]
    nodes = list(range(1,MaterialSizeList[0]+1))
    PassingNodes = ','.join(str(i) for i in nodes)
    Variable1 = "COORD"
    Headerflag = 'N'
    NewFileFlag = 'Y'
    Frames = frame
    MaterialName = MaterialList[0]
    DataFileName = 'AVW_nodal_coordinates'
    CallString = AbaqusBatLocation + '  CAE noGUI=".\lib\Get_Data_From_ODB"  -- -odbfilename "' + upgraded_inp_file.replace('\\','\\\\') + '" -partname ' + MaterialName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile ' + DataFileName + ' -headerflag ' + Headerflag + " -newfileflag " + NewFileFlag + " -frames " + Frames
#        print(CallString)
    subprocess.call(CallString)
    
    #May be needed for when I run it on a virtual machine because the files have to be uploaded
    time.sleep(3)
    
#        print(DataFileName)
    (xs, ys, zs) = getFEADataCoordinates(DataFileName+'.csv')
#    print(xs)
#########################################


#########################################
    # Check the distance for each point
    max_prolapse = -9999999999
    max_prolapse_absolute = -9999999999
#        print("Number of AVW points = ", len(xs))
    for j in range (len(xs)):
            distance_relative = shortest_distance(xs[j],ys[j],zs[j],a,b,c,d)
            distance_absolute = shortest_distance(xs[j],ys[j],zs[j],aBefore,bBefore,cBefore,dBefore)
            #            print(distance)
            if distance_relative > max_prolapse:
                max_prolapse = distance_relative
                max_prolapse_node = j+1
                
            if distance_absolute > max_prolapse_absolute:
                max_prolapse_absolute = distance_absolute
                max_prolapse_absolute_node = j+1

    print("Prolapse first time through: ", max_prolapse)
######################################################
##                    Negative prolapse code
##   If the prolapse is negative, then a different clinical measurement is used
#    if max_prolapse < 0:
#
#
## Create the interpolation
#        interpolator = interpolate.Rbf(xs, zs, ys, function='linear', smooth=1)
#
## Grab the highest Z value for midsaggital node
##            May need to go back and change this if there are small bulges with
##           negative prolapse values or the midsaggital isn't the highest Z
#        endZ = max(zs)
#
##            find distance between that point (X=0, Z=max, y = point/interpolation) and the next (X = 0, Z = max - 1, y = interpolated)
#        distance = 0.0
#        StepSize = 0.02
##            Distance along the vaginal wall that clinicians use to determine
##            prolapse size
##            desired_distance = 25
#        oldZ = endZ
#        oldY = interpolator(0, endZ)
#        while distance < desired_distance:
#            newZ = oldZ - StepSize
#            newY= interpolator(0, newZ)
#
## Add it to the total distance
#        
#            distance += pythag(newZ - oldZ, oldY - newY)
#            oldY = newY
#            oldZ = newZ
#
## Check to see if it is greater than the target
## If not loop
## If so, interpolate the new point
## Find the distance from that point to the planes that were found above            
##                zRange = np.arange(fromValue, toValue, INCREMENT * (1 if fromValue < toValue else -1))
#
#        distance_relative = shortest_distance(0,newY,newZ,a,b,c,d)
#        distance_absolute = shortest_distance(0,newY,newZ,aBefore,bBefore,cBefore,dBefore)
#        max_prolapse = distance_relative
#        max_prolapse_absolute = distance_absolute
##            print('Coordinates are: 0, ', newY, newZ)
#        
#        
#        
#######################################################
#        Continuing after adding negative prolapse code
    print("The prolapse size is: ", max_prolapse)
    print("It occurred at node ", max_prolapse_node, ": ", xs[max_prolapse_node-1], ", ", ys[max_prolapse_node-1], ", ", zs[max_prolapse_node-1])
    
    print("The absolute prolapse size is: ", max_prolapse_absolute)
    print("It occurred at node ", max_prolapse_node, ": ", xs[max_prolapse_absolute_node-1], ", ", ys[max_prolapse_absolute_node-1], ", ", zs[max_prolapse_absolute_node-1])
 
    
    #################### Adding code to find prolapse size based on Aa point
    Aa_point = Point(1.26804042,  -5.12414551,   -7.5848732)
    Aa_node = getXClosestNodes(Aa_point, 1, tissue_list[2], 'OPAL325_AVW_v6', INP_File)
    print("Aa NODE IS::::::", Aa_node)
    
    distance_relative = shortest_distance(xs[Aa_node-1],ys[Aa_node-1],zs[Aa_node-1],a,b,c,d)
    distance_absolute = shortest_distance(xs[Aa_node-1],ys[Aa_node-1],zs[Aa_node-1],aBefore,bBefore,cBefore,dBefore)
    
    print('New Distances:', distance_relative, distance_absolute)
    
    return(max_prolapse, max_prolapse_absolute, max_prolapse_node, plane_nodes)

'''
Function: calc_exposed_vaginal_length
'''
def calc_exposed_vaginal_length(ini_file, AVW_csv_file, inp_file, odb_file, frame):
    
    config = configparser.ConfigParser()
    config.sections()
    config.read(ini_file)

    AbaqusBatLocation= config["SYSTEM_INFO"]["AbaqusBatLocation"]
    
    THRESHOLD = 3 # Max Distance point can be from X axis

    (xs, ys, zs) = getFEADataCoordinates(AVW_csv_file)
    
#    Getting the middle nodes and finding the one with the largest z coordiante
    middle_nodes = []
    max_z = -999999
    for index, xval in enumerate(xs):
        if (abs(0 - xval) < THRESHOLD):
            middle_nodes.append((ys[index], zs[index]))
            if zs[index] > max_z:
                max_z = zs[index]
                start = (ys[index], zs[index])
   
    
    ## Setting up for Sorting the spline 
#    print('Unordered')
#    print(L)
    
    spline_ordered = [start]
    count = len(middle_nodes)
    
    distance_array=[0]
    used = []
    ## Sorting the spline 
    for i in range(0,count):
        least_dist = float("inf")
        least_indx = 0
    # loop through the points and see which one is the closest to the current point...
        # first point is the one with the "start" point which has the max z
        for j , point in enumerate(middle_nodes):
            if j not in used:
#                Calculate the distance between the last point added and the point that is being investigated
                dist = pythag(spline_ordered[-1][0] - point[0], spline_ordered[-1][1] - point[1])
                if dist < least_dist and j not in used:
                    least_dist = dist
                    least_indx = j
#       Keeping a log of the running total distance
        distance_array.append(least_dist+distance_array[-1])
        spline_ordered.append(middle_nodes[least_indx])
        used.append(least_indx)
    
#    print('First Attempt')
#    print(spline_ordered)
        # remove the first coordinate pair because it was just the starting one
    spline_ordered.pop(0)
    distance_array.pop(0)

# Higher y coordinate means it is the distal end of the AVW
#   If the last one has a higher y coordinate than the first one, the last one
#    is the distal end of the AVW and therefore the array will be flipped to put it first    
    if spline_ordered[0][0] < spline_ordered[-1][0]:
        spline_ordered = list(reversed(spline_ordered))
        
#    print('Correct Order')
#    print(spline_ordered)

#    print('Starting Coordiantes: ', spline_ordered[0])
    ## END Sorting the spline 


#   We need to find the end point for the exposed vaginal length (where the AVW leaves the GI Filler)
#    To do this we find the top of the middle PM Mid and then go up and out a
#        bit from it and find the closest GI Filler point to that
#           This is utilizing the intial coordinated from the inp file
    
#    print('first few points', spline_ordered[0:20])
    
    PM_mid_nodes = extractPointsForPartFrom(inp_file, 'OPAL325_PBody')
    PM_y_max = -9999999
    for i in range(0,len(PM_mid_nodes)):
        if abs(PM_mid_nodes[i][0] - 0) < THRESHOLD:
            if PM_mid_nodes[i][1] > PM_y_max:
               PM_y_max =  PM_mid_nodes[i][1]
               PM_y_max_z_coord = PM_mid_nodes[i][2]
    
    Center = Point(0, PM_y_max + 15, PM_y_max_z_coord - 5)
    GI_filler_end_node, GI_filler_end_node_coords = getXClosestNodes(Center, 1, 'OPAL325_GIfiller', inp_file)
#    print(Center)
    print('***********************', GI_filler_end_node)
#    

# Get the coordinates for that node from the end of the run
    material_name = 'OPAL325_GIfiller'
    nodes = GI_filler_end_node[0] + 1 # increase the node by 1 because the node numbers start at 1 instead of 0
    PassingNodes = str(nodes)
    Variable1 = "COORD"
    Headerflag = 'N'
    NewFileFlag = 'Y'
    Frames = frame
    MaterialName = material_name.upper() + '-1'
    DataFileName = "Temp_Coords"
    CallString = AbaqusBatLocation + '  CAE noGUI=".\lib\Get_Data_From_ODB"  -- -odbfilename "' + odb_file[:-4] + '" -partname ' + MaterialName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile "' + DataFileName + '" -headerflag ' + Headerflag + " -newfileflag " + NewFileFlag + " -frames " + Frames
#    print(CallString)
    subprocess.call(CallString)
    time.sleep(3)
        
    Coordinates = genfromtxt(DataFileName + '.csv', delimiter=',')

#   Set the end for the GI Filler    
    GI_Filler_End = (Coordinates[1],Coordinates[2])
#    print('GI Filler End coordiantes from the odb file: ', GI_Filler_End)
    
    # putting the coordinate pairs into an array
    X = np.array(spline_ordered).T # -> transpose
    
#    print(X)
#    print(np.array(X[0]))
#    print(np.array(X[1]))
#    print(distance_array)
    
#    t = list(range(0, count))   
    # https://stackoverflow.com/questions/12798695/spline-of-curve-python
#   create a function where you put in a distance along the curve and it gives you the x and y
    curve = interpolate.interp1d(distance_array, X)
    curve_y = interpolate.UnivariateSpline(distance_array, list(X[0]), k = 5)
    curve_z = interpolate.UnivariateSpline(distance_array, list(X[1]), k = 5)
    
    ## Set up for getting the length
    new_vals = [[],[]]
#   Create an array that is small distance steps from one end of the AVW to the other
    steps = np.arange(0,distance_array[-1], 0.1)
    dist = 0
    last_val = None



#    print(beststep)
    dist_array = [0]
    dist2_array = [0]

    ## summing the length    
    last_val = curve(steps[0])
    last_val2 = [curve_y(steps[0]), curve_z(steps[0])]
    ys = []
    zs = []
#    for i, step in enumerate(steps[1:bestindex]):

### Find the step that is the closest to the GI Point
    min_dist = 999999999    
    this_dist = []
    # loop through the steps (distances) to see which one is the closest
    for i, step in enumerate(steps):
#        val = curve(step)
#        dist = pythag(GI_Filler_End[0] - val[0], GI_Filler_End[1] - val[1])
        # distance between the end filler point and the current point along the AVW curve
        dist = pythag(GI_Filler_End[0] - curve_y(step), GI_Filler_End[1] - curve_z(step))
#        print(val)
#        print(dist)
        # check to see if it is the closest yet
        if dist < min_dist:
            min_dist = dist
            beststep = step
            bestindex = i
            
        val = curve(step)
        val2 = [curve_y(step), curve_z(step)]
#        print(step)
        new_vals[0].append(val[0])
        new_vals[1].append(val[1])
        
        ys.append(curve_y(step))
        zs.append(curve_z(step))
                
# distance between the current point and the last point
        this_dist.append(pythag(last_val2[0] - val2[0], last_val2[1] - val2[1]))

        # keep track of the distances in an array
        dist_array.append(dist_array[i] + pythag(last_val[0] - val[0], last_val[1] - val[1]))
        dist2_array.append(dist2_array[i] + pythag(last_val2[0] - val2[0], last_val2[1] - val2[1]))
        last_val = val
        last_val2 = val2
    
#    print('dist array', dist2_array)
    print('best end', curve(beststep))
#    add in the distance from the last point to the GI Filler for the times where it doesn't reach
    dist = dist_array[bestindex] + pythag(curve(beststep)[0] - GI_Filler_End[0], curve(beststep)[1] - GI_Filler_End[1])
    print('distance before final addition : ', dist2_array[bestindex])
#    print(curve_x(beststep), GI_Filler_End[0], curve_y(beststep), GI_Filler_End[1])
    dist2 = dist2_array[bestindex] + pythag(curve_y(beststep) - GI_Filler_End[0], curve_z(beststep) - GI_Filler_End[1])
        
##    print(dist2_array)   
##    print(this_dist)
#    
#    print(dist2)
##    print('xs : ', ys)    
#    print("What is the closest distance? ", min_dist, bestindex)
#    
##    print(beststep)
#    dist = 0
#    dist2 = 0
#    ## summing the length
#    
#    last_val = curve(steps[0])
#    last_val2 = [curve_x(steps[0]), curve_y(steps[0])]
#    xs = []
#    ys = []
#    delete_later = []
#    that_dist = []
#    for i, step in enumerate(steps[0:bestindex]):
#        val = curve(step)
#        val2 = [curve_x(step), curve_y(step)]
##        print(step)
#        new_vals[0].append(val[0])
#        new_vals[1].append(val[1])
#        
#        xs.append(curve_x(step))
#        ys.append(curve_y(step))
#        
#        
#        
#        that_dist.append(pythag(last_val2[0] - val2[0], last_val2[1] - val2[1]))
#        
#        dist += pythag(last_val[0] - val[0], last_val[1] - val[1])
#        dist2 += pythag(last_val2[0] - val2[0], last_val2[1] - val2[1])
#        last_val = val
#        last_val2 = val2
#        delete_later.append(dist2)
#
##    add in the distance from the last point to the GI Filler for the times where it doesn't reach
#    dist += pythag(last_val[0] - GI_Filler_End[0], last_val[1] - GI_Filler_End[1])
#    print('distance before final addition : ', dist2)
#    print(last_val2[0], GI_Filler_End[0], last_val2[1], GI_Filler_End[1])
#    dist2 += pythag(last_val2[0] - GI_Filler_End[0], last_val2[1] - GI_Filler_End[1])
    
#    print('xs : ', ys)    
    
#    print(delete_later)
#    print(that_dist)

##    print(new_vals[0], new_vals[1])
#    plt.plot(xs,ys)
#    plt.plot(new_vals[0], new_vals[1])

    
#    print('last value is: ', last_val)
    
    print("Exposed Vaginal Length:", dist, dist2)

##    Make a plot showing the surface and the line of points
#    surface = DataSet3d(xs, ys, zs)
#    ss.plot_Dataset(surface, DataSet3d([0] * len(new_vals[0]), new_vals[0], new_vals[1]))

    return dist2

'''
Function: calc_exposed_vaginal_length2
'''
def calc_exposed_vaginal_length2(GenericINPFile, ODBFile, INP_File, INI_file, frame, AVW_csv_file):
#"Redo soft tissue prolapse measurement to use a point at the top of the PM-Mid and top of PM-Body with same x coordinate:
    
    THRESHOLD = 3
    
    config = configparser.ConfigParser()
    config.sections()    
    config.read(INI_file)

    ##### Choose 2 points from PM-Mid and one from PBody
    ##### for MP-Mid I should choose the point with negative X that is about
    ##### 4mm up (Y coord) and the point with a positive X that is about 4mm up (Y coord)
    ##### for PBody, choose the point that is the closest to 0,0,0
#1) get the PM Body inner arc nodes
    PM_MID      = "OPAL325_PM_mid"
    connections = io.get_interconnections(INP_File, PM_MID)
    PM_Mid = io.get_dataset_from_file(INP_File, PM_MID)
    starting_index, ending_index = find_starting_ending_points_for_inside(PM_Mid)
    innerNodes = findInnerNodes(PM_Mid, connections, starting_index, ending_index)    

#2) find the point with the highest Y value = PM Mid top
    MaxY = -999999
    for i in innerNodes:
        if PM_Mid.node(i).y > MaxY:
            MaxY = PM_Mid.node(i).y
            midsaggital_x = PM_Mid.node(i).x
            PM_Mid_top_node = i
    PM_Mid_top_original = Point3D(PM_Mid.node(PM_Mid_top_node).x,PM_Mid.node(PM_Mid_top_node).y,PM_Mid.node(PM_Mid_top_node).z)
    
    
    print('PM_Mid_top_node', PM_Mid_top_node)
    
#3) collect all PM Body points who's x coords are PM Mid top.x +/- 3            
#4) Choose the top point = PM Body top
    PBODY       = "OPAL325_PBody"
    pbody_points = np.array(io.extractPointsForPartFrom(INP_File, PBODY))
#    print(pbody_points)
#    pbody_surface = DataSet3d(list(pbody_points[:, 0]), list(pbody_points[:, 1]), list(pbody_points[:, 2]))
    MaxY = -999999
#    print(pbody_surface)
    for i in range(0,len(pbody_points)):
        if abs(pbody_points[i][0] - midsaggital_x) < 5:
            if pbody_points[i][1] > MaxY:
                MaxY = pbody_points[i][1]
                pbody_top_middle_node = i

#5) set PM Body top.x = PM MId top.x
#    set the x coordinate to that of PM Top s (midsaggital_x) o that we get the midsaggital line
    pbody_top_original = Point3D(midsaggital_x, pbody_points[pbody_top_middle_node][1], pbody_points[pbody_top_middle_node][2])


#6) Create a line with PBody top and PM Mid top
#    prolapse_measurement_line_absolute = Line3D(PM_Mid_top_original, pbody_top_original)
    
    AbaqusBatLocation= config["SYSTEM_INFO"]["AbaqusBatLocation"]        
   
    #   COORD is the tag for the coordinate for the ODB file
    node_property = "COORD"

#    Use the last time step to get final position
    frames = frame
#    MaterialName = PM_MID.upper() + '-1' + ';' + PBODY.upper() + '-1'
    nodes = str(PM_Mid_top_node) + ';' + str(pbody_top_middle_node)
    
    odb_filename = ODBFile[:-4]

    material_names = PM_MID.upper() + '-1' + ';' + PBODY.upper() + '-1'
    
    points = get_odb_data(material_names, nodes, node_property, frames, AbaqusBatLocation, odb_filename)

    PM_Mid_top_deformed = Point3D(midsaggital_x, points[0][1], points[0][2])
    pbody_top_deformed = Point3D(midsaggital_x, points[1][1], points[1][2])
       
#    plane_nodes = [PM_Mid_top_node + 1, pbody_top_middle_node + 1]
    
#    prolapse_measurement_line_deformed = Line3D(PM_Mid_top_deformed, pbody_top_deformed)

    PM_mid_nodes = extractPointsForPartFrom(INP_File, 'OPAL325_PBody')
    PM_y_max = -9999999
    for i in range(0,len(PM_mid_nodes)):
        if abs(PM_mid_nodes[i][0] - 0) < THRESHOLD:
            if PM_mid_nodes[i][1] > PM_y_max:
               PM_y_max =  PM_mid_nodes[i][1]
               PM_y_max_z_coord = PM_mid_nodes[i][2]
    
    Center = Point(0, PM_y_max + 15, PM_y_max_z_coord - 5)
    GI_filler_end_node, GI_filler_end_node_coords = getXClosestNodes(Center, 1, 'OPAL325_GIfiller', INP_File)
#    print(Center)
#    print('***********************', GI_filler_end_node)
#    

# Get the coordinates for that node from the end of the run
    material_name = 'OPAL325_GIfiller'
    nodes = GI_filler_end_node[0] + 1 # increase the node by 1 because the node numbers start at 1 instead of 0
    PassingNodes = str(nodes)
    Variable1 = "COORD"
    Headerflag = 'N'
    NewFileFlag = 'Y'
    Frames = frame
    MaterialName = material_name.upper() + '-1'
    DataFileName = "Temp_Coords"
    CallString = AbaqusBatLocation + '  CAE noGUI=".\lib\Get_Data_From_ODB"  -- -odbfilename "' + ODBFile[:-4] + '" -partname ' + MaterialName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile "' + DataFileName + '" -headerflag ' + Headerflag + " -newfileflag " + NewFileFlag + " -frames " + Frames
#    print(CallString)
    subprocess.call(CallString)
    time.sleep(3)
        
    Coordinates = genfromtxt(DataFileName + '.csv', delimiter=',')

#   Set the end for the GI Filler    
    GI_Filler_End = (Coordinates[1],Coordinates[2])

############### Above is getting the lines that will define the exposure
############### Now get the AVW points in the middle that are on the correct 
############### side of those lines

    
    (xs, ys, zs) = getFEADataCoordinates(AVW_csv_file)

    THRESHOLD = 3 # Max Distance point can be from X axis
    
#    Getting the middle nodes and finding the one with the largest z coordiante
    middle_nodes = []
    max_z = -999999
    for index, xval in enumerate(xs):
        if (abs(0 - xval) < THRESHOLD):
#            whole_matrix = np.array([[midsaggital_x, PM_Mid_top_original.y, PM_Mid_top_original.z], [midsaggital_x, pbody_top_original.y, pbody_top_original.z], [midsaggital_x, ys[index], zs[index]]], dtype='float')

#            whole_matrix = np.array([[midsaggital_x, PM_Mid_top_deformed.y, PM_Mid_top_deformed.z], [midsaggital_x, pbody_top_deformed.y, pbody_top_deformed.z], [midsaggital_x, ys[index], zs[index]]], dtype='float')
            whole_matrix = np.array([[midsaggital_x, PM_Mid_top_deformed.y, PM_Mid_top_deformed.z], [midsaggital_x, pbody_top_deformed.y, pbody_top_deformed.z], [midsaggital_x, ys[index], zs[index]]], dtype='float')

# The line below was used for the abstracts
#            whole_matrix = np.array([[midsaggital_x, pbody_top_original.y, pbody_top_original.z], [midsaggital_x, pbody_top_deformed.y, pbody_top_deformed.z], [midsaggital_x, ys[index], zs[index]]], dtype='float')
            side_of_line = np.sign(np.linalg.det(whole_matrix.astype(float)))
#            print(side_of_line)

#            if side_of_line < 0:
            if -1 < 0:
                middle_nodes.append((xs[index], ys[index], zs[index]))

                if zs[index] > max_z:
                    max_z = zs[index]
                    start = (ys[index], zs[index])

    
    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################
    spline_ordered = [start]
    count = len(middle_nodes)
    
    distance_array=[0]
    used = []
    ## Sorting the spline 
    for i in range(0,count):
        least_dist = float("inf")
        least_indx = 0
        for j , point in enumerate(middle_nodes):
            if j not in used:
#                Calculate the distance between the last point added and the point that is being investigated
                dist = pythag(spline_ordered[-1][0] - point[1], spline_ordered[-1][1] - point[2])
                if dist < least_dist and j not in used:
                    least_dist = dist
                    least_indx = j
#       Keeping a log of the running total distance
        distance_array.append(least_dist+distance_array[-1])
        spline_ordered.append((middle_nodes[least_indx][1], middle_nodes[least_indx][2]))
        used.append(least_indx)
    
#    print('First Attempt')
#    print(spline_ordered)
    

    spline_ordered.pop(0)
    distance_array.pop(0)
    
    # Higher y coordinate means it is the distal end of the AVW
#   If the last one has a higher y coordinate than the first one, the last one
#    is the distal end of the AVW and therefore the array will be flipped to put it first    
    if spline_ordered[0][0] < spline_ordered[-1][0]:
        spline_ordered = list(reversed(spline_ordered))


    plt.scatter([item[1] for item in spline_ordered],[item[0] for item in spline_ordered], color = 'r', marker = '.')

#    print('spline ordered:', spline_ordered)
#    print('distance_array:', distance_array)    
    ###########&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&    
    X = np.array(spline_ordered).T # -> transpose
    
    # https://stackoverflow.com/questions/12798695/spline-of-curve-python
#   create a function where you put in a distance along the curve and it gives you the x and y
    curve = interpolate.interp1d(distance_array, X)
    curve_y = interpolate.UnivariateSpline(distance_array, list(X[0]), k = 5)
    curve_z = interpolate.UnivariateSpline(distance_array, list(X[1]), k = 5)
    
    ## Set up for getting the length
    new_vals = [[],[]]
#   Create an array that is small distance steps from one end of the AVW to the other
    steps = np.arange(0,distance_array[-1], 0.1)
    dist = 0
    last_val = None



#    print(beststep)
    dist_array = [0]
    dist2_array = [0]
    ## summing the length
    
    last_val = curve(steps[0])
    last_val2 = [curve_y(steps[0]), curve_z(steps[0])]
    ys = []
    zs = []
#    for i, step in enumerate(steps[1:bestindex]):

### Find the step that is the closest to the GI Point
    # I think: Start at the distal end and start adding up the distances...
#    also keep track of which point is the closest to the GI Filler end
#    Then find the distance along the AVW from that point (calcuated below)
#    and add in the distance from that point to the GI-Filler end
   
    this_dist = []
    correct_side_of_line = 0
    line_side_array = []
    print('pm deformed and gi filler:', PM_Mid_top_deformed, GI_Filler_End)
    for i, step in enumerate(steps):
            
        val = curve(step)
        val2 = [curve_y(step), curve_z(step)]

#   Get the coordinates of the step
        new_vals[0].append(val[0])
        new_vals[1].append(val[1])
        
        ys.append(curve_y(step))
        zs.append(curve_z(step))

        
#        whole_matrix = np.array([[midsaggital_x, spline_ordered[0][0], spline_ordered[0][1]], [midsaggital_x, pbody_top_deformed.y, pbody_top_deformed.z], [midsaggital_x, ys[-1], zs[-1]]], dtype='float')
#        whole_matrix = np.array([[midsaggital_x, spline_ordered[0][0], spline_ordered[0][1]], [midsaggital_x, GI_Filler_End[0], GI_Filler_End[1]], [midsaggital_x, ys[-1], zs[-1]]], dtype='float')
        whole_matrix = np.array([[midsaggital_x, PM_Mid_top_deformed.y, PM_Mid_top_deformed.z], [midsaggital_x, GI_Filler_End[0], GI_Filler_End[1]], [midsaggital_x, ys[-1], zs[-1]]], dtype='float')
        #        whole_matrix = np.array([[midsaggital_x, PM_Mid_top_deformed.y, PM_Mid_top_deformed.z], [midsaggital_x, pbody_top_deformed.y, pbody_top_deformed.z], [midsaggital_x, ys[-1], zs[-1]]], dtype='float')
        side_of_line = np.sign(np.linalg.det(whole_matrix.astype(float)))
#        print(i)
#        print(side_of_line)
        line_side_array.append(side_of_line)
        if side_of_line < 0:
            correct_side_of_line = 1
        
#        if you have been on the correct side before, but now cross over, stop
#            else, add distnce if you have already been on the correct side
        if correct_side_of_line == 1 and int(side_of_line) == 1:
            break
        elif correct_side_of_line == 1:
            
#        Maybe the distance from the distal end?
            this_dist.append(pythag(last_val2[0] - val2[0], last_val2[1] - val2[1]))
            
            # add the last distance to the distance between the previous point
#            and the current point
            dist_array.append(dist_array[-1] + pythag(last_val[0] - val[0], last_val[1] - val[1]))
#            dist2_array.append(dist2_array[i] + pythag(last_val2[0] - val2[0], last_val2[1] - val2[1]))
        last_val = val
        last_val2 = val2

#    print(line_side_array)
#    add in the distance from the last point to the GI Filler for the times where it doesn't reach
#    if dist_array[-1] > 1:
#    dist = dist_array[-1] + pythag(ys[-1] - GI_Filler_End[0], zs[-1] - GI_Filler_End[1])

    plt.plot(zs, ys, color = 'r')
    plt.plot(np.array([PM_Mid_top_deformed.z,GI_Filler_End[1]]),np.array([PM_Mid_top_deformed.y,GI_Filler_End[0]]), color = 'r')

#    print('plot old spline')
#    plt.scatter(zs,ys)
#    plt.show()

    dist = dist_array[-1]
    #    else:
#        dist = dist_array[-1]
    if side_of_line < 0:
        dist = dist + pythag(ys[-1] - GI_Filler_End[0], zs[-1] - GI_Filler_End[1])
    elif correct_side_of_line == 0:
        dist = 0
    print('distance before going to GI_Filler: ', dist_array[-1])
#    print('distance before final addition : ', dist2_array[bestindex])
#    print(curve_x(beststep), GI_Filler_End[0], curve_y(beststep), GI_Filler_End[1])
#    dist2 = dist2_array[bestindex] + pythag(curve_y(beststep) - GI_Filler_End[0], curve_z(beststep) - GI_Filler_End[1])        
    print('last point:', last_val2, last_val)
    print("Exposed Vaginal Length:", dist)
#    print('dist array:', dist_array)
#    print('line_side_array:', line_side_array)    
    return dist
    ###########&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

##   If the last one has a higher y coordinate than the first one, the last one
##    is the distal end of the AVW and therefore the array will be flipped to put it first    
#    if spline_ordered[0][0] < spline_ordered[-1][0]:
#        spline_ordered = list(reversed(spline_ordered))
        
        
#        
#    X = np.array(spline_ordered).T # -> transpose
#    
#
#    curve_y = interpolate.UnivariateSpline(distance_array, list(X[0]), k = 5)
#    curve_z = interpolate.UnivariateSpline(distance_array, list(X[1]), k = 5)
#    
##    print(X)
#    ## Set up for getting the length
##    new_vals = [[],[]]
##   Create an array that is small distance steps from one end of the AVW to the other
#    steps = np.arange(0,distance_array[-1], 1)
#    dist = 0
##    last_val = None
#
#
#    ## summing the length
#    
##    last_val2 = [curve_x(steps[0]), curve_y(steps[0])]
#    new_ys = []
#    new_zs = []
##    for i, step in enumerate(steps[1:bestindex]):
#
#### Find the step that is the closest to the GI Point
##    min_dist = 999999999    
##    this_dist = []
#    for i, step in enumerate(steps):
#
#        new_ys.append(curve_y(step)[()])
#        new_zs.append(curve_z(step)[()])

##########################################
#    # Check the distance for each point
#    max_prolapse = -9999999999
#    max_prolapse_absolute = -9999999999
##        print("Number of AVW points = ", len(xs))
#    for j in range (len(middle_nodes)):
##                whole_matrix = np.array([[midsaggital_x, PM_Mid_top_original.y, PM_Mid_top_original.z], [midsaggital_x, pbody_top_original.y, pbody_top_original.z], [midsaggital_x, new_xs[j], new_ys[j]]], dtype='float')
#        whole_matrix = np.array([[midsaggital_x, PM_Mid_top_original.y, PM_Mid_top_original.z], [midsaggital_x, pbody_top_original.y, pbody_top_original.z], [midsaggital_x, middle_nodes[j][1], middle_nodes[j][2]]], dtype='float')
#
#        side_of_line = np.sign(np.linalg.det(whole_matrix.astype(float)))
#        side_of_line2 = np.sign((pbody_top_original.y -PM_Mid_top_original.y)*(middle_nodes[j][2] - pbody_top_original.z) - (middle_nodes[j][1] - pbody_top_original.y)*(pbody_top_original.z - PM_Mid_top_original.z))
#
#        distance_relative = prolapse_measurement_line_deformed.distance(Point3D(middle_nodes[j])) * side_of_line * -1
#        distance_absolute = prolapse_measurement_line_absolute.distance(Point3D(middle_nodes[j])) * side_of_line * -1
##        print(middle_nodes[j], float(distance_absolute), side_of_line, side_of_line2)
#        #            print(distance)
#        if distance_relative > max_prolapse:
#            max_prolapse = float(distance_relative)
#            max_prolapse_node = j
#            
#        if distance_absolute > max_prolapse_absolute:
#            max_prolapse_absolute = float(distance_absolute)
#            max_prolapse_absolute_node = j
#
#    print('original code :', max_prolapse, max_prolapse_absolute)
#
#    return(max_prolapse, max_prolapse_absolute, max_prolapse_node, plane_nodes)
#    return(float(max_prolapse), float(max_prolapse_absolute), new_max_prolapse_node, plane_nodes)
    
#def calc_exposed_vaginal_length2(ini_file, AVW_csv_file, inp_file, odb_file, frame):
#    
#    config = configparser.ConfigParser()
#    config.sections()
#    config.read(ini_file)
#
#    AbaqusBatLocation= config["SYSTEM_INFO"]["AbaqusBatLocation"]    
#    THRESHOLD = 3 # Max Distance point can be from X axis
#    (xs, ys, zs) = getFEADataCoordinates(AVW_csv_file)    
#    
##    Getting the middle nodes and finding the one with the largest z coordiante
#    middle_nodes = []
#    max_z = -999999
#    for index, xval in enumerate(xs):
#        if (abs(0 - xval) < THRESHOLD):
#            middle_nodes.append((ys[index], zs[index]))
#            if zs[index] > max_z:
#                max_z = zs[index]
#                start = (ys[index], zs[index])
#   
#    
#    ## Setting up for Sorting the spline 
##    print('Unordered')
##    print(L)
#    
#    spline_ordered = [start]
#    count = len(middle_nodes)
#    
#    distance_array=[0]
#    used = []
#    ## Sorting the spline 
#    for i in range(0,count):
#        least_dist = float("inf")
#        least_indx = 0
#        for j , point in enumerate(middle_nodes):
#            if j not in used:
##                Calculate the distance between the last point added and the point that is being investigated
#                dist = pythag(spline_ordered[-1][0] - point[0], spline_ordered[-1][1] - point[1])
#                if dist < least_dist and j not in used:
#                    least_dist = dist
#                    least_indx = j
##       Keeping a log of the running total distance
#        distance_array.append(least_dist+distance_array[-1])
#        spline_ordered.append(middle_nodes[least_indx])
#        used.append(least_indx)
#    
##    print('First Attempt')
##    print(spline_ordered)
#    spline_ordered.pop(0)
#    distance_array.pop(0)
#
## Higher y coordinate means it is the distal end of the AVW
##   If the last one has a higher y coordinate than the first one, the last one
##    is the distal end of the AVW and therefore the array will be flipped to put it first    
#    if spline_ordered[0][0] < spline_ordered[-1][0]:
#        spline_ordered = list(reversed(spline_ordered))
#        
##    print('Correct Order')
##    print(spline_ordered)
#
##    print('Starting Coordiantes: ', spline_ordered[0])
#    ## END Sorting the spline 
#
#
##   We need to find the end point for the exposed vaginal length (where the AVW leaves the GI Filler)
##    To do this we find the top of the middle PM Mid and then go up and out a
##        bit from it and find the closest GI Filler point to that
##           This is utilizing the intial coordinated from the inp file
#    PM_mid_nodes = extractPointsForPartFrom(inp_file, 'OPAL325_PBody')
#    PM_y_max = -9999999
#    for i in range(0,len(PM_mid_nodes)):
#        if abs(PM_mid_nodes[i][0] - 0) < THRESHOLD:
#            if PM_mid_nodes[i][1] > PM_y_max:
#               PM_y_max =  PM_mid_nodes[i][1]
#               PM_y_max_z_coord = PM_mid_nodes[i][2]
#    
#    Center = Point(0, PM_y_max + 15, PM_y_max_z_coord - 5)
#    GI_filler_end_node, GI_filler_end_node_coords = getXClosestNodes(Center, 1, 'OPAL325_GIfiller', inp_file)
##    print(Center)
#    print('***********************', GI_filler_end_node)
##    
#
## Get the coordinates for that node from the end of the run
#    material_name = 'OPAL325_GIfiller'
#    nodes = GI_filler_end_node[0] + 1 # increase the node by 1 because the node numbers start at 1 instead of 0
#    PassingNodes = str(nodes)
#    Variable1 = "COORD"
#    Headerflag = 'N'
#    NewFileFlag = 'Y'
#    Frames = frame
#    MaterialName = material_name.upper() + '-1'
#    DataFileName = "Temp_Coords"
#    CallString = AbaqusBatLocation + '  CAE noGUI=".\lib\Get_Data_From_ODB"  -- -odbfilename "' + odb_file[:-4] + '" -partname ' + MaterialName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile "' + DataFileName + '" -headerflag ' + Headerflag + " -newfileflag " + NewFileFlag + " -frames " + Frames
##    print(CallString)
#    subprocess.call(CallString)
#    time.sleep(3)
#        
#    Coordinates = genfromtxt(DataFileName + '.csv', delimiter=',')
#
##   Set the end for the GI Filler    
#    GI_Filler_End = (Coordinates[1],Coordinates[2])
##    print('GI Filler End coordiantes from the odb file: ', GI_Filler_End)
#    
#    X = np.array(spline_ordered).T # -> transpose
#    
#    # https://stackoverflow.com/questions/12798695/spline-of-curve-python
##   create a function where you put in a distance along the curve and it gives you the x and y
#    curve = interpolate.interp1d(distance_array, X)
#    curve_y = interpolate.UnivariateSpline(distance_array, list(X[0]), k = 5)
#    curve_z = interpolate.UnivariateSpline(distance_array, list(X[1]), k = 5)
#    
#    ## Set up for getting the length
#    new_vals = [[],[]]
##   Create an array that is small distance steps from one end of the AVW to the other
#    steps = np.arange(0,distance_array[-1], 0.1)
#    dist = 0
#    last_val = None
#
#
#
##    print(beststep)
#    dist_array = [0]
#    dist2_array = [0]
#    ## summing the length
#    
#    last_val = curve(steps[0])
#    last_val2 = [curve_y(steps[0]), curve_z(steps[0])]
#    ys = []
#    zs = []
##    for i, step in enumerate(steps[1:bestindex]):
#
#### Find the step that is the closest to the GI Point
#    min_dist = 999999999    
#    this_dist = []
#    for i, step in enumerate(steps):
##        val = curve(step)
##        dist = pythag(GI_Filler_End[0] - val[0], GI_Filler_End[1] - val[1])
#        dist = pythag(GI_Filler_End[0] - curve_y(step), GI_Filler_End[1] - curve_z(step))
##        print(val)
##        print(dist)
#        if dist < min_dist:
#            min_dist = dist
#            beststep = step
#            bestindex = i
#            
#        val = curve(step)
#        val2 = [curve_y(step), curve_z(step)]
##        print(step)
#        new_vals[0].append(val[0])
#        new_vals[1].append(val[1])
#        
#        ys.append(curve_y(step))
#        zs.append(curve_z(step))
#        
#        
#        this_dist.append(pythag(last_val2[0] - val2[0], last_val2[1] - val2[1]))
#        
#        dist_array.append(dist_array[i] + pythag(last_val[0] - val[0], last_val[1] - val[1]))
#        dist2_array.append(dist2_array[i] + pythag(last_val2[0] - val2[0], last_val2[1] - val2[1]))
#        last_val = val
#        last_val2 = val2
#    
##    print(dist2_array)
##    add in the distance from the last point to the GI Filler for the times where it doesn't reach
#    dist = dist_array[bestindex] + pythag(curve(beststep)[0] - GI_Filler_End[0], curve(beststep)[1] - GI_Filler_End[1])
##    print('distance before final addition : ', dist2_array[bestindex])
##    print(curve_x(beststep), GI_Filler_End[0], curve_y(beststep), GI_Filler_End[1])
#    dist2 = dist2_array[bestindex] + pythag(curve_y(beststep) - GI_Filler_End[0], curve_z(beststep) - GI_Filler_End[1])        
#    
#    print("Exposed Vaginal Length:", dist, dist2)
#
###    Make a plot showing the surface and the line of points
##    surface = DataSet3d(xs, ys, zs)
##    ss.plot_Dataset(surface, DataSet3d([0] * len(new_vals[0]), new_vals[0], new_vals[1]))
#
#    return dist2

'''
Function: Aa_point
'''
def Aa_point(zs, ys, distance_array):

    distance_to_Aa = 25
    
    curve_y = interpolate.UnivariateSpline(distance_array, ys, k = 5)
    curve_z = interpolate.UnivariateSpline(distance_array, zs, k = 5)
    
    y_coord = curve_y(distance_to_Aa)
    print('ycoord inside: ', type(y_coord))
    y_coord = curve_y(distance_to_Aa)
    print(y_coord)
    
    return(y_coord.astype(np.float), curve_z(distance_to_Aa).astype(np.float))
    
#     # curve = interpolate.interp1d(distance_array, X)
    
    
#     config = configparser.ConfigParser()
#     config.sections()
#     config.read(ini_file)

#     AbaqusBatLocation= config["SYSTEM_INFO"]["AbaqusBatLocation"]
    
#     THRESHOLD = 3 # Max Distance point can be from X axis

#     (xs, ys, zs) = getFEADataCoordinates(AVW_csv_file)
    
    
# #    Getting the middle nodes and finding the one with the largest z coordiante
#     middle_nodes = []
#     max_z = -999999
#     for index, xval in enumerate(xs):
#         if (abs(0 - xval) < THRESHOLD):
#             middle_nodes.append((ys[index], zs[index]))
#             if zs[index] > max_z:
#                 max_z = zs[index]
#                 start = (ys[index], zs[index])
   
    
#     ## Setting up for Sorting the spline 
# #    print('Unordered')
# #    print(L)
    
#     spline_ordered = [start]
#     count = len(middle_nodes)
    
#     distance_array=[0]
#     used = []
#     ## Sorting the spline 
#     for i in range(0,count):
#         least_dist = float("inf")
#         least_indx = 0
#     # loop through the points and see which one is the closest to the current point...
#         # first point is the one with the "start" point which has the max z
#         for j , point in enumerate(middle_nodes):
#             if j not in used:
# #                Calculate the distance between the last point added and the point that is being investigated
#                 dist = pythag(spline_ordered[-1][0] - point[0], spline_ordered[-1][1] - point[1])
#                 if dist < least_dist and j not in used:
#                     least_dist = dist
#                     least_indx = j
# #       Keeping a log of the running total distance
#         distance_array.append(least_dist+distance_array[-1])
#         spline_ordered.append(middle_nodes[least_indx])
#         used.append(least_indx)
    
# #    print('First Attempt')
# #    print(spline_ordered)
#         # remove the first coordinate pair because it was just the starting one
#     spline_ordered.pop(0)
#     distance_array.pop(0)

# # Higher y coordinate means it is the distal end of the AVW
# #   If the last one has a higher y coordinate than the first one, the last one
# #    is the distal end of the AVW and therefore the array will be flipped to put it first    
#     if spline_ordered[0][0] < spline_ordered[-1][0]:
#         spline_ordered = list(reversed(spline_ordered))
        
# #    print('Correct Order')
# #    print(spline_ordered)


# #    print('GI Filler End coordiantes from the odb file: ', GI_Filler_End)
    
#     # putting the coordinate pairs into an array
#     X = np.array(spline_ordered).T # -> transpose
    
# #    print(X)
# #    print(np.array(X[0]))
# #    print(np.array(X[1]))
# #    print(distance_array)
    
# #    t = list(range(0, count))   
#     # https://stackoverflow.com/questions/12798695/spline-of-curve-python
# #   create a function where you put in a distance along the curve and it gives you the x and y
#     curve = interpolate.interp1d(distance_array, X)
#     # curve_y = interpolate.UnivariateSpline(distance_array, list(X[0]), k = 5)
#     # curve_z = interpolate.UnivariateSpline(distance_array, list(X[1]), k = 5)
    
#     Aa_point = curve(distance_to_Aa)


#     return(Aa_point)

'''
Function: apical_point

get the farthest back AVW point for the apical point (y coordinate, z coordinate)
'''
def apical_point(midline_points):

    return(midline_points[-1][0],midline_points[-1][1])
    
    # midline_points, distance_array = get_AVW_midline_nodes(AVW_csv_file, slice_x_value)
    
#     THRESHOLD = 3 # Max Distance point can be from X axis

#     (xs, ys, zs) = getFEADataCoordinates(AVW_csv_file)
    
    
# #    Getting the middle nodes and finding the one with the largest z coordiante
#     middle_nodes = []
#     max_z = -999999
#     for index, xval in enumerate(xs):
#         if (abs(0 - xval) < THRESHOLD):
#             middle_nodes.append((ys[index], zs[index]))
#             if zs[index] > max_z:
#                 max_z = zs[index]
#                 start = (ys[index], zs[index])
   
    
#     ## Setting up for Sorting the spline 
# #    print('Unordered')
# #    print(L)
    
#     spline_ordered = [start]
#     count = len(middle_nodes)
    
#     distance_array=[0]
#     used = []
#     ## Sorting the spline 
#     for i in range(0,count):
#         least_dist = float("inf")
#         least_indx = 0
#     # loop through the points and see which one is the closest to the current point...
#         # first point is the one with the "start" point which has the max z
#         for j , point in enumerate(middle_nodes):
#             if j not in used:
# #                Calculate the distance between the last point added and the point that is being investigated
#                 dist = pythag(spline_ordered[-1][0] - point[0], spline_ordered[-1][1] - point[1])
#                 if dist < least_dist and j not in used:
#                     least_dist = dist
#                     least_indx = j
# #       Keeping a log of the running total distance
#         distance_array.append(least_dist+distance_array[-1])
#         spline_ordered.append(middle_nodes[least_indx])
#         used.append(least_indx)
    
# #    print('First Attempt')
# #    print(spline_ordered)
#         # remove the first coordinate pair because it was just the starting one
#     spline_ordered.pop(0)
#     distance_array.pop(0)

# # Higher y coordinate means it is the distal end of the AVW
# #   If the last one has a higher y coordinate than the first one, the last one
# #    is the distal end of the AVW and therefore the array will be flipped to put it first    
#     if spline_ordered[0][0] < spline_ordered[-1][0]:
#         spline_ordered = list(reversed(spline_ordered))
        

    # return(spline_ordered[-1][0],spline_ordered[-1][1])


'''
Function: get_AVW_midline
'''
def get_AVW_midline(AVW_csv_file):
    THRESHOLD = 3 # Max Distance point can be from X axis

    (xs, ys, zs) = getFEADataCoordinates(AVW_csv_file)
    
    
#    Getting the middle nodes and finding the one with the largest z coordiante
    middle_nodes = []
    max_z = -999999
    for index, xval in enumerate(xs):
        if (abs(0 - xval) < THRESHOLD):
            middle_nodes.append((ys[index], zs[index]))
            if zs[index] > max_z:
                max_z = zs[index]
                start = (ys[index], zs[index])
   
    
    spline_ordered = [start]
    count = len(middle_nodes)
    
    distance_array=[0]
    used = []
    ## Sorting the spline 
    for i in range(0,count):
        least_dist = float("inf")
        least_indx = 0
    # loop through the points and see which one is the closest to the current point...
        # first point is the one with the "start" point which has the max z
        for j , point in enumerate(middle_nodes):
            if j not in used:
#                Calculate the distance between the last point added and the point that is being investigated
                dist = pythag(spline_ordered[-1][0] - point[0], spline_ordered[-1][1] - point[1])
                if dist < least_dist and j not in used:
                    least_dist = dist
                    least_indx = j
#       Keeping a log of the running total distance
        distance_array.append(least_dist+distance_array[-1])
        spline_ordered.append(middle_nodes[least_indx])
        used.append(least_indx)
    
#    print('First Attempt')
#    print(spline_ordered)
        # remove the first coordinate pair because it was just the starting one
    spline_ordered.pop(0)
    distance_array.pop(0)
    
    if spline_ordered[0][0] < spline_ordered[-1][0]:
        spline_ordered = list(reversed(spline_ordered))

    return (spline_ordered)


'''
Function: hymenal_ring

#### Adding 5/4/21
'''
def hymenal_ring(PM_Mid_file, PBody_file, INP_File):
# This is used to determine the distance that points are from the hymenal ring
    
    #"Redo soft tissue prolapse measurement to use a point at the top of the PM-Mid and top of PM-Body with same x coordinate:
    
    # config = configparser.ConfigParser()
    # config.sections()    
    # config.read(INI_file)

    ##### Choose 2 points from PM-Mid and one from PBody
    ##### for MP-Mid I should choose the point with negative X that is about
    ##### 4mm up (Y coord) and the point with a positive X that is about 4mm up (Y coord)
    ##### for PBody, choose the point that is the closest to 0,0,0
#1) get the PM Body inner arc nodes
    PM_MID      = "OPAL325_PM_mid"
    connections = io.get_interconnections(INP_File, PM_MID)
    PM_Mid = io.get_dataset_from_file(INP_File, PM_MID)
    starting_index, ending_index = find_starting_ending_points_for_inside(PM_Mid)
    innerNodes = findInnerNodes(PM_Mid, connections, starting_index, ending_index)    



#2) find the point with the highest Y value = PM Mid top
    MaxY = -999999
    for i in innerNodes:
        if PM_Mid.node(i).y > MaxY:
            MaxY = PM_Mid.node(i).y
            midsaggital_x = PM_Mid.node(i).x
            PM_Mid_top_node = i
    PM_Mid_top_original = Point3D(PM_Mid.node(PM_Mid_top_node).x,PM_Mid.node(PM_Mid_top_node).y,PM_Mid.node(PM_Mid_top_node).z)
    
#3) collect all PM Body points who's x coords are PM Mid top.x +/- 3            
#4) Choose the top point = PM Body top
    PBODY       = "OPAL325_PBody"
    pbody_points = np.array(io.extractPointsForPartFrom(INP_File, PBODY))
#    print(pbody_points)
#    pbody_surface = DataSet3d(list(pbody_points[:, 0]), list(pbody_points[:, 1]), list(pbody_points[:, 2]))
    MaxY = -999999
#    print(pbody_surface)
    for i in range(0,len(pbody_points)):
        if abs(pbody_points[i][0] - midsaggital_x) < 5:
            if pbody_points[i][1] > MaxY:
                MaxY = pbody_points[i][1]
                pbody_top_middle_node = i

#5) set PM Body top.x = PM MId top.x
#    set the x coordinate to that of PM Top s (midsaggital_x) o that we get the midsaggital line
    pbody_top_original = Point3D(midsaggital_x, pbody_points[pbody_top_middle_node][1], pbody_points[pbody_top_middle_node][2])


    (PM_xs, PM_ys, PM_zs) = getFEADataCoordinates(PM_Mid_file)
    (PBody_xs, PBody_ys, PBody_zs) = getFEADataCoordinates(PBody_file)

#    plt.scatter(PM_xs, PM_ys)

    MaxY = -999999
    for i in innerNodes:
        if PM_ys[i-1] > MaxY:
            MaxY = PM_ys[i-1]
            PM_Mid_top_node_deformed = i

    MaxY = -999999
    for i in range(0,len(PBody_xs)):
        if abs(PBody_xs[i] - midsaggital_x) < 5:
            if PBody_ys[i] > MaxY:
                MaxY = pbody_points[i][1]
                pbody_top_middle_node = i


# #6) Create a line with PBody top and PM Mid top
# #    prolapse_measurement_line_absolute = Line3D(PM_Mid_top_original, pbody_top_original)
    
#     AbaqusBatLocation= config["SYSTEM_INFO"]["AbaqusBatLocation"]        
   
#     #   COORD is the tag for the coordinate for the ODB file
#     node_property = "COORD"

# #    Use the last time step to get final position
#     frames = frame
# #    MaterialName = PM_MID.upper() + '-1' + ';' + PBODY.upper() + '-1'
#     nodes = str(PM_Mid_top_node) + ';' + str(pbody_top_middle_node)
    
#     odb_filename = ODBFile[:-4]

#     material_names = PM_MID.upper() + '-1' + ';' + PBODY.upper() + '-1'
    
#     points = get_odb_data(material_names, nodes, node_property, frames, AbaqusBatLocation, odb_filename)

    # PM_Mid_top_deformed = Point3D(midsaggital_x, points[0][1], points[0][2])
    # pbody_top_deformed = Point3D(midsaggital_x, points[1][1], points[1][2])
    print('PM_Mid_top_node_deformed', PM_Mid_top_node_deformed)
    PM_Mid_top_deformed = Point3D(midsaggital_x, PM_ys[PM_Mid_top_node_deformed - 1 ], PM_zs[PM_Mid_top_node_deformed -1])
    pbody_top_deformed = Point3D(midsaggital_x, PBody_ys[pbody_top_middle_node], PBody_zs[pbody_top_middle_node])
    
    
    
       
    return(PM_Mid_top_original,pbody_top_original,PM_Mid_top_deformed,pbody_top_deformed)
    

# def line_to_point_distance(prolapse_measurement_line_deformed,point):
    
#     # prolapse_measurement_line_deformed = Line3D(PM_Mid_top_deformed, pbody_top_deformed)

#     whole_matrix = np.array([[midsaggital_x, PM_Mid_top_original.y, PM_Mid_top_original.z], [midsaggital_x, pbody_top_original.y, pbody_top_original.z], [midsaggital_x, middle_nodes[j][1], middle_nodes[j][2]]], dtype='float')

#     side_of_line = np.sign(np.linalg.det(whole_matrix.astype(float)))
#     side_of_line2 = np.sign((pbody_top_original.y -PM_Mid_top_original.y)*(middle_nodes[j][2] - pbody_top_original.z) - (middle_nodes[j][1] - pbody_top_original.y)*(pbody_top_original.z - PM_Mid_top_original.z))

#     distance_relative = line.distance(point) * side_of_line * -1
#     distance_absolute = prolapse_measurement_line_absolute.distance(Point3D(middle_nodes[j])) * side_of_line * -1
    


'''
Function: get_AVW_midline_nodes

this gets the midline nodes and recreates the a spline of new nodes (y,z)
'''
def get_AVW_midline_nodes(AVW_csv_file, slice_x_value):
    
    (xs, ys, zs) = getFEADataCoordinates(AVW_csv_file)

    THRESHOLD =  3# Max Distance point can be from X axis
    # slice_x_value = 4
    
#    Getting the middle nodes and finding the one with the largest z coordiante
    middle_nodes = []
    max_z = -999999
    for index, xval in enumerate(xs):
        if (abs(slice_x_value - xval) < THRESHOLD):
            middle_nodes.append((ys[index], zs[index]))
            if zs[index] > max_z:
                max_z = zs[index]
                start = (ys[index], zs[index])
   
    
    spline_ordered = [start]
    count = len(middle_nodes)
    
    distance_array=[0]
    used = []
    ## Sorting the spline 
    for i in range(0,count):
        least_dist = float("inf")
        least_index = 0
    # loop through the points and see which one is the closest to the current point...
        # first point is the one with the "start" point which has the max z
        for j , point in enumerate(middle_nodes):
            if j not in used:
#                Calculate the distance between the last point added and the point that is being investigated
                for b in range(0,min(3,len(spline_ordered))):
                    # print(b)
                    dist = hypot(spline_ordered[-1*b - 1][0] - point[0], spline_ordered[-1*b - 1][1] - point[1])
                actual_dist = hypot(spline_ordered[-1][0] - point[0], spline_ordered[-1][1] - point[1])
                if dist < least_dist and j not in used:
                    least_dist = dist
                    least_index = j
                    actual_distance = actual_dist
                    # print(actual_dist)
#       Keeping a log of the running total distance
        distance_array.append(actual_distance + distance_array[-1])
        spline_ordered.append(middle_nodes[least_index])
        used.append(least_index)
    
#    print('First Attempt')
    # print(spline_ordered)
    # print(distance_array)
        # remove the first coordinate pair because it was just the starting one
    spline_ordered.pop(0)
    distance_array.pop(0)
    
    if spline_ordered[0][0] < spline_ordered[-1][0]:
        spline_ordered = list(reversed(spline_ordered))
        distance_array = list(reversed(distance_array))
        distance_array[:] = [max(distance_array)-element for element in distance_array ]
    
    
    plt.scatter([item[1] for item in spline_ordered],[item[0] for item in spline_ordered], color = 'b', marker = '.')
    # plt.show()    

    
    # smoothing out the data by averaging each 3 points
    # start with the first point (no averaging)
    new_spline_ordered = [spline_ordered[0]]
        
    # set new values for the points by averaging them with the point before and after
    # this is done because sometimes you have poorly performing data when you have 2 different rows
    for i in range(1,len(spline_ordered)-1):
        new_spline_ordered.append(((spline_ordered[i-1][0]+spline_ordered[i][0]+spline_ordered[i+1][0])/3,(spline_ordered[i-1][1]+spline_ordered[i][1]+spline_ordered[i+1][1])/3))
    new_spline_ordered.append(spline_ordered[-1])
    
    plt.scatter([item[1] for item in new_spline_ordered],[item[0] for item in new_spline_ordered], color = 'b', marker = '+')
    # plt.show()
    
    new_distance_array = [0]
    for i in range(1,len(new_spline_ordered)):    
        new_distance_array.append(hypot(new_spline_ordered[i][0] - new_spline_ordered[i-1][0], new_spline_ordered[i][1] - new_spline_ordered[i-1][1])+new_distance_array[-1])
    
    # print(distance_array)
    return (new_spline_ordered, new_distance_array)



'''
Function: midline_curve_nodes

takes the midline points and distance array and smooths it out
'''
def midline_curve_nodes(midline_points, distance_array):
    ys = [i[0] for i in midline_points]
    zs = [i[1] for i in midline_points]

    # doing interpolations for the coordinates with the distance array
    # this allows for the interpolation to be done to join the y and z
    # data together even though they're not a 1-to-1 function
    curve_y = interpolate.UnivariateSpline(distance_array, ys, k = 5)
    curve_z = interpolate.UnivariateSpline(distance_array, zs, k = 5)
    
    # the points are then createdto make the spacing for the points equal
    spaced_distance_array = np.linspace(0,distance_array[-1],51)    
    
    new_distance_array  = [0]
    previous_z = curve_z(0)
    previous_y = curve_y(0)
    new_zs = [previous_z]
    new_ys = [previous_y]
    for i in range (1,len(spaced_distance_array)):
        new_ys.append(float(curve_y(spaced_distance_array[i])))
        new_zs.append(float(curve_z(spaced_distance_array[i])))
    
        new_distance_array.append(hypot(new_ys[-1] - new_ys[-2], new_zs[-1]-new_zs[-2])+new_distance_array[-1])

    return(new_zs, new_ys, new_distance_array)

'''
Function: calc_prolapse_size_spline_line

finds the farthest distance from the hymenal ring to the the prolapse (AVW)
'''
def calc_prolapse_size_spline_line(GenericINPFile, INP_File, AVW_csv_file, PM_Mid_file, PBody_file):
#"Redo soft tissue prolapse measurement to use a point at the top of the PM-Mid and top of PM-Body with same x coordinate:

    print('Generic INP: ', GenericINPFile)
    # print('File for prolapse measuring: ', ODBFile)
    print('Specific INP: ', INP_File)
    # print('INI: ', INI_file)
    
    
###### Find the hymenal ring here...need to figure out what form to send it back with...points or lines...
    PM_Mid_top_original,pbody_top_original, PM_Mid_top_deformed, pbody_top_deformed = hymenal_ring(PM_Mid_file, PBody_file, INP_File)

    midsaggital_x = float(PM_Mid_top_original.x)
    prolapse_measurement_line_deformed = Line3D(PM_Mid_top_deformed, pbody_top_deformed)
    prolapse_measurement_line_absolute = Line3D(PM_Mid_top_original, pbody_top_original)
    # return(PM_Mid_top_original,pbody_top_original,PM_Mid_top_deformed,pbody_top_deformed)


    # get AVW midline from the csv file
    # spline_ordered = get_AVW_midline(AVW_csv_file)

    # (xs, ys, zs) = getFEADataCoordinates(AVW_csv_file)
    
    # gets a spline of nodes along the midline
    new_spline_ordered, new_distance_array = get_AVW_midline_nodes(AVW_csv_file, midsaggital_x)
    
    new_zs, new_ys, new_distance_array = midline_curve_nodes(new_spline_ordered, new_distance_array)
    
    #########################################
    # Check the distance for each AVW node to the deformed or undeformed line
    max_prolapse = -9999999999
    max_prolapse_absolute = -9999999999

    for j in range(len(new_zs)):
#                whole_matrix = np.array([[midsaggital_x, PM_Mid_top_original.y, PM_Mid_top_original.z], [midsaggital_x, pbody_top_original.y, pbody_top_original.z], [midsaggital_x, new_xs[j], new_ys[j]]], dtype='float')
        whole_matrix = np.array([[midsaggital_x, PM_Mid_top_original.y, PM_Mid_top_original.z], [midsaggital_x, pbody_top_original.y, pbody_top_original.z], [midsaggital_x, new_ys[j], new_zs[j]]], dtype='float')

        side_of_line = np.sign(np.linalg.det(whole_matrix.astype(float)))
        side_of_line2 = np.sign((pbody_top_original.y -PM_Mid_top_original.y)*(new_zs[j] - pbody_top_original.z) - (new_ys[j] - pbody_top_original.y)*(pbody_top_original.z - PM_Mid_top_original.z))

        # print(float(midsaggital_x))
        # print(new_ys[j])
        # print(new_zs[j])
        # print(Point3D(-2.63050294,-1.10570651,-31.12847607))
        # new_point = Point3D(-2.63050294,-1.10570651,-31.12847607)
        # print(side_of_line)
        # print(prolapse_measurement_line_deformed)
        # float_point = Point3D(midsaggital_x,float(new_ys[j]),float(new_zs[j]))
        # distance_relative = prolapse_measurement_line_deformed.distance(new_point) * side_of_line * -1
        # distance_relative = prolapse_measurement_line_deformed.distance(float_point) * side_of_line * -1
        distance_relative = prolapse_measurement_line_deformed.distance(Point3D(midsaggital_x,float(new_ys[j]),float(new_zs[j]))) * side_of_line * -1
        distance_absolute = prolapse_measurement_line_absolute.distance(Point3D(midsaggital_x,float(new_ys[j]),float(new_zs[j]))) * side_of_line * -1

        # distance_relative = prolapse_measurement_line_deformed.distance(Point3D(middle_nodes[j])) * side_of_line * -1
        # distance_absolute = prolapse_measurement_line_absolute.distance(Point3D(middle_nodes[j])) * side_of_line * -1
#        print(middle_nodes[j], float(distance_absolute), side_of_line, side_of_line2)
        #            print(distance)
        if distance_relative > max_prolapse:
            max_prolapse = float(distance_relative)
            max_prolapse_node = j
            
        if distance_absolute > max_prolapse_absolute:
            max_prolapse_absolute = float(distance_absolute)
            max_prolapse_absolute_node = j

    print('original code :', max_prolapse, max_prolapse_absolute)

    return(max_prolapse, max_prolapse_absolute)    

'''
Function: distance_to_hymenal_ring
'''
def distance_to_hymenal_ring(PM_Mid_top_original,pbody_top_original, PM_Mid_top_deformed, pbody_top_deformed, y, z):

    midsaggital_x = float(PM_Mid_top_original.x)
    prolapse_measurement_line_deformed = Line3D(PM_Mid_top_deformed, pbody_top_deformed)
    prolapse_measurement_line_absolute = Line3D(PM_Mid_top_original, pbody_top_original)
    
    whole_matrix = np.array([[midsaggital_x, PM_Mid_top_original.y, PM_Mid_top_original.z], [midsaggital_x, pbody_top_original.y, pbody_top_original.z], [midsaggital_x, y, z]], dtype='float')

    side_of_line = np.sign(np.linalg.det(whole_matrix.astype(float)))
    distance_relative = prolapse_measurement_line_deformed.distance(Point3D(midsaggital_x,float(y),float(z))) * side_of_line * -1
    distance_absolute = prolapse_measurement_line_absolute.distance(Point3D(midsaggital_x,float(y),float(z))) * side_of_line * -1
    
    return(float(distance_relative), float(distance_absolute))

'''
Function: distances_along_AVW
'''
def distances_along_AVW(PM_Mid_top_original,pbody_top_original, PM_Mid_top_deformed, pbody_top_deformed, ys, zs, distance_array):
    distance_to_line = -1
    i = 0
    # print("length of zs: ", len(zs))
    while distance_to_line < 0 and i < len(zs):
        distance_to_line, absolute_distance = distance_to_hymenal_ring(PM_Mid_top_original,pbody_top_original, PM_Mid_top_deformed, pbody_top_deformed, ys[i], zs[i])
        i += 1
    if i == len(zs):
        exposed_vaginal_length = 0
        apical_distance = distance_array[-1]
    else:
        distance_before_hymen = distance_array[i-1]
        while distance_to_line > 0:
            distance_to_line, absolute_distance = distance_to_hymenal_ring(PM_Mid_top_original,pbody_top_original, PM_Mid_top_deformed, pbody_top_deformed, ys[i], zs[i])
            i += 1
        exposed_vaginal_length = distance_array[i-1] - distance_before_hymen
        apical_distance = distance_array[-1] - distance_array[i-1]

    return(exposed_vaginal_length, apical_distance)    
        