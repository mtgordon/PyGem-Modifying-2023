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
from scipy import interpolate
import csv
from sympy import Point3D
from sympy.geometry import Line3D
from lib.workingWith3dDataSets import DataSet3d

def Calc_Reaction_Forces(path_base_file_name, output_base_filename, GenericINPFile, INI_file):
    
    config = configparser.ConfigParser()
    config.sections()
    config.read(INI_file)
    
    AbaqusBatLocation= config["SYSTEM_INFO"]["AbaqusBatLocation"]
    
    PartName = "OPAL325_CL_V6-1"
    PartNameCS = "OPAL325_CL_v6"
    Variable1 = "RF"
    Headerflag = 'Y'
    NewFileFlag = 'Y'
#            Frames = 'all' gives every step, Frames = 'last' is just the last one
    Frames = 'last'
    print('________________________________________')
    DataFileName = output_base_filename + '_Forces'
    Bnodes = getBnodes(GenericINPFile, PartNameCS)
    PassingNodes = ','.join(str(i) for i in Bnodes)

    CallString = AbaqusBatLocation + '  CAE noGUI=ODBMechensFunction_v2  -- -odbfilename "' + path_base_file_name + '" -partname ' + PartName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile "' + DataFileName + '" -headerflag ' + Headerflag + ' -newfileflag ' + NewFileFlag + ' -frames last'
    print(CallString)
    subprocess.call(CallString)
    time.sleep(3)
    
    PartName = "OPAL325_US_V6-1"
    PartNameCS = "OPAL325_US_v6"
    Variable1 = "RF"
    Headerflag = 'Y'
    NewFileFlag = 'N'
    Frames = 'last'
    Bnodes = getBnodes(GenericINPFile, PartNameCS)
    PassingNodes = ','.join(str(i) for i in Bnodes)
    CallString = AbaqusBatLocation + '  CAE noGUI=ODBMechensFunction_v2  -- -odbfilename "' + path_base_file_name + '" -partname ' + PartName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile "' + DataFileName + '" -headerflag ' + Headerflag + ' -newfileflag ' + NewFileFlag + ' -frames ' + Frames
    print('________________________________________')
    print(CallString)
    print('________________________________________')
    subprocess.call(CallString)
    time.sleep(3)
    
    headers = []
    forces = []
    with open(DataFileName + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) == 1:
                part = row[0]
            elif row[0] == 'Total Force Using Resultants =':
                headers.append(part + ' Total Resultant Force')
                forces.append(row[1])
            elif row[0] == 'Total Force Using Components =':
                headers.append(part + ' Total Component Force')
                forces.append(row[1])
                
    print(headers, forces)

    return(headers, forces)
## If there are multiple rows, get the last one
#    try:
#        data = csv[-1,:]
#    except IndexError:
#        data = csv


##### Taken from ExposedVaginalWallMeasurements.py
##### Cannot be imported from there because then it runs the file
##### Need to move the function to a file for the library
def getFEADataCoordinates(FileName):
    csv = np.genfromtxt (FileName, delimiter=",")

# If there are multiple rows, get the last one
    try:
        data = csv[-1,:]
    except IndexError:
        data = csv
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
  
# Function to find equation of plane. 
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
  
  
# Function to find distance 
def shortest_distance(x1, y1, z1, a, b, c, d):  
      
    d = (a * x1 + b * y1 + c * z1 + d) 
#    print("d = ", d)
    e = (math.sqrt(a * a + b * b + c * c))
#    print("e = ", e)
#    Might need to add a negative side if we want the other direction
    perpendicular_distance = -1*d/e
    return(perpendicular_distance)

###########################################################
# Read in the parameters from the ini file
###########################################################

def calc_prolapse_size(GenericINPFile, ODBFile, INP_File, INI_file):
#"Redo soft tissue prolapse measurement to use a point at the top of the PM-Mid and top of PM-Body with same x coordinate:


        # read from input file
#    ODBFile = base_file_name + '.odb'
#    INP_File = base_file_name + '.inp'
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
    pbody_surface = DataSet3d(list(pbody_points[:, 0]), list(pbody_points[:, 1]), list(pbody_points[:, 2]))
    MaxY = -999999
    for i in range(0,len(pbody_surface)):
        if abs(pbody_surface[i][0] - midsaggital_x) < 5:
            if pbody_surface[i][1] > MaxY:
                MaxY = pbody_surface[i][1]
                pbody_top_middle_node = i
#5) set PM Body top.x = PM MId top.x
#    set the x coordinate to that of PM Top s (midsaggital_x) o that we get the midsaggital line
    pbody_top_original = Point3D(midsaggital_x, pbody_surface[pbody_top_middle_node][1], pbody_surface[pbody_top_middle_node][2])


#6) Create a line with PBody top and PM Mid top
    prolapse_measurement_line_absolute = Line3D(PM_Mid_top_original, pbody_top_original)
    
    AbaqusBatLocation= config["SYSTEM_INFO"]["AbaqusBatLocation"]    
    
##########################################################################################################
## I need to get the coordinates of the points for the plane after the run
## ...for this I need the output file
    
#    Generate a temporary file to store the coordinates from the ODB file
#    that will be used to calculate the plane
    DataFileName = "Temp_Coords" # Not done yet
    
    #   COORD is the tag for the coordinate for the ODB file
    Variable1 = "COORD"
#   We don't want a header in our temp file
    Headerflag = 'N'
#    Do create a new file for the temp file
    NewFileFlag = 'Y'
#    Use the last time step to get final position
    Frames = 'last'
    
        
    MaterialName = PM_MID.upper() + '-1'
    PassingNodes = str(PM_Mid_top_node)
    
    upgraded_inp_file = ODBFile[:-4]
    CallString = AbaqusBatLocation + '  CAE noGUI=ODBMechensFunction_v2  -- -odbfilename "' + upgraded_inp_file.replace('\\','\\\\') + '" -partname ' + MaterialName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile ' + DataFileName + ' -headerflag ' + Headerflag + " -newfileflag " + NewFileFlag + " -frames " + Frames
    print('_________________________________________')
    print(CallString)
    print('_________________________________________')
    try:
        os.remove(DataFileName+'.csv')
    except:
        pass

    subprocess.call(CallString)
    time.sleep(3)
    
#        print(DataFileName)
    Coordinates = genfromtxt(DataFileName + '.csv', delimiter=',')
    PM_Mid_top_deformed = Point3D(midsaggital_x, Coordinates[1], Coordinates[2])
    
    
    
    MaterialName = PBODY.upper() + '-1'
    PassingNodes = str(pbody_top_middle_node)
    
    upgraded_inp_file = ODBFile[:-4]
    CallString = AbaqusBatLocation + '  CAE noGUI=ODBMechensFunction_v2  -- -odbfilename "' + upgraded_inp_file.replace('\\','\\\\') + '" -partname ' + MaterialName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile ' + DataFileName + ' -headerflag ' + Headerflag + " -newfileflag " + NewFileFlag + " -frames " + Frames
    print('_________________________________________')
    print(CallString)
    print('_________________________________________')
    try:
        os.remove(DataFileName+'.csv')
    except:
        pass

    subprocess.call(CallString)
    time.sleep(3)
    
#        print(DataFileName)
    Coordinates = genfromtxt(DataFileName + '.csv', delimiter=',')
    pbody_top_deformed = Point3D(midsaggital_x, Coordinates[1], Coordinates[2])
    
    prolapse_measurement_line_deformed = Line3D(PM_Mid_top_deformed, pbody_top_deformed)
    
    desired_distance = json.loads(config["PROLAPSE_SIZE"]["Vaginal_Wall_Distance"])


#7) Calculate distance between AVW points whose x val is PM Mid top.x +/- 3
#8) the largest distance of those points (hopefully correctly signed) is the prolapse size"    
    ###################################################################
    # Next get the points for the vaginal wall
#    PrintMaterialHeader = 'N'
#    NewOutputFile = 'Y'
    ML = ["OPAL325_AVW_v6"]
    MaterialList = ['OPAL325_AVW_V6-1']
    MaterialSizeList = [numOfNodes(GenericINPFile, ML[0])]
    nodes = list(range(1,MaterialSizeList[0]+1))
    PassingNodes = ','.join(str(i) for i in nodes)
    Variable1 = "COORD"
    Headerflag = 'N'
    NewFileFlag = 'Y'
    Frames = 'last'
    MaterialName = MaterialList[0]
    DataFileName = 'AVW_nodal_coordinates'
    CallString = AbaqusBatLocation + '  CAE noGUI=ODBMechensFunction_v2  -- -odbfilename "' + upgraded_inp_file.replace('\\','\\\\') + '" -partname ' + MaterialName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile ' + DataFileName + ' -headerflag ' + Headerflag + " -newfileflag " + NewFileFlag + " -frames " + Frames
#        print(CallString)
    subprocess.call(CallString)
    
    #May be needed for when I run it on a virtual machine because the files have to be uploaded
    time.sleep(3)
    
#        print(DataFileName)
    (xs, ys, zs) = getFEADataCoordinates(DataFileName+'.csv')
#########################################


#########################################
    # Check the distance for each point
    max_prolapse = -9999999999
    max_prolapse_absolute = -9999999999
#        print("Number of AVW points = ", len(xs))
    for j in range (len(xs)):
            distance_relative = prolapse_measurement_line_deformed.distance(Point3D(xs[j],ys[j],zs[j]))
            distance_absolute = prolapse_measurement_line_absolute.distance(Point3D(xs[j],ys[j],zs[j]))
            #            print(distance)
            if distance_relative > max_prolapse:
                max_prolapse = distance_relative
                max_prolapse_node = j+1
                
            if distance_absolute > max_prolapse_absolute:
                max_prolapse_absolute = distance_absolute
                max_prolapse_absolute_node = j+1


######################################################
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
    print("The prolapse size is: ", max_prolapse)
    print("It occurred at node ", max_prolapse_node, ": ", xs[max_prolapse_node-1], ", ", ys[max_prolapse_node-1], ", ", zs[max_prolapse_node-1])
    
    print("The absolute prolapse size is: ", max_prolapse_absolute)
    print("It occurred at node ", max_prolapse_node, ": ", xs[max_prolapse_absolute_node-1], ", ", ys[max_prolapse_absolute_node-1], ", ", zs[max_prolapse_absolute_node-1])
    
    return(max_prolapse, max_prolapse_absolute, max_prolapse_node)
    
    

def calc_prolapse_size_plane(GenericINPFile, ODBFile, INP_File, INI_file):
    # read from input file
#    ODBFile = base_file_name + '.odb'
#    INP_File = base_file_name + '.inp'
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
    
#    The ODB file that will be analyzed
#    Where the final node coordiantes are stored

    desired_distance = json.loads(config["PROLAPSE_SIZE"]["Vaginal_Wall_Distance"])
#    Upgrade_Necessary = 1
#    if Upgrade_Necessary:    
#        upgraded_inp_file = 'NewVersionODBFile'
#            
#        CallString = AbaqusBatLocation + ' -upgrade -job ' + upgraded_inp_file + ' -odb "' + ODBFile.replace('\\','\\\\') + '"'
##            print(CallString)
#
#        try:
#            os.remove(upgraded_inp_file + '.odb')
#        except:
#            pass
#        subprocess.call(CallString)
#    else:
######### This line was used successfully on Michigan computer, but didn't work on Luyun's Computer
##            upgraded_inp_file = ODBFile
#######################################
#        upgraded_inp_file = ODBFile[:-4]


    upgraded_inp_file = ODBFile[:-4]

#    
#   Turn the array of nodes into a string for sending
#    plane_nodes_string = ','.join(str(i) for i in plane_nodes)

#   COORD is the tag for the coordinate for the ODB file
    Variable1 = "COORD"
#   We don't want a header in our temp file
    Headerflag = 'N'
#    Do create a new file for the temp file
    NewFileFlag = 'Y'
#    Use the last time step to get final position
    Frames = 'last'
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
#        print(PassingNodes)
#        PassingNodes = str(i+1)
#        DataFileName = 'Temp_' + OutputFileName
#        try:
#            os.remove(ODBFile +'.odb')
#        except:
#            pass
#        CallString = 'abaqus -upgrade -job ' + ODBFile + ' -odb NewODBFile'
#        print(CallString)

#            print()
        CallString = AbaqusBatLocation + '  CAE noGUI=ODBMechensFunction_v2  -- -odbfilename "' + upgraded_inp_file.replace('\\','\\\\') + '" -partname ' + MaterialName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile ' + DataFileName + ' -headerflag ' + Headerflag + " -newfileflag " + NewFileFlag + " -frames " + Frames
        print('_________________________________________')
        print(CallString)
        print('_________________________________________')
        try:
            os.remove(DataFileName+'.csv')
        except:
            pass

        subprocess.call(CallString)
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
    Frames = 'last'
    MaterialName = MaterialList[0]
    DataFileName = 'AVW_nodal_coordinates'
    CallString = AbaqusBatLocation + '  CAE noGUI=ODBMechensFunction_v2  -- -odbfilename "' + upgraded_inp_file.replace('\\','\\\\') + '" -partname ' + MaterialName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile ' + DataFileName + ' -headerflag ' + Headerflag + " -newfileflag " + NewFileFlag + " -frames " + Frames
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


#####################################################
#                    Negative prolapse code
#   If the prolapse is negative, then a different clinical measurement is used
    if max_prolapse < 5:


# Create the interpolation
        interpolator = interpolate.Rbf(xs, zs, ys, function='linear', smooth=1)

# Grab the highest Z value for midsaggital node
#            May need to go back and change this if there are small bulges with
#           negative prolapse values or the midsaggital isn't the highest Z
        endZ = max(zs)

#            find distance between that point (X=0, Z=max, y = point/interpolation) and the next (X = 0, Z = max - 1, y = interpolated)
        distance = 0.0
        StepSize = 0.02
#            Distance along the vaginal wall that clinicians use to determine
#            prolapse size
#            desired_distance = 25
        oldZ = endZ
        oldY = interpolator(0, endZ)
        while distance < desired_distance:
            newZ = oldZ - StepSize
            newY= interpolator(0, newZ)

# Add it to the total distance
        
            distance += pythag(newZ - oldZ, oldY - newY)
            oldY = newY
            oldZ = newZ

# Check to see if it is greater than the target
# If not loop
# If so, interpolate the new point
# Find the distance from that point to the planes that were found above            
#                zRange = np.arange(fromValue, toValue, INCREMENT * (1 if fromValue < toValue else -1))

        distance_relative = shortest_distance(0,newY,newZ,a,b,c,d)
        distance_absolute = shortest_distance(0,newY,newZ,aBefore,bBefore,cBefore,dBefore)
        max_prolapse = distance_relative
        max_prolapse_absolute = distance_absolute
#            print('Coordinates are: 0, ', newY, newZ)
        
        
        
######################################################
#        Continuing after adding negative prolapse code
    print("The prolapse size is: ", max_prolapse)
    print("It occurred at node ", max_prolapse_node, ": ", xs[max_prolapse_node-1], ", ", ys[max_prolapse_node-1], ", ", zs[max_prolapse_node-1])
    
    print("The absolute prolapse size is: ", max_prolapse_absolute)
    print("It occurred at node ", max_prolapse_node, ": ", xs[max_prolapse_absolute_node-1], ", ", ys[max_prolapse_absolute_node-1], ", ", zs[max_prolapse_absolute_node-1])
    
    return(max_prolapse, max_prolapse_absolute, max_prolapse_node)