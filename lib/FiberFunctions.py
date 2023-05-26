# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 12:48:48 2017

@author: Jelaiya Jacob
"""

import math
import sys

from lib.Surface_Tools import pythag, plot_Dataset, find_starting_ending_points_for_inside, findInnerNodes
import lib.IOfunctions as io
from lib.ConnectingTissue import ConnectingTissue
from lib.workingWith3dDataSets import Point, DataSet3d
from lib.RemovePart import remove_connections
from scipy.optimize import fsolve
import numpy as np
import sympy

'''
Function: find_index_of_element_connectons
'''
def find_index_of_element_connectons(content):
    possible_starts = []
    for i, line in enumerate(content):
        if "*Element, type=CONN3D2" in line:
            possible_starts.append(i)    
        
    return possible_starts

'''
Function: try_int
'''
def try_int(n):
    try:
        return int(n)
    except:
        return False

#===============
# Function: get_connections_for_tissues
#
# Gets the points that link the AVW to CL
# Assumes data comes after "Weld,\n*Element, type=CONN3D2"
# The lines look like this "3, OPAL325_AVW_v6-1.14, OPAL325_Para_v6-1.15"
#===============
# Currently used with tis1 being the Fiber and tis2 being the AVW
def get_connections_for_tissues(tis1, tis2, file_name):
    
    content = io.openFile(file_name)
    con_nodes = {}
    
    # find all of the lines of connections (Assumes data comes after "Weld,\n*Element, type=CONN3D2")
    indexes = find_index_of_element_connectons(content)
    
    for start in indexes:
        index = start +1
        expected_next = int(content[index].split(",")[0])
        
        for i, line in enumerate(content[index:]):
#                split with , to make 3 parts: [0] the number, [1] and [2] the materials with their nodes
                data = line.split(",")
                if try_int(data[0]) == expected_next:
                    expected_next += 1
                    con1 = data[1]
                    con2 = data[2]
                    
                    # Check to see if this line referrs to the correct tissues
                    # If so get the number after the period (node number)
                    # The two parts are to make sure you have the node for tissue2 (AVW) and pair it with the corresponding tis1 (Fiber)
                    # Result is something like {7: 10, 1: 11, 14: 13, 16: 15, 5: 17, 3: 19, 4: 25, 8: 12, 9: 14, 10: 16, 11: 18, 12: 20}
                    # With the first number being the Fiber node and the second being the AVW node
                    if tis1 in con1 and tis2 in con2:
                        con_nodes[int(con1.split(".")[1]) - 1] = int(con2.split(".")[1]) - 1
                    elif tis1 in con2 and tis2 in con1:
                        con_nodes[int(con2.split(".")[1]) - 1] = int(con1.split(".")[1]) - 1
                else:
                    break
        if con_nodes != {}:
            break
    
    return con_nodes


# Function: CurveFibersInINP
#
# This function takes the apical supports (or other fibers), finds the attachment points,
# and tries to make them a certain length
def CurveFibersInINP(Part_Name1, Part_Name2, scale, inputFile, outputFile, dirVector, updatedPositiveP, updatedNegativeP,
                     positiveConnectionRemovePercent, negativeConnectionRemovePercent):

    updatedPositiveP = configure_start_points(updatedPositiveP)
    updatedNegativeP = configure_start_points(updatedNegativeP)

    #Part_Name1 = AVW, Part_Name2 = the fiber tissue
    # Getting the coordinates for the AVW in the correct form from the file being worked on
    FILE_NAME = inputFile
    AVWpoints = np.array(io.extractPointsForPartFrom(FILE_NAME, "OPAL325_AVW_v6"))
    AVW_surface = DataSet3d(list(AVWpoints[:, 0]), list(AVWpoints[:, 1]), list(AVWpoints[:, 2]))

    nodes, connections = io.extractPointsForPartFrom2(FILE_NAME, Part_Name2, get_connections=True) #nodes and their connections to each other
    
    # what gets returned? Are the AVW node or Fiber nodes?
    # Result is something like {7: 10, 1: 11, 14: 13, 16: 15, 5: 17, 3: 19, 4: 25, 8: 12, 9: 14, 10: 16, 11: 18, 12: 20}
    # With the first number being the Fiber node and the second being the AVW node
    con_from_1_to_2 = get_connections_for_tissues(Part_Name2, Part_Name1, FILE_NAME) # connections to the surface 
    # print("Testing this = ", con_from_1_to_2 )
    ct = ConnectingTissue(nodes, connections, con_from_1_to_2)
#    print(ct)
#    if(Part_Name2 == "OPAL325_Para_v6"):
#        
#       plot_Dataset(c
#   The keys might be the node numbers
    fibers = ct.fibers_keys
#    print("Fibers = ", fibers)

    #Keeps track of the total connection points for the positive fibers
    positive_connections = {}
    negative_connections = {}

    for i, fiber in enumerate(fibers): #loop through each fiber
#        print(fiber)
        starting_node_index = ct.starting_nodes[i] # getting indexes of nodes from the i fiber
        ending_node_index = ct.ending_nodes[i]

        # print('Original Staring node index at iter: ' + str(i) + ", index: " + str(starting_node_index))
        # print('Ending node index at iter: ' + str(i) + ", index: " + str(ending_node_index))

        starting_p = ct.node(starting_node_index) # getting nodes from the index number
        ending_p = ct.node(ending_node_index)

        # I believe the AVW node number that corresponds to where the fiber connects to the AVW
        avw_node_number = ct.avw_connections[ending_node_index]
        print()
        # node (coordinates) where the fiber connects to the AVW
        avw_node = AVW_surface.node(avw_node_number)

        if Part_Name2 == "OPAL325_CL_v6":
            if avw_node.x < 0:
                negative_connections[ending_node_index + 1] = ending_p.z
            else:
                positive_connections[ending_node_index + 1] = ending_p.z

        #Find the length of the original fiber
        OriginalFiberLength = 0


#        Get the original fiber length
        for j, NodeNumber in enumerate(fiber[:-1]): #loop through each node in the fiber except the last
            p = ct.node(NodeNumber)
            q = ct.node(fiber[j+1]) ### Error when it gets to last element of fiber
    
            # calculating distance between each node
            Distance = math.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2) #distance formula: there is some trouble with node index when it tries to calculate. Possibly caused by q variable
            OriginalFiberLength = OriginalFiberLength + Distance
        
#        print("Original Fiber Length = ", OriginalFiberLength)

# Percentage of normal length to make the new tissue (over 100%)
        IdealFiberLength = OriginalFiberLength*(scale + 1)
#        print("Ideal Fiber Length = ", IdealFiberLength)        
        
        StartingAmplitude = 1

        NodesPerCycle = 15
        NumberOfCycles = math.floor(len(fiber)/NodesPerCycle)
        new_ending_p = avw_node
        _dist = math.sqrt((starting_p.x - new_ending_p.x) ** 2 + (starting_p.y - new_ending_p.y) ** 2 + (starting_p.z - new_ending_p.z) ** 2)
#        print("AVW Node Coordinates: ", avw_node)
#        print("AVW Node Number: ", avw_node_number)
#        new_starting_p = avw_node
#        _dist = math.sqrt((new_starting_p.x - ending_p.x) ** 2 + (new_starting_p.y - ending_p.y) ** 2 + (new_starting_p.z - ending_p.z) ** 2)

#        _dist = math.sqrt((starting_p.x - ending_p.x) ** 2 + (starting_p.y - ending_p.y) ** 2 + (starting_p.z - ending_p.z) ** 2)

#avw_node
#        print("NumberOfCycles = ",NumberOfCycles)
#        print("Dist = ", _dist)
        if IdealFiberLength > _dist:
            sending = (IdealFiberLength, fiber, starting_node_index, ending_node_index, ct, dirVector,NumberOfCycles,avw_node)
            [CorrectAmp] = fsolve(ArcDistance, StartingAmplitude, args=sending)

        else:
            [CorrectAmp] = [0]
#        print("Correct Amp is = ", CorrectAmp)
        starting_p = ct.node(starting_node_index) # getting nodes from the index number
        ending_p = ct.node(ending_node_index)

        minY = starting_p.y # min and max y values from start and end nodes
        maxY = ending_p.y

        
        
        ##################################################################
        OldXRange = ending_p.x-starting_p.x
        OldYRange = ending_p.y-starting_p.y
        OldZRange = ending_p.z-starting_p.z
        
        ########################## Set the ending node to the correct location

        #TODO: If else for the updated point located here
        #TODO: New if branch to implement +/-
        if Part_Name2 == "OPAL325_CL_v6" or Part_Name2 == "OPAL325_US_v6" or Part_Name2 == "OPAL325_Para_v6":
            if np.sign(starting_p.x) < 0 and updatedNegativeP is not None:
                starting_p_alterable = updatedNegativeP
            elif np.sign(starting_p.x) > 0 and updatedPositiveP is not None:
                starting_p_alterable = updatedPositiveP
            else:
                starting_p_alterable = starting_p
        else:
            starting_p_alterable = starting_p

        #TODO: have these use the new point
        NewXRange = avw_node.x - starting_p_alterable.x
        NewYRange = avw_node.y - starting_p_alterable.y
        NewZRange = avw_node.z - starting_p_alterable.z


        FinalFiberLength = 0
        for j, NodeNumber in enumerate(fiber): #loop through each node in the fiber except the last to find the correct coordinates
    
            p = ct.node(NodeNumber)
            
            RangedpY = NumberOfCycles*(p.y - minY) / (maxY - minY) * math.pi # Set a y range between 0 and PI
            
            #TODO: have the newRange have the new point and the starting_p at the end (with oldrange) be the new point



            NewX = np.sign(p.x) * dirVector[0] * CorrectAmp * math.sin(RangedpY) + (p.x - starting_p.x) * NewXRange / OldXRange + starting_p_alterable.x # different so that it can be done to curve inwards from both sides of the AVW
            NewY = dirVector[1] * CorrectAmp * math.sin(RangedpY) + (p.y - starting_p.y) * NewYRange / OldYRange + starting_p_alterable.y
            NewZ = dirVector[2] * CorrectAmp * math.sin(RangedpY) + (p.z - starting_p.z) * NewZRange / OldZRange + starting_p_alterable.z
           
            new_p = Point(NewX, NewY, NewZ)
            p = new_p
            if j > 0:
                Distance = math.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2) #distance formula: there is some trouble with node index when it tries to calculate. Possibly caused by q variable
                FinalFiberLength = FinalFiberLength + Distance   
            q = new_p

            ct.update_node(NodeNumber, new_p)
#        print("Final Fiber Length = ", FinalFiberLength)
      
#    print(ct)

    write_part_to_inp(inputFile, outputFile, Part_Name2, ct)

    if Part_Name2 == "OPAL325_CL_v6":
        configure_remove_connections(positive_connections, positiveConnectionRemovePercent, Part_Name2, outputFile)
        configure_remove_connections(negative_connections, negativeConnectionRemovePercent, Part_Name2, outputFile)

    return


'''
Function: configure_remove_connections
Takes in a list of nodes that correspond to connections involving the specified part, the percentage of those
connections to be removed, and the INP file used. It is important to note that the percent given is the portion of
the original connections to be removed, not the connections being kept. The nodes passed into the remove_connections
function are those deleted from the INP file, so if you want more connections removed, make that passed in list larger.
The nodes that have the largest Z value are removed first.

For example (node list refers to the one passed to remove_connections):
percent = 1.0: node list size matches the original
percent = 0.7: node list size is 70% of the original
percent = 0.5: node list size is half the original
percent = 0.2: node list size is 20% of the original
percent = 0.0: node list size is empty
'''
def configure_remove_connections(connections, removePercent, partName, output):
    sorted_connections = []
    # Sort the nodes by largest Z coordinates first to smallest
    for i in sorted(connections, key=connections.get, reverse=True):
        sorted_connections.append(i)
    # Remove the positive connections
    cutoff = int(round(len(sorted_connections) * removePercent))
    remove_connections(sorted_connections[0:cutoff], partName, output)


'''
Function: configure_start_points
Takes in the string point from the run dictionary and converts it into the Point object for later use.
If no new point was given, then uses the point generated in the calculations.
If the given point string's format was incorrect, the program will end and notify the user.
'''
def configure_start_points(point):
    #Configure the updated points for the start of the fibers
    strList = point.split(',')
    floatList = []

    if strList[0] != "x": #Checks for th default, if so, then continue with unaltered point
        for coord in strList:
            try:
                floatList.append(float(coord))
            except ValueError:
                print('''\nERROR: Program has stopped, refer to the message below vvv
                Fiber start point must be in the following format:
                "x,y,z" (double quotes included with x,y,z replaced by the numeric values)
                
                The given rejected format was: ''' + point)
                sys.exit(1)
        return Point(floatList[0], floatList[1], floatList[2])
    else:
        return None


'''
Function: write_part_to_inp
'''
def write_part_to_inp(file_name, outputfile_name, part_name, data_set):

    # writes part to file
    # makes copy of file, 
    # reads from copy and writes data to current
    # deletes copy when finished

    open(outputfile_name, 'w').close()
    io.write_new_inp_file(file_name, part_name, outputfile_name, data_set)

'''
Function: ArcDistance
'''
def ArcDistance(a, *sending): # second function
   #(IdealFiberLength, fiber, starting_node_index, ending_node_index, ct, dirVector,NumberOfCycles,avw_node)
    IdealFiberLength, fiber, starting_node_index, ending_node_index, ct, dirVector, NumberOfCycles, avw_node = sending
    starting_p = ct.node(starting_node_index) # getting nodes from the index number
    ending_p = ct.node(ending_node_index)
    
#    print("InsideArcDistance")
    minY = starting_p.y # min and max y values from start and end nodes
    maxY = ending_p.y
    
    OldXRange = ending_p.x-starting_p.x
    OldYRange = ending_p.y-starting_p.y
    OldZRange = ending_p.z-starting_p.z
        
    ########################## Set the ending node to the correct location
    
    NewXRange = avw_node.x - starting_p.x
    NewYRange = avw_node.y - starting_p.y
    NewZRange = avw_node.z - starting_p.z    

    NewFiberLength = 0
    for j, NodeNumber in enumerate(fiber[:-1]): #loop through each node in the fiber except the last

        p = ct.node(NodeNumber)
        q = ct.node(fiber[j+1]) ### Error when it gets to last element of fiber
        
        RangedpY = NumberOfCycles*(p.y - minY) / (maxY - minY) * math.pi # Set a y range betweeen 0 and PI
        RangedqY = NumberOfCycles*(q.y - minY) / (maxY - minY) * math.pi # Set a y range betweeen 0 and PI

        # new x, y, z scale
#        RangedpY = (p.y - miny) / (maxy - miny) * math.pi # Set a range betweeen 0 and PI
#        RangedqY = (q.y - miny) / (maxy - miny) * math.pi # Set a range betweeen 0 and PI

        newpX = np.sign(p.x) * dirVector[0]* a * math.sin(RangedpY) + (p.x - starting_p.x) * NewXRange / OldXRange + starting_p.x # different so that it can be done to curve inwards from both sides of the AVW
        newpY = dirVector[1]* a * math.sin(RangedpY) +  (p.y - starting_p.y) * NewYRange / OldYRange + starting_p.y
        newpZ = dirVector[2]* a * math.sin(RangedpY) +  (p.z - starting_p.z) * NewZRange / OldZRange + starting_p.z

        newqX = np.sign(q.x) * dirVector[0]* a * math.sin(RangedqY) +  (q.x - starting_p.x) * NewXRange / OldXRange + starting_p.x
        newqY = dirVector[1]* a * math.sin(RangedqY) + (q.y - starting_p.y) * NewYRange / OldYRange + starting_p.y
        newqZ = dirVector[2]* a * math.sin(RangedqY) + (q.z - starting_p.z) * NewZRange / OldZRange + starting_p.z
        NewDistance = pythag( newpX - newqX , newpY - newqY , newpZ - newqZ)
        
        #NewDistance = math.sqrt((newpX - newqX) ** 2 + (newpY - newqY) ** 2 + (newpZ - newqZ) ** 2) #distance formula: there is some trouble with node index when it tries to calculate. Possibly caused by q variable
        NewFiberLength = NewFiberLength + NewDistance
    
#    if IdealFiberLength > 21: 
#        print("Amplitude", a, NewFiberLength)

#    print(NewFiberLength-IdealFiberLength)
    return NewFiberLength - IdealFiberLength

'''
Function: dist
'''
def dist(x1, x2, y1, y2, z1, z2):
    return ((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**0.5


'''
Function: getFiberLength

returns: average length of a fiber (since it may have multiple "lines")
'''
def getFiberLength(fiber, inputfile):
    nodes, connections = io.extractPointsForPartFrom2(inputfile, fiber, get_connections=True)
    lines = []
    mostRecent = -1


    #this following assumes that lines come in order (no interrupts with another line)
    #this section makes 'lines' or connected lists of nodes
    for connection in (connections):
        if connection[0] == mostRecent: #then its part of the same chain
            lines[len(lines) - 1].append(connection[1])
        else:
            lines.append([connection[0], connection[1]])
        mostRecent = connection[1]

    #for each line, go node by node and calculate the distance
    distances = []
    #print("#######" + inputfile)
    for line in lines:
        line[:] = [node - 1 for node in line] #subtracts 1 for every element in the array, prevents off by one error
        #print(line)
        distance = 0
        origin = nodes[line[0]]
        for nodeIndex in line:
            target = nodes[nodeIndex]
            distance += dist(origin[0], target[0], origin[1], target[1], origin[2], target[2])
            origin = target

        distances.append(distance)
        
    #return sum(distances)/len(distances)
    return distances

'''
Function: getFiberLengths

Calls <getFiberLength> n number of times, when n is the number of elements in the given fibers array
'''
def getFiberLengths(inputfile, fibers):
    fiberLengths = []
    for fiber in fibers:
        fiberLengths.append(getFiberLength(fiber, inputfile))
    return fiberLengths




'''
Function: CurvePARAFibersInINP

This function takes the apical supports (or other fibers), finds the attachemnt points,
and tries to make them a certain length
'''
def CurvePARAFibersInINP(Part_Name1, Part_Name2, scale, inputFile, outputFile, dirVector,  PM_Mid, connections):
#    g = 0
#    PM_Mid_new = PM_Mid.generatedCopy()
    #find the negative x point closest to the bottom center
    #scaleFactor = newWidth/originalWidth

    starting_index, ending_index = find_starting_ending_points_for_inside(PM_Mid)

#    #########
#    # Get nodes for the steart and end of the inner arch of the PM_Mid
#    starting_node = PM_Mid.node(starting_index)
#    ending_node = PM_Mid.node(ending_index)
#    #########


    innerNodes = findInnerNodes(PM_Mid, connections, starting_index, ending_index)
    
    ###########################################
# Adding to figure out how to correctly curve PARA

#    print('************************ STARTING HERE  ************************')    
#    for node_number in innerNodes:
#        print(node_number)
#        print(PM_Mid.zAxis[node_number])
#    print(PM_Mid.zAxis[:])
#    print(np.mean(PM_Mid.zAxis[:]))
#    print(np.min(PM_Mid.zAxis[:]))
#    print(np.max(PM_Mid.zAxis[:]))
#    print(np.ptp(PM_Mid.zAxis[:]))
    
    inner_node_x_coords = []
    inner_node_y_coords = []
    inner_node_z_coords = []
    
    for node_number in innerNodes:
        inner_node_x_coords.append(PM_Mid.xAxis[node_number])
        inner_node_y_coords.append(PM_Mid.yAxis[node_number])
        inner_node_z_coords.append(PM_Mid.zAxis[node_number])
        #            print('PM Node:', PM_Mid.xAxis[node_number])
    
    PM_Z_mean = np.mean(inner_node_z_coords)
#    PM_Z_min =  np.min(inner_node_z_coords)
#    PM_Z_max =  np.max(inner_node_z_coords)
########################################################

    
    
################################## default curve fiber below ###################
    #Part_Name1 = AVW, Part_Name2 = the fiber tissue
    # Getting the coodinates for the AVW in the correct form from the file being worked on
    FILE_NAME = inputFile
    AVWpoints = np.array(io.extractPointsForPartFrom(FILE_NAME, "OPAL325_AVW_v6"))
    AVW_surface = DataSet3d(list(AVWpoints[:, 0]), list(AVWpoints[:, 1]), list(AVWpoints[:, 2]))

    nodes, connections = io.extractPointsForPartFrom2(FILE_NAME, Part_Name2, get_connections=True) #nodes and their connections to each other
    
    # what gets returned? Are the AVW node or Fiber nodes?
    # Result is something like {7: 10, 1: 11, 14: 13, 16: 15, 5: 17, 3: 19, 4: 25, 8: 12, 9: 14, 10: 16, 11: 18, 12: 20}
    # With the first number being the Fiber node and the second being the AVW node
    con_from_1_to_2 = get_connections_for_tissues(Part_Name2, Part_Name1, FILE_NAME) # connections to the surface 
#    print("Testing this = ", con_from_1_to_2 )
    ct = ConnectingTissue(nodes, connections, con_from_1_to_2)
    ct2 = ConnectingTissue(nodes, connections, con_from_1_to_2)    

#    if(Part_Name2 == "OPAL325_Para_v6"):
#        
#       plot_Dataset(ct2.get_starts_of_fibers(), ct2.get_ends_of_fibers())

#   The keys might be the node numbers
    fibers = ct.fibers_keys
#    print("Fibers = ", fibers)
    
    z_mid_coord = PM_Z_mean
    for i, fiber in enumerate(fibers): #loop through each fiber
        
#        print('Fiber ', i)
        starting_node_index = ct.starting_nodes[i] # getting indexes of nodes from the i fiber
        ending_node_index = ct.ending_nodes[i]
        
        starting_p = ct.node(starting_node_index) # getting nodes from the index number
        ending_p = ct.node(ending_node_index)
                    
        # I believe the AVW node number that corresponds to where the fiber connects to the AVW
        avw_node_number = ct.avw_connections[ending_node_index]
        # node (coordiantes) where the fiber connects to the AVW
        avw_node = AVW_surface.node(avw_node_number)


        #Find the length of the origianl fiber
        OriginalFiberLength = 0
        
        #
#        min_Z_dist_to_PM = 999999

#        Get the original fiber length and the fiber node that is close to PM_mid
#        print("New Fiber")
        for j, NodeNumber in enumerate(fiber[:-1]): #loop through each node in the fiber except the last
        
            p = ct.node(NodeNumber)
            q = ct.node(fiber[j+1])


############################################################################
##            Below was done to try and find the node on the fiber that is closest to the PM Mid
##            However, this was still using the original fiber location, not the current one
#            print(min_Z_dist_to_PM)
##           Find the node that is next to the PM inner ring
#            if min_Z_dist_to_PM > abs(PM_Z_mean - p.z):
#                middle_node = NodeNumber
#                min_Z_dist_to_PM = abs(PM_Z_mean - p.z)
#                print('Min Dist Changed')
#                y_mid_coord = ct.node(middle_node).y
###############################################################################    
            # calculating distance between each node
            Distance = math.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2) #distance formula: there is some trouble with node index when it tries to calculate. Possibly caused by q variable
            OriginalFiberLength = OriginalFiberLength + Distance
        

#        Get a line for the fiber going from starting point to ending point
#        print('Starting point:', starting_p)
#        print('Starting point:', ending_p)
        
        new_starting_p = sympy.Point3D(starting_p.x,starting_p.y,starting_p.z)
        new_ending_p = sympy.Point3D(ending_p.x,ending_p.y,ending_p.z)

#        print('Starting point:', new_starting_p)
#        print('Starting point:', new_ending_p)        

        fiber_line = sympy.Line3D(new_starting_p, new_ending_p)
#        print('Fiber line:', fiber_line)
#        Check each inner node to see which one is the closest
#        Continue on with code



        min_dist_to_middle_node = 9999999
        
#        print('1')
#        Find the PM inner ring node that is closest to where the PARA passes through
        for node_number in innerNodes:
            
            if np.sign(PM_Mid.xAxis[node_number]) == np.sign(starting_p.x):
    #            print('PM Node:', PM_Mid.xAxis[node_number])
                PM_node = sympy.Point3D(PM_Mid.xAxis[node_number],PM_Mid.yAxis[node_number],PM_Mid.zAxis[node_number])
    #            print('PM Node:', PM_node)
                dist_from_fiber_line = fiber_line.distance(PM_node)
    #            print('Distance from line: ', dist_from_fiber_line)
    #            If they both have positive or both negative x coordinates AND the distance between their y coordinates is less than the previous minimum
    #            if ct.node(middle_node).x * PM_Mid.xAxis[node_number] > 0 and min_dist_to_middle_node > abs(PM_Mid.yAxis[node_number] - y_mid_coord):
                if min_dist_to_middle_node > abs(dist_from_fiber_line):
                
                    x_mid_coord = PM_Mid.xAxis[node_number]
                    y_mid_coord = PM_Mid.xAxis[node_number]
                    min_dist_to_middle_node = abs(dist_from_fiber_line)
                    z_mid_coord = PM_Mid.zAxis[node_number]
    #                min_dist_to_middle_node = math.sqrt((PM_Mid.xAxis[node_number] - ct.node(middle_node).x) ** 2 + (PM_Mid.yAxis[node_number] - ct.node(middle_node).x.y) ** 2)
        
#        Move the X coordinate in 2 mm to avoid PM_mid (sign is there to move
#        it positive if it is a negative number and negative if it is a positive number)
################# I think this is the location of the (or a) problem for fibers
#                that don't go through the PM Mid (stay on the proximal side)
#        print(g)
#        g += 1
#        print('X mid before: ', x_mid_coord)
        x_mid_coord += -2*np.sign(x_mid_coord)
#        print('X mid after: ', x_mid_coord)
#        print('Route it through here: ', x_mid_coord)
#        print('Instead of here: ',ct.node(middle_node).x)
        
        # Check to see if the cable is already going through the hole OR if the attachment point is before the PM mid
#        if abs(x_mid_coord) > abs(ct.node(middle_node).x) or avw_node.z > z_mid_coord:
        if avw_node.z > z_mid_coord:
            Adjust_PARA = 0
#            print("Comparing: ", abs(x_mid_coord), " to ", abs(ct.node(middle_node).x))
#            print("AND: ", avw_node.z, " to ", PM_Mid.zAxis[node_number])
#            print('$$$$$$$$$$$$$$$$$$$ DON"T WORRY ABOUT IT$$$$$$$$$$$$$$$$$$$$')
        else:
            Adjust_PARA = 1
#            print('@@@@@@@@@@@@@@@@@ WORRY ABOUT IT$$$$$$$$$$$$$$$$$$$$')        
        
#        print("Original Fiber Length = ", OriginalFiberLength)

# Percentage of normal length to make the new tissue (over 100%)
        IdealFiberLength = OriginalFiberLength*(scale + 1)
#        print("Ideal Fiber Length = ", IdealFiberLength)        
        
        StartingAmplitude = 1

        NodesPerCycle = 15
        NumberOfCycles = math.floor(len(fiber)/NodesPerCycle)
        new_ending_p = avw_node
#        print("Adjust it?", Adjust_PARA)
        
        z_mid_coord += 2
        
        if Adjust_PARA == 0:
            _dist = math.sqrt((starting_p.x - new_ending_p.x) ** 2 + (starting_p.y - new_ending_p.y) ** 2 + (starting_p.z - new_ending_p.z) ** 2)
        else:
            _first_dist = math.sqrt((starting_p.x - x_mid_coord) ** 2 + (starting_p.y - y_mid_coord) ** 2 + (starting_p.z - z_mid_coord) ** 2)
            _second_dist = math.sqrt((x_mid_coord - new_ending_p.x) ** 2 + (y_mid_coord- new_ending_p.y) ** 2 + (z_mid_coord - new_ending_p.z) ** 2)
            _dist = _first_dist + _second_dist
#        print("AVW Node Coordinates: ", avw_node)
#        print("AVW Node Number: ", avw_node_number)
#        new_starting_p = avw_node
#        _dist = math.sqrt((new_starting_p.x - ending_p.x) ** 2 + (new_starting_p.y - ending_p.y) ** 2 + (new_starting_p.z - ending_p.z) ** 2)

#        _dist = math.sqrt((starting_p.x - ending_p.x) ** 2 + (starting_p.y - ending_p.y) ** 2 + (starting_p.z - ending_p.z) ** 2)

#avw_node
#        print("NumberOfCycles = ",NumberOfCycles)
#        print("Dist = ", _dist)
#        print('2')
        if IdealFiberLength > _dist:
            sending = (IdealFiberLength, fiber, starting_node_index, ending_node_index, ct, dirVector,NumberOfCycles,avw_node)
            [CorrectAmp] = fsolve(ArcDistance, StartingAmplitude, args=sending)

        else:
            [CorrectAmp] = [0]
#        print("Correct Amp is = ", CorrectAmp)
        starting_p = ct.node(starting_node_index) # getting nodes from the index number
        ending_p = ct.node(ending_node_index)
                
        
        minY = starting_p.y # min and max y values from start and end nodes
        maxY = ending_p.y

        
        
        ##################################################################
        OldXRange = ending_p.x-starting_p.x
        OldYRange = ending_p.y-starting_p.y
        OldZRange = ending_p.z-starting_p.z
        
        ########################## Set the ending node to the correct location
        
        NewXRange = avw_node.x - starting_p.x
        NewYRange = avw_node.y - starting_p.y
        NewZRange = avw_node.z - starting_p.z
        
        
        if Adjust_PARA == 1:
#            first_NewXRange = avw_node.x - x_mid_coord # X (horizontal?) space that the points between the AVW and the PM Mid must cover
#            second_NewXRange = x_mid_coord - starting_p.x # covered by PM mid to insertion
            
            ####### Below is probably correct, just switching to see if it is wrong
            first_NewXRange = x_mid_coord - starting_p.x  # covered by PM mid to insertion
            second_NewXRange = avw_node.x - x_mid_coord # X (horizontal?) space that the points between the AVW and the PM Mid must cover
            NewXRange = first_NewXRange - second_NewXRange # combining them for total distance covered (can be negative)
#            print(first_NewXRange,second_NewXRange,NewXRange)

            first_NewZRange = z_mid_coord - starting_p.z # covered by PM mid to insertion
            second_NewZRange = avw_node.z - z_mid_coord  # X (horizontal?) space that the points between the AVW and the PM Mid must cover
            NewZRange = first_NewZRange + second_NewZRange # combining them for total distance covered (can be negative)
#            print(first_NewZRange,second_NewZRange,NewZRange)

#        print('3')
        FinalFiberLength = 0
        for j, NodeNumber in enumerate(fiber): #loop through each node in the fiber except the last to find the correct coordinates
    
            p = ct.node(NodeNumber)
            
            # I don't think this is correct now; below I am moving the x points which would move the y points...
            # how does this work normally. It seems like that would be wrong too.
            RangedpY = NumberOfCycles*(p.y - minY) / (maxY - minY) * math.pi # Set a y range betweeen 0 and PI
            if Adjust_PARA == 0:
#                print("Using the old equation to generate the point")
                NewX = np.sign(p.x) * dirVector[0]* CorrectAmp * math.sin(RangedpY) + (p.x - starting_p.x) * NewXRange / OldXRange + starting_p.x # different so that it can be done to curve inwards from both sides of the AVW
                NewY = dirVector[1]* CorrectAmp * math.sin(RangedpY) +  (p.y - starting_p.y) * NewYRange / OldYRange + starting_p.y
    #           The wiggle + how far away was the point from the original end * the ratio of new range / old range and then add in the starting position
                NewZ = dirVector[2]* CorrectAmp * math.sin(RangedpY) +  (p.z - starting_p.z) * NewZRange / OldZRange + starting_p.z

            else:
#                print("Using the new equation to generate the point")
                
                # if the X distance from this poin to the insertion divided by the old XRange is < the first distance fraction
                # so basically if the point is in the first section (determined by comparing the old X distance ratio to new X ratio)
                OldFraction = abs(p.x-starting_p.x)/OldXRange
#                print("Old Fraction: ", OldFraction)
#                print("YRange : ",NewYRange)
#                print("Old Test: ", abs((p.x - starting_p.x) / OldXRange), abs(_first_dist / _dist), "   New Test: ", OldFraction, _first_dist/_dist)
                if abs((p.x - starting_p.x) / OldXRange) < abs(_first_dist / _dist):
#                    NewX = np.sign(p.x) * dirVector[0]* CorrectAmp * math.sin(RangedpY) + (p.x - starting_p.x) * NewXRange / OldXRange + starting_p.x
                    NewX = np.sign(p.x) * dirVector[0]* CorrectAmp * math.sin(RangedpY) + ((p.x - starting_p.x) / OldXRange)/abs(_first_dist/_dist)*first_NewXRange + starting_p.x

                    NewFraction = abs(_dist/_first_dist * OldFraction)
#                    NewerX = starting_p.x + first_NewXRange*NewFraction
                    
                    NewY = dirVector[1]* CorrectAmp * math.sin(RangedpY) +  abs(OldFraction) * NewYRange + starting_p.y
    #           The wiggle + how far away was the point from the original end * the ratio of new range / old range and then add in the starting position
                    NewZ = dirVector[2]* CorrectAmp * math.sin(RangedpY) +  NewFraction * first_NewZRange + starting_p.z
#                    print("1st NewFraction: ", NewFraction)
#                    print("1st New Z Range: ", first_NewZRange)
#                    print("Two Xs and a Z", NewX,NewerX,NewZ)
#                    print("New Coords for 1st part", NewX, NewY, NewZ)
                else:
#                    b = 1/(1-(_dist/_first_dist))
#                    m = -1 * b * _dist / _first_dist
#                    NewFraction = m * OldFraction + b
#                    print("Newer Fraction: ", NewFraction)
                    NewFraction = ((p.x - starting_p.x) / OldXRange - abs(_first_dist/_dist))/abs(_second_dist/_dist) # hopefully this is true
#                    NewX = np.sign(p.x) * dirVector[0]* CorrectAmp * math.sin(RangedpY) + ((p.x - starting_p.x) / OldXRange)/(_first_dist/_dist)*first_NewXRange + starting_p.x
#                    print("NewFraction: ", NewFraction)
                    NewX = np.sign(p.x) * dirVector[0]* CorrectAmp * math.sin(RangedpY) + ((p.x - starting_p.x) / OldXRange - abs(_first_dist/_dist))/abs(_second_dist/_dist) * second_NewXRange + x_mid_coord
#                    print('percent along original path: ',(p.x - starting_p.x) / OldXRange, 'Amount that 1st part is: ', abs(_first_dist/_dist), 'Amount left: ', (p.x - starting_p.x) / OldXRange - abs(_first_dist/_dist), 'Fraction second part is to whole: ', abs(_second_dist/_dist), 'Fraction of second part :', ((p.x - starting_p.x) / OldXRange - abs(_first_dist/_dist))/abs(_second_dist/_dist))
                    NewY = dirVector[1]* CorrectAmp * math.sin(RangedpY) +  abs(OldFraction) * NewYRange + starting_p.y
    #           The wiggle + how far away was the point from the original end * the ratio of new range / old range and then add in the starting position
                    NewZ = dirVector[2]* CorrectAmp * math.sin(RangedpY) +  NewFraction * second_NewZRange + z_mid_coord
#                    print("2nd NewFraction : ", NewFraction)
#                    print("2nd New Z Range: ", second_NewZRange)
#                    print("New Coords for 2nd part", NewX, NewY, NewZ)
#            NewY = dirVector[1]* CorrectAmp * math.sin(RangedpY) +  (p.y - starting_p.y) * NewYRange / OldYRange + starting_p.y
##           The wiggle + how far away was the point from the original end * the ratio of new range / old range and then add in the starting position
#            NewZ = dirVector[2]* CorrectAmp * math.sin(RangedpY) +  (p.z - starting_p.z) * NewZRange / OldZRange + starting_p.z
                    
                    
            new_p = Point(NewX, NewY, NewZ)
            p = new_p
            if j > 0:
                Distance = math.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2) #distance formula: there is some trouble with node index when it tries to calculate. Possibly caused by q variable
                FinalFiberLength = FinalFiberLength + Distance   
            q = new_p

            ct.update_node(NodeNumber, new_p)
#        print("Final Fiber Length = ", FinalFiberLength)
            

#        print('4')
        
        if avw_node.z > z_mid_coord:
            Adjust_PARA = 0
##            print("Comparing: ", abs(x_mid_coord), " to ", abs(ct.node(middle_node).x))
##            print("AND: ", avw_node.z, " to ", PM_Mid.zAxis[node_number])
##            print('$$$$$$$$$$$$$$$$$$$ DON"T WORRY ABOUT IT$$$$$$$$$$$$$$$$$$$$')
        else:
            Adjust_PARA = 1
##            print('@@@@@@@@@@@@@@@@@ WORRY ABOUT IT$$$$$$$$$$$$$$$$$$$$')        
########################### FINDING THE MID POINT AGAIN USING THE CURRENT
            ############### FIBER NODES AND THEN DOING THINGS AGAIN
        if Adjust_PARA == 1:
            min_dist_from_fiber_line = 999999
            second_min_dist_from_fiber_line = 999999
            x_mid_coord = 0
            for j, NodeNumber in enumerate(fiber): #loop through each node in the fiber except the last to find the correct coordinates
        
                p = ct.node(NodeNumber)
#                print(p.z, z_mid_coord)
                if abs(p.z - z_mid_coord) < 2:
#                    print("Inside Loop 1")
                    for node_number in innerNodes:
                        if np.sign(PM_Mid.xAxis[node_number]) == np.sign(p.x):
#                            print(node_number)
                #            print('PM Node:', PM_Mid.xAxis[node_number])
                            PM_node = sympy.Point3D(PM_Mid.xAxis[node_number],PM_Mid.yAxis[node_number],PM_Mid.zAxis[node_number])
                #            print('PM Node:', PM_node)
                            dist_from_fiber_line = math.sqrt((p.x - PM_node.x) ** 2 + (p.y - PM_node.y) ** 2 + (p.z - PM_node.z) ** 2) #distance formula: there is some trouble with node index when it tries to calculate. Possibly caused by q variable
                            if dist_from_fiber_line < second_min_dist_from_fiber_line:
        #                        print("Inside Loop 2")
                                if dist_from_fiber_line < min_dist_from_fiber_line:     
                                    second_min_dist_from_fiber_line = min_dist_from_fiber_line                        
                                    min_dist_from_fiber_line = dist_from_fiber_line
                                    second_x_mid_coord = x_mid_coord
                                    x_mid_coord = PM_Mid.xAxis[node_number]
                                    y_mid_coord = PM_Mid.xAxis[node_number]
                                    z_mid_coord = PM_Mid.zAxis[node_number]
                                else:
                                    second_min_dist_from_fiber_line = dist_from_fiber_line
                                    second_x_mid_coord = PM_Mid.xAxis[node_number]
        #                        print(min_dist_from_fiber_line)
                            
            x_mid_coord = np.sign(x_mid_coord)*min(abs(x_mid_coord), abs(second_x_mid_coord))
    #        print('X mid before: ', x_mid_coord)
            x_mid_coord += -2*np.sign(x_mid_coord)
    #        print('X mid after: ', x_mid_coord)
    #        print('Route it through here: ', x_mid_coord)
    #        print('Instead of here: ',ct.node(middle_node).x)
            
        
#####################^^^^^^^^^^^^^^^^^ I was testing to see if a fiber needed to be adjusted after I did everything
#        
#        # Check to see if the cable is already going through the hole OR if the attachment point is before the PM mid
##        if abs(x_mid_coord) > abs(ct.node(middle_node).x) or avw_node.z > z_mid_coord:
#        if avw_node.z > z_mid_coord:
#            Adjust_PARA = 0
##            print("Comparing: ", abs(x_mid_coord), " to ", abs(ct.node(middle_node).x))
##            print("AND: ", avw_node.z, " to ", PM_Mid.zAxis[node_number])
##            print('$$$$$$$$$$$$$$$$$$$ DON"T WORRY ABOUT IT$$$$$$$$$$$$$$$$$$$$')
#        else:
#            Adjust_PARA = 1
##            print('@@@@@@@@@@@@@@@@@ WORRY ABOUT IT$$$$$$$$$$$$$$$$$$$$')        
#####################^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        
#        print("Original Fiber Length = ", OriginalFiberLength)

# Percentage of normal length to make the new tissue (over 100%)
        IdealFiberLength = OriginalFiberLength*(scale + 1)
#        print("Ideal Fiber Length = ", IdealFiberLength)        
        
        StartingAmplitude = 1

        NodesPerCycle = 15
        NumberOfCycles = math.floor(len(fiber)/NodesPerCycle)
        new_ending_p = avw_node
#        print("Adjust it?", Adjust_PARA)
        
        z_mid_coord += 2
#        print('5')
        if Adjust_PARA == 0:
            _dist = math.sqrt((starting_p.x - new_ending_p.x) ** 2 + (starting_p.y - new_ending_p.y) ** 2 + (starting_p.z - new_ending_p.z) ** 2)
        else:
            _first_dist = math.sqrt((starting_p.x - x_mid_coord) ** 2 + (starting_p.y - y_mid_coord) ** 2 + (starting_p.z - z_mid_coord) ** 2)
            _second_dist = math.sqrt((x_mid_coord - new_ending_p.x) ** 2 + (y_mid_coord- new_ending_p.y) ** 2 + (z_mid_coord - new_ending_p.z) ** 2)
            _dist = _first_dist + _second_dist
#        print("AVW Node Coordinates: ", avw_node)
#        print("AVW Node Number: ", avw_node_number)
#        new_starting_p = avw_node
#        _dist = math.sqrt((new_starting_p.x - ending_p.x) ** 2 + (new_starting_p.y - ending_p.y) ** 2 + (new_starting_p.z - ending_p.z) ** 2)

#        _dist = math.sqrt((starting_p.x - ending_p.x) ** 2 + (starting_p.y - ending_p.y) ** 2 + (starting_p.z - ending_p.z) ** 2)

#avw_node
#        print("NumberOfCycles = ",NumberOfCycles)
#        print("Dist = ", _dist)
        if IdealFiberLength > _dist:
            sending = (IdealFiberLength, fiber, starting_node_index, ending_node_index, ct2, dirVector,NumberOfCycles,avw_node)
            [CorrectAmp] = fsolve(ArcDistance, StartingAmplitude, args=sending)

        else:
            [CorrectAmp] = [0]
#        print("Correct Amp is = ", CorrectAmp)
        starting_p = ct2.node(starting_node_index) # getting nodes from the index number
        ending_p = ct2.node(ending_node_index)
                
        
        minY = starting_p.y # min and max y values from start and end nodes
        maxY = ending_p.y

        
        
#        ##################################################################
#        OldXRange = ending_p.x-starting_p.x
#        OldYRange = ending_p.y-starting_p.y
#        OldZRange = ending_p.z-starting_p.z
#        
#        ########################## Set the ending node to the correct location
        
        NewXRange = avw_node.x - starting_p.x
        NewYRange = avw_node.y - starting_p.y
        NewZRange = avw_node.z - starting_p.z
        
        
        if Adjust_PARA == 1:
#            first_NewXRange = avw_node.x - x_mid_coord # X (horizontal?) space that the points between the AVW and the PM Mid must cover
#            second_NewXRange = x_mid_coord - starting_p.x # covered by PM mid to insertion
            
            ####### Below is probably correct, just switching to see if it is wrong
            first_NewXRange = x_mid_coord - starting_p.x  # covered by PM mid to insertion
            second_NewXRange = avw_node.x - x_mid_coord # X (horizontal?) space that the points between the AVW and the PM Mid must cover
#            NewXRange = first_NewXRange - second_NewXRange # combining them for total distance covered (can be negative)
            NewXRange = abs(first_NewXRange) + abs(second_NewXRange) # combining them for total distance covered (can be negative)
#            print(first_NewXRange,second_NewXRange,NewXRange)

            first_NewZRange = z_mid_coord - starting_p.z # covered by PM mid to insertion
            second_NewZRange = avw_node.z - z_mid_coord  # X (horizontal?) space that the points between the AVW and the PM Mid must cover
            NewZRange = first_NewZRange + second_NewZRange # combining them for total distance covered (can be negative)
#            print(first_NewZRange,second_NewZRange,NewZRange)

#        print('6')
        FinalFiberLength = 0
        for j, NodeNumber in enumerate(fiber): #loop through each node in the fiber except the last to find the correct coordinates
    
            p = ct2.node(NodeNumber)
            
            # I don't think this is correct now; below I am moving the x points which would move the y points...
            # how does this work normally. It seems like that would be wrong too.
            RangedpY = NumberOfCycles*(p.y - minY) / (maxY - minY) * math.pi # Set a y range betweeen 0 and PI
            if Adjust_PARA == 0:
#                print("Using the old equation to generate the point")
                NewX = np.sign(p.x) * dirVector[0]* CorrectAmp * math.sin(RangedpY) + (p.x - starting_p.x) * NewXRange / OldXRange + starting_p.x # different so that it can be done to curve inwards from both sides of the AVW
                NewY = dirVector[1]* CorrectAmp * math.sin(RangedpY) +  (p.y - starting_p.y) * NewYRange / OldYRange + starting_p.y
    #           The wiggle + how far away was the point from the original end * the ratio of new range / old range and then add in the starting position
                NewZ = dirVector[2]* CorrectAmp * math.sin(RangedpY) +  (p.z - starting_p.z) * NewZRange / OldZRange + starting_p.z

            else:
#                print("Using the new equation to generate the point")
                
                # if the X distance from this poin to the insertion divided by the old XRange is < the first distance fraction
                # so basically if the point is in the first section (determined by comparing the old X distance ratio to new X ratio)
                OldFraction = abs(p.x-starting_p.x)/OldXRange
#                print("Old Fraction: ", OldFraction)
#                print("YRange : ",NewYRange)
#                print("Old Test: ", abs((p.x - starting_p.x) / OldXRange), abs(_first_dist / _dist), "   New Test: ", OldFraction, _first_dist/_dist)
#                If the point is in the fraction that would be before the PM Mid
                if abs((p.x - starting_p.x) / OldXRange) < abs(_first_dist / _dist):
#                    NewX = np.sign(p.x) * dirVector[0]* CorrectAmp * math.sin(RangedpY) + (p.x - starting_p.x) * NewXRange / OldXRange + starting_p.x
                    NewX = np.sign(p.x) * dirVector[0]* CorrectAmp * math.sin(RangedpY) + ((p.x - starting_p.x) / OldXRange)/abs(_first_dist/_dist)*first_NewXRange + starting_p.x

                    NewFraction = abs(_dist/_first_dist * OldFraction)
#                    NewerX is the same as NewX
#                    NewerX = starting_p.x + first_NewXRange*NewFraction
#                    if x_mid_coord > 0:
#                        print(NewX)
#                        print(NewerX)
                        
                    NewY = dirVector[1]* CorrectAmp * math.sin(RangedpY) +  abs(OldFraction) * NewYRange + starting_p.y
    #           The wiggle + how far away was the point from the original end * the ratio of new range / old range and then add in the starting position
                    NewZ = dirVector[2]* CorrectAmp * math.sin(RangedpY) +  NewFraction * first_NewZRange + starting_p.z
#                    print("1st NewFraction: ", NewFraction)
#                    print("1st New Z Range: ", first_NewZRange)
#                    print("Two Xs and a Z", NewX,NewerX,NewZ)
#                    print("New Coords for 1st part", NewX, NewY, NewZ)
                    
#                    if g == 5 or g==6:
#                    if x_mid_coord > 0:
#                        print("Point from 1st part: ", NewX, NewY, NewZ)
                        
                else:
#                    b = 1/(1-(_dist/_first_dist))
#                    m = -1 * b * _dist / _first_dist
#                    NewFraction = m * OldFraction + b
#                    print("Newer Fraction: ", NewFraction)
                    NewFraction = ((p.x - starting_p.x) / OldXRange - abs(_first_dist/_dist))/abs(_second_dist/_dist) # hopefully this is true
#                    NewX = np.sign(p.x) * dirVector[0]* CorrectAmp * math.sin(RangedpY) + ((p.x - starting_p.x) / OldXRange)/(_first_dist/_dist)*first_NewXRange + starting_p.x
#                    print("NewFraction: ", NewFraction)
                    NewX = np.sign(p.x) * dirVector[0]* CorrectAmp * math.sin(RangedpY) + ((p.x - starting_p.x) / OldXRange - abs(_first_dist/_dist))/abs(_second_dist/_dist) * second_NewXRange + x_mid_coord
#                    print('percent along original path: ',(p.x - starting_p.x) / OldXRange, 'Amount that 1st part is: ', abs(_first_dist/_dist), 'Amount left: ', (p.x - starting_p.x) / OldXRange - abs(_first_dist/_dist), 'Fraction second part is to whole: ', abs(_second_dist/_dist), 'Fraction of second part :', ((p.x - starting_p.x) / OldXRange - abs(_first_dist/_dist))/abs(_second_dist/_dist))
                    NewY = dirVector[1]* CorrectAmp * math.sin(RangedpY) +  abs(OldFraction) * NewYRange + starting_p.y
    #           The wiggle + how far away was the point from the original end * the ratio of new range / old range and then add in the starting position
                    NewZ = dirVector[2]* CorrectAmp * math.sin(RangedpY) +  NewFraction * second_NewZRange + z_mid_coord
#                    print("2nd NewFraction : ", NewFraction)
#                    print("2nd New Z Range: ", second_NewZRange)
#                    print("New Coords for 2nd part", NewX, NewY, NewZ)
#            NewY = dirVector[1]* CorrectAmp * math.sin(RangedpY) +  (p.y - starting_p.y) * NewYRange / OldYRange + starting_p.y
##           The wiggle + how far away was the point from the original end * the ratio of new range / old range and then add in the starting position
#            NewZ = dirVector[2]* CorrectAmp * math.sin(RangedpY) +  (p.z - starting_p.z) * NewZRange / OldZRange + starting_p.z
#                    if g == 5 or g == 6:
#                        print("Point from 2nd part: ", NewX, NewY, NewZ)
#                    if x_mid_coord > 0:
#                        print("Point from 2nd part: ", NewX, NewY, NewZ)
            new_p = Point(NewX, NewY, NewZ)
            p = new_p
            if j > 0:
                Distance = math.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2) #distance formula: there is some trouble with node index when it tries to calculate. Possibly caused by q variable
                FinalFiberLength = FinalFiberLength + Distance   
            q = new_p

            ct.update_node(NodeNumber, new_p)
#        print("Final Fiber Length = ", FinalFiberLength)
#        print('7')   
            
######################## END OF 2ND RUN THROUGH
        
    print('Para Function, Output File: ', outputFile)
    write_part_to_inp(inputFile, outputFile, Part_Name2, ct)
    return

def update_ligament(updated_Neg_P, updated_Pos_P, start_p):
    if np.sign(start_p.x) < 0 and updated_Neg_P is not None:
        return updated_Neg_P
    elif np.sign(start_p.x) > 0 and updated_Pos_P is not None:
        return updated_Pos_P
    else:
        return start_p
