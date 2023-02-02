#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:30:04 2017

@author: AaronR
"""

import numpy as np

'''
Function: getFEAData
'''
def getFEAData(FileName,nodes):
    csv = np.genfromtxt (FileName, delimiter=",")
#    print(csv)
#    [rows,columns] = csv.shape
#    print(rows)
#    print(columns)
    x=np.zeros(nodes)
    y=np.zeros(nodes)
    z=np.zeros(nodes)
    for i in range(0,nodes):
        x[i]=csv[i*3]
        y[i]=csv[i*3+1]
        z[i]=csv[i*3+2]
    return [x,y,z]

'''
Function: getInitialPositions
'''
def getInitialPositions(FileName,nodes):
    csv = np.genfromtxt (FileName, delimiter=",")
    x = np.array(csv[:,1])
    y = np.array(csv[:,2])
    z = np.array(csv[:,3])
    return [x,y,z]

'''
Function: pythag

Takes any number of parameters and pythags them
'''

def pythag(*args):
    squareSum = 0
    for arg in args:
        squareSum += arg**2
        
    return squareSum **0.5

##############################################

'''
Function: lowBoundIndex
'''
def lowBoundIndex(array, item):
    closestIndex = 0
    for i, num in enumerate(array):
        if num <= item:
            closestIndex = i
        else:
            break
    return closestIndex

'''
Function: yzDist
'''
def yzDist(origin, target):
    return ((origin.y-target.y)**2 + (origin.z-target.z)**2)**0.5

'''
Function: xyDist
'''
def xyDist(origin, target):
    return((origin.x-target.x)**2 + (origin.y-target.y)**2)**0.5

'''
Function: getCenterNodes
'''
def getCenterNodes(AVW, thresh):
    centerNodes = []
    for i, node in enumerate(AVW):
        if abs(node.x) < thresh:
            centerNodes.append(node)
    return centerNodes

'''
Function: findBottomMostIndex
'''
def findBottomMostIndex(centerNodes):
    largestZ = -999999 #"bottom" of the avw is in the positive z direction
    for i, node in enumerate(centerNodes):
        if node.z > largestZ:
            largestZ = node.z
            index = i

    return index

'''
Function: findClosestNodeIndex
'''
def findClosestNodeIndex(origin, centerNodes): #checks in the y and z coordinate only!
    MinDist = 999999
    for i, target in enumerate(centerNodes):
        dist = yzDist(origin, target)
        if dist < MinDist:
            MinDist = dist
            index = i

    return index

'''
Function: findClosestNodeIndex_Width
'''
def findClosestNodeIndex_Width(origin, sliceNodes): #checks in the x and y coordinate only!
    MinDist = 999999
    for i, target in enumerate(sliceNodes):
        dist = xyDist(origin, target)
        if dist < MinDist:
            MinDist = dist
            index = i

    return index

'''
Function: findRightMostIndex
'''
def findRightMostIndex(centerNodes):
    largestX = -999999
    for i, node in enumerate(centerNodes):
        if node.x > largestX:
            largestX = node.x
            index = i

    return index

'''
Function: getWidthSlice
'''
def getWidthSlice(AVW, thresh, zPos):
    sliceNodes = []
    
    zPlane = getAVW_z_from_percent(AVW, zPos)

    #zPlane = AVW[199-1].z
    #print("nodes")
    for i, node in enumerate(AVW):
        if abs(zPlane - node.z) < thresh:
            #print(i+1)
            sliceNodes.append(node)

    return sliceNodes

'''
Function: getAVWLength
'''
def getAVWLength(AVW):
    thresh = 2 #from the center, thresh*2 total strip
    centerNodes = getCenterNodes(AVW, thresh) #will be returned as an array of points

    bottomIndex = findBottomMostIndex(centerNodes)
    distance = 0

    origin = centerNodes[bottomIndex]
    centerNodes.pop(bottomIndex)


    while len(centerNodes) > 0:

        #find the closest new node
        closestNodeIndex = findClosestNodeIndex(origin, centerNodes)

        #add distance to the closest node
        distance += yzDist(origin, centerNodes[closestNodeIndex])

        #switch to new node and remove it
        origin = centerNodes[closestNodeIndex]
        centerNodes.pop(closestNodeIndex) #we don't want to use it anymore

    return distance

'''
Function: getAVWWidth
'''
def getAVWWidth(AVW):
    thresh = 2
    zPos = 0.2 #high z positions will not work
    nodeSlice = getWidthSlice(AVW, thresh, zPos)

    rightIndex = findRightMostIndex(nodeSlice)

    distance = 0

    origin = nodeSlice[rightIndex]
    #print(origin)
    #print(distance)
    nodeSlice.pop(rightIndex)
    xyz = 0

    while len(nodeSlice) > 0:

        #find the closest new node
        closestNodeIndex = findClosestNodeIndex_Width(origin, nodeSlice)
        #print(nodeSlice[closestNodeIndex])

        #add distance to the closest node
        distance += xyDist(origin, nodeSlice[closestNodeIndex])
        xyz += pythag((origin.x-nodeSlice[closestNodeIndex].x), (origin.y-nodeSlice[closestNodeIndex].y), (origin.z-nodeSlice[closestNodeIndex].z))
        #print(xyz)

        #switch to new node and remove it
        origin = nodeSlice[closestNodeIndex]
        nodeSlice.pop(closestNodeIndex) #we don't want to use it anymore

    return xyz

'''
Function: getAVW_z_from_percent
'''
def getAVW_z_from_percent(AVW, zPos):
    thresh = 2 #from the center, thresh*2 total strip
    centerNodes = getCenterNodes(AVW, thresh) #will be returned as an array of points

    bottomIndex = findBottomMostIndex(centerNodes)
    distance = 0

    origin = centerNodes[bottomIndex]
    orderedNodes = []
    orderedNodes.append(origin)

    centerNodes.pop(bottomIndex)

    distances = []

    while len(centerNodes) > 0:

        #find the closest new node
        closestNodeIndex = findClosestNodeIndex(origin, centerNodes)

        #add distance to the closest node
        distance += yzDist(origin, centerNodes[closestNodeIndex])
        distances.append(distance)
        #switch to new node and remove it
        origin = centerNodes[closestNodeIndex]
        orderedNodes.append(origin)
        centerNodes.pop(closestNodeIndex) #we don't want to use it anymore

    zPos = 0.2

    index = lowBoundIndex(distances, zPos*distance)
    firstNode = orderedNodes[index]

    return AVW[74-1].z

    try:
        secondNode = orderedNodes[index + 1]
    except IndexError:
        return firstNode.z


    if abs(distances[index] - distance*zPos) < abs (distances[index + 1] - distance*zPos):
        return firstNode.z
    return secondNode.z



    """
    try:
        slope = (secondNode.z-firstNode.z)/(distances[index + 1] - distances[index])
        intercept = firstNode.z - slope*distances[index]

        #a y=mx+b, x is distance and y is the desired z
        desired_z = slope*(zPos*distance) +intercept

    except IndexError: #its the last node...
        desired_z = orderedNodes[len(orderedNodes) - 1].z #just set it to the final point"""

    return z_coord