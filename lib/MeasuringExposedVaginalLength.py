#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:30:04 2017

@author: AaronR
"""

from scipy import interpolate
import numpy as np
import lib.ScalingFEM_Simplified as ss
from lib.workingWith3dDataSets import DataSet3d, Point
from lib.AVW_Measurements import getInitialPositions, pythag, getFEAData

'''
Script: MeasuringExposedVaginalLength.py
'''

#def measureExposedVaginalLength(AVW, GI_Filler)
def main():

    # This is hard coded because it is should be known
    # It should be passed if you want to make this reusable

    # Location of apical midline point on AVW (y,z)
    #start = (-11.91986417, 9.2637553) 
    #start = (-4.4, 17.5)


    #TODO - automatically get these points

    start = (-4.4, 17.5)
    # separation point between GI and AVW along mindline (y,z)    
    GI_Filler_End = (-21.14,-35.39)
    
    #OriginalFile = 'CL100_PARA100_PCM100_AVW100_CLSS0_Len1_CLStrain0_HL32_data.csv'
    OriginalFile = 'UnilateralCL100_PARA100_PCM50_AVW100_CLSS0_Len1_CLStrain0_HL38_data.csv'

    THRESHOLD = 1 # Max Distance point can be from X axis

    # Get the final dispacements from the OriginalFile (has 309 points)

    #TODO: make function to get total number of points
    #
    #length =     
    #

    (OriginalDeltaX, OriginalDeltaY, OriginalDeltaZ) = getFEAData(OriginalFile,309)

    # Get the intial positions of the coordinates
    (InitialX,InitialY,InitialZ) = getInitialPositions('InitialPositions.csv',309)


    # still unsure if this works -ab
    # Calculate the final positions of the points
    xs =InitialX+OriginalDeltaX
    ys =InitialY+OriginalDeltaY
    zs =InitialZ+OriginalDeltaZ
    
    L = []
    for indx, xval in enumerate(xs):
        #if (abs(0 - xval) < THRESHOLD):
        if abs(xval) < THRESHOLD:
            L.append((ys[indx], zs[indx]))
    
    
    ## Setting up for Sorting the spline   
    spline_ordered = [start]
    
    count = len(L)
    
    t=[0]
    used = []
    ## Sorting the spline 
    for i in range(0,count):
        
        least_dist = float("inf")
        least_indx = 0
        for j , point in enumerate(L):
            if j not in used:
                dist = pythag(spline_ordered[-1][0] - point[0], spline_ordered[-1][1] - point[1])
                if dist < least_dist and j not in used:
                    least_dist = dist
                    least_indx = j
        t.append(least_dist+t[-1])    
        spline_ordered.append(L[least_indx])
        used.append(least_indx)
    
    
    # bacause the starting point that was hard coded was not removed
    # from the group of points in L it found itself to be the next closest 
    # point, AKA its in spline_ordered twice so I remove the first one
#    spline_ordered.pop(0)  # 
    
    ## END Sorting the spline 
    X = np.array(spline_ordered).T # -> transpose
    
#    t = list(range(0, count))   
    # https://stackoverflow.com/questions/12798695/spline-of-curve-python
    f = interpolate.interp1d(t, X)
    
    ## Set up for getting the length
    new_vals = [[],[]]
    steps = np.arange(0,t[-1], .001)
    dist = 0
    last_val = None
    
#
### Find the step that is the closest to the GI Point
    min_dist = 999999999    
    for i, step in enumerate(steps):
        val = f(step)
        dist = pythag(GI_Filler_End[0] - val[0], GI_Filler_End[1] - val[1])
#        print(dist)
        if dist < min_dist:
            min_dist = dist
            beststep = step
            bestindex = i
    
    dist = 0            
    ## summing the length
    for i, step in enumerate(steps[bestindex:-1]):
        if i == 0:
            last_val = f(step)
            continue
        val = f(step)
        new_vals[0].append(val[0])
        new_vals[1].append(val[1])
        dist += pythag(last_val[0] - val[0], last_val[1] - val[1])
        last_val = val
        
    
    print("Distance,", dist)
    surface = DataSet3d(xs, ys, zs)
    
    ss.plot_Dataset(surface, DataSet3d([0] * len(new_vals[0]), new_vals[0], new_vals[1]))
    return dist