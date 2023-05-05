#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:52:57 2017
@author: Aaron Renfroe @ California Baptist University

"""
#### New functions loaded for new narrowing functions after sabbatical
from lib.Surface_Tools import pythag, plot_Dataset, find_starting_ending_points_for_inside, findInnerNodes
import math
######## #### #### #### #### #### 
import lib.Surface_Tools_Circle as ss
import numpy as np
from lib.workingWith3dDataSets import DataSet3d, Point
import lib.IOfunctions as io
from lib.ConnectingTissue import ConnectingTissue
from lib.AVW_Measurements import getAVWWidth, getAVWLength
from lib.FiberFunctions import getFiberLengths
from scipy.optimize import fsolve
from scipy import sqrt, sin, pi, interpolate, cos, integrate

'''
Function: widen_part
'''
def widen_part(part_name, file_name, scale):
    print("Widen Part: " + part_name)

    if (scale != 1):
        generated_surface = io.get_dataset_from_file(file_name, part_name).generatedCopy()
        unmodifiedLength = len(generated_surface.xAxis)

        generated_surface = ss.edges_to_try_and_correct_interpolation(generated_surface, generated_surface.generatedCopy())
    #======================================================
        new_XY_vals = ss.generate_values_for("x",generated_surface,scale, unmodifiedLength)
    #======================================================
#        print("old = ", generated_surface.xAxis)        
#        print("new = ", new_XY_vals[0])
        generated_surface.xAxis = new_XY_vals[0]
        generated_surface.yAxis = new_XY_vals[1]
        
        generated_surface = generated_surface.trim()
        # Write to File
       
        io.write_part_to_inp_file(file_name, part_name, generated_surface)
    
'''
Function: lengthen_and_shift_part
'''
def lengthen_and_shift_part(part_name, file_name, scale, shift):
    print("Lengthen and Shift Part: " + part_name)
#    Don't do anything if there's no scale or shift
    print(scale, shift)
    if (scale != 1 or shift != 0):
    
#        Get the data for the the part (typically the AVW)
        generated_surface = io.get_dataset_from_file(file_name, part_name).generatedCopy()
#       See how many nodes there are in the part
        unmodifiedLength = len(generated_surface.xAxis)


#        I beleive this adds points to the edges to make the interpolation better
#        Without it weird things can happen past the current data points
#        Past the data points matter when moving the nodes beyond the original boundaries
        generated_surface = ss.edges_to_try_and_correct_interpolation(generated_surface, generated_surface.generatedCopy())
        #======================================================
        #generate new Z Data       -- Z --        Adding Shift slides the surface down the given length. It is done here to shave 30 seconds off processing time
        newZYVals = ss.generate_values_for("z", generated_surface, scale, unmodifiedLength, shiftValue=shift)
        #======================================================
        generated_surface.zAxis = newZYVals[0]
        generated_surface.yAxis = newZYVals[1]
        generated_surface = generated_surface.trim()
        # Write to File
        
        io.write_part_to_inp_file(file_name, part_name, generated_surface)


'''
Function: curve_avw
'''
def curve_avw(part_name, file_name, original_file_name, hiatus, z_cutoff, rotate_angle, rot_point, HiatusLength):
    print("Curve and Rotate AVW")

    # Getting Nodes 
    generic_surface = io.get_dataset_from_file(original_file_name, part_name)
    generated_surface = io.get_dataset_from_file(file_name, part_name).generatedCopy()
    
    #Probably want better solution but prevents from AVW intersecting GIfiller
    #generated_surface = ss.corrective_shift(generated_surface, HiatusLength)

    #This does the curve
    generated_surface = ss.apply_curve_after_threshold(generic_surface, generated_surface, hiatus, z_cutoff, rotate_angle, rot_point)

    # Write to File
    io.write_part_to_inp_file(file_name, part_name, generated_surface)
    
    return generated_surface


'''
Function: rotate_part
'''
def rotate_part(part_name, file_name, rotate_angle, rot_point):
    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", part_name)
    generated_surface = io.get_dataset_from_file(file_name, part_name).generatedCopy()
    
#    generated_surface = ss.rotate_part(generated_surface, rotate_angle, rot_point)
    threshold = 20

    Xs = list()
    Ys = list()
    Zs = list()
    
    BoundaryFile = file_name
    BoundaryTissue = part_name
    np_points = np.array(io.extractPointsForPartFrom(BoundaryFile, BoundaryTissue))
    Bnodes = ss.getBnodes(BoundaryFile, BoundaryTissue)

#    print(BoundaryTissue)    
    for i in Bnodes:
        Xs.append(np_points[i-1,0])
        Ys.append(np_points[i-1,1])
        Zs.append(np_points[i-1,2])
        
# Also include the boundaries of PM_Mid for LA        
    if BoundaryTissue == "OPAL325_LA":
        SecondBoundaryTissue = "OPAL325_PM_mid"
        np_points = np.array(io.extractPointsForPartFrom(BoundaryFile, SecondBoundaryTissue))
        Bnodes = ss.getBnodes(BoundaryFile, SecondBoundaryTissue)
        for i in Bnodes:
            Xs.append(np_points[i-1,0])
            Ys.append(np_points[i-1,1])
            Zs.append(np_points[i-1,2])
            
            
    TissueBoundary = DataSet3d(list(Xs),list(Ys),list(Zs))
    
############################################################################
## The code below was used to apply all boundary conditions to all points on all tissues
##    #Now only the boundary points for that tissue will be used.
# 1) set arry of Tissues = ["OPAL325_ATFP",OPAL#@%_Para....]
# 2) for j in range 0, len(tissues)
# 3) if Tissues[j] != Boundary Tissue
# then
    Xs = list()
    Ys = list()
    Zs = list()
    #Tissues = ["OPAL325_ATFP", "OPAL325_Para_v6", "OPAL325_ATLA", "OPAL325_CL_v6", "OPAL325_US_v6", "OPAL325_LA", "OPAL325_PM_mid"]
    Tissues = ["OPAL325_ATFP", "OPAL325_Para_v6", "OPAL325_CL_v6", "OPAL325_US_v6", "OPAL325_LA", "OPAL325_PM_mid"]
    for mat in Tissues:
        if mat != BoundaryTissue and BoundaryTissue != "OPAL325_PM_mid": #hard-coded exclusion of PM_mid
            Bnodes = []
            np_otherPoints = np.array(io.extractPointsForPartFrom(BoundaryFile, mat))
            Bnodes = ss.getBnodes(BoundaryFile, mat)
            for i in Bnodes:
                Xs.append(np_otherPoints[i-1,0])
                Ys.append(np_otherPoints[i-1,1])
                Zs.append(np_otherPoints[i-1,2])


    AllBoundaries = DataSet3d(list(Xs),list(Ys),list(Zs)) #actually "other" boundaries

    generated_surface = ss.rotate_part_wboundaries(generated_surface, rotate_angle, rot_point, TissueBoundary, AllBoundaries, threshold)

    io.write_part_to_inp_file(file_name, part_name, generated_surface)
    return generated_surface
    
'''
Function: adjust_fibers
'''
def adjust_fibers(generated_surface, file_name, surface_name, connection_name):
    #Read the nodal coordinates defined part name out of defined file name
    nodes, connections = io.extractPointsForPartFrom(file_name, connection_name,get_connections=True)
    con_from_1_to_2 = ss.get_connections_for_tissues(connection_name,surface_name ,file_name)    
    ct = ConnectingTissue(nodes,connections, con_from_1_to_2)
    connections = ct.graph
    nodes_xyz = ct.nodes_as_xyz_list()
    
    for i, sp in enumerate(ct.starting_nodes):
        
        new_start_x = generated_surface.xAxis[con_from_1_to_2[sp]]
        new_start_y = generated_surface.yAxis[con_from_1_to_2[sp]]
        new_start_z = generated_surface.zAxis[con_from_1_to_2[sp]]
        
        endx = nodes_xyz[0][ct.ending_nodes[i]]
        endy = nodes_xyz[1][ct.ending_nodes[i]]
        endz = nodes_xyz[2][ct.ending_nodes[i]]
        
        node = ct.fibers_keys[i] 
        
        number_of_points = len(node)
        
        new_xps = np.linspace(new_start_x, endx, number_of_points)
        new_yps = np.linspace(new_start_y, endy, number_of_points)
        new_zps = np.linspace(new_start_z, endz, number_of_points)
        
        node = ct.fibers_keys[i]   #[0,24,25,26,27] this length will equal new_fiber_xp
        
        for i, val in enumerate(node):
            
            point = Point(new_xps[i], new_yps[i], new_zps[i])
            ct.update_node(val, point)
            
    return ct

'''
Function: average
'''
def average(array):
    return sum(array)/len(array)

'''
Function: takeMeasurements
'''
def takeMeasurements(output, AVW, Fibers, GenericINPFile, OutputINPfile):

    tempOrigList = io.extractPointsForPartFrom(GenericINPFile, AVW)
    tempOutList = io.extractPointsForPartFrom(OutputINPfile, AVW)
    originalPoints = [] #as a list of Point types
    for p in tempOrigList:
        originalPoints.append(Point(p[0], p[1], p[2]))

    newPoints = []
    for p in tempOutList:
        newPoints.append(Point(p[0], p[1], p[2]))

    AVWWidthOriginal = getAVWWidth(originalPoints)
    AVWLengthOriginal = getAVWLength(originalPoints)
    AVWWidthNew = getAVWWidth(newPoints)
    AVWLengthNew = getAVWLength(newPoints)
    oldFiberLengths = getFiberLengths(GenericINPFile, Fibers)
    newFiberLengths = getFiberLengths(OutputINPfile, Fibers)


    f = open(output, "w")
    f.write("============================\n")
    f.write("===---ORIGINAL GENERIC---===\n")
    f.write("============================\n")
    f.write("AVW Width\n\t" + str(AVWWidthOriginal) + "\n")
    f.write("AVW Length\n\t " + str(AVWLengthOriginal) + "\n")
    for i, fiber in enumerate(Fibers):
        f.write(fiber + " Lengths (Average = " + str(average(oldFiberLengths[i])) +  ")\n")
        for distance in oldFiberLengths[i]:
            f.write('\t' + str(distance) + '\n')

    f.write('\n\n')

    f.write("============================\n")
    f.write("===----OUTPUT GENERIC---====\n")
    f.write("============================\n")
    f.write("AVW Width\n\t" + str(AVWWidthNew) + "\n")
    f.write("AVW Length\n\t" + str(AVWLengthNew) + "\n")
    for i, fiber in enumerate(Fibers):
        f.write(fiber + " Lengths (Average = " + str(average(newFiberLengths[i])) + ")\n")
        for distance in newFiberLengths[i]:
            f.write('\t' + str(distance) + '\n')

    f.write('\n\n')

    f.write("============================\n")
    f.write("=====------SCALES------=====\n")
    f.write("============================\n")
    f.write("AVW Width\n\t" + str(AVWWidthNew/AVWWidthOriginal) + '\n')
    f.write("AVW Length\n\t" + str(AVWLengthNew/AVWLengthOriginal) + '\n')
    for i, fiber in enumerate(Fibers):
        f.write(fiber + " Lengths (Average = " + str(average(newFiberLengths[i])/average(oldFiberLengths[i])) + ")\n")
        for j in range(0, len(newFiberLengths[i])):
            f.write('\t' + str(newFiberLengths[i][j]/oldFiberLengths[i][j]) + '\n')

    f.close()
    return

################# Functions that are no longer used #################

'''
Function: lengthen_part

Replaced by <lengthen_and_shift_part>
'''
def lengthen_part(part_name, file_name, scale):
    if (scale != 1):
        generated_surface = io.get_dataset_from_file(file_name, part_name)
        
        generated_surface = ss.edges_to_try_and_correct_interpolation(generated_surface, generated_surface.generatedCopy())
    #======================================================
    #generate new Z Data       -- Z -- 
        newZYVals = ss.generate_values_for("z", generated_surface, scale)
    #======================================================
        generated_surface.zAxis = newZYVals[0]
        generated_surface.yAxis = newZYVals[1]
        
        generated_surface = generated_surface.trim()
        
        # Write to File
        
        io.write_part_to_inp_file(file_name, part_name, generated_surface)
        
'''
Function: narrow_distal_avw
'''
def narrow_distal_avw(part_name, file_name, original_file_name):
    print("Narrow Distal End of AVW")

    # Getting Nodes 
    generic_surface = io.get_dataset_from_file(original_file_name, part_name)
    pre_curved_surface = io.get_dataset_from_file(file_name, part_name).generatedCopy()
    generated_surface = io.get_dataset_from_file(file_name, part_name).generatedCopy()
    
########################################################################################################################################################################
# Pasting Function to work with

#Set constant variables    
    PM_z_coord = -24
    z_threshold = 20
    max_scaling = 0.5
    wave_width = 40
    yInterpolator = interpolate.Rbf(pre_curved_surface.xAxis,pre_curved_surface.zAxis,pre_curved_surface.yAxis,function = 'linear', smooth = 1)

# Check each point to see if it is within the threshold or not
    for index in range(0,len(pre_curved_surface.zAxis)):
        z_coord = pre_curved_surface.zAxis[index]

        if abs(z_coord - PM_z_coord) < z_threshold:

#            print(index)
# Calculate scaling factor
#            print(generated_surface.xAxis[index],generated_surface.yAxis[index],generated_surface.zAxis[index])
            scaling_factor = max_scaling * (1 - abs(z_coord - PM_z_coord)/z_threshold)
#            print('Scale = ', scaling_factor)
# Generate the equation using that scaling factor
#            Amplitude_func = lambda A : integrate.quad(lambda x: sqrt(1+(-A*2*pi/wave_width*sin(2*pi*x/wave_width))**2),0,wave_width)[0]/wave_width-(1+scaling_factor)
#            The Amplitude of the waves is determine by howclose the z coordinate is to the PM_z_coord as determine by SCALING_FACTOR
            Amplitude_func = lambda A : integrate.quad(lambda x: sqrt(1+(-A*2*pi/wave_width*cos(2*pi*x/wave_width))**2),0,wave_width)[0]/wave_width-(1+scaling_factor)
            A_initial_guess = 1
            Amplitude = fsolve(Amplitude_func, A_initial_guess)[0]
#            print('Amp = ', Amplitude)
            
            
# Find the distance from the point to the midline
            dist_to_center = ss.getSurfaceDistanceForXatZ(pre_curved_surface.xAxis[index],z_coord,yInterpolator)
#            print('Dist to center = ', dist_to_center)
# Find the coordinates on the sine wave to be at the same distance from the midline
#            Wave_func = lambda x_final : integrate.quad(lambda x: sqrt(1+(-Amplitude*2*pi/wave_width*sin(2*pi*x/wave_width))**2),0,x_final)[0]-dist_to_center
            Wave_func = lambda x_final : integrate.quad(lambda x: sqrt(1+(-Amplitude*2*pi/wave_width*cos(2*pi*x/wave_width))**2),0,x_final)[0]-dist_to_center
            x_final_initial_guess = dist_to_center*(1-max_scaling)
            new_x_coord = round(fsolve(Wave_func, x_final_initial_guess)[0],3)
#            print('********************')
#            print(new_x_coord)
            delta_y = Amplitude*cos(new_x_coord*2*pi/wave_width)
#            print(delta_y)
# Find the new coordinates for the point (newX, newY+oldY, oldZ)
            generated_surface.yAxis[index] = pre_curved_surface.yAxis[index]+delta_y
            generated_surface.xAxis[index] = new_x_coord
#            print(generated_surface.xAxis[index],generated_surface.yAxis[index],generated_surface.zAxis[index])

# Update it to the file
    generated_surface = generated_surface.trim()
    io.write_part_to_inp_file(file_name, part_name, generated_surface)

    return generated_surface

##############################################################################
##############################################################################
##############################################################################
########################## Non-Sabbatical Code Below #########################
##############################################################################
##############################################################################
##############################################################################
''' 
Section: Non-Sabbatical Code
'''

'''
Function: narrow_distal_avw_narrow
'''
def narrow_distal_avw_narrow(part_name, file_name, original_file_name, PM_Mid, PM_connections, AVW_connections):
    print("Narrow Distal End of AVW")

#   How far in the z direction (+/-) will be affected by the narrowing
    z_threshold = 5    


################### Probably delete ##########################################
# #Set constant variables    
#     PM_z_coord = -32

#     max_scaling = 0.2
    # wave_width = 40
    # yInterpolator = interpolate.Rbf(pre_curved_surface.xAxis,pre_curved_surface.zAxis,pre_curved_surface.yAxis,function = 'linear', smooth = 1)
##############################################################################

    # Getting Nodes
    # Generic AVW (Maybe never used)
    generic_surface = io.get_dataset_from_file(original_file_name, part_name)
    # AVW that won't be modified
    pre_curved_surface = io.get_dataset_from_file(file_name, part_name).generatedCopy()
    # AVW that will be modified/narrowed
    generated_surface = io.get_dataset_from_file(file_name, part_name).generatedCopy()

    # Find the starting nodes for the AVW edges (starting at the apex)
    neg_start, pos_start = find_edge_starting_ending_points_for_AVW(pre_curved_surface)
    # print(neg_start, pos_start)
    
    # 
    [negative_edges, positive_edges] = findEdgeNodes(generated_surface, AVW_connections, neg_start, pos_start)
    
    # print('edges:', negative_edges, positive_edges)
    
    all_edge_nodes = negative_edges+positive_edges
    
    # print(all_edge_nodes)
# Now that I have the edge nodes, check to see which ones are closest to the z...
    ### get within the threshold then grab min/max?
    
    
#   Get the mean of the Z coordinates for the inner arch on PM_Mid
    starting_index, ending_index = find_starting_ending_points_for_inside(PM_Mid)
    innerNodes = findInnerNodes(PM_Mid, PM_connections, starting_index, ending_index)
      
    inner_node_x_coords = []
    inner_node_y_coords = []
    inner_node_z_coords = []
    
    for node_number in innerNodes:
        inner_node_x_coords.append(PM_Mid.xAxis[node_number])
        inner_node_y_coords.append(PM_Mid.yAxis[node_number])
        inner_node_z_coords.append(PM_Mid.zAxis[node_number])
   
    # This is the average z coordinate of the inner arch of the PM_Mid
    PM_Z_mean = np.mean(inner_node_z_coords)
#    PM_Z_min =  np.min(inner_node_z_coords)
#    PM_Z_max =  np.max(inner_node_z_coords)
########################################################
    
    # Find the AVW edge nodes that are at that are close to that Z location
    # I think I need to find top/bottom negative/positive to use for later
    AVW_negative_top_y = -9999
    AVW_negative_bottom_y = 9999
    

    for node in negative_edges:
        index = node - 1
        z_coord = pre_curved_surface.zAxis[index]

        if abs(z_coord - PM_Z_mean) < z_threshold:
            if pre_curved_surface.yAxis[index] > AVW_negative_top_y:
                AVW_negative_top_y = pre_curved_surface.yAxis[index]

            if pre_curved_surface.yAxis[index] < AVW_negative_bottom_y:
                AVW_negative_bottom_y = pre_curved_surface.yAxis[index]

                
    AVW_positive_top_y = -9999
    AVW_positive_bottom_y = 9999

    for node in positive_edges:
        index = node - 1
        z_coord = pre_curved_surface.zAxis[index]

        if abs(z_coord - PM_Z_mean) < z_threshold:
            if pre_curved_surface.yAxis[index] > AVW_positive_top_y:
                AVW_positive_top_y = pre_curved_surface.yAxis[index]
                # AVW_positive_top_x = pre_curved_surface.xAxis[index]
            if pre_curved_surface.yAxis[index] < AVW_positive_bottom_y:
                AVW_positive_bottom_y = pre_curved_surface.yAxis[index]
                # AVW_positive_bottom_x = pre_curved_surface.xAxis[index]
                
                
    
    y_threshold = 3
    
    AVW_negative_top_x = 9999
    AVW_negative_bottom_x = 9999
    for node in negative_edges:
        index = node - 1
        if abs(pre_curved_surface.zAxis[index] - PM_Z_mean) < z_threshold and abs(pre_curved_surface.yAxis[index] - AVW_negative_top_y) < y_threshold:
            if pre_curved_surface.xAxis[index] < AVW_negative_top_x:
                AVW_negative_top_x = pre_curved_surface.xAxis[index]            
            if pre_curved_surface.xAxis[index] < AVW_negative_bottom_x:
                AVW_negative_bottom_x = pre_curved_surface.xAxis[index]
                
    AVW_positive_top_x = -9999
    AVW_positive_bottom_x = -9999
    for node in positive_edges:
        index = node - 1
        if abs(pre_curved_surface.zAxis[index] - PM_Z_mean) < z_threshold and abs(pre_curved_surface.yAxis[index] - AVW_positive_top_y) < y_threshold:
            if pre_curved_surface.xAxis[index] > AVW_positive_top_x:
                AVW_positive_top_x = pre_curved_surface.xAxis[index]            
            if pre_curved_surface.xAxis[index] > AVW_positive_bottom_x:
                AVW_positive_bottom_x = pre_curved_surface.xAxis[index]

# ######################### Old Way #############################################

#     # Find the max and min Y coordinates of the AVW which are the y coordinate
#     # for top and bottom avw at PM
#     top_y = -9999
#     bottom_y = 9999
#     for index in range(0,len(pre_curved_surface.zAxis)):
#         z_coord = pre_curved_surface.zAxis[index]

#         if abs(z_coord - PM_Z_mean) < z_threshold:
#             if pre_curved_surface.yAxis[index] > top_y:
#                 top_y = pre_curved_surface.yAxis[index]
#             elif pre_curved_surface.yAxis[index] < bottom_y:
#                 bottom_y = pre_curved_surface.yAxis[index]
    # 
    ######################################################################
    ######################################################################
    ######################################################################
    # top_y = 0.33
    # bottom_y = -2
    ######################################################################
    ######################################################################
    ######################################################################
    
    # # Put other AVW points in groups depending on which they are closest to
    # # If closest to max point, check if x is larger than max x or less than min x
    # # Same for closest to min point
    # # Difference between max and min xs gives the width
    # top_min_x = 9999
    # top_max_x = -9999
    # bottom_min_x = 9999
    # bottom_max_x = -9999
    # # for node in all_edge_nodes:
    # #     index = node - 1
    # for index in range(0,len(pre_curved_surface.zAxis)):
    #     z_coord = pre_curved_surface.zAxis[index]
    #     if abs(z_coord - PM_Z_mean) < z_threshold*2:
    #         # print(pre_curved_surface.xAxis[index],pre_curved_surface.yAxis[index],pre_curved_surface.zAxis[index])
    #         # If the y coordinate is closer to the top than it is to the bottom
    #         if abs(pre_curved_surface.yAxis[index] - top_y) < abs(pre_curved_surface.yAxis[index] - bottom_y):
    #             if pre_curved_surface.xAxis[index] < top_min_x:
    #                 top_min_x = pre_curved_surface.xAxis[index]
    #             if pre_curved_surface.xAxis[index] > top_max_x:
    #                 top_max_x = pre_curved_surface.xAxis[index]
    #         else:
    #             if pre_curved_surface.xAxis[index] < bottom_min_x:
    #                 bottom_min_x = pre_curved_surface.xAxis[index]
    #             if pre_curved_surface.xAxis[index] > bottom_max_x:
    #                 bottom_max_x = pre_curved_surface.xAxis[index]
    # top_avw_width = top_max_x - top_min_x
    # bottom_avw_width = bottom_max_x - bottom_min_x
                    
    # Find nodes Inner Arch PM Nodes that are closest to the Y coordiantes for top and bottom
    # Determine the distances between the archs
    pm_best_neg_top_y = 9999
    pm_best_pos_top_y = 9999
    pm_best_neg_bottom_y = 9999
    pm_best_pos_bottom_y = 9999
    for i, x_coord in enumerate(inner_node_x_coords):
        # and the x coordinae is negative
        if x_coord < 0: 
            # if the coordinate is above the top of the AVW
            if inner_node_y_coords[i] > AVW_negative_top_y:
                # and it is closer to the top AVW than the previous best point
                if inner_node_y_coords[i] - AVW_negative_top_y < pm_best_neg_top_y - AVW_negative_top_y:
                    pm_best_neg_top_y = inner_node_y_coords[i]
                    pm_best_neg_top_x = x_coord
                    pm_best_neg_top_z = inner_node_z_coords[i]
            elif inner_node_y_coords[i] > AVW_negative_bottom_y:
                if inner_node_y_coords[i] - AVW_negative_bottom_y < pm_best_neg_bottom_y - AVW_negative_bottom_y:
                    pm_best_neg_bottom_y = inner_node_y_coords[i]
                    pm_best_neg_bottom_x = x_coord
                    pm_best_neg_bottom_z = inner_node_z_coords[i]            
            # if inner_node_y_coords[i] > positive_top_y:
            #     # and it is closer to the top AVW than the previous best point
            #     if inner_node_y_coords[i] - positive_top_y < best_pos_top_y - positive_top_y:
            #         best_pos_top_y = inner_node_y_coords[i]
            #         best_pos_top_x = x_coord
            #         best_pos_top_z = inner_node_z_coords[i]
        else:
            if inner_node_y_coords[i] > AVW_positive_top_y:
                if inner_node_y_coords[i] - AVW_positive_top_y < pm_best_pos_top_y - AVW_positive_top_y:
                    pm_best_pos_top_y = inner_node_y_coords[i]
                    pm_best_pos_top_x = x_coord
                    pm_best_pos_top_z = inner_node_z_coords[i]
            elif inner_node_y_coords[i] > AVW_positive_bottom_y:
                if inner_node_y_coords[i] - AVW_positive_bottom_y < pm_best_pos_bottom_y - AVW_positive_bottom_y:
                    pm_best_pos_bottom_y = inner_node_y_coords[i]
                    pm_best_pos_bottom_x = x_coord     
                    pm_best_pos_bottom_z = inner_node_z_coords[i]
                # and the x coordinae is negative
            # if x_coord < 0: 

            # else:
            #     # and it is closer to the top AVW than the previous best point
            #     if inner_node_y_coords[i] - bottom_y < best_pos_bottom_y - bottom_y:
            #         best_pos_bottom_y = inner_node_y_coords[i]
            #         best_pos_bottom_x = x_coord     
            #         best_pos_bottom_z = inner_node_z_coords[i]
    # top_pm_width =  best_pos_top_x - best_neg_top_x
    # bottom_pm_width = best_pos_bottom_x - best_neg_bottom_x
    top_pm_average_z = (pm_best_neg_top_z + pm_best_pos_top_z) / 2
    bottom_pm_average_z = (pm_best_neg_bottom_z + pm_best_pos_bottom_z) / 2
                    
    
    
    
    
    if abs(AVW_negative_top_x) > abs(pm_best_neg_top_x):
        top_neg_scaling_factor = 1 - pm_best_neg_top_x/AVW_negative_top_x
    else:
        top_neg_scaling_factor = 0
        
    if abs(AVW_positive_top_x) > abs(pm_best_pos_top_x):
        top_pos_scaling_factor = 1 - pm_best_pos_top_x/AVW_positive_top_x
    else:
        top_pos_scaling_factor = 0

    if abs(AVW_negative_bottom_x) > abs(pm_best_neg_bottom_x):
        bottom_neg_scaling_factor = 1 - pm_best_neg_bottom_x/AVW_negative_bottom_x
    else:
        bottom_neg_scaling_factor = 0
        
    if abs(AVW_positive_bottom_x) > abs(pm_best_pos_bottom_x):
        bottom_pos_scaling_factor = 1 - pm_best_pos_bottom_x/AVW_positive_bottom_x
    else:
        bottom_pos_scaling_factor = 0
    
    # if top_pm_width < top_avw_width:
    #     top_max_scaling_factor = 1 - top_pm_width/top_avw_width
    # else:
    #     top_max_scaling_factor = 0
    # if bottom_pm_width < bottom_avw_width:
    #     bottom_max_scaling_factor = 1 - bottom_pm_width/bottom_avw_width
    # else:
    #     bottom_max_scaling_factor = 0

    # print('***')    
    # print('pm tops and bottoms of x', pm_best_neg_top_x, pm_best_pos_top_x, pm_best_neg_bottom_x, pm_best_pos_bottom_x)
    # print('AVW tops and bottoms of x', AVW_negative_top_x, AVW_positive_top_x, AVW_negative_bottom_x, AVW_positive_bottom_x)
    # print(top_neg_scaling_factor, top_pos_scaling_factor, bottom_neg_scaling_factor, bottom_pos_scaling_factor)

    # print('***')
    # print('pm_z:', PM_Z_mean)
    # print('top y', top_y)
    # print('bottom y', bottom_y)
    # print('top and bottom xs:', top_min_x,top_max_x,bottom_min_x,bottom_max_x)
    # print('pm top and bottom ys', best_pos_top_y, best_neg_top_y, best_pos_bottom_y, best_neg_bottom_y)
    # print('avw widths', top_avw_width, bottom_avw_width)
    # print('pm widths', top_pm_width, bottom_pm_width)
    # print('top and bottom scaling factors', top_max_scaling_factor, bottom_max_scaling_factor)
    # print('top and bottom pm Z', top_pm_average_z,bottom_pm_average_z)
    

    
    # Determine the scaling factors
    # When looking at a node, check to see if it is in the Z range, and then which Y it is the closest to
    # Use that scaling to determine the position



    
########################################################################################################################################################################
# Pasting Function to work with


# Check each point to see if it is within the threshold or not
    for index in range(0,len(pre_curved_surface.zAxis)):
        z_coord = pre_curved_surface.zAxis[index]

        if abs(z_coord - PM_Z_mean) < z_threshold + 5:

#            print(index)
# Calculate scaling factor
#            print(generated_surface.xAxis[index],generated_surface.yAxis[index],generated_surface.zAxis[index])
            
            # # The closer it is to the PM (PM_z_coord) the more it needs to be adjusted. Adjusted between max_scaling and 0
            # scaling_factor = max_scaling * (1 - abs(z_coord - PM_z_coord)/z_threshold)
            
            
            
            # Changing to checking if it is closer to the top y coordinate or
            # bottom and then assigning the correct scaling factor
            # If closer to the top avw/pm interface than the bottom
            if abs(pre_curved_surface.yAxis[index] - top_pm_average_z) < abs(pre_curved_surface.yAxis[index] - bottom_pm_average_z):
                if abs(z_coord - top_pm_average_z) < z_threshold:
                    if pre_curved_surface.xAxis[index] < 0:
                        scaling_factor = top_neg_scaling_factor * (1 - abs(z_coord - top_pm_average_z)/z_threshold)
                    else:
                        scaling_factor = top_pos_scaling_factor * (1 - abs(z_coord - top_pm_average_z)/z_threshold)
                else:
                    scaling_factor = 0
            else:
                if abs(z_coord - bottom_pm_average_z) < z_threshold:
                    if pre_curved_surface.xAxis[index] < 0:
                        scaling_factor = bottom_neg_scaling_factor * (1 - abs(z_coord - bottom_pm_average_z)/z_threshold) 
                    else:
                        scaling_factor = bottom_pos_scaling_factor * (1 - abs(z_coord - bottom_pm_average_z)/z_threshold) 
                else:
                    scaling_factor = 0
            
#            print('Scale = ', scaling_factor)
# Generate the equation using that scaling factor
            # Amplitude_func = lambda A : integrate.quad(lambda x: sqrt(1+(-A*2*pi/wave_width*sin(2*pi*x/wave_width))**2),0,wave_width)[0]/wave_width-(1+scaling_factor)
            # A_initial_guess = 1
            # Amplitude = fsolve(Amplitude_func, A_initial_guess)[0]
#            print('Amp = ', Amplitude)
            
            
# Find the distance from the point to the midline
            # dist_to_center = ss.getSurfaceDistanceForXatZ(pre_curved_surface.xAxis[index],z_coord,yInterpolator)
#            print('Dist to center = ', dist_to_center)
# Find the coordinates on the sine wave to be at the same distance from the midline
            # Wave_func = lambda x_final : integrate.quad(lambda x: sqrt(1+(-Amplitude*2*pi/wave_width*sin(2*pi*x/wave_width))**2),0,x_final)[0]-dist_to_center
            # x_final_initial_guess = dist_to_center*(1-max_scaling)
            # new_x_coord = round(fsolve(Wave_func, x_final_initial_guess)[0],3)
            new_x_coord = pre_curved_surface.xAxis[index]*(1-scaling_factor)
#            print('********************')
#            print(new_x_coord)
            # delta_y = Amplitude*cos(new_x_coord*2*pi/wave_width)
#            print(delta_y)
# Find the new coordinates for the point (newX, newY+oldY, oldZ)
            generated_surface.yAxis[index] = pre_curved_surface.yAxis[index]
            generated_surface.xAxis[index] = new_x_coord
#            print(generated_surface.xAxis[index],generated_surface.yAxis[index],generated_surface.zAxis[index])

# Update it to the file
    generated_surface = generated_surface.trim()
    io.write_part_to_inp_file(file_name, part_name, generated_surface)

    return generated_surface

'''
Function: narrow_distal_avw_curve_down
'''
def narrow_distal_avw_curve_down(part_name, file_name, original_file_name):
    print("Narrow Distal End of AVW")

    # Getting Nodes 
    generic_surface = io.get_dataset_from_file(original_file_name, part_name)
    pre_curved_surface = io.get_dataset_from_file(file_name, part_name).generatedCopy()
    generated_surface = io.get_dataset_from_file(file_name, part_name).generatedCopy()
    
########################################################################################################################################################################
# Pasting Function to work with

#Set constant variables    
    PM_z_coord = -24
    z_threshold = 20
    max_scaling = 0.5
    wave_width = 40
    yInterpolator = interpolate.Rbf(pre_curved_surface.xAxis,pre_curved_surface.zAxis,pre_curved_surface.yAxis,function = 'linear', smooth = 1)

# Check each point to see if it is within the threshold or not
    #loop through all of the AVW nodes
    for index in range(0,len(pre_curved_surface.zAxis)):
        z_coord = pre_curved_surface.zAxis[index]

#       Check to see if it is close to the PM
        if abs(z_coord - PM_z_coord) < z_threshold:

# Calculate scaling factor
#            print(generated_surface.xAxis[index],generated_surface.yAxis[index],generated_surface.zAxis[index])
            
            # The closer it is to the PM (PM_z_coord) the more it needs to be adjusted. Adjusted between max_scaling and 0
            scaling_factor = max_scaling * (1 - abs(z_coord - PM_z_coord)/z_threshold)
#            print('Scale = ', scaling_factor)
# Generate the equation using that scaling factor
            Amplitude_func = lambda A : integrate.quad(lambda x: sqrt(1+(-A*2*pi/wave_width*sin(2*pi*x/wave_width))**2),0,wave_width)[0]/wave_width-(1+scaling_factor)
            A_initial_guess = 1
            Amplitude = fsolve(Amplitude_func, A_initial_guess)[0]
#            print('Amp = ', Amplitude)
            
            
# Find the distance from the point to the midline
            dist_to_center = ss.getSurfaceDistanceForXatZ(pre_curved_surface.xAxis[index],z_coord,yInterpolator)
#            print('Dist to center = ', dist_to_center)
# Find the coordinates on the sine wave to be at the same distance from the midline
            Wave_func = lambda x_final : integrate.quad(lambda x: sqrt(1+(-Amplitude*2*pi/wave_width*sin(2*pi*x/wave_width))**2),0,x_final)[0]-dist_to_center
            x_final_initial_guess = dist_to_center*(1-max_scaling)
            new_x_coord = round(fsolve(Wave_func, x_final_initial_guess)[0],3)
#            print('********************')
#            print(new_x_coord)
            delta_y = Amplitude*cos(new_x_coord*2*pi/wave_width)
#            print(delta_y)
# Find the new coordinates for the point (newX, newY+oldY, oldZ)
            generated_surface.yAxis[index] = pre_curved_surface.yAxis[index]+delta_y
            generated_surface.xAxis[index] = new_x_coord
#            print(generated_surface.xAxis[index],generated_surface.yAxis[index],generated_surface.zAxis[index])

# Update it to the file
    generated_surface = generated_surface.trim()
    io.write_part_to_inp_file(file_name, part_name, generated_surface)

    return generated_surface


'''
Function: find_edge_starting_ending_points_for_AVW
'''
def find_edge_starting_ending_points_for_AVW(AVW):

    closestNegativeIndex = -1
    closestPositiveIndex = -1

    closestNegativeDist = 999999
    closestPositiveDist = 999999

    x_pos_point = 9999
    x_neg_point = -9999
    z_point = 9999

# first look for the -x side
    for i, (x, y, z) in enumerate(AVW.zipped()):
        if x < 0:
            distance = ((x-x_neg_point)**2+(z-z_point)**2)**0.5
            if distance < closestNegativeDist:
                closestNegativeIndex = i
                closestNegativeDist = distance
        else:
            distance = ((x-x_pos_point)**2+(z-z_point)**2)**0.5
            if distance < closestPositiveDist:
                closestPositiveIndex = i
                closestPositiveDist = distance

    return closestNegativeIndex, closestPositiveIndex

'''
Function: findEdgeNodes
'''
def findEdgeNodes(AVW_surface, AVW_connections, neg_start, pos_start):
    
    current_index = neg_start
    neg_node_list = [neg_start + 1]

    delta_x = 0
    delta_y = 999
    delta_z = 999
    i = 0

    while (delta_x / (delta_y**2 + delta_z**2 + delta_x**2)**0.5) < (1/2**0.5):
        lowest_x = 9999
        print(AVW_connections[current_index])
        for connection in AVW_connections[current_index]:
            if connection not in neg_node_list:
                corrected_index = connection - 1
                if AVW_surface.xAxis[corrected_index] < lowest_x:
                    next_node = connection
                    lowest_x = AVW_surface.xAxis[corrected_index]
                    current_index = corrected_index
        delta_x = abs(AVW_surface.xAxis[neg_node_list[-1]-1] - AVW_surface.xAxis[next_node-1])
        delta_y = abs(AVW_surface.yAxis[neg_node_list[-1]-1] - AVW_surface.yAxis[next_node-1])
        delta_z = abs(AVW_surface.zAxis[neg_node_list[-1]-1] - AVW_surface.zAxis[next_node-1])
        # print('deltas:', delta_x, delta_y, delta_z)
        neg_node_list.append(next_node)
        i=i+1
    neg_node_list.pop(-1)
      

    current_index = pos_start
    pos_node_list = [pos_start + 1]

    delta_x = 0
    delta_y = 999
    delta_z = 999
    i = 0
    # while delta_x < delta_y or delta_x < delta_z:
    while (delta_x / (delta_y**2 + delta_z**2 + delta_x**2)**0.5) < (1/2**0.5):
        highest_x = -9999
        print(AVW_connections[current_index])
        for connection in AVW_connections[current_index]:
            if connection not in pos_node_list:
                corrected_index = connection - 1
                if AVW_surface.xAxis[corrected_index] > highest_x:
                    next_node = connection
                    highest_x = AVW_surface.xAxis[corrected_index]
                    current_index = corrected_index
        delta_x = abs(AVW_surface.xAxis[pos_node_list[-1]-1] - AVW_surface.xAxis[next_node-1])
        delta_y = abs(AVW_surface.yAxis[pos_node_list[-1]-1] - AVW_surface.yAxis[next_node-1])
        delta_z = abs(AVW_surface.zAxis[pos_node_list[-1]-1] - AVW_surface.zAxis[next_node-1])
        # print('deltas:', delta_x, delta_y, delta_z)
        pos_node_list.append(next_node)
        i=i+1
    pos_node_list.pop(-1)

    return neg_node_list, pos_node_list


# def narrow_distal_avw_narrow_and_curve(part_name, file_name, original_file_name, PM_Mid, PM_connections, AVW_connections):
#     print("Narrow Distal End of AVW")

# #   How far in the z direction (+/-) will be affected by the narrowing
#     z_threshold = 5    

#     # Getting Nodes
#     # Generic AVW (Maybe never used)
#     generic_surface = io.get_dataset_from_file(original_file_name, part_name)
#     # AVW that won't be modified
#     pre_curved_surface = io.get_dataset_from_file(file_name, part_name).generatedCopy()
#     # AVW that will be modified/narrowed
#     generated_surface = io.get_dataset_from_file(file_name, part_name).generatedCopy()

#     # Find the starting nodes for the AVW edges (starting at the apex)
#     neg_start, pos_start = find_edge_starting_ending_points_for_AVW(pre_curved_surface)
#     print(neg_start, pos_start)
    
#     # 
#     [negative_edges, positive_edges] = findEdgeNodes(generated_surface, AVW_connections, neg_start, pos_start)
    
#     print('edges:', negative_edges, positive_edges)
    
#     all_edge_nodes = negative_edges+positive_edges
    
#     print(all_edge_nodes)
# # Now that I have the edge nodes, check to see which ones are closest to the z...
#     ### get within the threshold then grab min/max?
    
    
# #   Get the mean of the Z coordinates for the inner arch on PM_Mid
#     starting_index, ending_index = find_starting_ending_points_for_inside(PM_Mid)
#     innerNodes = findInnerNodes(PM_Mid, PM_connections, starting_index, ending_index)
      
#     inner_node_x_coords = []
#     inner_node_y_coords = []
#     inner_node_z_coords = []
    
#     for node_number in innerNodes:
#         inner_node_x_coords.append(PM_Mid.xAxis[node_number])
#         inner_node_y_coords.append(PM_Mid.yAxis[node_number])
#         inner_node_z_coords.append(PM_Mid.zAxis[node_number])
   
#     # This is the average z coordinate of the inner arch of the PM_Mid
#     PM_Z_mean = np.mean(inner_node_z_coords)
    
#     # Find the AVW edge nodes that are at that are close to that Z location
#     # I think I need to find top/bottom negative/positive to use for later
#     AVW_negative_top_y = -9999
#     AVW_negative_bottom_y = 9999
    

#     for node in negative_edges:
#         index = node - 1
#         z_coord = pre_curved_surface.zAxis[index]

#         if abs(z_coord - PM_Z_mean) < z_threshold:
#             if pre_curved_surface.yAxis[index] > AVW_negative_top_y:
#                 AVW_negative_top_y = pre_curved_surface.yAxis[index]

#             if pre_curved_surface.yAxis[index] < AVW_negative_bottom_y:
#                 AVW_negative_bottom_y = pre_curved_surface.yAxis[index]

                
#     AVW_positive_top_y = -9999
#     AVW_positive_bottom_y = 9999

#     for node in positive_edges:
#         index = node - 1
#         z_coord = pre_curved_surface.zAxis[index]

#         if abs(z_coord - PM_Z_mean) < z_threshold:
#             if pre_curved_surface.yAxis[index] > AVW_positive_top_y:
#                 AVW_positive_top_y = pre_curved_surface.yAxis[index]
#                 # AVW_positive_top_x = pre_curved_surface.xAxis[index]
#             if pre_curved_surface.yAxis[index] < AVW_positive_bottom_y:
#                 AVW_positive_bottom_y = pre_curved_surface.yAxis[index]
#                 # AVW_positive_bottom_x = pre_curved_surface.xAxis[index]
                
                
    
#     y_threshold = 3
    
#     AVW_negative_top_x = 9999
#     AVW_negative_bottom_x = 9999
#     for node in negative_edges:
#         index = node - 1
#         if abs(pre_curved_surface.zAxis[index] - PM_Z_mean) < z_threshold and abs(pre_curved_surface.yAxis[index] - AVW_negative_top_y) < y_threshold:
#             if pre_curved_surface.xAxis[index] < AVW_negative_top_x:
#                 AVW_negative_top_x = pre_curved_surface.xAxis[index]            
#             if pre_curved_surface.xAxis[index] < AVW_negative_bottom_x:
#                 AVW_negative_bottom_x = pre_curved_surface.xAxis[index]
                
#     AVW_positive_top_x = -9999
#     AVW_positive_bottom_x = -9999
#     for node in positive_edges:
#         index = node - 1
#         if abs(pre_curved_surface.zAxis[index] - PM_Z_mean) < z_threshold and abs(pre_curved_surface.yAxis[index] - AVW_positive_top_y) < y_threshold:
#             if pre_curved_surface.xAxis[index] > AVW_positive_top_x:
#                 AVW_positive_top_x = pre_curved_surface.xAxis[index]            
#             if pre_curved_surface.xAxis[index] > AVW_positive_bottom_x:
#                 AVW_positive_bottom_x = pre_curved_surface.xAxis[index]
            
#     # Find nodes Inner Arch PM Nodes that are closest to the Y coordiantes for top and bottom
#     # Determine the distances between the archs
#     pm_best_neg_top_y = 9999
#     pm_best_pos_top_y = 9999
#     pm_best_neg_bottom_y = 9999
#     pm_best_pos_bottom_y = 9999
#     for i, x_coord in enumerate(inner_node_x_coords):
#         # and the x coordinae is negative
#         if x_coord < 0: 
#             # if the coordinate is above the top of the AVW
#             if inner_node_y_coords[i] > AVW_negative_top_y:
#                 # and it is closer to the top AVW than the previous best point
#                 if inner_node_y_coords[i] - AVW_negative_top_y < pm_best_neg_top_y - AVW_negative_top_y:
#                     pm_best_neg_top_y = inner_node_y_coords[i]
#                     pm_best_neg_top_x = x_coord
#                     pm_best_neg_top_z = inner_node_z_coords[i]
#             elif inner_node_y_coords[i] > AVW_negative_bottom_y:
#                 if inner_node_y_coords[i] - AVW_negative_bottom_y < pm_best_neg_bottom_y - AVW_negative_bottom_y:
#                     pm_best_neg_bottom_y = inner_node_y_coords[i]
#                     pm_best_neg_bottom_x = x_coord
#                     pm_best_neg_bottom_z = inner_node_z_coords[i]            
#         else:
#             if inner_node_y_coords[i] > AVW_positive_top_y:
#                 if inner_node_y_coords[i] - AVW_positive_top_y < pm_best_pos_top_y - AVW_positive_top_y:
#                     pm_best_pos_top_y = inner_node_y_coords[i]
#                     pm_best_pos_top_x = x_coord
#                     pm_best_pos_top_z = inner_node_z_coords[i]
#             elif inner_node_y_coords[i] > AVW_positive_bottom_y:
#                 if inner_node_y_coords[i] - AVW_positive_bottom_y < pm_best_pos_bottom_y - AVW_positive_bottom_y:
#                     pm_best_pos_bottom_y = inner_node_y_coords[i]
#                     pm_best_pos_bottom_x = x_coord     
#                     pm_best_pos_bottom_z = inner_node_z_coords[i]

#     top_pm_average_z = (pm_best_neg_top_z + pm_best_pos_top_z) / 2
#     bottom_pm_average_z = (pm_best_neg_bottom_z + pm_best_pos_bottom_z) / 2
                    
    
#     if abs(AVW_negative_top_x) > abs(pm_best_neg_top_x):
#         top_neg_scaling_factor = 1 - pm_best_neg_top_x/AVW_negative_top_x
#     else:
#         top_neg_scaling_factor = 0
        
#     if abs(AVW_positive_top_x) > abs(pm_best_pos_top_x):
#         top_pos_scaling_factor = 1 - pm_best_pos_top_x/AVW_positive_top_x
#     else:
#         top_pos_scaling_factor = 0

#     if abs(AVW_negative_bottom_x) > abs(pm_best_neg_bottom_x):
#         bottom_neg_scaling_factor = 1 - pm_best_neg_bottom_x/AVW_negative_bottom_x
#     else:
#         bottom_neg_scaling_factor = 0
        
#     if abs(AVW_positive_bottom_x) > abs(pm_best_pos_bottom_x):
#         bottom_pos_scaling_factor = 1 - pm_best_pos_bottom_x/AVW_positive_bottom_x
#     else:
#         bottom_pos_scaling_factor = 0

#     print('***')    
#     print('pm tops and bottoms of x', pm_best_neg_top_x, pm_best_pos_top_x, pm_best_neg_bottom_x, pm_best_pos_bottom_x)
#     print('AVW tops and bottoms of x', AVW_negative_top_x, AVW_positive_top_x, AVW_negative_bottom_x, AVW_positive_bottom_x)
#     print(top_neg_scaling_factor, top_pos_scaling_factor, bottom_neg_scaling_factor, bottom_pos_scaling_factor)

#     print('***')


#     yInterpolator = interpolate.Rbf(pre_curved_surface.xAxis,pre_curved_surface.zAxis,pre_curved_surface.yAxis,function = 'linear', smooth = 1)
#     dist_to_center_top_neg = ss.getSurfaceDistanceForXatZ(AVW_negative_top_x,pm_best_neg_top_z,yInterpolator)
#     dist_to_center_top_pos = ss.getSurfaceDistanceForXatZ(AVW_positive_top_x,pm_best_pos_top_z,yInterpolator)
    
#     dist_to_center_bottom_neg = ss.getSurfaceDistanceForXatZ(AVW_negative_bottom_x,pm_best_neg_bottom_z,yInterpolator)
#     dist_to_center_bottom_pos = ss.getSurfaceDistanceForXatZ(AVW_positive_bottom_x,pm_best_pos_bottom_z,yInterpolator)

#     AVW_top_distance = dist_to_center_top_pos - dist_to_center_top_neg 
#     AVW_bottom_distance = dist_to_center_bottom_pos - dist_to_center_bottom_neg 
    
#     print(dist_to_center_top_neg, dist_to_center_top_pos, dist_to_center_bottom_neg, dist_to_center_bottom_pos)
#     print('AVW top distance: ', AVW_top_distance)
#     print('AVW bottom distance: ', AVW_bottom_distance)
# ###### Set up the interpolations...set up an interpolation for each level
# # ..then determine which to use based on how close the z coordinate is to the
#     # PM_Mid coordiante

# #*****************Width below should be set in a smarter way*****************
#     wave_width = 41
    
#     Amplitude = 0
#     best_top_amp = 0
#     best_bot_amp = 0
#     interpolations = [yInterpolator]
#     while best_top_amp == 0 or best_bot_amp == 0:
#         interpolating_surface = pre_curved_surface
#         Amplitude += 1
#         print(Amplitude)
#         for index in range(0,len(pre_curved_surface.zAxis)):
#             z_coord = pre_curved_surface.zAxis[index]
#             if abs(z_coord - PM_Z_mean) < z_threshold + 5:
#                 x_coord = pre_curved_surface.xAxis[index]
#                 #  want to go between -wave_width/2 and wave_width/2 being between -pi/2 and pi/2
#                 delta_y = Amplitude*cos(x_coord*pi/wave_width)
#                 interpolating_surface.yAxis[index] = pre_curved_surface.yAxis[index]-delta_y
#         interpolations.append(interpolate.Rbf(interpolating_surface.xAxis,interpolating_surface.zAxis,interpolating_surface.yAxis,function = 'linear', smooth = 1))
#         # need to check if the distance to arch is greater than the
#         # original distance to the AVW edge
#         yInterpolator = interpolations[-1]
#         dist_to_center_top_neg = ss.getSurfaceDistanceForXatZ(pm_best_neg_top_x,pm_best_neg_top_z,yInterpolator)
#         dist_to_center_top_pos = ss.getSurfaceDistanceForXatZ(pm_best_pos_top_x,pm_best_pos_top_z,yInterpolator)
        
#         dist_to_center_bottom_neg = ss.getSurfaceDistanceForXatZ(pm_best_neg_bottom_x,pm_best_neg_bottom_z,yInterpolator)
#         dist_to_center_bottom_pos = ss.getSurfaceDistanceForXatZ(pm_best_pos_bottom_x,pm_best_pos_bottom_z,yInterpolator)
#         # check for both positive and negative side before it is good
#         # check for top and bottom as we need an ideal for both
        
#         print("AVW distance, top dists:", AVW_top_distance, dist_to_center_top_pos, dist_to_center_top_neg, dist_to_center_top_pos - dist_to_center_top_neg)
#         print("AVW distance, bot dists:", AVW_bottom_distance, dist_to_center_bottom_pos, dist_to_center_bottom_neg, dist_to_center_bottom_pos - dist_to_center_bottom_neg)
        
#         if AVW_top_distance  < dist_to_center_top_pos - dist_to_center_top_neg and best_top_amp == 0:
#             best_top_amp = Amplitude
            
#         if AVW_bottom_distance  < dist_to_center_bottom_pos - dist_to_center_bottom_neg and best_bot_amp == 0:
#             best_bot_amp = Amplitude
    


# ##############################################################################
# #################### Curve Code Below ########################################
# ##############################################################################
# # Check each point to see if it is within the threshold or not
#     #loop through all of the AVW nodes
#     for index in range(0,len(pre_curved_surface.zAxis)):

        
        
#         ##### Narrowing Code Below ################
#         ##### Narrowing Code Below ################
#         ##### Narrowing Code Below ################                
#         z_coord = pre_curved_surface.zAxis[index]

#         if abs(z_coord - PM_Z_mean) < z_threshold + 5:

# #            print(index)
# # Calculate scaling factor
# #            print(generated_surface.xAxis[index],generated_surface.yAxis[index],generated_surface.zAxis[index])
            
#             # # The closer it is to the PM (PM_z_coord) the more it needs to be adjusted. Adjusted between max_scaling and 0
#             # scaling_factor = max_scaling * (1 - abs(z_coord - PM_z_coord)/z_threshold)
            
            
            
#             # Changing to checking if it is closer to the top y coordinate or
#             # bottom and then assigning the correct scaling factor
#             # If closer to the top avw/pm interface than the bottom
#             if abs(pre_curved_surface.yAxis[index] - top_pm_average_z) < abs(pre_curved_surface.yAxis[index] - bottom_pm_average_z):
#                 # and within the threshold to narrow
#                 if abs(z_coord - top_pm_average_z) < z_threshold:
#                     node_amp = math.ceil((1- abs(z_coord - top_pm_average_z)/z_threshold)*best_top_amp)
#                     # print(node_amp, z_coord, top_pm_average_z, z_threshold, best_top_amp)
#                 else:
#                     node_amp = 0
#                     # # if the node is in the -x direction
#                     # if pre_curved_surface.xAxis[index] < 0:
#                     #     scaling_factor = top_neg_scaling_factor * (1 - abs(z_coord - top_pm_average_z)/z_threshold)
#                     # # if the node is in the +x direction
#                     # else:
#                     #     scaling_factor = top_pos_scaling_factor * (1 - abs(z_coord - top_pm_average_z)/z_threshold)
#                 # else:
#                 #     scaling_factor = 0
#             else:
#                 if abs(z_coord - bottom_pm_average_z) < z_threshold:
#                     node_amp = math.ceil((1- abs(z_coord - bottom_pm_average_z)/z_threshold)*best_bot_amp)
#                     # print(node_amp, z_coord, bottom_pm_average_z, z_threshold, best_bot_amp)
#                 else:
#                     node_amp = 0
#             # print('node amp =:', node_amp)
#             # print(len(interpolations))
                    
#             yInterpolator = interpolations[node_amp]
#             dist_to_center = ss.getSurfaceDistanceForXatZ(pre_curved_surface.xAxis[index],z_coord,yInterpolator)
            
#             new_x = fsolve(lambda x: ss.getSurfaceDistanceForXatZ(x,z_coord,yInterpolator)-dist_to_center, pre_curved_surface.zAxis[index],xtol = 1)
#             # print(new_x)
#             # print(z_coord)
#             new_y = yInterpolator(new_x[0],z_coord)
            
#             generated_surface.yAxis[index] = new_y
#             generated_surface.xAxis[index] = new_x[0]
            
#             # dist_func = lambda x_final : ss.getSurfaceDistanceForXatZ(interpolation.xAxis[index],z_coord,yInterpolator)-dist_to_center
#             # x_final_initial_guess = dist_to_center*(1-max_scaling)
#             # new_x_coord = round(fsolve(Wave_func, x_final_initial_guess)[0],3)


#                 #     if pre_curved_surface.xAxis[index] < 0:
#                 #         scaling_factor = bottom_neg_scaling_factor * (1 - abs(z_coord - bottom_pm_average_z)/z_threshold) 
#                 #     else:
#                 #         scaling_factor = bottom_pos_scaling_factor * (1 - abs(z_coord - bottom_pm_average_z)/z_threshold) 
#                 # else:
#                 #     scaling_factor = 0
        
        
        

        
        
#         ##### Narrowing Code Above ################        
#         ##### Narrowing Code Above ################
#         ##### Narrowing Code Above ################        
        

# #         z_coord = pre_curved_surface.zAxis[index]

# # #       Check to see if it is close to the PM
# #         if abs(z_coord - PM_Z_mean) < z_threshold:

# # # Calculate scaling factor
# # #            print(generated_surface.xAxis[index],generated_surface.yAxis[index],generated_surface.zAxis[index])
            
# #             # The closer it is to the PM (PM_z_coord) the more it needs to be adjusted. Adjusted between max_scaling and 0
# #             scaling_factor = max_scaling * (1 - abs(z_coord - PM_Z_mean)/z_threshold)
# # #            print('Scale = ', scaling_factor)
# # Generate the equation using that scaling factor


# # ########## MOVE OUTSIDE OF THE LOOP ##################
# #             yInterpolator = interpolate.Rbf(pre_curved_surface.xAxis,pre_curved_surface.zAxis,pre_curved_surface.yAxis,function = 'linear', smooth = 1)
# #             max_scaling = bottom_pos_scaling_factor
            
# # #           Trying to solve for the amplitude that will give the increased distance from center the correct amount
# #             Amplitude_func = lambda A : integrate.quad(lambda x: sqrt(1+(-A*2*pi/wave_width*sin(2*pi*x/wave_width))**2),0,wave_width)[0]/wave_width-(1+scaling_factor)
# #             A_initial_guess = 1
# #             Amplitude = fsolve(Amplitude_func, A_initial_guess)[0]
            
# #             # print("index:", index)
# #             # print('Amp = ', Amplitude)
            
            
# # # Find the distance from the point to the midline
# #             dist_to_center = ss.getSurfaceDistanceForXatZ(pre_curved_surface.xAxis[index],z_coord,yInterpolator)
# #             # print('Dist to center = ', dist_to_center)
# # # Find the coordinates on the sine wave to be at the same distance from the midline
# #             Wave_func = lambda x_final : integrate.quad(lambda x: sqrt(1+(-Amplitude*2*pi/wave_width*sin(2*pi*x/wave_width))**2),0,x_final)[0]-dist_to_center
# #             x_final_initial_guess = dist_to_center*(1-max_scaling)
# #             new_x_coord = round(fsolve(Wave_func, x_final_initial_guess)[0],3)
# # #            print('********************')
# #             # print('Old x:', pre_curved_surface.xAxis[index])
# #             # print('New x:', new_x_coord)
# #             delta_y = Amplitude*cos(new_x_coord*2*pi/wave_width)
# #             # print(delta_y)
# # # Find the new coordinates for the point (newX, newY+oldY, oldZ)
# #             generated_surface.yAxis[index] = pre_curved_surface.yAxis[index]-delta_y
# #             generated_surface.xAxis[index] = new_x_coord
# # #            print(generated_surface.xAxis[index],generated_surface.yAxis[index],generated_surface.zAxis[index])

# # Update it to the file
#     generated_surface = generated_surface.trim()
#     io.write_part_to_inp_file(file_name, part_name, generated_surface)

#     return generated_surface    

'''
Function: narrow_distal_avw_narrow
'''
def narrow_distal_avw_narrow(part_name, file_name, original_file_name, PM_Mid, PM_connections, AVW_connections):
    print("Narrow Distal End of AVW")

#   How far in the z direction (+/-) will be affected by the narrowing
    z_threshold = 5

    # Getting Nodes
    # Generic AVW (Maybe never used)
    generic_surface = io.get_dataset_from_file(original_file_name, part_name)
    # AVW that won't be modified
    pre_curved_surface = io.get_dataset_from_file(file_name, part_name).generatedCopy()
    # AVW that will be modified/narrowed
    generated_surface = io.get_dataset_from_file(file_name, part_name).generatedCopy()

    # Find the starting nodes for the AVW edges (starting at the apex)
    neg_start, pos_start = find_edge_starting_ending_points_for_AVW(pre_curved_surface)
    print(neg_start, pos_start)
    
    # 
    [negative_edges, positive_edges] = findEdgeNodes(generated_surface, AVW_connections, neg_start, pos_start)
    
    print('edges:', negative_edges, positive_edges)
    
    all_edge_nodes = negative_edges+positive_edges
    
    print(all_edge_nodes)
# Now that I have the edge nodes, check to see which ones are closest to the z...
    ### get within the threshold then grab min/max?
    
    
#   Get the mean of the Z coordinates for the inner arch on PM_Mid
    starting_index, ending_index = find_starting_ending_points_for_inside(PM_Mid)
    innerNodes = findInnerNodes(PM_Mid, PM_connections, starting_index, ending_index)
      
    inner_node_x_coords = []
    inner_node_y_coords = []
    inner_node_z_coords = []
    
    for node_number in innerNodes:
        inner_node_x_coords.append(PM_Mid.xAxis[node_number])
        inner_node_y_coords.append(PM_Mid.yAxis[node_number])
        inner_node_z_coords.append(PM_Mid.zAxis[node_number])
   
    # This is the average z coordinate of the inner arch of the PM_Mid
    PM_Z_mean = np.mean(inner_node_z_coords)
########################################################
    
    # Find the AVW edge nodes that are at that are close to that Z location
    # I think I need to find top/bottom negative/positive to use for later
    AVW_negative_top_y = -9999
    AVW_negative_bottom_y = 9999
    
    for node in negative_edges:
        index = node - 1
        z_coord = pre_curved_surface.zAxis[index]

#       If the node is close to the inner PM arch
        if abs(z_coord - PM_Z_mean) < z_threshold:
            # if the y coordinate is higher than the current highest coordinate
            #     record it as the highest one
            if pre_curved_surface.yAxis[index] > AVW_negative_top_y:
                AVW_negative_top_y = pre_curved_surface.yAxis[index]

#               if the y coordiante is lower than the current lowest coordiante
                #  record it as the lowest one
            if pre_curved_surface.yAxis[index] < AVW_negative_bottom_y:
                AVW_negative_bottom_y = pre_curved_surface.yAxis[index]

                
    AVW_positive_top_y = -9999
    AVW_positive_bottom_y = 9999

# same thing for the positive x side
    for node in positive_edges:
        index = node - 1
        z_coord = pre_curved_surface.zAxis[index]

        if abs(z_coord - PM_Z_mean) < z_threshold:
            if pre_curved_surface.yAxis[index] > AVW_positive_top_y:
                AVW_positive_top_y = pre_curved_surface.yAxis[index]
                # AVW_positive_top_x = pre_curved_surface.xAxis[index]
            if pre_curved_surface.yAxis[index] < AVW_positive_bottom_y:
                AVW_positive_bottom_y = pre_curved_surface.yAxis[index]
                # AVW_positive_bottom_x = pre_curved_surface.xAxis[index]
                
    y_threshold = 3
    
    AVW_negative_top_x = 9999
    AVW_negative_bottom_x = 9999
    for node in negative_edges:
        index = node - 1
#           if the node is close to the arc in the z direction AND
        #       if it is close to the top of the avw
        if abs(pre_curved_surface.zAxis[index] - PM_Z_mean) < z_threshold and abs(pre_curved_surface.yAxis[index] - AVW_negative_top_y) < y_threshold:
            if pre_curved_surface.xAxis[index] < AVW_negative_top_x:
                AVW_negative_top_x = pre_curved_surface.xAxis[index]            
        if abs(pre_curved_surface.zAxis[index] - PM_Z_mean) < z_threshold and abs(pre_curved_surface.yAxis[index] - AVW_negative_bottom_y) < y_threshold:
            if pre_curved_surface.xAxis[index] < AVW_negative_bottom_x:
                AVW_negative_bottom_x = pre_curved_surface.xAxis[index]
                
    AVW_positive_top_x = -9999
    AVW_positive_bottom_x = -9999
    for node in positive_edges:
        index = node - 1
        if abs(pre_curved_surface.zAxis[index] - PM_Z_mean) < z_threshold and abs(pre_curved_surface.yAxis[index] - AVW_positive_top_y) < y_threshold:
            if pre_curved_surface.xAxis[index] > AVW_positive_top_x:
                AVW_positive_top_x = pre_curved_surface.xAxis[index]            
        if abs(pre_curved_surface.zAxis[index] - PM_Z_mean) < z_threshold and abs(pre_curved_surface.yAxis[index] - AVW_positive_bottom_y) < y_threshold:
            if pre_curved_surface.xAxis[index] > AVW_positive_bottom_x:
                AVW_positive_bottom_x = pre_curved_surface.xAxis[index]

                    
    # Find nodes Inner Arch PM Nodes that are closest to the Y coordiantes for top and bottom
    # Determine the distances between the archs
    pm_best_neg_top_y = 9999
    pm_best_pos_top_y = 9999
    pm_best_neg_bottom_y = 9999
    pm_best_pos_bottom_y = 9999
    pm_top_neg_set = 0
    pm_top_pos_set = 0
    pm_bot_neg_set = 0
    pm_bot_pos_set = 0
    for i, x_coord in enumerate(inner_node_x_coords):
        # and the x coordinae is negative
        if x_coord < 0: 
            # if the coordinate is above the top of the AVW
            if inner_node_y_coords[i] > AVW_negative_top_y:
                # and it is closer to the top AVW than the previous best point
                if abs(inner_node_y_coords[i] - AVW_negative_top_y) < abs(pm_best_neg_top_y - AVW_negative_top_y):
                    pm_best_neg_top_y = inner_node_y_coords[i]
                    pm_best_neg_top_x = x_coord
                    pm_best_neg_top_z = inner_node_z_coords[i]
                    pm_top_neg_set = 1
            elif inner_node_y_coords[i] > AVW_negative_bottom_y:
                if abs(inner_node_y_coords[i] - AVW_negative_bottom_y) < abs(pm_best_neg_bottom_y - AVW_negative_bottom_y):
                    pm_best_neg_bottom_y = inner_node_y_coords[i]
                    pm_best_neg_bottom_x = x_coord
                    pm_best_neg_bottom_z = inner_node_z_coords[i]
                    pm_bot_neg_set = 1
            # if inner_node_y_coords[i] > positive_top_y:
            #     # and it is closer to the top AVW than the previous best point
            #     if inner_node_y_coords[i] - positive_top_y < best_pos_top_y - positive_top_y:
            #         best_pos_top_y = inner_node_y_coords[i]
            #         best_pos_top_x = x_coord
            #         best_pos_top_z = inner_node_z_coords[i]
        else:
            if inner_node_y_coords[i] > AVW_positive_top_y:
                if abs(inner_node_y_coords[i] - AVW_positive_top_y) < abs(pm_best_pos_top_y - AVW_positive_top_y):
                    pm_best_pos_top_y = inner_node_y_coords[i]
                    pm_best_pos_top_x = x_coord
                    pm_best_pos_top_z = inner_node_z_coords[i]
                    pm_top_pos_set = 1
            elif inner_node_y_coords[i] > AVW_positive_bottom_y:
                if abs(inner_node_y_coords[i] - AVW_positive_bottom_y) < abs(pm_best_pos_bottom_y - AVW_positive_bottom_y):
                    pm_best_pos_bottom_y = inner_node_y_coords[i]
                    pm_best_pos_bottom_x = x_coord     
                    pm_best_pos_bottom_z = inner_node_z_coords[i]
                    pm_bot_pos_set = 1
                # and the x coordinae is negative
            # if x_coord < 0: 

            # else:
            #     # and it is closer to the top AVW than the previous best point
            #     if inner_node_y_coords[i] - bottom_y < best_pos_bottom_y - bottom_y:
            #         best_pos_bottom_y = inner_node_y_coords[i]
            #         best_pos_bottom_x = x_coord     
            #         best_pos_bottom_z = inner_node_z_coords[i]
    # top_pm_width =  best_pos_top_x - best_neg_top_x
    # bottom_pm_width = best_pos_bottom_x - best_neg_bottom_x
                    
                    
    #  Get the z coordinates for where the PM is high and low
    if pm_top_neg_set and pm_top_pos_set:
        top_pm_average_z = (pm_best_neg_top_z + pm_best_pos_top_z) / 2
    else:
        top_pm_average_z = 9999
    if pm_bot_neg_set and pm_bot_pos_set:
        bottom_pm_average_z = (pm_best_neg_bottom_z + pm_best_pos_bottom_z) / 2
    else:
        bottom_pm_average_z = 9999
                    
    
    
    
    bottom_x_buffer = 2
    
    if abs(AVW_negative_top_x) > abs(pm_best_neg_top_x):
        top_neg_scaling_factor = 1 - pm_best_neg_top_x/AVW_negative_top_x
    else:
        top_neg_scaling_factor = 0
        
    if abs(AVW_positive_top_x) > abs(pm_best_pos_top_x):
        top_pos_scaling_factor = 1 - pm_best_pos_top_x/AVW_positive_top_x
    else:
        top_pos_scaling_factor = 0

    if pm_bot_neg_set == 0:
        pm_best_neg_bottom_x = pm_best_neg_top_x
    if abs(AVW_negative_bottom_x) + bottom_x_buffer > abs(pm_best_neg_bottom_x):
        bottom_neg_scaling_factor = 1 - pm_best_neg_bottom_x/(AVW_negative_bottom_x - bottom_x_buffer)
    else:
        bottom_neg_scaling_factor = 0
        
    if pm_bot_pos_set == 0:
        pm_best_pos_bottom_x = pm_best_pos_top_x
    if abs(AVW_positive_bottom_x) > abs(pm_best_pos_bottom_x):
        bottom_pos_scaling_factor = 1 - pm_best_pos_bottom_x/(AVW_positive_bottom_x + bottom_x_buffer)
    else:
        bottom_pos_scaling_factor = 0
    
    # if top_pm_width < top_avw_width:
    #     top_max_scaling_factor = 1 - top_pm_width/top_avw_width
    # else:
    #     top_max_scaling_factor = 0
    # if bottom_pm_width < bottom_avw_width:
    #     bottom_max_scaling_factor = 1 - bottom_pm_width/bottom_avw_width
    # else:
    #     bottom_max_scaling_factor = 0

    print('***')    
    print('pm tops and bottoms of x', pm_best_neg_top_x, pm_best_pos_top_x, pm_best_neg_bottom_x, pm_best_pos_bottom_x)
    print('AVW tops and bottoms of x', AVW_negative_top_x, AVW_positive_top_x, AVW_negative_bottom_x, AVW_positive_bottom_x)
    print('scaling factor, top neg, top pos, bot neg, bot pos:', top_neg_scaling_factor, top_pos_scaling_factor, bottom_neg_scaling_factor, bottom_pos_scaling_factor)

    print('***')
    # print('pm_z:', PM_Z_mean)
    # print('top y', top_y)
    # print('bottom y', bottom_y)
    # print('top and bottom xs:', top_min_x,top_max_x,bottom_min_x,bottom_max_x)
    # print('pm top and bottom ys', best_pos_top_y, best_neg_top_y, best_pos_bottom_y, best_neg_bottom_y)
    # print('avw widths', top_avw_width, bottom_avw_width)
    # print('pm widths', top_pm_width, bottom_pm_width)
    # print('top and bottom scaling factors', top_max_scaling_factor, bottom_max_scaling_factor)
    # print('top and bottom pm Z', top_pm_average_z,bottom_pm_average_z)
    
#### going to skip checking if it needs to be scaled for now since I know it does for this one
    # if the quadrant has scaling factor of > 0
    # do the finding the pm node again with the distance to minimize being
        # abs(abs(AVWx - PMx) - abs(AVWy-PMy))
    # once you have the new PM node locations, the scaling factor is the same as before
    # for each node, figure out the distance from it is going in the y direction and move it
        # in in the x direction that some amount
    # Find nodes Inner Arch PM Nodes that are closest to the Y coordiantes for top and bottom
    # Determine the distances between the archs
    pm_best_neg_top_y = 9999
    pm_best_pos_top_y = 9999
    pm_best_neg_bottom_y = 9999
    pm_best_pos_bottom_y = 9999
    pm_top_neg_set = 0
    pm_top_pos_set = 0
    pm_bot_neg_set = 0
    pm_bot_pos_set = 0
    for i, x_coord in enumerate(inner_node_x_coords):
        # and the x coordinae is negative
        if x_coord < 0: 
            # if the coordinate is above the top of the AVW
            # if inner_node_y_coords[i] > AVW_negative_top_y:
                # and it is closer to the top AVW than the previous best point
                if abs(abs(inner_node_y_coords[i] - AVW_negative_top_y) - abs(inner_node_x_coords[i] - AVW_negative_top_x)) < abs(abs(pm_best_neg_top_y - AVW_negative_top_y) - abs(pm_best_neg_top_x - AVW_negative_top_x)):
                    pm_best_neg_top_y = inner_node_y_coords[i]
                    pm_best_neg_top_x = x_coord
                    pm_best_neg_top_z = inner_node_z_coords[i]
                    pm_top_neg_set = 1
            # elif inner_node_y_coords[i] > AVW_negative_bottom_y:
                # if abs(abs(inner_node_y_coords[i] - AVW_negative_bottom_y) - abs(inner_node_x_coords[i] - AVW_negative_bottom_x)) < abs(abs(pm_best_neg_bottom_y - AVW_negative_bottom_y) - abs(pm_best_neg_bottom_x - AVW_negative_bottom_x)):
                # # if abs(inner_node_y_coords[i] - AVW_negative_bottom_y) < abs(pm_best_neg_bottom_y - AVW_negative_bottom_y):
                #     pm_best_neg_bottom_y = inner_node_y_coords[i]
                #     pm_best_neg_bottom_x = x_coord
                #     pm_best_neg_bottom_z = inner_node_z_coords[i]
                #     pm_bot_neg_set = 1
            # if inner_node_y_coords[i] > positive_top_y:
            #     # and it is closer to the top AVW than the previous best point
            #     if inner_node_y_coords[i] - positive_top_y < best_pos_top_y - positive_top_y:
            #         best_pos_top_y = inner_node_y_coords[i]
            #         best_pos_top_x = x_coord
            #         best_pos_top_z = inner_node_z_coords[i]
        else:
            # if inner_node_y_coords[i] > AVW_positive_top_y:
                if abs(abs(inner_node_y_coords[i] - AVW_positive_top_y) - abs(inner_node_x_coords[i] - AVW_positive_top_x)) < abs(abs(pm_best_pos_top_y - AVW_positive_top_y) - abs(pm_best_pos_top_x - AVW_positive_top_x)):
                # if abs(inner_node_y_coords[i] - AVW_positive_top_y) < abs(pm_best_pos_top_y - AVW_positive_top_y):
                    pm_best_pos_top_y = inner_node_y_coords[i]
                    pm_best_pos_top_x = x_coord
                    pm_best_pos_top_z = inner_node_z_coords[i]
                    pm_top_pos_set = 1
            # elif inner_node_y_coords[i] > AVW_positive_bottom_y:
                # if abs(abs(inner_node_y_coords[i] - AVW_positive_bottom_y) - abs(inner_node_x_coords[i] - AVW_positive_bottom_x)) < abs(abs(pm_best_pos_bottom_y - AVW_positive_bottom_y) - abs(pm_best_pos_bottom_x - AVW_positive_bottom_x)):
                # # if abs(inner_node_y_coords[i] - AVW_positive_bottom_y) < abs(pm_best_pos_bottom_y - AVW_positive_bottom_y):
                #     pm_best_pos_bottom_y = inner_node_y_coords[i]
                #     pm_best_pos_bottom_x = x_coord     
                #     pm_best_pos_bottom_z = inner_node_z_coords[i]
                #     pm_bot_pos_set = 1

    if pm_top_neg_set and pm_top_pos_set:
        top_pm_average_z = (pm_best_neg_top_z + pm_best_pos_top_z) / 2
    else:
        top_pm_average_z = 9999
    # if pm_bot_neg_set and pm_bot_pos_set:
    #     bottom_pm_average_z = (pm_best_neg_bottom_z + pm_best_pos_bottom_z) / 2
    # else:
    #     bottom_pm_average_z = 9999    
    
    bottom_x_buffer = 2
    top_x_buffer = 2
    
    if abs(AVW_negative_top_x) > abs(pm_best_neg_top_x):
        top_neg_scaling_factor = 1 - pm_best_neg_top_x/(AVW_negative_top_x - top_x_buffer)
    else:
        top_neg_scaling_factor = 0
        
    if abs(AVW_positive_top_x) > abs(pm_best_pos_top_x):
        top_pos_scaling_factor = 1 - pm_best_pos_top_x/(AVW_positive_top_x + top_x_buffer)
    else:
        top_pos_scaling_factor = 0

    # if pm_bot_neg_set == 0:
    #     pm_best_neg_bottom_x = pm_best_neg_top_x
    # if abs(AVW_negative_bottom_x) + bottom_x_buffer > abs(pm_best_neg_bottom_x):
    #     bottom_neg_scaling_factor = 1 - pm_best_neg_bottom_x/(AVW_negative_bottom_x - bottom_x_buffer)
    # else:
    #     bottom_neg_scaling_factor = 0
        
    # if pm_bot_pos_set == 0:
    #     pm_best_pos_bottom_x = pm_best_pos_top_x
    # if abs(AVW_positive_bottom_x) > abs(pm_best_pos_bottom_x):
    #     bottom_pos_scaling_factor = 1 - pm_best_pos_bottom_x/(AVW_positive_bottom_x + bottom_x_buffer)
    # else:
    #     bottom_pos_scaling_factor = 0

    print('***')    
    print('pm tops and bottoms of x', pm_best_neg_top_x, pm_best_pos_top_x, pm_best_neg_bottom_x, pm_best_pos_bottom_x)
    print('AVW tops and bottoms of x', AVW_negative_top_x, AVW_positive_top_x, AVW_negative_bottom_x, AVW_positive_bottom_x)
    print('scaling factor, top neg, top pos, bot neg, bot pos:', top_neg_scaling_factor, top_pos_scaling_factor, bottom_neg_scaling_factor, bottom_pos_scaling_factor)

    print('***')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Determine the scaling factors
    # When looking at a node, check to see if it is in the Z range, and then which Y it is the closest to
    # Use that scaling to determine the position



    
########################################################################################################################################################################
# Pasting Function to work with


# Check each point to see if it is within the threshold or not
    for index in range(0,len(pre_curved_surface.zAxis)):
        top_scaling_used = 0
        z_coord = pre_curved_surface.zAxis[index]

#       if the AVW node is close to where it would pass through the PM Mid
        if abs(z_coord - PM_Z_mean) < z_threshold + 5:

            # print(index)
# Calculate scaling factor
#            print(generated_surface.xAxis[index],generated_surface.yAxis[index],generated_surface.zAxis[index])
            
            # # The closer it is to the PM (PM_z_coord) the more it needs to be adjusted. Adjusted between max_scaling and 0
            # scaling_factor = max_scaling * (1 - abs(z_coord - PM_z_coord)/z_threshold)
            
            
            
            # Changing to checking if it is closer to the top y coordinate or
            # bottom and then assigning the correct scaling factor
            # If closer to the top avw/pm interface than the bottom
            if (abs(pre_curved_surface.yAxis[index] - (AVW_positive_top_y + AVW_negative_top_y)/2) < abs(pre_curved_surface.yAxis[index] - (AVW_positive_bottom_y + AVW_negative_bottom_y)/2)):
                if abs(z_coord - top_pm_average_z) < z_threshold:
                    if pre_curved_surface.xAxis[index] < 0:
                        # scaling_factor = top_neg_scaling_factor * (1 - abs(z_coord - top_pm_average_z)/z_threshold)
                        scaling_factor = top_neg_scaling_factor
                        top_scaling_used = 1
                    else:
                        # scaling_factor = top_pos_scaling_factor * (1 - abs(z_coord - top_pm_average_z)/z_threshold)
                        scaling_factor = top_pos_scaling_factor
                        top_scaling_used = 1
                else:
                    scaling_factor = 0
            else:
                if abs(z_coord - bottom_pm_average_z) < z_threshold:
                    if pre_curved_surface.xAxis[index] < 0:
                        # scaling_factor = bottom_neg_scaling_factor * (1 - abs(z_coord - bottom_pm_average_z)/z_threshold) 
                        scaling_factor = bottom_neg_scaling_factor
                    else:
                        # scaling_factor = bottom_pos_scaling_factor * (1 - abs(z_coord - bottom_pm_average_z)/z_threshold)
                        scaling_factor = bottom_pos_scaling_factor
                else:
                    scaling_factor = 0
            
            # print('Scale = ', scaling_factor)
# Generate the equation using that scaling factor
            # Amplitude_func = lambda A : integrate.quad(lambda x: sqrt(1+(-A*2*pi/wave_width*sin(2*pi*x/wave_width))**2),0,wave_width)[0]/wave_width-(1+scaling_factor)
            # A_initial_guess = 1
            # Amplitude = fsolve(Amplitude_func, A_initial_guess)[0]
#            print('Amp = ', Amplitude)
            
            
# Find the distance from the point to the midline
            # dist_to_center = ss.getSurfaceDistanceForXatZ(pre_curved_surface.xAxis[index],z_coord,yInterpolator)
#            print('Dist to center = ', dist_to_center)
# Find the coordinates on the sine wave to be at the same distance from the midline
            # Wave_func = lambda x_final : integrate.quad(lambda x: sqrt(1+(-Amplitude*2*pi/wave_width*sin(2*pi*x/wave_width))**2),0,x_final)[0]-dist_to_center
            # x_final_initial_guess = dist_to_center*(1-max_scaling)
            # new_x_coord = round(fsolve(Wave_func, x_final_initial_guess)[0],3)
            new_x_coord = pre_curved_surface.xAxis[index]*(1-scaling_factor)
#            print('********************')
#            print(new_x_coord)
            # delta_y = Amplitude*cos(new_x_coord*2*pi/wave_width)
#            print(delta_y)
# Find the new coordinates for the point (newX, newY+oldY, oldZ)
            if top_scaling_used:
                generated_surface.yAxis[index] = pre_curved_surface.yAxis[index] - abs(pre_curved_surface.xAxis[index] - new_x_coord)
            else:
                generated_surface.yAxis[index] = pre_curved_surface.yAxis[index]
            generated_surface.xAxis[index] = new_x_coord
#            print(generated_surface.xAxis[index],generated_surface.yAxis[index],generated_surface.zAxis[index])

# Update it to the file
    generated_surface = generated_surface.trim()
    io.write_part_to_inp_file(file_name, part_name, generated_surface)

    return generated_surface