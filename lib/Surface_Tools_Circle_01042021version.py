#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Jun  7 19:56:24 2017
@author: Aaron Renfroe @ California Baptist University
"""


import numpy as np
from scipy import interpolate, optimize
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from lib.workingWith3dDataSets import DataSet3d, Point
from lib.IOfunctions import extractPointsForPartFrom
from lib.workingWithCatenarys import CatenaryCurve
from scipy import sqrt, sin, pi, interpolate, cos, integrate
from scipy.optimize import fsolve
from time import sleep
import sys


INCREMENT = 0.02
INCREMENT = .1
SMOOTH_VAL = 1

'''
Main Function
Opens csv and reads in values
Currently scales X Values
'''

def find_yz_properties_of_x_plane(x, surface):
    maxz = float("-inf")
    y_at_max_z = 0
    minz = float("inf")
    y_at_min_z = 0
    
    for index in range(0,len(surface.zAxis)):
        # finding z at x close to 0
        val = surface.zAxis[index]
        x_val = surface.xAxis[index]
        if x - 3 < x_val < x + 3:
            if maxz < val:
                maxz = val
                y_at_max_z = surface.yAxis[index]
            if minz > val:
                minz = val
                y_at_min_z = surface.yAxis[index]
                
    return maxz, y_at_max_z, minz, y_at_min_z

#=============================
# Dr. Gordons Curve Algorithm
#=============================
def apply_curve_after_threshold(generic_surface, generated_surface, hiatus, z_cutoff, rotate_angle, rot_point):
    
    # =======
    # STEP 1: Have an interpolated surface where you can give it X and Z and it will give you Y
    # =======
    # given zx you get y
    
    generic_interpolator_zx_y = interpolate.Rbf(generic_surface.zAxis, generic_surface.xAxis, generic_surface.yAxis, function="linear")
    
    # ===============================================================
    # STEP 2: Have a 3D spline for the endpoints of the original  AVW
    # STEP 3: Have a 3D spline for the endpoints of the stretched AVW
    # Note I Just went ahead and used the surface interpolators instead
    # ===============================================================
    
    
    # given zx you get y
    generated_interpolator_zx_y = interpolate.Rbf(generated_surface.zAxis, generated_surface.xAxis, generated_surface.yAxis, function="linear", smooth=5)
#    generated_interpolator_zx_y = interpolate.Rbf(generated_surface.zAxis, generated_surface.xAxis, generated_surface.yAxis, function="thin-plate")
    # ====================
    # Prepping for Step 4
    # ====================
    
    
    # Moved the For loop in to a function for reuability: see -> find_yz_properties_of_x_plane
    # getting maxz, y_at_max_z, minz, y_at_min_z of x plane as tuple
    (maxz_at_x0_plane, y_at_max_z_x0, minz_at_x0_plane, y_at_min_z_x0) = find_yz_properties_of_x_plane( hiatus.x, generated_surface)
    (generic_maxz_at_x0_plane, generic_y_at_max_z_x0, generic_minz_at_x0_plane, generic_y_at_min_z_x0) = find_yz_properties_of_x_plane( hiatus.x, generic_surface)    
    minZ_generic = min(generic_surface.zAxis)+1
    
    
    slope = (minz_at_x0_plane - maxz_at_x0_plane)/(y_at_min_z_x0 - y_at_max_z_x0)
        
    end_y = generic_interpolator_zx_y(minZ_generic, 0)
    

#    I should really let it go down to the bottom of the rotated generic surface because that is where the GI_Filler will end
    
    rotated_generic = rotate_part(generic_surface, rotate_angle, rot_point)
    rotated_zy_properties = find_yz_properties_of_x_plane( hiatus.x, rotated_generic) # returns maxz, y_at_max_z, minz, y_at_min_z of x plane as tuple
    
    ### I believe this is the point you wanted
    rotated_generic_minz_on_hiatus_x = rotated_zy_properties[2] # index is 3 becuase min_z is the third value returned
    
    # Force a crash for faster testing
    #raise NotImplementedError ("Testing function: this crash occured becasue a dev was testing code and did not want to continue execution. This error can be removed by comment")

    p2 = Point(0, generic_y_at_min_z_x0, generic_minz_at_x0_plane)
    
#    print('p2 original:', p2)

    #print("z_cutoff = " + str(z_cutoff))
#    print('p1,p2:', p1,p2)
    z_cutoff = optimize_cutoff_value(z_cutoff, slope, generated_interpolator_zx_y, maxz_at_x0_plane, rotated_generic_minz_on_hiatus_x, minz_at_x0_plane, p2, max(generated_surface.zAxis))
    print("Here it is :", z_cutoff)   

    min_generated_z = min(generated_surface.zAxis)



    AVW_Angle = math.degrees(math.atan2(-1 * np.sign(slope),abs(slope)))
#    print(AVW_Angle)
#    AVW_Angle = math.degrees(math.atan2(-1 * np.sign(slope),abs(slope))) + 20
#    print(AVW_Angle)
    
# #############------------------------------------------------------------------
# #############------------------------------------------------------------------
# #############------------------------------------------------------------------
# #############------------------------------------------------------------------
# #############------------------------------------------------------------------
# #def apply_curve_after_threshold(generic_surface, generated_surface, hiatus, z_cutoff, rotate_angle, rot_point):

#     pre_curved_surface = generated_surface
#     generated_surface = generated_surface
    
# ########################################################################################################################################################################
# # Pasting Function to work with

# #Set constant variables    
# #    PM_z_coord = -24
# #    z_threshold = 20
#     max_scaling = 0.35
#     wave_width = 36
#     distal_center= -2.63
# #    yInterpolator = interpolate.Rbf(pre_curved_surface.xAxis,pre_curved_surface.zAxis,pre_curved_surface.yAxis,function = 'linear', smooth = 1)

# # Check each point to see if it is within the threshold or not
# #    for index in range(0,len(pre_curved_surface.zAxis)):
# #        z_coord = pre_curved_surface.zAxis[index]

# #        if abs(z_coord - PM_z_coord) < z_threshold:

# #            print(index)
# # Calculate scaling factor
# #            print(generated_surface.xAxis[index],generated_surface.yAxis[index],generated_surface.zAxis[index])
#     scaling_factor = max_scaling
# #            print('Scale = ', scaling_factor)
# # Generate the equation using that scaling factor
#     # possibly good one, but with factors of 2 in it
# #    Amplitude_func = lambda A : integrate.quad(lambda x: sqrt(1+(-A*2*pi/wave_width*sin(2*pi*x/wave_width))**2),0,wave_width)[0]/wave_width-(1+scaling_factor)
#     Amplitude_func = lambda A : integrate.quad(lambda x: sqrt(1+(-A*pi/wave_width*sin(pi*x/wave_width))**2),0,wave_width)[0]/wave_width-(1+scaling_factor)
#     #            The Amplitude of the waves is determine by howclose the z coordinate is to the PM_z_coord as determine by SCALING_FACTOR
# #    Amplitude_func = lambda A : integrate.quad(lambda x: sqrt(1+(-A*2*pi/wave_width*sin(2*pi*x/wave_width))**2),0,wave_width)[0]/wave_width-(1+scaling_factor)
#     A_initial_guess = 1
#     Amplitude = optimize.fsolve(Amplitude_func, A_initial_guess)[0]

#     print('Amp:', Amplitude)


# #############------------------------------------------------------------------
# #############------------------------------------------------------------------
# #############------------------------------------------------------------------
# #############------------------------------------------------------------------
# #############------------------------------------------------------------------
# #############------------------------------------------------------------------
    
    
    #-----------------------------------------------------------------
    for index in range(0,len(generated_surface.zAxis)):
        val = generated_surface.zAxis[index]
#        print(index)
        # ================================================================================
        # STEP 4: Check the Z coordinate of a point to determine if it is below the cutoff
        # ================================================================================
#        print('Z CUTOFF:', z_cutoff)
        if val < z_cutoff:
            x_val = generated_surface.xAxis[index]
            y_val = generated_surface.yAxis[index]
            # ==================================================================================
            # NO STEP 5
            # ==================================================================================
            # STEP 6: Take the X coordinate and find the point on the interpolated surface given 
            # the X coordinate from the point and the Z coordinate of the cutoff. This gives you
            # a point that is on the same ZY plane (X coordinate) as the point you care about, 
            # and is at the point of the cutoff.
            # ==================================================================================
            
            # result of step 6
            # p1 is cutoff point on the surface
            p1 = Point(x_val, generated_interpolator_zx_y(z_cutoff, x_val), z_cutoff)
            
            # ===============================================================================
            # STEP 7: Take the X coordinate and find the point on the stretched AVW 3D spline 
            # with that X coordinate
            # ===============================================================================
            # Might need a y here, maybe because it's a spline we don't need a y
            # but we will need a y for the next step
        
            # ========================================
            # STEP 8: finding the length of the "rope"
            # This looks to be the distance from the z cutoff to the end of the AVW
            # ========================================
            length_of_droop = abs(getSurfaceDistance(p1.z, min_generated_z, x_val, generated_interpolator_zx_y))
            
            # ==================================================================================
            # STEP 9: Take the X coordinate, and find the point on the original 3D distal spline with that X coordinate
            # ==================================================================================

            end_y = generic_interpolator_zx_y(minZ_generic, x_val)
            
            #######----------------------------------------------


            x_threshold = 1200
####################################################################
####################################################################
####################################################################
#           p2 is the point at the end of the AVW. p1 is the point at the cuttoff
#           these two points are used to put the new point between  
            # print('p2 before:', p2)
            if abs(x_val) < x_threshold:
#                p2 = Point(x_val, end_y, minZ_generic)
                delta_x = 0
                delta_y = 0
            else:
#                print(x_val)
                delta_x = (x_threshold + (abs(x_val)- x_threshold)*2**0.5 / 2)* np.sign(x_val) - x_val
                delta_y = -(x_threshold - abs(x_val))*2**0.5 / 2
#                p2 = Point((x_threshold + (abs(x_val)- x_threshold)*2**0.5 / 2)* np.sign(x_val), end_y + (x_threshold - abs(x_val))*2**0.5 / 2, minZ_generic)
#                print('**previous x plane and new coordinates:', x_val, p2)
                print('****x_val, threshold, y_val, end_y:', x_val, x_threshold, y_val, end_y)
            
                print('delta_x and y', delta_x, delta_y)
                print('possible other delta x:', )
            p2 = Point(x_val, end_y, minZ_generic)
            print('**previous x plane and new coordinates:', index, x_val, p2)
            # ===============================================================
            # STEP 10: Make your catenary curve using the points from 9 and 6
            # ===============================================================
            
#            p2_first_rotation = rotate_point(p2.x, p2.y, p2.z, AVW_Angle, p1)

#           NEED TO ROTATE AGAIN, BUT THIS TIME ABOUT THE Z AXIS
            
#            print('after first rotation:', p2_first_rotation)
#            p2_first_rotation = p2
#            
#            cat_angle = math.degrees(math.atan2(p2_first_rotation.x - p1.x, p2_first_rotation.y - p1.y))
##            cat_angle = cat_angle * np.sign(p2_first_rotation.x)
#
#            p2_rotated = rotate_point_about_z(p2_first_rotation.x, p2_first_rotation.y, p2_first_rotation.z, cat_angle, p1)
##            print('p2 after being rotated back...the x should match xval from above:', p2_rotated, p1, cat_angle)
#            p2 = p2_rotated

#            print('after second rotation:', p2_rotated)
#??????????????????????????????????????
#            if p2_rotated.y < p1.y + 0.1:
#                if index == 21 - 1:
#                    print("21!")
#                p2_rotated.y = p1.y + 0.1
                
#            direct length between points
            sl = pythag(p2.y - p1.y, p2.z - p1.z)
            if sl <= length_of_droop:

                point_1 = np.array([p1.y, p1.z])
                point_2 = np.array([p2.y, p2.z])
                arc_length = length_of_droop
                                
                bisection_point = [(point_1[0]+point_2[0])/2, (point_1[1]+point_2[1])/2]
                                               
                cord_vector = point_1 - point_2
                cord_slope = cord_vector[1]/cord_vector[0]

                if cord_slope == 0:
                    bisector_slope = 999999999
                else:
                    bisector_slope = -1/cord_slope

                y_intercept = bisection_point[1]-bisector_slope * bisection_point[0]
                
                data = (point_1, point_2, arc_length, bisector_slope, y_intercept)

                zGuess = np.array([point_1[0] - 1])
                x_c = fsolve(circle_center, zGuess, args = data)[0]

                y_c = bisector_slope * x_c + y_intercept
                
                center = np.array([x_c,y_c])

                r = np.linalg.norm(point_1 - center)
#
#                print('##arc length', arc_length)
#                print('##point_1:', point_1)
#                print('##point_2:', point_2)
#                print('##center:', center)
#                print('##r:', r)
                
#                The way the circle code was originally written 
                z_c = y_c
                y_c = x_c
                
                
#                This is the arc_length from the takeoff point to the point in question
                wanted_arc_length = abs(getSurfaceDistance(p1.z, val, x_val, generated_interpolator_zx_y))
                
                
                distance_fraction = wanted_arc_length / length_of_droop
                print('distance_fraction',distance_fraction)
#                print("wanted : ", wanted_arc_length)
#                if (length_of_droop < wanted_arc_length):
#                    print("Oh lord help me", generated_end_point.z, val)
                # =======================================================================================
                # STEP 11: Trace along your catenary curve from point 6 until you hit the wanted distance 
                # =======================================================================================
                circle = [y_c, z_c, r, p1]
#                print('original_point :', x_val, y_val, val)
#                print('$$$$$$$', circle, wanted_arc_length)
                yz_tuple = calc_new_point_on_circle_curve(wanted_arc_length, circle)
                if math.isnan(yz_tuple[0]):
                    print('THINGS WENT POORLY (NAN in yz_tuple)')
                    sys.exit()
                    yz_tuple[0] = y_val
#                yz_tuple = calc_new_point_on_curve(wanted_arc_length, ccc, p1)
#                print('drooped point 0 (tilt back):', p2_rotated.x, yz_tuple[0], yz_tuple[1])

#                print('yz_tuple1', yz_tuple)
#                final_point = rotate_point_about_z(p2_rotated.x - delta_x * distance_fraction, yz_tuple[0], yz_tuple[1], -cat_angle, p1)
                final_point = Point(p2.x - delta_x * distance_fraction, yz_tuple[0] - delta_y * distance_fraction, yz_tuple[1])

                ##### Testing because the end points seem to be messed up currently
                if distance_fraction > 0.98:
                    final_point = p2
#                print('final point...this should have an x value between the x and x_val:', final_point, p1, -cat_angle, p2_rotated.x)
#                print('point before rotation:', p2_rotated.x, yz_tuple[0], yz_tuple[1])
#                print('drooped point 1:', point_unfirst_rotated)
#                final_point = rotate_point(point_unfirst_rotated.x, point_unfirst_rotated.y, point_unfirst_rotated.z, -AVW_Angle, p1)
#                print('drooped point 2:', final_point)
#                final_point = rotate_point(p2_rotated.x, yz_tuple[0], yz_tuple[1], AVW_Angle, p1)                
#                final_point = Point(p2.x, yz_tuple[0], yz_tuple[1])
                generated_surface.yAxis[index] = final_point.y
                generated_surface.xAxis[index] = final_point.x
#                if (yz_tuple[1] < -100):
#                    
#                    generated_surface.zAxis[index] = (p1.z + p2.z)/2.0
#                    print("x+x+x+x+x+x+x+x+x+x+x+x- ERROR #2 -x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+")
#                    print("Curve Function has Exploded! Cover your heads")
#                    print("Something Went Wrong with The Drooping\n Process "
#                          + "Probably means that p1 and p2 two are really close"
#                          + "\n and the length was something unreasonable for the math to work out."
#                          +" See output Below. Average of p1.z and p2.z was used instead")
#                    print(" z: ", yz_tuple[1])
#                    print(" a: ", ccc.a)
#                    print("P1", p1)
#                    print("P2", p2)
#                    print("length", length_of_droop)
#                    print("")
#                    print("x+x+x+x+x+x+x+x+x+x+x+x- END ERROR -x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+")
#                    #raise ValueError("Something WentWrong with The Drooping Process, Probably means that p1 and p2 two are really close and the lingth was something unreasonable for the math to work out See output")
#                else:
#                    generated_surface.zAxis[index] = final_point.z
                generated_surface.zAxis[index] = final_point.z
            # Triggered When a droop length is shorter than direct distance  
            else:
                print("lengthError")

    return generated_surface
        



def circle_center(z2, *data):
    # print('z:',z2)
    p1, p2, arc_dist, m, b = data
    
#    print('p1:', p1)
#    print('p2:', p2)
#    print('arc_dist:', arc_dist)
#    # print('m and b:', m, b)
    
    x_c = z2[0]
    y_c= m * x_c + b
    
    
    center = np.array([x_c,y_c])
    
    
    # F = np.empty((2))
    # print(p1)
#    print('center:', center)
    
    # print(np.linalg.norm(np.array([5,0]) - np.array([10,0])))
    # print('1', np.linalg.norm(p1 - center))
    # print('2', np.linalg.norm(p2 - center))
    
    
    # F[0] = np.linalg.norm(p1 - center) - np.linalg.norm(p2 - center)
    # F[0] = z2

    
    vector_1 = p1-center
    vector_2 = p2-center
    r = np.linalg.norm(vector_1)
    
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
#    print('dot product:', dot_product)

    whole_matrix = np.array([[1, p1[0], p1[1]], [1, p2[0], p2[1]], [1, center[0], center[1]]], dtype='float')
#    print('matrix:', whole_matrix)
    side_of_line = np.sign(np.linalg.det(whole_matrix.astype(float))) 
#    print('side of line:', side_of_line)
    
    arccros_val = np.arccos(dot_product)
    
    if math.isnan(arccros_val):
        arccros_val = 3.14159

#    print('arccros_val:', arccros_val)
    if side_of_line < 0:
        angle = 2*3.14159 - arccros_val
    else:
        angle = arccros_val

    # print('angle:', angle*57.3, 'r:', r)
    # print('r:', r)
    # F[1] = arc_dist - r*angle
#    print('attempted arc length:', center[0], r*angle)
    
    arc_diff = arc_dist - r*angle
    # print(arc_diff)
    
    # F[1] = z2
    # arc_diff = z2
    # F[3] = 
    # print('F:', F)
    # print('new arc dist:', r*angle)
    
#    sleep(0.5)
    return(arc_diff)


    
"""
start_z is initial z position

interpolator_zx_y is a function: when passed a z, x params it will give you a y value for the surface
max_z and min_z are the z boundaries of the generated surface
p2 is the bottom point located at (0, y, minz)

minZ_generic is the Z coordinate of P2
p2 is the end connection for X = 0
    z_cutoff = optimize_cutoff_value(z_cutoff, slope, generated_interpolator_zx_y, maxz_at_x0_plane, minz_generic, minz_at_x0_plane, p2)

"""

def optimize_cutoff_value(start_z, slope, interpolator_zx_y, max_z, z_of_filler, min_z_x0, p2, zmax):

    bottom_third_z = z_of_filler+(zmax-z_of_filler)/2

#   Trying a change 6/10/2020
    start_z = bottom_third_z - 3*INCREMENT
#    start_z = z_of_filler + 3*INCREMENT

#    AVW_Angle = math.degrees(math.atan(1/slope))
    AVW_Angle = math.degrees(math.atan2(-1 * np.sign(slope),abs(slope)))

# While we are above the end of the filler but less than the top of the AVW...
#hard coded -25 to keep the cutoff above the distal endpoint fow the avw
#### not sure why we wanted to keep it above -25 so I took it out
#    print('bottom limit:', z_of_filler)
#    print('top limit:',  bottom_third_z - INCREMENT)
#    print('starting point:', start_z)
    while (z_of_filler < start_z < bottom_third_z - INCREMENT):
#        print('looking at this z for cutoff:', start_z)
#     start_z is where the catenary curve would start
        y = interpolator_zx_y(start_z, 0)
#        p1 is the starting point of the catenary curve
        p1 = Point(0, y, start_z)

# get the distance between point 1 and the bottom of the AVW...
# this is the length that the catenary curve would have to be
        
        #AVW_Angle is made negative in rotate_point
#        p2_rotated = rotate_point(p2.x, p2.y, p2.z, AVW_Angle, p1)
        #p1 = Point(0, y, p2_rotated.z - abs(p1.z - p2_rotated.z))
        
#        This becomes the arc length
#        print('things before length:', p1, min_z_x0)
        length = getSurfaceDistance(p1.z, min_z_x0, 0, interpolator_zx_y)
#        print('length:', length)
        
#       For some reason the length is multilied by -1, so I'll do it again to give myself a positive
        length *= -1.0

        ####################################################################################################################################
        ####################################################################################################################################
        ####################################################################################################################################
        ####################################################################################################################################
        ####################################################################################################################################
        ####################################################################################################################################


        point_1 = np.array([p1.y, p1.z])
        point_2 = np.array([p2.y, p2.z])
        arc_length = length
        
#        print(arc_length)
        bisection_point = [(point_1[0]+point_2[0])/2, (point_1[1]+point_2[1])/2]
        
        
        # calculate the distance between a point and the bisection point (half the cord length)
        # half_cord_len = np.linalg.norm(point_1-bisection_point)
        
        cord_vector = point_1 - point_2
        cord_slope = cord_vector[1]/cord_vector[0]
        # print(cord_slope)
        if cord_slope == 0:
            bisector_slope = 999999999
        else:
            bisector_slope = -1/cord_slope
        # print(bisector_slope)
        y_intercept = bisection_point[1]-bisector_slope * bisection_point[0]
        # print(y_intercept)
        
        
        data = (point_1, point_2, arc_length, bisector_slope, y_intercept)
        zGuess = np.array([bisection_point[0]])
        
#        print('p1,p2:', point_1, point_2)
        
#        actually the y coordinate
        x_c = fsolve(circle_center, zGuess, args = data)
        # print(x_c)
#        actually the z coordinate
        y_c = bisector_slope * x_c + y_intercept
        # print(y_c)
#        print('point_1:', point_1)
#        print('center:', x_c, y_c)
        slope_of_radius = (point_1[0] - x_c)/(point_1[1] - y_c)
#        print('slope_of_radius:', slope_of_radius)
        slope_of_tangent = -1/slope_of_radius
#        print('slope_of_tangent:', slope_of_tangent)
        
        tangent_angle = -1*math.degrees(math.atan2(abs(slope_of_tangent),np.sign(slope_of_tangent)))
#        slope_of_tangent = math.degrees(slope_of_tangent)
#        print('tangent_slope:', tangent_angle)
#        print('tangent:', slope_of_tangent)
        sleep(.3)
#        if abs(slope_of_tangent) < 999:
#            p1_y += .001
#        else:
#            print('call it a day')
#            break
        
        ####################################################################################################################################
        ####################################################################################################################################
        ####################################################################################################################################
        ####################################################################################################################################
        ####################################################################################################################################
        ####################################################################################################################################
                                        

#        curve = CatenaryCurve(p1, p2_rotated, abs(length))
#
#        p3_rotated = Point(p1.x, p1.y+INCREMENT, curve.f(p1.y+INCREMENT))
#        p3_unrotated = rotate_point(p3_rotated.x, p3_rotated.y, p3_rotated.z, -AVW_Angle, p1)
#
#        curve_angle = math.atan2((p3_unrotated.y-p1.y),-(p3_unrotated.z-p1.z))
#
#        curve_angle = math.degrees(curve_angle)

#        print("##################################")
#
#        print('looking at this z:', start_z)
#        print('AVW Angle:', AVW_Angle)
#        print('curve angle:', tangent_angle)
#        print('angle difference', AVW_Angle - tangent_slope)
# Trying a change 6/10/20
#        if AVW_Angle - curve_angle < -5:
        if AVW_Angle - tangent_angle < 0:
            start_z += INCREMENT
# Trying a change 6/10/20
#        elif AVW_Angle - curve_angle > -1:
        elif AVW_Angle - tangent_angle > 5:
            start_z -= INCREMENT
        else:
            print('STOPPING')
            return start_z


#        start_z -= INCREMENT*10


        """
#   new_z appears to be the z coordinate corresponding to the starting point of the curve        
        new_z = curve.f(p1.y-INCREMENT)
        #print("Cat Curve y : ", p1.y)
        #print("Cat Curve New Z :", new_z)
        curve_delta_z = new_z - p1.z

        #print("###########")
        #print(p1)
        #print(p2)
        #print(p1.y, p1.z)
        #print(p1.y - INCREMENT, new_z)
        #print("length = " + str(length))
        #print([curve_delta_z, surface_delta_z, abs(curve_delta_z-surface_delta_z)])
        #print("curve delta : ",curve_delta_z)
        #print("Initial delta : ",surface_delta_z)
        if start_z > bottom_third_z:
            return start_z
        if abs(curve_delta_z - surface_delta_z) < threshold:  
            return start_z

        #elif (abs(surface_delta_z) > abs(curve_delta_z)):
        #elif surface_delta_z < curve_delta_z:
        if curve_delta_z > 0 or surface_delta_z > curve_delta_z:
            start_z += INCREMENT
        else:
            start_z -= INCREMENT"""
    return start_z


def calc_new_point_on_curve(distance, curve, start_point):
    if distance < 0.001:
        return [start_point.y, start_point.z]

    cur_distance = 0
    z = start_point.z
    y = start_point.y
    old_y, old_z = y, z
    
    p = Point(0, y, z)
    prev_curv_distance = cur_distance
    while cur_distance < distance:
        old_z, old_y = z, y
        prev_curv_distance = cur_distance
#        p.y = y + INCREMENT
        p.y = y + .001
        p.z = curve.f(p.y)
        # Concern
        # I think that it looks like it thinks it hasn't gone as far as it really has
        cur_distance = curve.arc_length(p)
        z = p.z
        y = p.y

    y = old_y*(cur_distance-distance)/(cur_distance-prev_curv_distance) + y*(distance-prev_curv_distance)/(cur_distance-prev_curv_distance)
    z = curve.f(y)
    
    return [y, z]


def calc_new_point_on_circle_curve(distance, data):
#    print('desired arc length',distance)
    y_c = data[0]
    z_c = data[1]
#    print('center:', y_c,z_c)
    r = data[2]
#    print('radius:', r)
    start_point = data[3]
    
#    print('desired sweep distance:', distance)
#    negative because we want to go counter clockwise
    sweep_angle = -1 * distance / r

#    print('sweep_angle', sweep_angle*57.3)
    current_angle = math.asin((start_point.y - y_c)/r)
    new_angle = sweep_angle + current_angle
    
    y = y_c + r * math.sin(new_angle)
    z = z_c + r * math.cos(new_angle)
    
#    print('current_angle:', current_angle*57.3)
#    print('new_angle:', new_angle*57.3)
#    print('new y:', y)
#    print('new z:', z)
#     
#    
##    print_flag = data[4]
##    print('start_point',start_point)
##    circle = [y_c, z_c, r]
#    if distance < 0.001:
#        return [start_point.y, start_point.z]
#
#    cur_distance = 0
##    print('starting_z', start_point.z)
#    z = start_point.z
##    print('outside z', z)
#    y = start_point.y
#    old_y, old_z = y, z
#    
#    p = Point(0, y, z)
##    prev_curv_distance = cur_distance
#    while cur_distance < distance:
##        coordinates before moving slightly along
#        old_z, old_y = z, y
##        p.y = y + INCREMENT
##        print('inside z', z)
#        p.z = z - .001
##       This could give 1 of 2 roots, how do I know which I need? putting -1 in
##       there for now because most of my y values will be negative
##        print('**')
##        print(z_c)
##        print(y_c)
##        print(r)
##        print(z)
#        if (p.z - z_c)**2 > r**2:
#            print('----')
#    #        print(old_z)
#            print('z coord of new point', p.z)
#    #        print(old_y)
#            print('y coord of new point', p.y)      
#            print(cur_distance)
#        p.y = (-1*(r**2 - (p.z - z_c)**2)**0.5 + y_c)
#        
#        if math.isnan(p.y):
#            # Concern
#            # I think that it looks like it thinks it hasn't gone as far as it really has
#            print('----')
#    #        print(old_z)
#            print('z coord of new point', p.z)
#    #        print(old_y)
#            print('y coord of new point', p.y)
#        cur_distance += ((old_z - p.z)**2+(old_y - p.y)**2)**0.5
##        print('cur_distance',cur_distance)
#        z = p.z
#        y = p.y
##        sleep(.5)
##    y = old_y*(cur_distance-distance)/(cur_distance-prev_curv_distance) + y*(distance-prev_curv_distance)/(cur_distance-prev_curv_distance)
##    z = curve.f(y)
#    
    return [y, z]

def edges_to_try_and_correct_interpolation(generic, generated):
    xmin = min(generated.xAxis)
    xmax = max(generated.xAxis)
    
    for i, val in enumerate(generated.xAxis):
        if (xmax - 5) < val < (xmin + 5):
            generated.xAxis.append(val * 30)
            generated.yAxis.append(generated.yAxis[i])
            generated.zAxis.append(generated.zAxis[i])
    
    minZ_generic = min(generic.zAxis)
    z0 = max(generic.zAxis)
    y0 = min(generic.yAxis)
    for index in range(0,len(generic.zAxis)):
        if generic.zAxis[index] < (minZ_generic  + 0.3):
            
            y1 = generated.yAxis[index]
            z1 = generated.zAxis[index]
            if not z1 == z0:
                slope = (y1 - y0) / (z1 - z0)
                z = z1*30
                new_y = slope*(z - z1) + y1
                generated.xAxis.append(generated.xAxis[index])
                generated.yAxis.append(new_y)
                generated.zAxis.append(z)
        
    return generated
# Generates new value for the given axis 


def getLengthFromDist(apex, z):
    return abs(apex-z)




############################################
#    This function takes the tissue (typically the AVW) and moves points
#    to either shift or scale (stretch) in a direction
def generate_values_for(axis, dataSet, scale, unmodifiedLength, shiftValue=0):
    
#   Variables to be set below depending on which 
#    The direction that the movement will take place in
    mainAxis = None
#    The direction 
    sliceAxis = None
    interpolatingAxis = None
    referencePoint = 0

#   Scaling in the x happens when widening the AVW
#        we're going to give it a certain X and it will move along that constant X
#        by varying the Z to see how far away it is and to move the point
    if axis == "x":
#        The axis that will be changed/stepped in
        mainAxis = dataSet.xAxis
#        The axis that will be held constant
        sliceAxis = dataSet.zAxis
#        The axis that will be fed back from the interpolation after being given
#        the other two
        interpolatingAxis = dataSet.yAxis
        #scale = (scale - 1) / 2.0 + 1 # because we are starting in the center and lengthening the left and right side
        #scale can remain what was passed


#   scaling in the z happens for lengthening the AVW and shifting the apcial location
#        we're going to give it a certain X and it will move along that constant X
#        by varying the Z to see how far away it is and to move the point
    elif axis == "z":
#        The axis that will be changed/stepped in
        mainAxis = dataSet.zAxis
#        The axis that will be held constant
        sliceAxis = dataSet.xAxis
#        The axis that will be fed back from the interpolation after being given
#        the other two        
        interpolatingAxis = dataSet.yAxis
        
#       This is considered the top of the AVW
        referencePoint = max(dataSet.zAxis)
        
#    Get an interpolated surface from the data
    interpolator = interpolate.Rbf(mainAxis, sliceAxis, interpolatingAxis, function='linear', smooth=SMOOTH_VAL)
    
    generated_Axis_Values = []
    generated_Y_Values = []
    newMainCoord = None

    if axis == "x":
        #TODO: read in widths[] and lengths[] automatically
        widths = [41, 41, 60, 50, 40, 60, 41]
        lengths = [-1000, 0, 10, 15, 20, 40, 1000]
        apex = max(dataSet.zAxis)

        desiredWidth = interpolate.interp1d(lengths, widths, kind = 'cubic', fill_value="extrapolate")
        dataSet = []
        maxScale = 5
        xnew = np.arange(0, 60, 0.1)
        ynew = desiredWidth(xnew)
        plt.plot(xnew, ynew)
        plt.show()
        #print(len(mainAxis))
    for i in range(0,unmodifiedLength):
        #print(i)
        if axis == "x":
            """
            #distance = getSurfaceDistanceForXatZ(mainAxis[i], sliceAxis[i], interpolator)

            #Need a function to put in distance (from AVW apex) and receive length
            length = getSurfaceDistance(apex, sliceAxis[i], mainAxis[i], interpolator)

            #TODO - 
            scale = desiredWidth((-length))/41"""
            #print(str(length) + " , " + str(scale) + ", " + str(i+1))
            distance = getSurfaceDistanceForXatZ(mainAxis[i], sliceAxis[i], interpolator)
            scaledDistance = (distance * scale) # might want to divide scale by two
            newMainCoord = find_val_for_distance(scaledDistance,sliceAxis[i],interpolator, referencePoint)
        
        
        elif axis == "z":
           distance = getSurfaceDistance(referencePoint, mainAxis[i], sliceAxis[i], interpolator)
#           print("Distance = ",distance)
           scaledDistance = (distance * float(scale)) + shiftValue  
            
           newMainCoord= find_val_for_distance(scaledDistance,sliceAxis[i],interpolator, referencePoint)
               
               
        generated_Axis_Values.append(newMainCoord[0])
        generated_Y_Values.append(newMainCoord[1])
    
    
    return(generated_Axis_Values, generated_Y_Values) 
    
    
'''

Calculates Surface Distance from x = 0 to x = givenPoint

'''
def getSurfaceDistanceForXatZ(endX, atZ, yInterpolator):
    # list of x values from 0 to endpoint incremented by INCREMENT
    #xRange = np.arange(0.0,endX, INCREMENT * (1 if endX >= 0 else -1)) #[0, 0.02, 0.04, 0.06...endX]
    xRange = np.arange(0.0,endX, INCREMENT * (1 if endX >= 0 else -1)) #[0, 0.02, 0.04, 0.06...endX]
    distance = 0.0
    oldY = yInterpolator(0, atZ)
    
    for index in range(1,len(xRange)):
        interptedY = yInterpolator(xRange[index], atZ)
        distance += pythag(xRange[index - 1] - xRange[index], oldY - interptedY)
        oldY = interptedY
    
    # if endX is negitive distance should be negitive as well
    if endX < 0 and distance > 0:
        distance *= -1.0
    return distance


'''
Calculates Surface Distance from Zmax to given z value

'''
def getSurfaceDistance(fromValue, toValue, forSlice, interpolator):
    
    # If adjusting X  axis 
    #   fromValue will be  0 
    #   toValue will be x value on surface to calculate distance to
    #   forSlice is a value on the Z Axis that the distance is being calculated along
    #
    #If adjusting Z  axis 
    #   fromValue will be  Max Value Of Z
    #   toValue will be Z value on surface to calculate distance to
    #   forSlice is a value on the X Axis that the distance is being calculated along
    
    # list of main values from 0 to endpoint incremented by INCREMENT
    
    zRange = np.arange(fromValue, toValue, INCREMENT * (1 if fromValue < toValue else -1))

    distance = 0.0
    oldY = interpolator(fromValue, forSlice)
    
    
    for index in range(1,len(zRange)):
        interptedY = interpolator(zRange[index], forSlice)
        
        distance += pythag(zRange[index - 1] - zRange[index], oldY - interptedY)
        oldY = interptedY
        
            
        
    # if endX is negative distance should be negative aswell
    if fromValue == 0:
        if toValue < 0 and distance > 0:
            distance *= -1.0
    else:
        distance *= -1.0
    return distance

'''
Calculates new XValue for Scaled Surface distance
The value currently returned is close to the desired distance
if x is incremented by 0.1 and the change in y is near zero then 
deltaX and deltaY pythag-ed results in a value close to deltaX which results in a falsly high value incorrect value
'''
def find_val_for_distance(distance, z, interper, referencePoint):
    # if this function was called from generate_values_for under the else "z" then 
    # z = an x value, reverance point = the top of the surface, max z
    # xvale is a z value
    # oldx is a z value
    
    xVal = referencePoint
    oldX = referencePoint
    currentDistance = 0.0
    
    oldY = interper(xVal, z) 
    #sets off set to positive or negitive depending on which way we want to go
    #if distance is negitive we want the offsetValue to be negitive
    offsetValue = INCREMENT if distance > 0 else INCREMENT * -1
    
    # returns last X before desiered distance
    while True:
        
        xVal += offsetValue
        interpedY = interper(xVal, z)
        
        deltaX = oldX - xVal
        deltaY = oldY - interpedY
        
        prevDistance = currentDistance
        currentDistance += pythag(deltaX, deltaY) * (INCREMENT / offsetValue)
        
        oldY = interpedY
        # breaks out of while loop if case is met before setting Old XValue
        if abs(currentDistance) > abs(distance):
            finalStep = (distance - prevDistance)/(currentDistance - prevDistance)
            oldOldX = oldX
            oldX = oldX + (offsetValue * finalStep)
            
            #-------- trying to calculate final distance to new point 
            interpedY = interper(oldX, z)
            deltaX = oldX - xVal
            deltaY = oldY - interpedY
            
            currentDistance = prevDistance
            currentDistance += pythag(deltaX, deltaY) * ((oldX - oldOldX) / offsetValue)
            break
        else:
            oldX = xVal
            
    # Returns last value before current distance is greater that desired distance
    #print(str(oldY) + " " + str(yInterpolator(oldX, z)))
    
    return (oldX, interper(oldX, z),currentDistance)


# takes any number of parameters and pythags them
def pythag(*args):
    squareSum = 0
    for arg in args:
        squareSum += arg**2
        
    return squareSum **0.5

def euclidean_distance_between_points(points):
    dist_sum = 0
    for i, p in enumerate(points):
        try:
            if(i+1 < len(points)):
                p2 = Point(points[i+1][0], points[i+1][1],points[i+1][2])
                p1 = Point(p[0], p[1], p[2])
                dist_sum += pythag(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)
        except:
            break
    return dist_sum

def plot_Dataset(*args):
    
    fig = plt.figure(figsize=plt.figaspect(0.5) * 1)
#    #get current Axes
#        
    ax = fig.gca(projection='3d')
    
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
    for i ,(arg, c) in enumerate(zip(args, colors)):
        ax.scatter(arg.xAxis, arg.yAxis, arg.zAxis, label=i+1, c=colors[i])
        
        
#    ax.axis('equal')
    scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
#    ax.auto_scale_xyz(*[[-30, 30]]*3)
    plt.legend(loc='upper left');
    plt.show()
            
#======================
# Give a DataSet, angle in degrees, and a point of rotation
# Return Dataset rotated on the x axis about the point of rotation
#======================
            
def rotate_part(cloud, angle, rotPoint):
    
    # we create a copy so we don't modify the original
    rotCloud = cloud.generatedCopy()
    if angle > 0:
        # i = index, ans xyz are the values at that index
        for i, (x, y, z) in enumerate(cloud.zipped()):
            
            p = rotate_point(x,y,z,angle, rotPoint)

            rotCloud.modify_point(p,i)
        
    return rotCloud


def getClosestNewBoundIndex(x, y, z, newBoundsList):
    MinDistToNewBound = 999999
    closest_node_index = 0
    for i in range(0,len(newBoundsList)):
        p = newBoundsList[i]
        DistToBound = ((x - p.x)**2 + (y - p.y)**2 + (z - p.z)**2)**0.5
        if(DistToBound < MinDistToNewBound):
            MinDistToNewBound = DistToBound
            closest_node_index = i
    return closest_node_index


def DistToNewBound(x, y, z, newBoundsList, index):
    p = newBoundsList[index]
    return ((x - p.x)**2 + (y - p.y)**2 + (z - p.z)**2)**0.5

def rotate_part_wboundaries(cloud, angle, rotPoint, TissueBoundary, AllBoundaries, threshold):
    # we create a copy so we don't modify the original
    rotCloud = cloud.generatedCopy()
    if angle > 0:

        did_angle_hit_bound = []
        angle_moved = []
        Touching_Threshold = 3
        newBoundsList = []
        newBoundaryAngles = []
        max_Angles = []

        for i, (x, y, z) in enumerate(cloud.zipped()):

            Angles = []
            MinDistToBound = 999999

            ############################################################
            ## PART 1: Finding the closest boundary (checking itself) ##
            ############################################################
            for j, (xb, yb, zb) in enumerate(TissueBoundary.zipped()):
                DistToBound = ((x - xb)**2 + (y - yb)**2 + (z - zb)**2)**0.5
                if MinDistToBound > DistToBound:
                    MinDistToBound = DistToBound

            if MinDistToBound > threshold:
                max_angle = angle
            else:   
                max_angle = angle*MinDistToBound/threshold

            Angles.append(max_angle)

        
        
            angle_steps = 10
            MinDistToBound = 999999
            closest_angle = max_angle
            
            ###################################################################
            ## PART 2: Finding the closest boundary (checking other tissues) ##
            ###################################################################
            for j, (xb, yb, zb) in enumerate(AllBoundaries.zipped()):
                previous_distance = ((x - xb)**2 + (y - yb)**2 + (z - zb)**2)**0.5
                for h in range(1,angle_steps):
                    p = rotate_point(x,y,z,max_angle*h/angle_steps, rotPoint)
                    DistToBound = ((p.x - xb)**2 + (p.y - yb)**2 + (p.z - zb)**2)**0.5
                    #If distance increases (and is not the 1st step)
                    #then previous angle is closest

                    if DistToBound < Touching_Threshold and DistToBound < previous_distance:
                        MinDistToBound = previous_distance
                        closest_angle = max_angle*(h-1)/angle_steps
                        break

                    if DistToBound > previous_distance:
                        if h == 1: #then  it was moving away, not closer
                            pass
                        elif previous_distance < MinDistToBound:
                            MinDistToBound = previous_distance
                            closest_angle = max_angle*(h-1)/angle_steps
                        break
                    #If distance is decreasing, keep going
                    else:
                        previous_distance = DistToBound
                        
            #If we are really close to another boundary node, this node will
            #become a new boundary node
            if MinDistToBound < Touching_Threshold:
                closest_angle *= 0.9
                did_angle_hit_bound.append(True)
                newBoundsList.append(i)
                newBoundaryAngles.append(closest_angle)
                Angles.append(closest_angle)

            #standard scaling
            #elif MinDistToBound < threshold:
            #    max_angle = max_angle*MinDistToBound/threshold 
            #    did_angle_hit_bound.append(False)

            else:
                did_angle_hit_bound.append(False)
            

            #Angles.append(closest_angle)

            max_Angles.append(Angles)

        ####################################################################
        ## PART 3: Checking itself on new boundaries that might of formed ##
        ####################################################################
        if newBoundsList: #only if there are new boundaries
            for i, (x, y, z) in enumerate(cloud.zipped()):

                MinDistToNewBound = 999999
                max_angle = max_Angles[i][0]
                MinNewAngle = max_angle
 
                #finds the closest new Boundary and how far it would have to rotate back IF we were to rotate it back
                for j in newBoundsList:
                    oldPoint = cloud.node(j)
                    original_distance = ((x-oldPoint.x)**2+(y-oldPoint.y)**2+(z-oldPoint.z)**2)**0.5

                    newBoundIndex = newBoundsList.index(j)
                    coefficient = math.e**(-(original_distance/10)**2) #e^-(x^2) type

                    new_angle = coefficient*newBoundaryAngles[newBoundIndex]+(1-coefficient)*max_angle

                    if new_angle < MinNewAngle:
                        MinNewAngle = new_angle
                        
                    if original_distance < MinDistToNewBound:
                        MinDistToNewBound = original_distance


                if MinDistToNewBound < threshold: #If it's close enough, then we will consider it as a limit
                    max_Angles[i].append(MinNewAngle)

        #Finally, it the results from checking itself, checking others, and checking for new boundaries
        #and rotates it the minimum amount one of them had rotated
        for i, node_angles in enumerate(max_Angles):
            p = rotCloud.node(i)
            m = rotate_point(p.x,p.y,p.z,min(node_angles), rotPoint)

            rotCloud.modify_point(m, i)
                    
    return rotCloud



def rotate_point(x, y, z, angle, rotPoint):

    #angle multipled by -1 because it is rotating in the negative direction...
    #at least how we were thinking about it when rotating the LA
    angle = -math.radians(angle)
    rotZ = z - rotPoint.z
    rotY = y - rotPoint.y
    newY = rotY*math.cos(angle) - rotZ*math.sin(angle) + rotPoint.y
    newZ = rotY*math.sin(angle) + rotZ*math.cos(angle) + rotPoint.z
    return Point(x, newY, newZ)


def rotate_point_about_z(x, y, z, angle, rotPoint):

    #angle multipled by -1 because it is rotating in the negative direction...
    #at least how we were thinking about it when rotating the LA
#    angle = math.radians(angle)
#    rotY = y - rotPoint.y
#    rotX = x - rotPoint.x
#    newX = rotX*math.cos(angle) - rotY*math.sin(angle) + rotPoint.x
#    newY = rotX*math.sin(angle) + rotY*math.cos(angle) + rotPoint.y


    angle = -1*math.radians(angle)
    rotY = y - rotPoint.y
    rotX = x - rotPoint.x
    newX = rotX*math.cos(angle) + rotY*math.sin(angle) + rotPoint.x
    newY = -1*rotX*math.sin(angle) + rotY*math.cos(angle) + rotPoint.y

    print('rotY, angle, sin(angle), part 2:', rotY, angle, math.sin(angle), rotY*math.sin(angle))

#    newX = rotX*math.cos(angle) - rotY*math.sin(angle) + rotPoint.x
#    newY = rotX*math.sin(angle) + rotY*math.cos(angle) + rotPoint.y

    return Point(newX, newY, z)

def optimize_rot_angle(ideal_dist, rot_point, ps_point, gi_fill_point):
    
#    a = np.array([gi_fill_point.x, gi_fill_point.y, gi_fill_point.z])
#    b = np.array([rot_point.x, rot_point.y, rot_point.z])
#    c = np.array([ps_point.x, ps_point.y, ps_point.z])
#    
#    ba = a - b
#    bc = c - b
#    
#    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#    orig_angle = np.arccos(cosine_angle)
    
    minimized_angle  = optimize.minimize(return_difference_given_angle, 0, (ideal_dist, rot_point, ps_point, gi_fill_point), bounds=[(0,180)]).x
    
    return minimized_angle
    

def return_difference_given_angle(angle, *params):
    
    # expanding the parameters in to variables
    (ideal_dist, rot_point, ps_point, gi_fill_point) = params
    new_point = rotate_point(gi_fill_point.x, gi_fill_point.y, gi_fill_point.z, angle, rot_point)
    distance = euclidean_distance_between_points([[0, new_point.y, new_point.z], [0, ps_point.y, ps_point.z]])
    
    return abs(distance - ideal_dist)

def getFakeData():
    fake_x = []
    fake_y = []
    fake_z = []

    for i in range(0, 10):
        for j in range(0, 10):
            fake_x.append(i-5)
            fake_y.append(j)
            fake_z.append(0)

    return DataSet3d(fake_x, fake_y, fake_z)


def surface_plotter(surface):
    x = surface.xAxis
    y = surface.yAxis
    z = surface.zAxis
    
    xMin = min(x)
    xMax = max(x)
    zMin = min(z)
    zMax = max(z)
    
    SurfacePlot = 1
    
    if SurfacePlot == 1:
    
        Surface = interpolate.Rbf(x,z,y, function='linear')
        
        xSurf = np.linspace(xMin-20,xMax+20,40)
        zSurf = np.linspace(zMin-20,zMax+20,40)
                
        xData = []
        yData = []
        zData = []
        
        for i in range(0,len(xSurf)):
            for j in range(0,len(zSurf)):
                
                xData = np.append(xData,xSurf[i])
                yData = np.append(yData,Surface(xSurf[i],zSurf[j]))
                zData = np.append(zData,zSurf[j])
        
        fig = plt.figure(figsize=plt.figaspect(0.5)*1.5)
        ax = fig.gca(projection='3d')
        
        ax.plot_trisurf(xData, zData, yData, cmap=cm.jet, linewidth=0.2)
        

        for ii in range (0,5):
        #for ii in range(0,360,30):
                ax.view_init(elev=10., azim=ii*72)
        #        ax.auto_scale_xyz([-30, 30], [-30, 30], [-30, 30])
                plt.savefig("FEASurface"+str(ii)+".png")
        #       plt.show()
        plt.show()

def parseNumsFromCSV(nodeIDs, line):
  line.replace(" ", "") #delete spaces
  numList = line.split(",")
  for x in numList:
    nodeIDs.append(int(x)) #int(x) converts it from string to integer

def isBoundaryLine(boundarySets, line):
  if "*Nset" in line:
    line = line.split(", ")
    name = line[1].split("=")[1] #selects _PickedSet5 from nset=_PickedSet5
    if name in boundarySets:
        return True
  return False

def getName(line):
  line = line.split(", ")
  name = line[3].split("=")[1].strip() #selects _PickedSet5 from nset=_PickedSet5
  return name

def getNodes(file, line):
  numberLines = [] #lines containing lists of numbers in the INP file
  line = next(file)
  while "*" not in line:
    numberLines.append(line)
    line = next(file)
    
  nodeIDs = []
  for line in numberLines:
      parseNumsFromCSV(nodeIDs, line)
  return nodeIDs


#This, and the functions it calls, can be moved to IOFunctions
def getBnodes(file_location, BoundaryTissue):#look for *Boundary and grab the next line after dropping the _ and only grabbing PickedSET### storing them in an array
    BNodes = []
    boundarySets = []
    with open(file_location, 'r') as file:
        line = "h"
        while "** BOUNDARY CONDITIONS" not in line:
            line = next(file)
        while "** INTERACTIONS" not in line:
                if "*Boundary" in line:
                    boundarySets.append(next(file).split(",")[0])
                line = next(file)
    
    with open(file_location, 'r') as file: #items are above *Boundary lines, must be reopened
        line = "line"
        while True:
            if isBoundaryLine(boundarySets, line):
                material = getName(line)
                if BoundaryTissue in material:
                    
                    if "generate" in line: #then it is parameters for an instruction
                        instructions = getNodes(file, line)
                        start = instructions[0]
                        end = instructions[1]
                        increment = instructions[2]
                        nodes = np.arange(start, end + 1, increment).tolist()
                        BNodes = BNodes + nodes

                    else: #it's just a list of nodes
                        newNodes = getNodes(file, line)
                        for n in newNodes:
                            if n not in BNodes:
                                BNodes.append(n)
                else:
                    del line
                try:
                    line = next(file)
                except StopIteration:
                    break
            else:
                try:
                    line = next(file)
                except StopIteration:
                    break
        #BNodes.sort()
    return BNodes


#Technically an IO function but I think of it more of a dataSet operation
def numOfNodes(INPFile, Material):
    points = extractPointsForPartFrom(INPFile, Material)
    return len(points)


#####################
## UNUSED FUNCTION ##
#####################

#This function used to shift the AVW up to prevent curve from intersecting GIFiller
#z_cutoff was moved back and this function also introduced some errors to the curve function
def corrective_shift(generated_surface, HiatusLength):
    if HiatusLength > 40 or HiatusLength < 33:
        return generated_surface

    zMax = max(generated_surface.zAxis)
    zMin = min(generated_surface.zAxis)
    shiftValue = (-2/7)*HiatusLength+(80/7) #linear function that has points (33,2) and (40,0)

    percent = 0.5
    how_far_from_minZ_equalsPercent = 20 #at this distance from the minimum point, the coefficient equals percent
    how_far_from_minZ_equalsPercent = (zMax - zMin * percent)

    #log below is natural log
    k = math.log(1/percent)/(how_far_from_minZ_equalsPercent**2) #used in e^-k(x^2)

    for i, (x, y, z) in enumerate(generated_surface.zipped()):
        coefficient = 1 - math.e**-(k*(z-zMin)**2)
        new_point = Point(x, y + coefficient*shiftValue, z)
        generated_surface.modify_point(new_point, i)
    return generated_surface

#####################
## UNUSED FUNCTION ##
#####################

#This function is used for seeing what the catenary curve looks like
def plotCatenaryCurve(curve, p1, p2):
    Ys = []
    Zs =[]
    y = float(p1.y)
    print("Plotting point")
    print(p2)
    plt.figure()
    while y < p2.y:
        Ys.append(y)
        Zs.append(-curve.f(y))
        y += INCREMENT

    plt.plot(Zs, Ys)
    plt.show()

#####################
## UNUSED FUNCTION ##
#####################

def generate_surface_center_third (generic, generated, scale, shiftValue = 0):
    #this is the surface we will be modifying
    #newSurface = generated.generatedCopy()

    newSurface = generic.generatedCopy() #will retain old Z coordinates
    
    xmin = min(generated.xAxis)
    xmax = max(generated.xAxis)
    zmin = min(generated.zAxis)
    zmax = max(generated.zAxis)
    zLength = zmax-zmin
    xLowBound = xmin + (xmax-xmin) * 1/3
    xHighBound = xmin + (xmax-xmin) * 2/3

    #if scale is not a factor but in mm
    #xLength = abs(xmax - xmin)
    #factor = (xLength+scale)/xLength)

    centerThirdDist = abs(xmax - xmin)/6 #returns center third (1/3)*(1/2), 1/2 because split along x=0
    centerXs = []
    centerYs = []
    centerZs = []

    for i, x in enumerate(generated.xAxis):
        if x > xLowBound and x < xHighBound and generated.zAxis[i] > zmax - 0.8*(zmax - zmin) :
            centerXs.append(generated.xAxis[i])
            centerYs.append(generated.yAxis[i])
            centerZs.append(generated.zAxis[i])

    #New Center nodes used to create surface
    newInterpolator = interpolate.Rbf(centerXs, centerZs, centerYs, function='linear', smooth=SMOOTH_VAL)

    oldInterpolator = interpolate.Rbf(generic.xAxis, generic.zAxis, generic.yAxis, function='linear', smooth=SMOOTH_VAL)

    referencePoint = 0

    for i in range(0,len(generic.xAxis)):
        distance = getSurfaceDistanceForXatZ(generic.xAxis[i], generic.zAxis[i], oldInterpolator)
        scaledDistance = (distance*scale)
        newMainCoord = find_val_for_distance(scaledDistance,generic.zAxis[i],newInterpolator,referencePoint)
        #[x value, new y value]
        #if generated.zAxis[i] > zmax - 0.8*(zmax - zmin):
        newSurface.xAxis[i] = newMainCoord[0]
        newSurface.yAxis[i] = newMainCoord[1] + shiftValue

    return newSurface

#####################
## UNUSED FUNCTION ##
#####################

def AVW_put_above_GI (GI_Filler, AVW):
    newAVW = AVW.generatedCopy()
    shiftValue = 1
    top = 0.05

    yMax = max(GI_Filler.yAxis)
    yMin = min(GI_Filler.yAxis)
    yLength = yMax - yMin
    yLimit = yMin + yLength * (1 - top) #top 5% 

    sectionXs = []
    sectionYs = []
    sectionZs = []

    for i, (x, y, z) in enumerate(GI_Filler.zipped()):
        if y > yLimit:
            sectionXs.append(x)
            sectionYs.append(y)
            sectionZs.append(z)

    interpolator_zx_y = interpolate.Rbf(sectionZs, sectionXs, sectionYs, function="linear")

    AVW_minY = min(AVW.yAxis)
    for i, (x, y, z) in enumerate(AVW.zipped()):
        if y == AVW_minY:
            lowestPoint = AVW.node(i)

    newY = interpolator_zx_y(lowestPoint.z, lowestPoint.x) + shiftValue
    moveUp = newY - AVW_minY

    for i, (x, y, z) in enumerate(AVW.zipped()):
        newAVW.modify_point(Point(x, y + moveUp, z), i)

    return newAVW

#####################
## UNUSED FUNCTION ##
#####################

def getCorrectScale(zMax, zMin, z, scales):
    zLength = zMax - zMin

    newVal = (z - zMin)/zLength
    newVal *= len(scales)-1

    index = math.floor(newVal)

    scale1 = scales[index]
    try:
        scale2 = scales[index + 1]
    except IndexError: #it is the last element
        correctScale = scales[index]
        return correctScale

    correctScale = scale1 + (scale2 - scale1)*(newVal-index)

    return correctScale

def scale_PM_Mid_Width(PM_Mid, connections, scaleFactor):
    PM_Mid_new = PM_Mid.generatedCopy()
    #find the negative x point closest to the bottom center
    #scaleFactor = newWidth/originalWidth

    starting_index, ending_index = find_starting_ending_points_for_inside(PM_Mid)

    #########
    # Get hiatus length to find scale factor HERE:
    starting_node = PM_Mid.node(starting_index)
    ending_node = PM_Mid.node(ending_index)
    hLength = starting_node.distance(ending_node)
    #########
    print(hLength)

    innerNodes = findInnerNodes(PM_Mid, connections, starting_index, ending_index)

    
    starting_index, ending_index = find_starting_ending_points_for_outside(PM_Mid)
    outerNodes = findOuterNodes(PM_Mid, connections, starting_index, ending_index)

    for i in range(0, len(PM_Mid.xAxis)):
    	point = PM_Mid.node(i)
    	distToInner = findClosestNode(PM_Mid, innerNodes, point)
    	distToOuter = findClosestNode(PM_Mid, outerNodes, point)
    	distanceFactor = (distToOuter)/(distToInner+distToOuter)
    	PM_Mid_new.xAxis[i] = PM_Mid.xAxis[i]+(scaleFactor-1)*distanceFactor*PM_Mid.xAxis[i]

    for node in range(0, len(PM_Mid.xAxis)):
    	PM_Mid_new.zAxis[node] += 10

    #for node in outerNodes:
    #    PM_Mid_new.zAxis[node] += 10

    return PM_Mid_new

def findClosestNode(tissue, nodeList, point):
	minDist = 999999
	for nodeIndex in nodeList:
		distance = tissue.node(nodeIndex).distance(point)
		if distance < minDist:
			minDist = distance

	return minDist
	
def findClosestNodeNumber(tissue, nodeList, point):
	minDist = 999999
	for nodeIndex in nodeList:
		distance = tissue.node(nodeIndex).distance(point)
		if distance < minDist:
			minDist = distance
			node = nodeIndex
	return node

def findInnerNodes(PM_Mid, connections, starting_index, ending_index):
    current_index = starting_index
    innerList = [starting_index]
    previous_angle = 3.14159

    while current_index != ending_index:
        current_point = PM_Mid.node(current_index)

        next_node = -1
        smallest_angle = 99
        for connection in connections[current_index]:
            corrected = connection - 1
            if corrected not in innerList:
                angle_to_point = findXYAngle(current_point, PM_Mid.node(corrected))
                if angle_to_point < smallest_angle and abs(angle_to_point-previous_angle) < 90*math.pi/180: #negative angle = clockwise
                    next_node = corrected
                    smallest_angle = angle_to_point

        if(next_node==-1):
        	raise RuntimeError("Failed to determine tissue boundaries when trying to widen PM_MID")
        innerList.append(next_node)
        current_index = next_node
        previous_angle = smallest_angle
    return innerList

def findOuterNodes(PM_Mid, connections, starting_index, ending_index):
    current_index = starting_index
    innerList = [starting_index]
    previous_angle = 3.14159/2

    while current_index != ending_index:
        current_point = PM_Mid.node(current_index)

        next_node = -1
        largest_angle = -99
        for connection in connections[current_index]:
            corrected = connection - 1
            if corrected not in innerList:
                angle_to_point = findXYAngle(current_point, PM_Mid.node(corrected))
                if angle_to_point > largest_angle and abs(angle_to_point-previous_angle) < 90*math.pi/180: #negative angle = clockwise
                    next_node = corrected
                    largest_angle = angle_to_point

        if(next_node==-1):
            break
        innerList.append(next_node)
        current_index = next_node
        previous_angle = largest_angle
    return innerList


def findXYAngle(point, new_point):
	delta = Point(new_point.x-point.x, new_point.y-point.y, new_point.z-point.z)
	angle_of_delta = math.atan2(delta.y,delta.x)
	return angle_of_delta


# I think this is trying to find the starting point for the inner curve of the
# PM_Mid
def find_starting_ending_points_for_inside(PM_Mid):

    closestNegativeIndex = -1
    closestPositiveIndex = -1

    closestNegativeDist = 999999
    closestPositiveDist = 999999

    yMin = min(PM_Mid.yAxis)

    for i, (x, y, z) in enumerate(PM_Mid.zipped()):
        distance = (x*x+(y-yMin)**2)**0.5
        if x < 0:
            if distance < closestNegativeDist:
                closestNegativeIndex = i
                closestNegativeDist = distance
        elif distance < closestPositiveDist:
            closestPositiveIndex = i
            closestPositiveDist = distance

    return closestNegativeIndex, closestPositiveIndex

def find_starting_ending_points_for_outside(PM_Mid):
    furthestNegativeIndex = -1
    furthestPositiveIndex = -1
    mostNegativeX = 999999
    mostPositiveX = -999999

    for i, (x, y, z) in enumerate(PM_Mid.zipped()):
        if x < mostNegativeX:
            furthestNegativeIndex = i
            mostNegativeX = x
        if x > mostPositiveX:
            furthestPositiveIndex = i
            mostPositiveX = x

    return furthestNegativeIndex, furthestPositiveIndex