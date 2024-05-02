import generate_pca_points_AVW as gic
import re
import Bottom_Tissue_SA_Final as bts
import time
import csv
import pandas as pd
import PCA_data
import os
import predict_funtions as pf
import matplotlib.pyplot as plt
import PointsExtractionTesting
import numpy as np
from math import hypot
from scipy import interpolate
""""
This files purpose is to verify the shape of the generated cylinders from "automate_febio.py",
we generate the bottom pca points given the .feb  &+ .log file for both the inner and outer radius of the given cylinder
"""

# File path to .feb & log file
feb_name = 'D:\\Gordon\\Automate FEB Runs\\2024_4_29 auto\\_Part5_E(0.78)_Pressure(0.06)_Inner_Radius(2.3)_Outer_Radius(4.5).feb'
log_name = 'D:\\Gordon\\Automate FEB Runs\\2024_4_29 auto\\_Part5_E(0.78)_Pressure(0.06)_Inner_Radius(2.3)_Outer_Radius(4.5).log'

# Parameters for functions below
obj = 'Object5'
window_width = 0.3
num_pts = 9
spline_ordered = 0

logCoordinates = gic.extract_coordinates_from_final_step(log_name, feb_name, obj)


"""
Function: generate_cylinder_bottom(numpts, extract_pts, window_width)

This function takes in a desired amount of points (numpts), the point cloud (extract_pts), and desired window size
We then generate the "bottom" points of the cylinder by finding the ymin value for each z_value index, which
we determine through the numpts passed in.
"""
def generate_outer_cylinder_bottom(numpts, extract_pts, window_width):
   #initialize maxz, minz, & best points array which we will be returning
   best_points = []
   maxz = 0
   minz = np.infty

   # iterate through each element within extract_pts
   for ele in extract_pts:
      # if current elements z value is greater than maxz, then update maxz
      if ele[1][2] > maxz:
         maxz = ele[1][2]
      # if current elements z value is less than minz, then update minz
      if ele[1][2] < minz and ele[1][2] >= 0:
         minz = ele[1][2]

   # divide up z points using linspace given the desired numpts from user
   z_values = np.linspace(minz, maxz, numpts)

   # determine width of 2nd window
   window2_width = ((maxz - minz)/(numpts-1)) / 2

   # iterate through each z-value in z_values
   for i, z in enumerate(z_values):
      # initialize ymin to infinity which we will be updating later
      ymin = np.infty
      # iterate through each element within extract_pts
      for ele in extract_pts:
         # determine whether the "|X| < window_width"
         if abs(ele[1][0]) < window_width:
            # determine whether the "|Z - current_z_in_loop| < window2_width"
            if abs(ele[1][2] - z) < window2_width:
               # determine if "y < ymin"
               if ele[1][1] < ymin:
                  # update ymin to equal y & assign z value
                  ymin = ele[1][1]
                  zvalue = ele[1][2]
      # append values to best_points after ymin and z value are determined
      best_points.append([0, ymin, zvalue])

   return best_points

# assign cylinder_bottom equal to generate_outer_cylinder_bottom given parameters.
cylinder_bottom = generate_outer_cylinder_bottom(num_pts, logCoordinates, window_width)

# Sort logCoordinates into regular 2d array ready for plotting
logStripped = []
for ele in logCoordinates:
   logStripped.append(ele[1])

# convert arrays to np.arrays
logStripped = np.array(logStripped)
cylinder_bottom = np.array(cylinder_bottom)

"""
Function: plot_cylinder_bottom(cylinder, cylinder_bottom)

Simple helper function which takes in regular cylinder & calculated cylinder bottom coordinates and
plots them along the same graph to compare differences between the two
"""
def plot_cylinder_bottom(cylinder, cylinder_bottom):
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   # Plot points
   ax.scatter(cylinder[:, 0], cylinder[:, 1], cylinder[:, 2], c='g', marker='o')
   ax.scatter(cylinder_bottom[:, 0], cylinder_bottom[:, 1], cylinder_bottom[:, 2], c='r', marker='X')
   # Set labels
   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')
   # Set aspect ratio
   ax.set_box_aspect([1, 1, 1])
   plt.show()
   return 0

# Call plot function
#plot_cylinder_bottom(logStripped, cylinder_bottom)

#TODO: Create generate_inner_cylinder_bottom
#TODO: THIS FUNCTION IS NOT FINISHED, DIRECTLY PORTED FROM genertate_outer_cylinder_bottom
"""
Function: generate_inner_cylinder_bottom(numpts, extract_pts, window_width)
"""
def generate_inner_cylinder_bottom(numpts, extract_pts, window_width):
   # initialize maxz, minz, & best points array which we will be returning
   best_points = []
   maxz = 0
   minz = np.infty

   # iterate through each element within extract_pts
   for ele in extract_pts:
      # if current elements z value is greater than maxz, then update maxz
      if ele[1][2] > maxz:
         maxz = ele[1][2]
      # if current elements z value is less than minz, then update minz
      if ele[1][2] < minz and ele[1][2] >= 0:
         minz = ele[1][2]

   # divide up z points using linspace given the desired numpts from user
   z_values = np.linspace(minz, maxz, numpts)
   # determine width of 2nd window
   window2_width = ((maxz - minz) / (numpts - 1)) / 2

   # iterate through each z-value in z_values
   for i, z in enumerate(z_values):
      # initialize ymin to infinity which we will be updating later
      ymin = np.infty
      # iterate through each element within extract_pts
      for ele in extract_pts:
         # determine whether the "|X| < window_width"
         if abs(ele[1][0]) < window_width:
            # determine whether the "|Z - current_z_in_loop| < window2_width"
            if abs(ele[1][2] - z) < window2_width:
               # determine if "y < ymin"
               if abs(ele[1][1]) < ymin and ele[1][1] < 0:
                  # update ymin to equal y & assign z value
                  ymin = abs(ele[1][1])
                  zvalue = ele[1][2]
      # append values to best_points after ymin and z value are determined
      best_points.append([0, ymin, zvalue])

   return best_points

"""
this is a helper function to get the distances between the spline points and then return the 
ys and zs for the spline points

Parameters:
The coordinate list of the desired points on the cylinder

Returns:
ys and zs array that dont include x values
the distance array between each point
"""
def get_spline_points(coords_list):
   #this is to get rid of the x values since they are all zeros
   ys_zs = coords_list[:, 1:]
   #calculates the new distances between the points
   new_distance_array = [0]
   for i in range(1, len(ys_zs)):
      new_distance_array.append(hypot(ys_zs[i][0] - ys_zs[i-1][0],
                                      ys_zs[i][1] - ys_zs[i-1][1]) + new_distance_array[-1])
   print("New Distance Array: ", new_distance_array)


   print("BEST POINTS", ys_zs)
   return ys_zs, new_distance_array
"""
This function is to generate the 2d coordinates of our cylinder model. it is similar to a function in generate_pca_points but does some
things differently.

Parameters:
takes in a coordinates list that is an array of arrays that contain the x, y, and z values

Returns:
newys and newzs

"""
def generate_2d_coords_for_cylinder_pca(coords_list):

   spline_list, dist_array = get_spline_points(coords_list)
   #gets all of the ys and zs and inserts them into their own arrays
   ys = [i[0] for i in spline_list]
   zs = [i[1] for i in spline_list]

   #uses a function from interpolate that calculates TODO: find out what UnivariateSpline does
   curve_y = interpolate.UnivariateSpline(dist_array, ys, k = 5)
   curve_z = interpolate.UnivariateSpline(dist_array, zs, k = 5)
   #finds the equal amount of space between each element
   spaced_distace_array = np.linspace(0, dist_array[-1], num_pts)

   #calls curve_y to find a curve to find y and z coordinate of the curve
   previous_y = curve_y(0).tolist()
   previous_z = curve_z(0).tolist()
   previous_y = np.array(previous_y)
   previous_z = np.array(previous_z)

   new_ys = [previous_y]
   new_zs = [previous_z]

   #does the same as above but for all ys and zs
   for i in range(1, len(spaced_distace_array)):
      new_ys.append(float(curve_y(spaced_distace_array[i])))
      new_zs.append(float(curve_z(spaced_distace_array[i])))

   return new_ys + new_zs

cylinder_inner_bottom = generate_inner_cylinder_bottom(num_pts, logCoordinates, window_width)
cylinder_inner_bottom = np.array(cylinder_inner_bottom)
plot_cylinder_bottom(logStripped, cylinder_bottom)

#TODO: plot this to see what it outputs
#y,z = generate_2d_coords_for_cylinder_pca(cylinder_bottom)
#this allows us to plot the newly made points generated by the function above
"""
x_zeros = np.zeros_like(y)
coordinates_with_zeros_x = np.column_stack((x_zeros, y, z))
plot_cylinder_bottom(coordinates_with_zeros_x, logStripped)
"""
