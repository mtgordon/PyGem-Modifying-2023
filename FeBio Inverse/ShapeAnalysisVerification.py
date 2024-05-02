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
num_pts = 15
spline_ordered = 0

logCoordinates = []
logCoordinates.append(gic.extract_coordinates_from_final_step(log_name, feb_name, obj))


"""
   Function: generate_outer_cylinder_bottom

   Generates the "bottom" points of a cylinder based on the provided parameters.

   Parameters:
   - numpts (int): Desired number of points.
   - extract_pts (list): Point cloud represented as a list of points, each point in the form [x, y, z].
   - window_width (float): Width of the search window for selecting points.

   Returns:
   - best_points (list): List of selected points representing the "bottom" of the cylinder, each point represented as [y, z].
"""
def generate_outer_cylinder_bottom(numpts, extract_pts, window_width):
   # Initialize variables
   best_points = []  # List to store the selected points
   maxz = 0  # Maximum z-value encountered in the point cloud
   minz = np.infty  # Minimum z-value encountered in the point cloud

   # Iterate through each element within extract_pts
   for ele in extract_pts:
      # Update maxz if the current element's z-value is greater than maxz
      if ele[2] > maxz:
         maxz = ele[2]
      # Update minz if the current element's z-value is less than minz and greater than or equal to 0
      if ele[2] < minz and ele[2] >= 0:
         minz = ele[2]

   # Divide up z points using linspace given the desired numpts from user
   z_values = np.linspace(minz, maxz, numpts)

   # Determine the width of the second search window
   window2_width = ((maxz - minz) / (numpts - 1)) / 2

   # Iterate through each z-value in z_values
   for i, z in enumerate(z_values):
      # Initialize ymin to infinity
      ymin = np.infty
      # Iterate through each element within extract_pts
      for ele in extract_pts:
         # Determine whether the absolute value of X is less than window_width
         if abs(ele[0]) < window_width:
            # Determine whether the absolute value of Z - current_z_in_loop is less than window2_width
            if abs(ele[2] - z) < window2_width:
               # Determine if Y is less than ymin
               if ele[1] < ymin:
                  # Update ymin and assign the corresponding z value
                  ymin = ele[1]
                  zvalue = ele[2]
      # Append values to best_points after ymin and z value are determined
      best_points.append([ymin, zvalue])

   return best_points

# assign cylinder_bottom equal to generate_outer_cylinder_bottom given parameters.
cylinder_bottom = generate_outer_cylinder_bottom(num_pts, logCoordinates[0], window_width)




"""
   Function: generate_inner_cylinder_bottom
 
    Generates the "bottom" points of the inner cylinder based on the provided parameters.

    Parameters:
    - numpts (int): Desired number of points.
    - extract_pts (list): Point cloud represented as a list of points, each point in the form [x, y, z].
    - window_width (float): Width of the search window for selecting points.

    Returns:
    - best_points (list): List of selected points representing the "bottom" of the inner cylinder, each point represented as [y, z].
"""
def generate_inner_cylinder_bottom(numpts, extract_pts, window_width):
   # Initialize variables
   best_points = []  # List to store the selected points
   maxz = 0  # Maximum z-value encountered in the point cloud
   minz = np.infty  # Minimum z-value encountered in the point cloud

   # Iterate through each element within extract_pts
   for ele in extract_pts:
      # Update maxz if the current element's z-value is greater than maxz
      if ele[2] > maxz:
         maxz = ele[2]
      # Update minz if the current element's z-value is less than minz and greater than or equal to 0
      if ele[2] < minz and ele[2] >= 0:
         minz = ele[2]

   # Divide up z points using linspace given the desired numpts from user
   z_values = np.linspace(minz, maxz, numpts)

   # Determine the width of the second search window
   window2_width = ((maxz - minz) / (numpts - 1)) / 2

   # Iterate through each z-value in z_values
   for i, z in enumerate(z_values):
      # Initialize ymin to infinity
      ymin = np.infty
      # Iterate through each element within extract_pts
      for ele in extract_pts:
         # Determine whether the absolute value of X is less than window_width
         if abs(ele[0]) < window_width:
            # Determine whether the absolute value of Z - current_z_in_loop is less than window2_width
            if abs(ele[2] - z) < window2_width:
               # Determine if Y is less than ymin and negative
               if abs(ele[1]) < ymin and ele[1] < 0:
                  # Update ymin and assign the corresponding z value
                  ymin = abs(ele[1])
                  zvalue = ele[2]
      # Append values to best_points after ymin and z value are determined
      best_points.append([ymin, zvalue])

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


def get_distance_and_coords(ys, zs):
   coords_2d = []
   for i in range(len(ys)):
      temparr = []
      temparr.append(ys[i])  # Append individual elements instead of the entire array
      temparr.append(zs[i])  # Append individual elements instead of the entire array
      coords_2d.append(temparr)

   # Calculate the new distances between the points
   new_distance_array = [0]
   for i in range(1, len(coords_2d)):
      distance = hypot(coords_2d[i][0] - coords_2d[i - 1][0], coords_2d[i][1] - coords_2d[i - 1][1])
      new_distance_array.append(distance + new_distance_array[-1])

   print("New Distance Array:", new_distance_array)

   return coords_2d, new_distance_array
#get_spline_points(cylinder_bottom)


"""
This function is to generate the 2d coordinates of our cylinder model. it is similar to a function in generate_pca_points but does some
things differently.

Parameters:
takes in a coordinates list that is an array of arrays that contain the x, y, and z values

Returns:
newys and newzs
"""
def generate_2d_coords_for_cylinder_pca(coords_list):
   #  to fix, ctrl z till this is gone.
   X,Y,Z = gic.get_x_y_z_values(coords_list)

   y_and_z_coords, dist_array = get_distance_and_coords(Y, Z)
   #gets all of the ys and zs and inserts them into their own arrays
   ys = [i[0] for i in y_and_z_coords]
   zs = [i[1] for i in y_and_z_coords]

   #uses a function from interpolate that calculates TODO: find out what UnivariateSpline does
   curve_y = interpolate.UnivariateSpline(dist_array, ys, k = 5)
   curve_z = interpolate.UnivariateSpline(dist_array, zs, k = 5)
   #  finds the equal amount of space between each element
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
































# ****************************************
# PLOTTING STUFF ON MATPLOTLIB FOR VISUALIZATIONS


#TODO: Sort logCoordinates into regular 2d array ready for plotting
#logStripped = []
#for ele in logCoordinates:
#   logStripped.append(ele[1])

# convert arrays to np.arrays
#logStripped = np.array(logStripped)
#cylinder_bottom = np.array(cylinder_bottom)

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