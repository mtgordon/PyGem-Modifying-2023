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

# File path to .feb & log file
feb_name = 'D:\\Gordon\\Automate FEB Runs\\2024_4_29 auto\\_Part5_E(0.78)_Pressure(0.06)_Inner_Radius(2.3)_Outer_Radius(4.5).feb'
log_name = 'D:\\Gordon\\Automate FEB Runs\\2024_4_29 auto\\_Part5_E(0.78)_Pressure(0.06)_Inner_Radius(2.3)_Outer_Radius(4.5).log'

# Parameters for functions below
obj = 'Object5'
window_width = 0.3
num_pts = 9

logCoordinates = gic.extract_coordinates_from_final_step(log_name, feb_name, obj)

#TODO: Create functioning generate_cylinder_bottom method
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
plot_cylinder_bottom(logStripped, cylinder_bottom)
