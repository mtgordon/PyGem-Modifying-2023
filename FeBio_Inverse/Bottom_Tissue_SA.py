import datetime

import dateutil.utils
from sklearn.preprocessing import StandardScaler
import xml.etree.ElementTree as ET
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import ttest_ind
import numpy as np
import itertools
from math import cos, radians, sin, hypot
from copy import copy
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
#import cv2
from scipy import interpolate
from subprocess import call
import re
import time
import seaborn as sns
import numpy as np
import csv
import glob
import generate_pca_points_AVW as gic
import PostProcess_FeBio as proc
import PCA_data as pcd
import pandas as pd
import re
#Added the necessary parameters for the fetching of the files that are going to be used in the SA analysis
first_file_flag = True
current_date = datetime.datetime.now()
date_prefix = str(current_date.year) + '_' + str(current_date.month)  + '_' + str(current_date.day)
Results_Folder = "C:\\Users\\phine\\OneDrive\\Desktop\\FEBio files\\Pycharm Results"
csv_filename = Results_Folder + '\\' + date_prefix + '_intermediate.csv'

object_list=['Object16']
obj_coords_list=[]

#intended for loop to fetch the bottom tissue feb files and also name the log file
for feb_name in glob.glob("C:\\Users\\phine\\OneDrive\\Desktop\\FEBio files\\Test_post_process_driver\\_Part2_E(1)_Part8_E(1.9579)_Part9_E(1)_Part12_E(1.7333)_Part27_E(2.2972).feb"):
    int_log_name = feb_name.split(".f")
    int_log_name[1] = ".log"
    log_name = int_log_name[0] + int_log_name[1]

    csv_row = []
    csv_headers = []

    # Get the pure file name that just has the material parameters
    file_params = int_log_name[0].split('\\')[-1]

    for obj in object_list:
        obj_coords_list.append(gic.extract_coordinates_from_final_step(log_name, feb_name, obj))

'''The function below is meant to extract the top ten coordinates in from the bottom tisue by the finding the lowest x
value in percentages. 1 top coordinate from every ten percent of coordinates'''
def extract_ten_coordinates_block(node_list):
    sublist = node_list[0]

    sorted_list = sorted(sublist, key=lambda x: x[1][1])
    selected_coordinates = []

    subset_size = int(len(sorted_list) * 0.1)
    for i in range(0, 10 * subset_size, subset_size):
        subset = sorted_list[i:i + subset_size]
        min_x = min(subset, key=lambda x: x[1][0])
        selected_coordinates.append([min_x[1]])
    print(selected_coordinates)
    return selected_coordinates

'''This function is meant to fetch the coordinates and fit them into different lists. Each one will be coinciding with
its entry into the first list'''
def get_x_y_z_values(ten_coords):
    x_values = [entry[0][0] for entry in ten_coords]
    y_values = [entry[0][1] for entry in ten_coords]
    z_values = [entry[0][2] for entry in ten_coords]
    return x_values, y_values, z_values

def slice_value(coord_list):
  mid_value =(max(coord_list)+min(coord_list))/2.0
  return mid_value
'''This function's main purpose is to get the midline of the bottom tissue across the same axis (Z in this case)'''
def get_bottom_tissue_midline(xs, ys, zs):
    THRESHOLD = 0.3  # Max Distance point can be from Z axis was 0.3
    slice_z_value = slice_value(zs)

    # Getting the middle nodes and finding the one with the largest z coordinate
    middle_nodes = []
    min_x = np.inf
    for index, zval in enumerate(zs):
        if abs(slice_z_value - zval) < THRESHOLD:
            middle_nodes.append((xs[index], ys[index]))
            if xs[index] < min_x:
                min_x = xs[index]
                start = (xs[index], ys[index])

    spline_ordered = [start]

    # Iteratively add points to the spline_ordered list based on minimum distance
    while len(middle_nodes) > 0:
        distances = np.array([hypot(spline_ordered[-1][0] - point[0], spline_ordered[-1][1] - point[1])
                              for point in middle_nodes])
        # Find the index of the point with the minimum distance
        min_index = np.argmin(distances)

        # Add the point with the minimum distance to the spline_ordered list
        spline_ordered.append(middle_nodes[min_index])

        # Remove the selected point from the middle_nodes list
        middle_nodes.pop(min_index)

    # Remove the first coordinate pair because it was just the starting one
    spline_ordered.pop(0)
    print(spline_ordered)

    if spline_ordered[0][0] < spline_ordered[-1][0]:
        spline_ordered = list(reversed(spline_ordered))

    # Smoothing out the data
    new_spline_ordered = [spline_ordered[0]]
    for i in range(1, len(spline_ordered) - 1):
        new_spline_ordered.append(((spline_ordered[i - 1][0] + spline_ordered[i][0] + spline_ordered[i + 1][0]) / 3,
                                   (spline_ordered[i - 1][1] + spline_ordered[i][1] + spline_ordered[i + 1][1]) / 3))
    new_spline_ordered.append(spline_ordered[-1])

    # Calculate new distances
    new_distance_array = [0]
    for i in range(1, len(spline_ordered)):
        new_distance_array.append(
            hypot(spline_ordered[i][0] - spline_ordered[i - 1][0],
                  spline_ordered[i][1] - spline_ordered[i - 1][1]) + new_distance_array[-1])

    return spline_ordered, new_distance_array

'''This function below is what is supposed to generate the midline and do the PCA spline on the bottom tissue the
resulting points will then be taken through the PCA analysis code from the PCA_data file so that it can get its own PCA
data'''
def generate_2d_bottom_tissue(coords_list):
    X,Y,Z = get_x_y_z_values(coords_list)
    sp_ordered, d_array = get_bottom_tissue_midline(X, Y, Z)

    xs = [i[0] for i in sp_ordered]
    ys = [i[1] for i in sp_ordered]
    curve_x = interpolate.UnivariateSpline(d_array, xs, k=5)
    curve_y = interpolate.UnivariateSpline(d_array, ys, k=5)

    spaced_distance_array = np.linspace(0, d_array[-1], 10)
    new_distance = 0
    new_distance_array = [0]
    previous_x = curve_x(0)
    previous_y = curve_y(0)
    new_xs = [previous_x]
    new_ys = [previous_y]

    for i in range(1, len(spaced_distance_array)):
        new_xs.append(float(curve_x(spaced_distance_array[i])))
        new_ys.append(float(curve_y(spaced_distance_array[i])))
    print(len(new_ys))

    plt.plot(new_ys, new_xs, color='b')
    plt.xlabel("Y coordinates")
    plt.ylabel("X coordinates")

    plt.show()

    return new_xs + new_ys
def process_bottom_tissue_features(csvfile):
    int_df = pd.read_csv(csvfile)
    pc_df = int_df.iloc[:, 4:len(int_df.columns)]
    # int_df = pd.read_csv("intermediate_pc_data", header=None)
    total_result_PC, pca = pcd.PCA_(pc_df)

    PC_scores = total_result_PC[['principal component 1', 'principal component 2']]
    print(PC_scores)

    final_df = pd.concat([int_df.loc[:, ["File Name", "Apex", "E1", "E2"]], PC_scores], axis=1)
    final_df.to_csv(Results_Folder + '\\' + date_prefix + "_features.csv", index=False)

coordinates = extract_ten_coordinates_block(obj_coords_list)
bottom_pc_points = generate_2d_bottom_tissue(coordinates)
# apex = proc.find_apex(obj_coords_list[1])










