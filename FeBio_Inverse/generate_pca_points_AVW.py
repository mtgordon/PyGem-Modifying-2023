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

def  generate_2d_coords_for_pca(coords_list):
  #coords_list = extract_coordinates_from_final_step(log_file_name,feb_file_name,obj_name)
  X,Y,Z = get_x_y_z_values(coords_list)
  spline_list,dist_array=get_AVW_midline(X,Y,Z)
  xs=[i[0] for i in spline_list]
  ys=[i[1] for i in spline_list ]

  curve_x = interpolate.UnivariateSpline(dist_array, xs, k = 5)
  curve_y = interpolate.UnivariateSpline(dist_array, ys, k = 5)

  spaced_distance_array = np.linspace(0, dist_array[-1], 15)#changed from from 51 to 15 and the return the 15 points


  new_distance = 0
  new_distance_array  = [0]
  previous_x = curve_x(0).tolist()
  previous_y = curve_y(0).tolist()
  new_xs = [previous_x]
  new_ys = [previous_y]

  for i in range (1,len(spaced_distance_array)):
      new_xs.append(float(curve_x(spaced_distance_array[i])))
      new_ys.append(float(curve_y(spaced_distance_array[i])))


  #TODO we have to make  this csv_filename dynamic but that will be later.

  # csv_filename= log_file_name.split("\\")[-1]
  # csv_filename = csv_filename.split(".lo")[0] + "_intermediate_csv"

  # save_to_csv(csv_filename,new_xs,new_ys)
  return new_xs + new_ys
#---------------------------------------------------------------------------------------------------------------------------------------------
#Helpers to the main list

# This function returns a list contains all the nodes' ids,
def extract_node_id_list_from_feb(file_name, node_name, get_connections=False):
    """
    This function extracts the IDs of nodes from an FEB file and returns them as a list.

    Parameters:
    file_name (str): The name of the FEB file to extract node IDs from.
    node_name (str): The name of the node element in the FEB file.
    get_connections (bool, optional): Whether to also extract the connections between nodes. Default is False.

    Returns:
    node_id_list (list): A list containing the IDs of nodes.
    """
    tree = ET.parse(file_name)
    root = tree.getroot()

    node_id_list = []  # an empty list

    # Find the 'Nodes' element with the target_name attribute
    nodes_element = root.find(f".//Nodes[@name='{node_name}']")

    # Check if the 'Nodes' element is found
    if nodes_element is not None:
        # Iterate over each 'node' element in the found 'Nodes' element
        # for node in nodes_element.iter('node'):
        #     node_id_list.append(tuple(float(coordinate) for coordinate in node.text.split(',')))

        node_id_list = [node.attrib['id'] for node in nodes_element.findall('node')]
    else:
        print(f"No 'Nodes' element with the name '{node_name}' found.")
    return node_id_list

def extract_coordinates_from_final_step(log_file_path,feb_file_path,object_value):
    index_list=extract_node_id_list_from_feb(feb_file_path,object_value)
    with open(log_file_path) as file:
        lines = file.readlines()

    # Find the index of the last step
    last_step_index = len(lines) - 1
    while last_step_index >= 0 and not lines[last_step_index].startswith('Step'):
        last_step_index -= 1
    if last_step_index < 0:
        raise ValueError('No Step found in the FEBio log file')

    # Extract the x, y, and z coordinates for the last step
    last_step_coords = []
    for line in lines[last_step_index+3:]:
        line = line.strip()
        if not line:
            break
        cols = line.split()
        node_index = (cols[0])
        if node_index in index_list:
          coords = [float(c) for c in cols[1:]]
          last_step_coords.append([node_index, coords])

    return last_step_coords

def get_x_y_z_values(node_list):
  x_values = [entry[1][0] for entry in node_list]
  y_values = [entry[1][1] for entry in node_list]
  z_values = [entry[1][2] for entry in node_list]
  return x_values, y_values, z_values


def slice_value(coord_list):
  mid_value =(max(coord_list)+min(coord_list))/2.0

  return mid_value

def get_AVW_midline(xs, ys, zs):
    THRESHOLD = 0.3  # Max Distance point can be from Z axis was 0.3 TODO:0.3 seems to be hard coded
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

def save_to_csv(file_name,x_coords,y_coords):
  with open(file_name,'a',newline='') as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(x_coords + y_coords)
