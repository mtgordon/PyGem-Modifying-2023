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
#import generate_int_csvs as gic
import PostProcess_FeBio as proc
import PCA_data
import pandas as pd
import re
import Bottom_Tissue_SA_Final as bts
import CylinderFunctions
import lib.IOfunctions as IO

current_date = datetime.datetime.now()
date_prefix = str(current_date.year) + '_' + str(current_date.month)  + '_' + str(current_date.day)
Results_Folder = "D:\\Gordon\\Automate FEB Runs\\2024_5_9_NewModel\\Test_Folder_5.20"  # INTERMEDIATE CSV ENDS UP HERE
Target_Folder = "D:\\Gordon\\Automate FEB Runs\\2024_5_9_NewModel\\Test_Folder_5.20\\Part1_E(2.37)Part2_E(1.3)Part5_E(2.02)Part7_E(1.94)Part8_E(2.85)Pressure(0.08)Inner_Radius(0.5812)Outer_Radius(1.3154).feb"  # LOOK HERE FOR THE FEB FILES
csv_filename = Results_Folder + '\\' + date_prefix + '_intermediate.csv'
date_prefix = str(current_date.year) + '_' + str(current_date.month)  + '_' + str(current_date.day)

object_list = ["Object8"]  # MAKE SURE THIS MATCHES THE OBJECTS IN THE CURRENTLY USED MODEL
part_list = ["Part1"]
obj_coords_list = []
file_num = 0
numCompPCA = 2

first_file_flag = True
GENERATE_INTERMEDIATE_FLAG = True
final_csv_flag = False


def getCylinderEdgePoints(feb_file, part_list):
    if isinstance(part_list, str):
        part_list = [part_list]

    tree = ET.parse(feb_file)
    root = tree.getroot()

    element_counts = {}

    for part in part_list:
        # Debug: print the part being processed
        print(f"Processing part: {part}")

        # Find the Elements tag for the current part
        elements = root.find('.//Elements[@type="hex8"][@name="{}"]'.format(part))

        if elements is not None:
            # Debug: confirm Elements tag found
            print(f"Found Elements tag for part: {part}")

            # Find all elem elements within the Elements
            elem_elements = elements.findall('elem')

            # Iterate over each elem element
            for elem in elem_elements:
                # Extract the text content containing the numbers
                numbers_text = elem.text.strip()
                # Convert text to list of integers
                elem_ids = [int(num.strip()) for num in numbers_text.split(',')]

                # Increment the count for each node ID
                for node_id in elem_ids:
                    if node_id in element_counts:
                        element_counts[node_id] += 1
                    else:
                        element_counts[node_id] = 1

    edge_elements_dictionary = {}
    edge_element_ids = []
    for elem in element_counts:
        if (element_counts[elem] == 2):
            edge_element_ids.append(elem)

    for node in root.findall('.//Nodes/node'):
        # Get the ID of the current node
        current_node_id = int(node.get('id'))

        # Check if the current node ID is in the set of elem IDs
        if current_node_id in edge_element_ids:
            # Extract coordinates from the node text
            coordinates = [float(coordinate) for coordinate in node.text.split(',')]
            # Add the node ID and coordinates to the dictionary
            edge_elements_dictionary[current_node_id] = coordinates
    print("All edges: ", edge_elements_dictionary)
    return edge_elements_dictionary

def getRadiiFromEdges(edge_elements_dictionary):
    # Get dictionary containing nodes of top inner & outer radii
    topedges_dictionary = {}
    for edge, value in edge_elements_dictionary.items():
        if value[2] == 4:
            topedges_dictionary[edge] = value

    CylinderFunctions.plot_3d_points(topedges_dictionary)
    max_value = -float('inf')
    for key, value in topedges_dictionary.items():
        if value[1] > max_value:
            max_key = key
            max_value = value[1]

    startpoint = max_key

    #TODO: dictionary starting point value messes up, dictionaries are perfect except one point
    outer_radius_dict, inner_radius_dict = IO.find_closest_points(topedges_dictionary, startpoint, 0.2)
    CylinderFunctions.plot_3d_points(outer_radius_dict)
    CylinderFunctions.plot_3d_points(inner_radius_dict)


    return topedges_dictionary

edge_elements_dictionary = getCylinderEdgePoints(Target_Folder, part_list)
getRadiiFromEdges(edge_elements_dictionary)




if GENERATE_INTERMEDIATE_FLAG:

    for feb_name in glob.glob(Target_Folder):

        int_log_name = feb_name.split(".f")
        int_log_name[1] = ".log"
        log_name = int_log_name[0]+int_log_name[1]

        csv_row = []

        # Get the pure file name that just has the material parameters
        file_params = int_log_name[0].split('\\')[-1]

        proc.generate_int_csvs(file_params, object_list, log_name, feb_name, first_file_flag, csv_filename)

        if first_file_flag:
            first_file_flag = False

        # sleep to give the file time to reach directory
        time.sleep(1)
        file_num += 1
        print(str(file_num) + ": " + file_params)
        obj_coords_list = []

if final_csv_flag:
    print('Generating PC File')
    proc.process_features(csv_filename, Results_Folder, date_prefix, numCompPCA)