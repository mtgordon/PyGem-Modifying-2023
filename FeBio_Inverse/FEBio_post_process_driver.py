import datetime

import dateutil.utils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xml.etree.ElementTree as ET
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
import ShapeAnalysisVerification as sav
import PCA_data
import pandas as pd
import re
import Bottom_Tissue_SA_Final as bts
import CylinderFunctions
import lib.IOfunctions as IO
import CylinderFunctions as cf

current_date = datetime.datetime.now()
date_prefix = str(current_date.year) + '_' + str(current_date.month)  + '_' + str(current_date.day)
Results_Folder = "D:\\Gordon\\Automate FEB Runs\\2024_5_9_NewModel\\TEST_FOLDER_6.4"  # INTERMEDIATE CSV ENDS UP HERE
Target_Folder = "D:\\Gordon\\Automate FEB Runs\\2024_5_9_NewModel\\TEST_FOLDER_6.4\\*.feb"  # LOOK HERE FOR THE FEB FILES
csv_filename = Results_Folder + '\\' + date_prefix + '_intermediate.csv'
date_prefix = str(current_date.year) + '_' + str(current_date.month)  + '_' + str(current_date.day)

object_list = ['Levator Ani Side 2']  # MAKE SURE THIS MATCHES THE OBJECTS IN THE CURRENTLY USED MODEL
part_list = ['Part1', 'Part3', 'Part7', 'Part10', 'Part11']
obj_coords_list = []
file_num = 0
numCompPCA = 3

first_file_flag = True
GENERATE_INTERMEDIATE_FLAG = False
final_csv_flag = True
plot_points_on_spline = False



if GENERATE_INTERMEDIATE_FLAG:

    for feb_name in glob.glob(Target_Folder):

        int_log_name = feb_name.split(".f")
        int_log_name[1] = ".log"
        log_name = int_log_name[0]+int_log_name[1]


        csv_row = []

        # Get the pure file name that just has the material parameters
        file_params = int_log_name[0].split('\\')[-1]

        edge_elements_dictionary = sav.getCylinderEdgePoints(feb_name, part_list)
        tree = ET.parse(feb_name)
        root = tree.getroot()
        extract_points = CylinderFunctions.get_initial_points_from_parts(root, part_list)
        inner_radius, outer_radius = sav.getRadiiFromEdges(edge_elements_dictionary, cf.findLargestZ(extract_points), log_name, feb_name, object_list[0])

        proc.generate_int_csvs(file_params, object_list, log_name, feb_name, first_file_flag, csv_filename, inner_radius, outer_radius, )

        if first_file_flag:
            first_file_flag = False

        # sleep to give the file time to reach directory
        time.sleep(1)
        file_num += 1
        print(str(file_num) + ": " + file_params)
        obj_coords_list = []

if final_csv_flag:
    print('Generating PC File')
    filepath = proc.process_features(csv_filename, Results_Folder, date_prefix, numCompPCA)
    print("File path: ", filepath)