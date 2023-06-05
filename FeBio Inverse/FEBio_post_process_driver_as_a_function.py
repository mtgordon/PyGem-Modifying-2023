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
# import cv2
from scipy import interpolate
from subprocess import call
import re
import time
import seaborn as sns
import numpy as np
import csv
import glob
import generate_int_csvs as gic
import PostProcess_FeBio as proc
import PCA_data
import pandas as pd
import re

# filepath = "C:\\Users\\Elijah Brown\\Desktop\\Bio Research\\Post Process\\*.feb"
object_list = ['Object2', 'Object8']
intermediate_csv = '_intermediate.csv'
Results_Folder = 'D:\\Gordon\\Automate FEB Runs\\2023_5_23 auto'


def febio_post_function(filepath, object_list, intermediate_csv, Results_Folder, first_int_file_flag=True):
    current_date = datetime.datetime.now()
    date_prefix = str(current_date.year) + '_' + str(current_date.month) + '_' + str(current_date.day)
    GENERATE_INTERMEDIATE_FLAG = True
    csv_filename = Results_Folder + '\\' + date_prefix + intermediate_csv
    obj_coords_list = []
    file_num = 0

    if GENERATE_INTERMEDIATE_FLAG:

        for feb_name in glob.glob(filepath):

            int_log_name = feb_name.split(".f")
            int_log_name[1] = ".log"
            log_name = int_log_name[0] + int_log_name[1]

            csv_row = []
            csv_header = []

            # Get the pure file name that just has the material parameters
            file_params = int_log_name[0].split('\\')[-1]

            # Get the changed material properties
            paren_pattern = re.compile(r'(?<=\().*?(?=\))')  # find digits in parentheses
            prop_result = paren_pattern.findall(file_params)
            prop_final = []
            for prop in prop_result:
                prop = float(prop)
                if prop != 1.0:
                    prop_final.append(prop)

            # Get the coordinates for each object in list
            for obj in object_list:
                obj_coords_list.append(gic.extract_coordinates_from_final_step(log_name, feb_name, obj))
                print('Extracting... ' + obj + ' for ' + file_params)

            # Get the PC points for Object2
            pc_points = gic.generate_2d_coords_for_pca(obj_coords_list[0])

            # Begin building the row to be put into the intermediate csv
            csv_row.append(file_params)  # file params
            csv_row.append(proc.find_apex(obj_coords_list[1]))  # apex FIX

            csv_row.extend(prop_final)
            csv_row.extend(pc_points)  # the 30 pc coordinates

            if first_int_file_flag:
                csv_header.append('File Name')
                csv_header.append('E1')
                csv_header.append('E2')
                csv_header.append('Apex')
                coord = 'x'
                for i in range(2):
                    if i == 1:
                        coord = 'y'
                    for j in range(15):
                        csv_header.append(coord + str(j + 1))

                with open(csv_filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(csv_header)
                    writer.writerow(csv_row)
                    first_int_file_flag = False
            else:
                with open(csv_filename, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(csv_row)

            # sleep to give the file time to reach directory

            time.sleep(1)
            file_num += 1
            print(str(file_num) + ": " + file_params)
            obj_coords_list = []

        process_features(csv_filename, date_prefix)


# use the generated csv to get the 2 PC scores
def process_features(csv_file, date_prefix):
    int_df = pd.read_csv(csv_file)
    pc_df = int_df.iloc[:, 4:len(int_df.columns)]
    # int_df = pd.read_csv("intermediate_pc_data", header=None)
    total_result_PC, pca = PCA_data.PCA_(pc_df)

    PC_scores = total_result_PC[['principal component 1', 'principal component 2']]
    print(PC_scores)

    final_df = pd.concat([int_df.loc[:, ["File Name", "E1", "E2", "Apex"]], PC_scores], axis=1)
    final_df.to_csv(Results_Folder + '\\' + date_prefix + "_features.csv", index=False)
