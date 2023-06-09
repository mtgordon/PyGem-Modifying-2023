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
import PCA_data
import pandas as pd
import re
import Bottom_Tissue_SA_Final as bts
first_file_flag = True
first_int_file_flag = True
last_object = False
current_date = datetime.datetime.now()
date_prefix = str(current_date.year) + '_' + str(current_date.month)  + '_' + str(current_date.day)
Results_Folder = "C:\\Users\\phine\\OneDrive\\Desktop\\FEBio files\\Pycharm Results"
csv_filename = Results_Folder + '\\' + date_prefix + '_intermediate.csv'

object_list = ['Object2', 'Object8','Object16']
obj_coords_list = []
file_num = 0

GENERATE_INTERMEDIATE_FLAG = True


if GENERATE_INTERMEDIATE_FLAG:

    for feb_name in glob.glob("C:\\Users\\phine\\OneDrive\\Desktop\\FEBio files\\Test_post_process_driver\\*.feb"):

        int_log_name = feb_name.split(".f")
        int_log_name[1] = ".log"
        log_name = int_log_name[0]+int_log_name[1]
        file_params = int_log_name[0].split('\\')[-1]
        if first_int_file_flag:
            proc.generate_int_csvs(file_params, object_list, log_name, feb_name,first_int_file_flag,csv_filename)
            first_int_file_flag = False
        else:
            proc.generate_int_csvs(file_params, object_list, log_name, feb_name,first_int_file_flag,csv_filename)
        file_num += 1
        print('Completed Iteration ' + str(file_num) + ": " + file_params)
        obj_coords_list = []

proc.process_features(csv_filename,Results_Folder,date_prefix)
print('TESTING')
# use the generated csv to get the 2 PC scores
