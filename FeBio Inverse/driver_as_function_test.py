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
import FEBio_post_process_driver_as_a_function as ppd

#filepath = "C:\\Users\\Elijah Brown\\Desktop\\Bio Research\\Post Process\\*.feb"

#filepath = "/FeBio Inverse/_Part2_E(1.05)_Part2_v(1.15)_Part9_E(0.80)_Part9_v(1)_Part27_E(0.90)_Part27_v(1).feb"
object_list = ['Object2', 'Object8']
intermediate_csv = '_intermediate.csv'
Results_Folder = 'D:\\Gordon\\Automate FEB Runs\\2023_5_23 auto'
filepath = "C:\\Users\\EGRStudent\\Desktop\\Test_post_process_driver\\*.feb"
ppd.febio_post_function(filepath, object_list, intermediate_csv, Results_Folder)
