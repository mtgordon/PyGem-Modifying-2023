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
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
#import cv2
from scipy import interpolate
from skimage import io
from skimage import data
from subprocess import call
import re
import time
import seaborn as sns
import numpy as np
import csv
import glob
import generate_int_csvs as gic

for feb_name in glob.glob("C:\\Users\\EGRStudent\\PycharmProjects\\PyGem-Modifying-2023\\FeBio Inverse\\*.feb"):

    int_log_name = feb_name.split(".")
    int_log_name[1] = ".log"
    log_name = int_log_name[0]+int_log_name[1]

    gic.generate_2d_coords_for_pca(log_name, feb_name, "Object2")

