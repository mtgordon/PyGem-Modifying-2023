from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import ttest_ind
import numpy as np
import itertools
from math import cos, radians, sin
from copy import copy
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
#import cv2
from skimage import io
from skimage import data
from subprocess import call
import re
import time
import seaborn as sns
import PCA_and_Graphs_test as pag
import PCA_data_Copy
import csv
import glob
import generate_int_csvs as gic
import PostProcess_FeBio as proc
import PCA_data as pdd
import pandas as pd
from PCA_and_Graphs_test import do_PCA_and_basic_graphs


plot_df = pd.read_csv("C:\\Users\\phine\\Downloads\\combo_intermediate_as_of_5_26.csv")
print(plot_df)

no_points = 15
do_PCA_and_basic_graphs(plot_df, no_points)



