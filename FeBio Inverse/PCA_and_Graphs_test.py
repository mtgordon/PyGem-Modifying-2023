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
import PCA_data as pcd
import time
import seaborn as sns
'''Overall bjective of the first function is to cumulatively do all the graph work before the pca is done.'''
def do_PCA_and_basic_graphs(pca_trials_df, no_points):
    #"print("new function")
    data_plot_basic(pca_trials_df, no_points)
    plt.show()

    #performing PCA on the dataframe
    pca_df, pca = pcd.PCA_(pca_trials_df)

    pca_df.iloc[4:]

    for pc_no in range (1,3):
        print("pc_no:", pc_no)
        mode_shape_graphs_hold_basic(pca, pca_df, pc_no)
        plt.show()

    # pc_no = 1
    # print("pc_no:", pc_no)
    # mode_shape_graphs_hold_basic(pca, pca_df, pc_no)
    # plt.show()
    # # plt.show()

def data_plot_basic(pca_trials_df, no_points):
    #print("new data plot")

    print('headers:', list(pca_trials_df.columns))
    #for loop to get all the xs and ys for plotting
    for index, row in pca_trials_df.iterrows():
        xs = []
        ys = []
        for i in range(no_points):
            xs.append(row['x'+str(i+1)])
            ys.append(row['y'+str(i+1)])

    #plot the data points
        plt.plot(xs, ys)


#This function plots the mode shape graph of a dataframe given the pca analysis
def mode_shape_graphs_hold_basic(pca, pca_df, pc_no):

    headers = list(pca_df.columns)
    non_pca_columns = [x for x in headers if not re.match('[x|y]\\d+', x)]

    only_coord_df = pca_df.drop(columns = non_pca_columns)
    coords_means = only_coord_df.mean(axis=0)

    # stds = pca_df.std(axis = 0, skipna = True)


    # Magnitudes (coefficients) of the variance plotted
    magnitudes = [-2, -1, 0, 1, 2]

    # pc_stds = [stds['principal component 1'], stds['principal component 2']]

    print("test 1")
    for mag in magnitudes:
        if mag == 0:
            line_color = 'k'
            line_style = '--'
        else:
            line_color = 'r'
            line_style = '--'
        shape_mode_xs = []
        shape_mode_ys = []
        for i in range(0,len(coords_means)):
            coordinate = pca.components_[pc_no - 1][i]*mag*(pca.explained_variance_[pc_no - 1]**0.5) + coords_means[i]
            if i < (len(coords_means)/2):
                shape_mode_xs.append(coordinate)
            else:
                shape_mode_ys.append(coordinate)
        # axes = plt.gca()

        plt.plot(shape_mode_xs, shape_mode_ys, color = line_color, linestyle = line_style, zorder = 1, linewidth = 0.5)

    plt.axis('off')
#        plt.show()

#-----------------------------------------------------------------------------------------------------------------------

# def PCA_and_graphs(pca_trials_df, no_points):
#     data_plot_v2(pca_trials_df, no_points, 0)
#
#     fig_name = 'raw_subject_data'
#
#     #    im = plt.imread('sacrum.png')
#     #    plt.imshow(im, origin = 'upper', extent = [-68,7,-25,100])
#     #
#     #    new_ellipse = copy(pubic_ellipse)
#     #    plt.gca().add_patch(new_ellipse)
#     #    plt.plot([0,PS_point[0]], [0,-1*PS_point[1]], '--', color = 'k', linewidth = SCIPP_linewidth)
#     #    axes = plt.gca()
#     #    axes.set_ylim([-75, 50])
#     #    axes.set_xlim([-50, 15])
#
#     # plot_accessories(pubic_ellipse, PS_point, SCIPP_linewidth)
#
#     # data_plot(pca_trials_df, no_points, 0)
#     data_plot_v2(pca_trials_df, no_points, 0)
#
#     # file_name = fig_file_prefix + fig_name + '.' + fig_ext
#     # emf_file_name = fig_file_prefix + fig_name + '.emf'
#     # plt.savefig(results_folder + '\\' + file_name, format=fig_ext, dpi=plot_dpi)
#
#     # call([r"C:\\Program Files\\Inkscape\\bin\\inkscape.exe", results_folder + '\\' + file_name, "--export-filename",
#     #       results_folder + '\\' + emf_file_name])
#     plt.show()
#
# def data_plot_v2(pca_trials_df, no_points, show):
#     #print('headers:', list(pca_trials_df.columns))
#     for index, row in pca_trials_df.iterrows():
#         num_coords = no_points*2
#         xs = []
#         ys = []
#
#         for i in range(4,len(row)):
#             if i < 4+no_points:
#                 xs.append(row[i])
#             else:
#                 ys.append(row[i])
#         # for i in range(no_points):
#         #     xs.append(row['x'+str(i+1)])
#         #     ys.append(row['y'+str(i+1)])
#         axes = plt.gca()
#         axes.set_ylim([-10, 10])
#         axes.set_xlim([0, 10])
#         plt.plot(xs, np.multiply(ys,-1), color = 'red', linewidth = 0.5)
#         #plt.scatter(xs, np.multiply(ys,-1), color = 'blue')
#     #plt.axis('off')
#     if show == 1:
#         plt.show()
#
# def pca_plot_2vars_v2(pca_df,show):
#     # variable = 'age'
#     # states = states
# # def data_plot_v2(pca_trials_df, no_points, show, category_header, categories, ss, line_colors):
#     # Plot the results of the 1st to principal components
#     # print(variable, states)
#     fig = plt.figure(figsize = (8,8))
#     ax = fig.add_subplot(1,1,1)
#     ax.set_xlabel('Principal Component 1', fontsize = 15)
#     ax.set_ylabel('Principal Component 2', fontsize = 15)
#     ax.set_title('2 component PCA', fontsize = 20)
#
#     # separation_column = variable #'target'
#     # separation_values = states #
#
#
#     #separation_column = 'state'
#     # values = ['failure', 'success']
#
#
#     # colors = ['r', 'g', 'b']
# #     for value, color in zip(states, colors):
# #
# # #        indicesToKeep = pca_df['target'] == value
# # #        target_filter_df = pca_df.loc[indicesToKeep, 'principal component 1']
# #         target_filter_df = pca_df[pca_df[variable].isin([value])]
# #         # point_styles = ['none', color]
# # #        full_df[full_df.time.isin(['pre'])]
# #         # for state, fill_style in zip(separation_values, point_styles):
# #         #     indicesToKeep = target_filter_df[separation_column] == state
# #         ax.scatter(target_filter_df['principal component 1']
# #                , target_filter_df['principal component 2']
# #                , edgecolors = color
# #                , s = 50
# #                , facecolors = color)
# # #        print(pca_df)
# # #        print(indicesToKeep)
# # #        print(pca_df.loc[indicesToKeep, 'principal component 1'])
# #         # if value == 'success':
# #         #     success_pca_df = target_filter_df.loc[indicesToKeep, 'principal component 1']
# #         # else:
# #         #     failure_pca_df = target_filter_df.loc[indicesToKeep, 'principal component 1']
# #     # print('pca_plot_2vars_v2')
# #     ax.legend(states)
# #    ax.grid()
#     if show == 1:
#         plt.show()
#
# def mode_shape_graphs(pca, pca_df):
#     print('Explained Variance Ratio:', pca.explained_variance_ratio_)
#
#     headers = list(pca_df.columns)
#     non_pca_columns = [x for x in headers if not re.match('[x|y]\\d+', x)]
#
#     only_coord_df = pca_df.drop(columns=non_pca_columns)
#     coords_means = only_coord_df.mean(axis=0)
#
#     stds = pca_df.std(axis=0, skipna=True)
#
#     magnitudes = [-2, -1, 0, 1, 2] # Magnitudes (coefficients) of the variance plotted
#
#     # pc_stds = [stds['principal component 1', 'principal component 2']]
#     figure_num = 3
#     fig = plt.figure(1)
#     for eigenvector in range(0, len(pca.components_)):
#         for mag in magnitudes:
#             figure_num += mag
#             if mag == 0:
#                 line_color = 'k'
#                 line_style = '-'
#             else:
#                 line_color = 'k'
#                 line_style = '--'
#             shape_mode_xs = []
#             shape_mode_ys = []
#             for i in range(0, len(coords_means)):
#                 coordinate = pca.components_[eigenvector][i] * mag * (pca.explained_variance_[eigenvector] ** 0.5) + coords_means[i]
#                 if i < (len(coords_means) / 2):
#                     shape_mode_xs.append(coordinate)
#                 else:
#                     shape_mode_ys.append(coordinate)
#
#             ax = fig.add_subplot(111)
#             label_tag = str(mag)
#             plt.plot(shape_mode_xs, np.multiply(shape_mode_ys, -1) ,label = label_tag, linestyle=line_style)
#
#         #plt.axes().set_aspect('equal', 'datalim')
#         # plt.axis('off')
#         plt.show()
