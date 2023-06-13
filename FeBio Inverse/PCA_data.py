# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:32:40 2020

@author: mgordon
"""
from math import hypot
import datetime

from scipy.interpolate import interpolate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from scipy.stats import ttest_ind
import numpy as np
import re
from scipy.interpolate import UnivariateSpline

def do_PCA_get_data(pca_trials_df, PS_point, no_points, fig_file_prefix, results_folder, group_category, groups):
    pca_df, pca = PCA_(pca_trials_df)

    file_name = results_folder + '\\PCA_shape_data.csv'

    PCA_shape_data(pca, pca_df, file_name)


####################################################
########## Do the PCA ##############################
####################################################

def PCA_(pca_trials_df):
    ### Notes
    '''
    full_df is all of the data about the subject and run
    pca_trials_df is all of the data about the trials that are used for the pca
    pca_trials_data_df is the data needed for pca_analysis
    pca_df is the pca_trials_df PLUS the results of the pca
    '''

    # headers = pca_trials_df.columns
    # non_pca_columns = [x for x in headers if not re.match('[x|y]\\d+', x)]
    #
    # print('first:', pca_trials_df)
    # pca_trials_data_df = pca_trials_df.drop(columns = non_pca_columns)
    #TODO: use the raw csv for right now as it is only the coordinate data
    #pca_trials_data_df = pca_trials_df.drop(columns = ["File Name","E1","E2","E3","Apex"])
    if isinstance(pca_trials_df, list):
        pca_trials_data_df = pca_trials_df[0]
    else:
        pca_trials_data_df = pca_trials_df

    # print('second:', pca_trials_data_df)
    x = StandardScaler(with_std = False).fit_transform(pca_trials_data_df)
#    x = StandardScaler().fit_transform(pca_trials_data_df)
#    print(x)
    # How many components to look at
    PCA_components = 2

    # Genreate the PCA and record the principal components
    pca = PCA(n_components = PCA_components)
    #principalComponents = pca.fit_transform(x)
    principalComponents = pca.fit_transform(x)

    IDs = pca_trials_data_df.index


    # print('#############################################')


    # Making a dataframe from the principle components
    if PCA_components == 2:
        pc_df = pd.concat([pd.DataFrame(IDs, columns = ['SubjectID']), pd.DataFrame(principalComponents, columns = ['principal component 1', 'principal component 2'])], axis = 1)
        pc_df.set_index('SubjectID', inplace=True)
        pca_df = pd.concat([pca_trials_data_df, pc_df], axis = 1)

    elif PCA_components == 3:
        pc_df = pd.concat([pd.DataFrame(IDs, columns = ['SubjectID']), pd.DataFrame(principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])], axis = 1)
        pc_df.set_index('SubjectID', inplace=True)
        pca_df = pd.concat([pca_trials_data_df, pc_df], axis = 1)

    return(pca_df, pca)


def pca_Ttest(pca_df, Ttest_var, Ttest_states):
    Ttest_data = [0,0]
    # print(Ttest_states)
    # print(pca_df.columns)
    Ttest_data[0] = pca_df[pca_df[Ttest_var]==Ttest_states[0]]
    Ttest_data[1] = pca_df[pca_df[Ttest_var]==Ttest_states[1]]

#    failure_data = pca_df[pca_df[Ttest_var]==Ttest_states[0]]
#    success_data = pca_df[pca_df[Ttest_var]==Ttest_states[1]]

    ttest_results_PC1 = ttest_ind(Ttest_data[0]['principal component 1'], Ttest_data[1]['principal component 1'])
    ttest_results_PC2 = ttest_ind(Ttest_data[0]['principal component 2'], Ttest_data[1]['principal component 2'])
    # print(ttest_results)
    # print("%.2g" % ttest_results[0])

    print('PC1 t-Test results: (statistic:', "%.2g" % ttest_results_PC1[0], ', p value:', "%.2g" % ttest_results_PC1[1], ')')
    print('PC2 t-Test results: (statistic:', "%.2g" % ttest_results_PC2[0], ', p value:', "%.2g" % ttest_results_PC2[1], ')')
    # import sys
    # sys.exit()
    # print(ttest_ind(Ttest_data[0]['principal component 1'], Ttest_data[1]['principal component 1']))
    # print(ttest_ind(Ttest_data[0]['principal component 2'], Ttest_data[1]['principal component 2']))


def subject_and_PCs(pca_df, file_name, group_category):
    print("OUTPUT FILE")

    # times = pca_df['time'].unique()
    # states = pca_df['state'].unique()
    # # print(pca_df)
    # new_headers = []
    # for unique_time in times:
    #     for unique_state in states:
    #         # print(unique_time ,unique_state)
    #         new_headers.append(unique_time + '_' + unique_state)

    pc_array = np.array(['principal component 1','principal component 2'])

    newer_headers = []
    # for prefix in new_headers:
    for suffix in pc_array:
        # newer_headers.append(prefix + '_' + suffix)
        newer_headers.append(suffix)

    indices = pca_df.index.unique()
    pc_db = pd.DataFrame(index = indices, columns = newer_headers)

    for index, row in pca_df.iterrows():
        # print(row)
        pc1 = row['principal component 1']
        pc2 = row['principal component 2']
        targets = row[group_category]

        # prefix = row['time'] + '_' + row['state']
        pc1_column = 'principal component 1'
        pc2_column = 'principal component 2'
        # target_column = prefix + group_category

        pc_db.loc[index,pc1_column] = pc1
        pc_db.loc[index,pc2_column] = pc2
        pc_db.loc[index,group_category] = targets

    pc_db.to_csv(file_name)


def PCA_shape_data(pca, pca_df, pc_no, file_name):
    headers = list(pca_df.columns)
    non_pca_columns = [x for x in headers if not re.match('[x|y]\\d+', x)]

    only_coord_df = pca_df.drop(columns = non_pca_columns)
    coords_means = only_coord_df.mean(axis=0)

    x_coefficients_PC1 = pca.components_[0][0:int(len(pca.components_[0])/2)]
    y_coefficients_PC1 = pca.components_[0][int(len(pca.components_[0])/2):]

    x_means = coords_means[0:int(len(pca.components_[0])/2)]
    y_means = coords_means[int(len(pca.components_[0])/2):]

    x_coefficients_PC2 = pca.components_[1][0:int(len(pca.components_[0])/2)]
    y_coefficients_PC2 = pca.components_[1][int(len(pca.components_[0])/2):]

    PCA_shape_data = pd.DataFrame(list(zip(x_coefficients_PC1,x_means,y_coefficients_PC1,y_means,x_coefficients_PC2,y_coefficients_PC2)),
                                              columns =['LP_PC1_x_coefficient','LP_x_intercept','LP_PC1_y_coefficient','LP_y_intercept','LP_PC2_x_coefficient','LP_PC2_y_coefficient'])

    PCA_shape_data.index += 1

    PCA_shape_data = PCA_shape_data.rename_axis('point_number')

    PCA_shape_data.to_csv(file_name)


def PCA_and_Score(pca_trials_df, other_data_df):


    ### Notes
    '''
    full_df is all of the data about the subject and run
    pca_trials_df is all of the data about the trials that are used for the pca
    pca_trials_data_df is the data needed for pca_analysis
    pca_df is the pca_trials_df PLUS the results of the pca
    '''

    headers = pca_trials_df.head()
    non_pca_columns = [x for x in headers if not re.match('[x|y]\\d+', x)]

    pca_trials_data_df = pca_trials_df.drop(columns = non_pca_columns)


    x = StandardScaler(with_std = False).fit_transform(pca_trials_data_df)

    # How many components to look at
    PCA_components = 2

    # Genreate the PCA and record the principal components
    pca = PCA(n_components = PCA_components)
    #principalComponents = pca.fit_transform(x)
    principalComponents = pca.fit_transform(x)

    IDs = pca_trials_data_df.index


    # print('#############################################')


    # Making a dataframe from the principle components
    if PCA_components == 2:
        pc_df = pd.concat([pd.DataFrame(IDs, columns = ['SubjectID']), pd.DataFrame(principalComponents, columns = ['principal component 1', 'principal component 2'])], axis = 1)
        pc_df.set_index('SubjectID', inplace=True)
        pca_df = pd.concat([pca_trials_df, pc_df], axis = 1)

    elif PCA_components == 3:
        pc_df = pd.concat([pd.DataFrame(IDs, columns = ['SubjectID']), pd.DataFrame(principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])], axis = 1)
        pc_df.set_index('SubjectID', inplace=True)
        pca_df = pd.concat([pca_trials_df, pc_df], axis = 1)

    return(pca_df, pca)


#TODO: area for testing, so I can just run this file directly
# test_df = pd.read_csv('intermediate_pc_data.csv')
# result_PC, pca= PCA_(test_df)
# result_PC.to_csv('test.csv', columns=['principal component 1', 'principal component 2'], index=False)

#Recommended noise scale value = 0.05,(.01 is too close to original graph),(.1 is very jumpy with the values)
# noise scale is the parameter that goes into the std param in the np.random.normal function used to add the noise

'''This new function adds noise to a spline that has already been made and it reads from a csv'''

def add_noise_to_csv(csv_filename, noise_scale):
    Results_Folder = "C:\\Users\\phine\\OneDrive\\Desktop\\FEBio files\\Pycharm Results"
    current_date = datetime.datetime.now()
    date_prefix = str(current_date.year) + '_' + str(current_date.month) + '_' + str(current_date.day)
    data_df = pd.read_csv(csv_filename)
    first_row = True
    pc_header = []
    pc_header.append('File Name')
    pc_header.append('E1')
    pc_header.append('E2')
    pc_header.append('E3')
    pc_header.append('Apex')
    coord = 'x'
    for i in range(2):
        if i == 1:
            coord = 'y'
        for j in range(15):
            pc_header.append(coord + str(j + 1))
    coord = 'Bx'
    for i in range(2):
        if i == 1:
            coord = 'By'
        for j in range(10):
            pc_header.append(coord + str(j + 1))
    pc_df = pd.DataFrame(columns=pc_header)

    for index, row in data_df.iterrows():
        pc_rows = []
        file_and_e_values = row.iloc[:5].tolist()
        avw_xs= row.iloc[5:20].tolist()
        avw_ys = row.iloc[20:35].tolist()
        bottom_xs = row.iloc[35:45].tolist()
        bottom_ys = row.iloc[45:].tolist()

        noise_avw_xs = avw_xs + np.random.normal(0, noise_scale, len(avw_xs))
        noise_avw_ys = avw_ys + np.random.normal(0, noise_scale, len(avw_ys))
        noise_bottom_xs = bottom_xs + np.random.normal(0, noise_scale, len(bottom_xs))
        noise_bottom_ys = bottom_ys + np.random.normal(0, noise_scale, len(bottom_ys))

        splined_avw_xs, splined_avw_ys = get_noise_spline(noise_avw_xs, noise_avw_ys)
        splined_bottom_xs , splined_bottom_ys = get_noise_spline(noise_bottom_xs, noise_bottom_ys)

        apex = min(splined_avw_ys)

        file_and_e_values[-1] = apex

        pc_rows.extend(file_and_e_values)
        pc_rows.extend(splined_avw_xs)
        pc_rows.extend(splined_avw_ys)
        pc_rows.extend(splined_bottom_xs)
        pc_rows.extend(splined_bottom_ys)
        pc_df.loc[index] = pc_rows

    pc1_df =pc_df.iloc[:, 5:35]
    pcbottom_df = pc_df.iloc[:, 35:len(pc_df.columns)]
    # int_df = pd.read_csv("intermediate_pc_data", header=None)
    total_result_PC1, pca = PCA_(pc1_df)
    total_result_PCB, pca = PCA_([pcbottom_df])

    PC_scores = total_result_PC1[['principal component 1', 'principal component 2']]
    PC_scores_bottom = total_result_PCB[['principal component 1', 'principal component 2']]

    print(PC_scores)
    print(PC_scores_bottom)

    PC_scores = PC_scores.rename(columns={'principal component 1': 'principal component 1 AVW',
                                          'principal component 2': 'principal component 2 AVW'})
    PC_scores_bottom = PC_scores_bottom.rename(columns={'principal component 1': 'principal component 1 Bottom Tissue',
                                                        'principal component 2': 'principal component 2 Bottom Tissue'})

    final_df = pd.concat([pc_df.loc[:, ["File Name", "E1", "E2", "E3", "Apex"]], PC_scores, PC_scores_bottom], axis=1)
    file_path = Results_Folder + '\\' + date_prefix + "_modified_test.csv"
    final_df.to_csv(Results_Folder + '\\' + date_prefix + "_modified_test.csv", index=False)



'''This is the old spline function that is part of a process and adds noise to already splined points, which are also
points that have been analysed already'''
def add_noise_to_spline(spline_ordered_list, dist_array, noise_scale):
    xs = [i[0] for i in spline_ordered_list]
    ys = [i[1] for i in spline_ordered_list]

    curve_x = interpolate.UnivariateSpline(dist_array, xs, k=5)
    curve_y = interpolate.UnivariateSpline(dist_array, ys, k=5)
    spaced_distance_array = np.linspace(0, dist_array[-1], 15)

    #The code above fits a spline of 15 points through the original xs and ys. The end result is a spline curve of
    # 15 points.

    new_distance = 0
    new_distance_array = [0]
    previous_x = curve_x(0)
    previous_y = curve_y(0)
    new_xs = [previous_x]
    new_ys = [previous_y]

    for i in range(1, len(spaced_distance_array)):
        new_xs.append(float(curve_x(spaced_distance_array[i])))
        new_ys.append(float(curve_y(spaced_distance_array[i])))

    #Once we have the 15 points in a spline, the next block of code adds noise to that spline and appends the new points
    #onto two lists, one for x coordinates and one with y coordinates. After that we run through the same process of
    #adding a spline to it and then graphing the noisy coordinates' spline.

    noise_x = new_xs + np.random.normal(0, noise_scale, len(new_xs))
    noise_y = new_ys + np.random.normal(0, noise_scale, len(new_ys))


    return noise_x, noise_y

def get_noise_spline(xs, ys):
    middle_nodes = []
    minimum_x = np.inf
    for index, x in enumerate(xs):
        middle_nodes.append((x, ys[index]))
        if x < minimum_x:
            minimum_x = x
            start = (x, ys[index])

    noise_spline = [start]
    # Iteratively add points to the spline_ordered list based on minimum distance
    while len(middle_nodes) > 0:
        distances = np.array([hypot(noise_spline[-1][0] - point[0], noise_spline[-1][1] - point[1])
                              for point in middle_nodes])
        # Find the index of the point with the minimum distance
        minimum_index = np.argmin(distances)

        # Add the point with the minimum distance to the spline_ordered list
        noise_spline.append(middle_nodes[minimum_index])

        # Remove the selected point from the middle_nodes list
        middle_nodes.pop(minimum_index)

    # Remove the first coordinate pair because it was just the starting one
    noise_spline.pop(0)

    if noise_spline[0][0] < noise_spline[-1][0]:
        noise_spline = list(reversed(noise_spline))

    noise_distance_array = [0]
    for i in range(1, len(noise_spline)):
        noise_distance_array.append(
            hypot(noise_spline[i][0] - noise_spline[i - 1][0],
                  noise_spline[i][1] - noise_spline[i - 1][1]) + noise_distance_array[-1])

    spline_xs = [i[0] for i in noise_spline]
    spline_ys = [i[1] for i in noise_spline]

    curve_x = UnivariateSpline(noise_distance_array, spline_xs, k=5)
    curve_y = UnivariateSpline(noise_distance_array, spline_ys, k=5)



    if len(spline_ys)<11:
        noise_spaced_distance_array = np.linspace(0, noise_distance_array[-1], len(spline_ys))
    else:
        noise_spaced_distance_array = np.linspace(0, noise_distance_array[-1], 15)
    new_distance = 0
    new_distance_array = [0]
    previous_noise_x = curve_x(0)
    previous_noise_y = curve_y(0)
    new_noise_xs = [previous_noise_x]
    new_noise_ys = [previous_noise_y]

    for i in range(1, len(noise_spaced_distance_array)):
        new_noise_xs.append(float(curve_x(noise_spaced_distance_array[i])))
        new_noise_ys.append(float(curve_y(noise_spaced_distance_array[i])))

    return new_noise_xs, new_noise_ys





