# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:32:40 2020

@author: mgordon
"""
import os
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn import joblib
import pandas as pd
from scipy.stats import ttest_ind
import numpy as np
import re


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

    # headers = pca_trials_df.head()
    # non_pca_columns = [x for x in headers if not re.match('[x|y]\\d+', x)]

    # print('first:', pca_trials_df)
    # pca_trials_data_df = pca_trials_df.drop(columns = non_pca_columns)
    # TODO: use the raw csv for right now as it is only the coordinate data
    pca_trials_data_df = pca_trials_df

    # print('second:', pca_trials_data_df)
    x = StandardScaler(with_std=False).fit_transform(pca_trials_data_df)
    #    x = StandardScaler().fit_transform(pca_trials_data_df)
    #    print(x)
    # How many components to look at
    PCA_components = 2

    # Genreate the PCA and record the principal components
    pca = PCA(n_components=PCA_components)
    # principalComponents = pca.fit_transform(x)
    principalComponents = pca.fit_transform(x)

    IDs = pca_trials_data_df.index

    # print('#############################################')

    # Making a dataframe from the principle components
    if PCA_components == 2:
        pc_df = pd.concat([pd.DataFrame(IDs, columns=['SubjectID']), pd.DataFrame(principalComponents,
                                                                                  columns=['principal component 1',
                                                                                           'principal component 2'])],
                          axis=1)
        pc_df.set_index('SubjectID', inplace=True)
        pca_df = pd.concat([pca_trials_df, pc_df], axis=1)

    elif PCA_components == 3:
        pc_df = pd.concat([pd.DataFrame(IDs, columns=['SubjectID']), pd.DataFrame(principalComponents,
                                                                                  columns=['principal component 1',
                                                                                           'principal component 2',
                                                                                           'principal component 3'])],
                          axis=1)
        pc_df.set_index('SubjectID', inplace=True)
        pca_df = pd.concat([pca_trials_df, pc_df], axis=1)

    pca_path = save_model_path()
    print(pca_path)
    save_PCA_pickle(pca, pca_path)
    pca = load_PCA_pickle(pca_path)

    return (pca_df, pca)


"""
TEST SAVING FUNCTION
"""


def save_model_path():
    if not os.path.exists('PCAModel'):
        os.makedirs('PCAModel')

    # Determine the model name
    suffix = 1
    model_name = os.path.join('PCAModel', f"pca{suffix}")

    while os.path.exists(model_name):
        suffix += 1
        model_name = os.path.join('PCAModel', f"pca{suffix}")

    return model_name

#
# def save_PCA_joblib(pca):
#     pca_model = save_model_path(pca)
#     joblib.dump(pca, pca_model)
#     print(pca_model)
#     return pca_model
#
#
# pca = joblib.load('pca_model.pkl')


def save_PCA_pickle(pca, pca_path):
    with open(pca_path, 'wb') as f:
        pickle.dump(pca, f)
    return pca_path


def load_PCA_pickle(pca_model):
    with open(pca_model, 'rb') as f:
        pca = pickle.load(f)
    return pca


def pca_Ttest(pca_df, Ttest_var, Ttest_states):
    Ttest_data = [0, 0]
    # print(Ttest_states)
    # print(pca_df.columns)
    Ttest_data[0] = pca_df[pca_df[Ttest_var] == Ttest_states[0]]
    Ttest_data[1] = pca_df[pca_df[Ttest_var] == Ttest_states[1]]

    #    failure_data = pca_df[pca_df[Ttest_var]==Ttest_states[0]]
    #    success_data = pca_df[pca_df[Ttest_var]==Ttest_states[1]]

    ttest_results_PC1 = ttest_ind(Ttest_data[0]['principal component 1'], Ttest_data[1]['principal component 1'])
    ttest_results_PC2 = ttest_ind(Ttest_data[0]['principal component 2'], Ttest_data[1]['principal component 2'])
    # print(ttest_results)
    # print("%.2g" % ttest_results[0])

    print('PC1 t-Test results: (statistic:', "%.2g" % ttest_results_PC1[0], ', p value:', "%.2g" % ttest_results_PC1[1],
          ')')
    print('PC2 t-Test results: (statistic:', "%.2g" % ttest_results_PC2[0], ', p value:', "%.2g" % ttest_results_PC2[1],
          ')')
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

    pc_array = np.array(['principal component 1', 'principal component 2'])

    newer_headers = []
    # for prefix in new_headers:
    for suffix in pc_array:
        # newer_headers.append(prefix + '_' + suffix)
        newer_headers.append(suffix)

    indices = pca_df.index.unique()
    pc_db = pd.DataFrame(index=indices, columns=newer_headers)

    for index, row in pca_df.iterrows():
        # print(row)
        pc1 = row['principal component 1']
        pc2 = row['principal component 2']
        targets = row[group_category]

        # prefix = row['time'] + '_' + row['state']
        pc1_column = 'principal component 1'
        pc2_column = 'principal component 2'
        # target_column = prefix + group_category

        pc_db.loc[index, pc1_column] = pc1
        pc_db.loc[index, pc2_column] = pc2
        pc_db.loc[index, group_category] = targets

    pc_db.to_csv(file_name)


def PCA_shape_data(pca, pca_df, pc_no, file_name):
    headers = list(pca_df.columns)
    non_pca_columns = [x for x in headers if not re.match('[x|y]\\d+', x)]

    only_coord_df = pca_df.drop(columns=non_pca_columns)
    coords_means = only_coord_df.mean(axis=0)

    x_coefficients_PC1 = pca.components_[0][0:int(len(pca.components_[0]) / 2)]
    y_coefficients_PC1 = pca.components_[0][int(len(pca.components_[0]) / 2):]

    x_means = coords_means[0:int(len(pca.components_[0]) / 2)]
    y_means = coords_means[int(len(pca.components_[0]) / 2):]

    x_coefficients_PC2 = pca.components_[1][0:int(len(pca.components_[0]) / 2)]
    y_coefficients_PC2 = pca.components_[1][int(len(pca.components_[0]) / 2):]

    PCA_shape_data = pd.DataFrame(
        list(zip(x_coefficients_PC1, x_means, y_coefficients_PC1, y_means, x_coefficients_PC2, y_coefficients_PC2)),
        columns=['LP_PC1_x_coefficient', 'LP_x_intercept', 'LP_PC1_y_coefficient', 'LP_y_intercept',
                 'LP_PC2_x_coefficient', 'LP_PC2_y_coefficient'])

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

    pca_trials_data_df = pca_trials_df.drop(columns=non_pca_columns)

    x = StandardScaler(with_std=False).fit_transform(pca_trials_data_df)

    # How many components to look at
    PCA_components = 2

    # Genreate the PCA and record the principal components
    pca = PCA(n_components=PCA_components)
    # principalComponents = pca.fit_transform(x)
    principalComponents = pca.fit_transform(x)

    IDs = pca_trials_data_df.index

    # print('#############################################')

    # Making a dataframe from the principle components
    if PCA_components == 2:
        pc_df = pd.concat([pd.DataFrame(IDs, columns=['SubjectID']), pd.DataFrame(principalComponents,
                                                                                  columns=['principal component 1',
                                                                                           'principal component 2'])],
                          axis=1)
        pc_df.set_index('SubjectID', inplace=True)
        pca_df = pd.concat([pca_trials_df, pc_df], axis=1)

    elif PCA_components == 3:
        pc_df = pd.concat([pd.DataFrame(IDs, columns=['SubjectID']), pd.DataFrame(principalComponents,
                                                                                  columns=['principal component 1',
                                                                                           'principal component 2',
                                                                                           'principal component 3'])],
                          axis=1)
        pc_df.set_index('SubjectID', inplace=True)
        pca_df = pd.concat([pca_trials_df, pc_df], axis=1)

    return (pca_df, pca)

# TODO: area for testing, so I can just run this file directly
# test_df = pd.read_csv('intermediate_pc_data.csv')
# result_PC, pca= PCA_(test_df)
# result_PC.to_csv('test.csv', columns=['principal component 1', 'principal component 2'], index=False)
