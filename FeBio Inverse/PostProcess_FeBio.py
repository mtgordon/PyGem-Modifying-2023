"""
This file is designed to use the IO functions relating to FeBio, which can entail feb files that are XML input files,
and log files that are txt output files. The goal is to replicate the same process as the post-processing currently
implemented for the INP files with Abaqus.
"""

import test_yutian.functions.IOfunctions_Feb as IO
import generate_pca_points_AVW as gic
import re
import Bottom_Tissue_SA_Final as bts
import time
import csv
import pandas as pd
import PCA_data
import os
import predict_funtions as pf
import ShapeAnalysisVerification as sav
import numpy as np
import matplotlib.pyplot as plt



#TODO: Change the following to preffered numbers
window_width = 0.3
num_pts = 9
spline_ordered = 0
startingPointColHeader = 'inner_x'
secondPartStart = 'outer_x'
numCompPCA = 2
stringList = ['inner_x', 'outer_x']

# TEST PROCESS FEATURES
#TODO: finish this later today at the end so that we can use string list to go through and have the least amount of hard
#      coding

# def process_features(csv_file, Results_Folder, date_prefix):
#     int_df = pd.read_csv(csv_file)
#     for i, header in enumerate(stringList):
#         if i < len(stringList) - 1:
#             next_header = stringList[i + 1]
#
#         currentStartIndex = int_df.columns[int_df.columns.str.contains(header)].tolist()
#         currentIndex = int_df.columns.get_loc(currentStartIndex[0])
#         nextStartIndex = int_df.columns[int_df.columns.str.contains(next_header)].tolist()
#         nextIndex = int_df.columns.get_loc(nextStartIndex[0])
#         pc1_df = int_df.iloc[:, currentIndex:nextIndex] #TODO: HARD CODED _ CHANGE LATER
#         pcbottom_df = int_df.iloc[:, nextIndex:len(int_df.columns)]
#         # int_df = pd.read_csv("intermediate_pc_data", header=None)
#         total_result_PC1, pca1 = PCA_data.PCA_(pc1_df, numCompPCA)
#         total_result_PCB, pcaB = PCA_data.PCA_([pcbottom_df], numCompPCA)
#
#     #TODO: MAKE THESE WORK WITH ANY OBJECT NAME OR PC COMPONENT NUMBER
#     PC_scores = total_result_PC1[['principal component 1', 'principal component 2']]
#     PC_scores_bottom = total_result_PCB[['principal component 1', 'principal component 2']]
#
#     print("PC_Scores: ", PC_scores)
#     print("PC_scores_bottom", PC_scores_bottom)
#
#     PC_scores = PC_scores.rename(columns = {'principal component 1': 'principal component 1 AVW','principal component 2':'principal component 2 AVW'})
#     PC_scores_bottom  = PC_scores_bottom.rename(columns={'principal component 1': 'principal component 1 Bottom Tissue',
#                                           'principal component 2': 'principal component 2 Bottom Tissue'})
#
#     final_df = pd.concat([int_df.loc[:, ["File Name", "E5", "Pressure", "Inner_Radius", "Outer_Radius"]], PC_scores, PC_scores_bottom], axis=1)
#     if not os.path.exists(Results_Folder):
#         os.makedirs(Results_Folder)
#     file_name = pf.get_file_name(csv_file)
#
#     file_path = Results_Folder + '\\' + file_name + '_' + date_prefix + "_modified_train.csv"
#
#     final_df.to_csv(file_path, index=False)
#     return file_path, pca1, pcaB

def process_features(csv_file, Results_Folder, date_prefix):
    int_df = pd.read_csv(csv_file)
    pointStartIdx = int_df.columns[int_df.columns.str.contains(startingPointColHeader)].tolist()
    startIdx = int_df.columns.get_loc(pointStartIdx[0])
    secondStartIdx = int_df.columns[int_df.columns.str.contains(secondPartStart)].tolist()
    secondIdx = int_df.columns.get_loc(secondStartIdx[0])
    pc1_df = int_df.iloc[:, startIdx:secondIdx] #TODO: HARD CODED _ CHANGE LATER
    pcbottom_df = int_df.iloc[:, secondIdx:len(int_df.columns)]
    # int_df = pd.read_csv("intermediate_pc_data", header=None)
    total_result_PC1, pca1 = PCA_data.PCA_(pc1_df, numCompPCA)
    total_result_PCB, pcaB = PCA_data.PCA_([pcbottom_df], numCompPCA)

    #TODO: MAKE THESE WORK WITH ANY OBJECT NAME OR PC COMPONENT NUMBER
    PC_scores = total_result_PC1[['principal component 1', 'principal component 2']]
    PC_scores_bottom = total_result_PCB[['principal component 1', 'principal component 2']]

    print("PC_Scores: ", PC_scores)
    print("PC_scores_bottom", PC_scores_bottom)

    PC_scores = PC_scores.rename(columns = {'principal component 1': 'principal component 1 AVW','principal component 2':'principal component 2 AVW'})
    PC_scores_bottom  = PC_scores_bottom.rename(columns={'principal component 1': 'principal component 1 Bottom Tissue',
                                          'principal component 2': 'principal component 2 Bottom Tissue'})

    final_df = pd.concat([int_df.loc[:, ["File Name", "E5", "Pressure", "Inner_Radius", "Outer_Radius"]], PC_scores, PC_scores_bottom], axis=1)
    if not os.path.exists(Results_Folder):
        os.makedirs(Results_Folder)
    file_name = pf.get_file_name(csv_file)

    file_path = Results_Folder + '\\' + file_name + '_' + date_prefix + "_modified_train.csv"

    final_df.to_csv(file_path, index=False)
    return file_path, pca1, pcaB

def find_apex(coordList):
    min_y = coordList[0][1][1]
    for coord in coordList:
        if coord[1][1] < min_y:
            min_y = coord[1][1]

    return min_y




"""
This function was originally made to generate a csv intermediate file for FeBio post-processing. Its original purpose
was to work with the model from summer of 2023 which had an APEX and pc_points_bottom.

For the summer of 2024 we changed it to work with a cylinder that has Pressure, Inner Radius, Outer Radius. 
this was done by commenting out the line "pc_points_bottom = bts.generate_2d_bottom_tissue(bts.extract_ten_coordinates_block(obj_coords_list[2]))"
and line "apex = find_apex(obj_coords_list[1])" To revert back to old model simply change headers and then uncomment the lines metioned 
above. Also uncomment the second for loop starting at "coord = 'Bx'"
"""
def generate_int_csvs(file_params,object_list,log_name,feb_name,first_int_file_flag,csv_filename):
    obj_coords_list = []

    csv_row = []
    csv_header = []

    # Get the pure file name that just has the material parameters


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
        print('Extracting... ' + obj + ' for ' + file_params)
        obj_coords_list.append(gic.extract_coordinates_from_final_step(log_name, feb_name, obj))

    #TODO: THIS IS WHERE THE INT CSV DATA IS OBTAINED


    # gets obj_coords_list to [[x, y, z]]
    object_coords_list = []
    for ele in obj_coords_list[0]:
        temparray = []
        temparray.append(ele[1][0])
        temparray.append(ele[1][1])
        temparray.append(ele[1][2])
        object_coords_list.append(temparray)
    obj_coords_list[0] = object_coords_list

    #THIS USES THE FUNCTION THAT WE CREATED TO GENERATE THE OUTER AND INNER POINTS THEN PASSING THEM
    #IN TO FIND THE 2D_COORDS FOR THE PCA POINTS
    outer_points = sav.generate_outer_cylinder_bottom(num_pts, obj_coords_list[0], window_width)
    inner_points = sav.generate_inner_cylinder_bottom(num_pts, obj_coords_list[0], window_width)
    outer_pc_points = sav.generate_2d_coords_for_cylinder_pca(outer_points)
    inner_pc_points = sav.generate_2d_coords_for_cylinder_pca(inner_points) #REPLACE WITH CYLINDER POINTS



    #pc_points_bottom = bts.generate_2d_bottom_tissue(bts.extract_ten_coordinates_block(obj_coords_list[2])) #TODO: Errors due to not enough objects (0, 2, 1 idx should be looked at)


    # Get the PC points for Object2
    # Begin building the row to be put into the intermediate csv
    csv_row.append(file_params)  # file params

    #apex = find_apex(obj_coords_list[1])
    # apex FIX
    csv_row.extend(prop_final)
    #csv_row.append(apex)
    csv_row.extend(inner_pc_points)
    csv_row.extend(outer_pc_points)
    #csv_row.extend(pc_points_bottom)  # the 30 pc coordinates

    if first_int_file_flag:
        #TODO: Have the headers loop through the file params... Done?
        csv_header.append('File Name')
        csv_header.append('E5')
        csv_header.append('Pressure')
        csv_header.append('Inner_Radius')
        csv_header.append('Outer_Radius')



        #coord = 'inner_x'
        coord = startingPointColHeader
        # TODO: This is purely for the coordinate headers (ADJUST 15 FOR THE MAX NUMBER OF COORDINATE HEADERS)
        for i in range(4):
            if i == 1:
                coord = 'inner_y'
            elif i == 2:
                coord = 'outer_x'
            elif i == 3:
                coord = 'outer_y'
            for j in range(num_pts):
                csv_header.append(coord + str(j + 1))
        #TODO: commented this out because we do not have a points for 'bottom'
        """
        
        coord = 'Bx'
        for i in range(2):
            if i == 1:
                coord = 'By'
            for j in range(10):
                csv_header.append(coord + str(j + 1))
        """

        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)
            writer.writerow(csv_row)

    else:
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_row)

    # sleep to give the file time to reach directory
    time.sleep(1)

    return csv_filename
