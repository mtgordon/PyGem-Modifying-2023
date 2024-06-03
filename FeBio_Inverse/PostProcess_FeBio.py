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
import predict_functions as pf
import ShapeAnalysisVerification as sav
import numpy as np
import matplotlib.pyplot as plt
import CylinderFunctions



#TODO: Change the following to preferred numbers
window_width = 0.3
num_pts = 9
spline_ordered = 0
startingPointColHeader = 'inner_y'
secondPartStart = 'outer_x'
numCompPCA = 2
#stringList = ['inner_x', 'outer_x']
stringList = ['inner_y', 'outer_y', 'inner_radius_x', 'outer_radius_x']

"""
    Generate modified training CSV files with principal component scores from the original file.

    Parameters:
        csv_file (str): The path to the original CSV file.
        Results_Folder (str): The folder path where the modified CSV file will be saved.
        date_prefix (str): The prefix to be added to the modified CSV file name.
        numCompPCA (int): The number of principal components to retain.

    Returns:
        file_path (str): The path of the generated modified training CSV file.
        pca1 (object): The PCA model object for the top tissue.
        pcaB (object): The PCA model object for the bottom tissue.

    Example:
        >>> csv_file = "data.csv"
        >>> Results_Folder = "results"
        >>> date_prefix = "2024_05_08"
        >>> numCompPCA = 3
        >>> file_path, pca1, pcaB = process_features(csv_file, Results_Folder, date_prefix, numCompPCA)
    """
def process_features(csv_file, Results_Folder, date_prefix, numCompPCA):
    # Read the input CSV file into a pandas DataFrame
    int_df = pd.read_csv(csv_file)

    # Iterate through the headers
    for i, header in enumerate(stringList):
        # If not the last header, get the next header
        if i < len(stringList) - 1:
            next_header = stringList[i + 1]

        # Get the start and end indices of the current header's columns
        currentStartIndex = int_df.columns[int_df.columns.str.contains(header)].tolist()
        currentIndex = int_df.columns.get_loc(currentStartIndex[0])

        nextStartIndex = int_df.columns[int_df.columns.str.contains(next_header)].tolist()
        nextIndex = int_df.columns.get_loc(nextStartIndex[0])

        # Slice the DataFrame to get the columns for the current header
        pc1_df = int_df.iloc[:, currentIndex:nextIndex]  # TODO: HARD CODED _ CHANGE LATER
        pcbottom_df = int_df.iloc[:, nextIndex:len(int_df.columns)]

        # Perform PCA on the sliced DataFrames
        total_result_PC1, pca1 = PCA_data.PCA_(pc1_df, numCompPCA)
        total_result_PCB, pcaB = PCA_data.PCA_([pcbottom_df], numCompPCA)

        # Get the principal component scores
        PC_scores = total_result_PC1.iloc[:, :numCompPCA]
        PC_scores_bottom = total_result_PCB.iloc[:, :numCompPCA]

        # Rename the column headers to "Principal Component i Inner/Outer Radius"
        PC_scores = PC_scores.rename(
            columns={f'inner_x{i + 1}': f'Principal Component {i + 1} Inner Radius' for i in range(numCompPCA)})
        PC_scores_bottom = PC_scores_bottom.rename(
            columns={f'outer_x{i + 1}': f'Principal Component {i + 1} Outer Radius' for i in range(numCompPCA)})

        # Concatenate the DataFrames to create the final DataFrame
        final_df = pd.concat([int_df.loc[:, ["File Name", "Part3_E", "Pressure", "Inner_Radius", "Outer_Radius"]],
                              PC_scores, PC_scores_bottom], axis=1)

        # Create the directory if it doesn't exist
        if not os.path.exists(Results_Folder):
            os.makedirs(Results_Folder)

        # Get the base file name
        file_name = pf.get_file_name(csv_file)

        # Construct the file path for the modified CSV
        file_path = os.path.join(Results_Folder, f'{file_name}_{date_prefix}_modified_train.csv')

        # Save the final DataFrame to a CSV file
        final_df.to_csv(file_path, index=False)

        # Return the file path and PCA models
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
def generate_int_csvs(file_params, object_list, log_name, feb_name, first_int_file_flag, csv_filename, inner_radius, outer_radius, current_run_dict, plot_points_on_spline):
    obj_coords_list = []

    csv_row = []
    csv_header = []

    # Get the pure file name that just has the material parameters

    prop_final = []
    # Get the changed material properties
    for key, value in current_run_dict.items():
        prop = float(value)
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

    # THIS USES THE FUNCTION THAT WE CREATED TO GENERATE THE OUTER AND INNER POINTS THEN PASSING THEM
    # IN TO FIND THE 2D_COORDS FOR THE PCA POINTS
    outer_points = sav.generate_outer_cylinder_bottom(num_pts, object_coords_list, window_width)

    inner_points = sav.generate_inner_cylinder_bottom(num_pts, object_coords_list, window_width)

    inner_radius = sav.get_2d_coords_from_dictionary(inner_radius)
    outer_radius = sav.get_2d_coords_from_dictionary(outer_radius)

    inner_radius_pc_points = sav.generate_2d_coords_for_cylinder_pca(inner_radius, num_pts)
    outer_radius_pc_points = sav.generate_2d_coords_for_cylinder_pca(outer_radius, num_pts)

    if plot_points_on_spline:
        inner_radius_pc_points_plot = CylinderFunctions.pair_points(inner_radius_pc_points)
        outer_radius_pc_points_plot = CylinderFunctions.pair_points(outer_radius_pc_points)
        CylinderFunctions.plot_pc_points(inner_radius_pc_points_plot)
        CylinderFunctions.plot_pc_points(outer_radius_pc_points_plot)

    outer_pc_points = sav.generate_2d_coords_for_cylinder_pca(outer_points, num_pts)
    inner_pc_points = sav.generate_2d_coords_for_cylinder_pca(inner_points, num_pts) #REPLACE WITH CYLINDER POINTS



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
    csv_row.extend(inner_radius_pc_points)
    csv_row.extend(outer_radius_pc_points)
    #csv_row.extend(pc_points_bottom)  # the 30 pc coordinates

    if first_int_file_flag:
        #TODO: Have the headers loop through the file params... Done?
        csv_header.append('File Name')
        for key, value in current_run_dict.items():
            csv_header.append(key)

        #coord = 'inner_x'
        coord = startingPointColHeader
        # TODO: This is purely for the coordinate headers (ADJUST 15 FOR THE MAX NUMBER OF COORDINATE HEADERS)
        for i in range(8):
            if i == 1:
                coord = 'inner_z'
            elif i == 2:
                coord = 'outer_y'
            elif i == 3:
                coord = 'outer_z'
            elif i == 4:
                coord = 'inner_radius_x'
            elif i == 5:
                coord = 'inner_radius_y'
            elif i == 6:
                coord = 'outer_radius_x'
            elif i == 7:
                coord = 'outer_radius_y'
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
