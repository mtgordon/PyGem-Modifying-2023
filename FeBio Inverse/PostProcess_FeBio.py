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


def process_features(csv_file, Results_Folder, date_prefix):
    int_df = pd.read_csv(csv_file)
    pc1_df = int_df.iloc[:, 5:35]
    pcbottom_df = int_df.iloc[:, 35:len(int_df.columns)]
    # int_df = pd.read_csv("intermediate_pc_data", header=None)
    total_result_PC1, pca1 = PCA_data.PCA_(pc1_df)
    total_result_PCB, pcaB = PCA_data.PCA_([pcbottom_df])

    PC_scores = total_result_PC1[['principal component 1', 'principal component 2']]
    PC_scores_bottom = total_result_PCB[['principal component 1', 'principal component 2']]

    print(PC_scores)
    print(PC_scores_bottom)

    PC_scores = PC_scores.rename(columns = {'principal component 1': 'principal component 1 AVW','principal component 2':'principal component 2 AVW'})
    PC_scores_bottom  = PC_scores_bottom.rename(columns={'principal component 1': 'principal component 1 Bottom Tissue',
                                          'principal component 2': 'principal component 2 Bottom Tissue'})

    final_df = pd.concat([int_df.loc[:, ["File Name", "E1", "E2","E3","Apex"]], PC_scores, PC_scores_bottom], axis=1)
    if not os.path.exists(Results_Folder):
        os.makedirs(Results_Folder)
    file_name = pf.get_file_name(csv_file)

    file_path = Results_Folder + '\\' + file_name + date_prefix + "_modified_train.csv"

    final_df.to_csv(file_path, index=False)
    return file_path, pca1, pcaB

def find_apex(coordList):
    min_y = coordList[0][1][1]
    for coord in coordList:
        if coord[1][1] < min_y:
            min_y = coord[1][1]

    return min_y
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
        obj_coords_list.append(gic.extract_coordinates_from_final_step(log_name, feb_name, obj))
        print('Extracting... ' + obj + ' for ' + file_params)
        if obj == 'Object16':
            last_object = True
    pc_points = gic.generate_2d_coords_for_pca(obj_coords_list[0])
    pc_points_bottom = bts.generate_2d_bottom_tissue(bts.extract_ten_coordinates_block(obj_coords_list[2]))
    # Get the PC points for Object2
    # Begin building the row to be put into the intermediate csv
    csv_row.append(file_params)  # file params
    apex = find_apex(obj_coords_list[1])
    # apex FIX
    csv_row.extend(prop_final)
    csv_row.append(apex)
    csv_row.extend(pc_points)
    csv_row.extend(pc_points_bottom)  # the 30 pc coordinates

    if first_int_file_flag:
        csv_header.append('File Name')
        csv_header.append('E1')
        csv_header.append('E2')
        csv_header.append('E3')
        csv_header.append('Apex')
        coord = 'x'
        for i in range(2):
            if i == 1:
                coord = 'y'
            for j in range(15):
                csv_header.append(coord + str(j + 1))
        coord = 'Bx'
        for i in range(2):
            if i == 1:
                coord = 'By'
            for j in range(10):
                csv_header.append(coord + str(j + 1))

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

