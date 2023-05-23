"""
This file is responsible for running FeBio in conjunction with the feb_variables.csv file so that
a run is completed for each row, then is post processed
"""
import csv
import datetime
import os.path
import sys
import subprocess
import xml.etree.ElementTree as ET
import generate_int_csvs as gic
import PostProcess_FeBio as proc
import pandas as pd
import re
import time
import PCA_data

'''
Function: RunFEBinFeBio
Takes in the input feb file along with the location of FeBio on the system and runs it via 
the command line. The output log file name can also be given.
'''
def RunFEBinFeBio(inputFileName, FeBioLocation, outputFileName=None):
    CallString = '\"' + FeBioLocation + '\" -i \"' + inputFileName + '\"'
    if outputFileName is not None:
        CallString += ' -o \"' + outputFileName + '\"'
    print(CallString)
    subprocess.call(CallString)

'''
Function: updateProperties
Takes in a specified part name, finds the corresponding material, changes the modulus of the material, 
then saves a new input file with the name relating to the changed part (part, property, new value)
'''
def updateProperties(origFile, fileTemp):
    # Parse original FEB file
    tree = ET.parse(origFile)
    root = tree.getroot()

    # Locate the Mesh Domains section to find which parts have which materials
    meshDomains = root.find('MeshDomains')
    for domain in meshDomains:
        for partProp in current_run_dict.keys():
            partName = partProp.split('_')[0]
            propName = partProp.split('_')[1]
            if domain.attrib['name'] == partName:
                for mat in tree.find('Material'):
                    if mat.attrib['name'] == domain.attrib['mat']:
                        newValue = float(mat.find(propName).text) * float(current_run_dict[partProp])
                        mat.find(propName).text = str(newValue)

    # using UTF-8 encoding does not bring up any issues (can be changed if needed)
    newInputFile = Results_Folder + '\\' + fileTemp + '.feb'
    tree.write(newInputFile, xml_declaration=True, encoding='ISO-8859-1')

    return newInputFile


'''
Function: new_check_normal_run
Takes in a log file and checks for the normal termination indicator, notifying that post processing
can be done on the file
'''
def new_check_normal_run(log_file_path):
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
        last_lines = [line.strip() for line in lines[-10:]]
        for line in last_lines:
            if "N O R M A L   T E R M I N A T I O N" in line:
                return True
        return False


# FeBio Variables
dictionary_file = 'feb_variables.csv'
FeBioLocation = 'C:\\Program Files\\FEBioStudio2\\bin\\febio4.exe'
originalFebFilePath = 'C:\\Users\\Elijah Brown\\Desktop\\Bio Research\\Results\\Curve_and_Flat_and_CL_and_Filler_meshed_v4_v2_log_included.feb'

#Get data from the Run_Variables file
# Newer code (2/14)
run_file = open(dictionary_file)
DOE_dict = csv.DictReader(run_file)

# Flag to indicate it the first time through the loop so that it will create a
# new file and add headers (MAY REMOVE)
first_file = 1

# Flag to indicate the first time through the while loop for the load search
first_iter = True

#Have the default material variables be 1 (100%) so they do not change if no variable is given
default_dict = {
    'Part2_E': 1,
    'Part8_E': 1,
    'Part9_E': 1,
    'Part12_E': 1,
    'Part27_E': 1
}

current_run_dict = default_dict.copy()

# Post Processing Variables
first_file_flag = True
current_date = datetime.datetime.now()
date_prefix = str(current_date.year) + '_' + str(current_date.month)  + '_' + str(current_date.day)
object_list = ['Object2', 'Object8']
obj_coords_list = []
file_num = 0
GENERATE_INTERMEDIATE_FLAG = True
Results_Folder = 'C:\\Users\\Elijah Brown\\Desktop\\Bio Research\\Results'

# Run the FEB files for each sat of variables in the run_variables csv
for row in DOE_dict:
    print('Row:', row)

    # Initialize current run dictionary
    for key in DOE_dict.fieldnames:
        if key in default_dict.keys():
            current_run_dict[key] = row[key]
        else:
            if key != 'Run Number':
                print(key, 'is an invalid entry')
                sys.exit()

    #generation of current run file template based on attributes
    fileTemplate = ''
    for key in current_run_dict:
        param = '_' + str(key) + '(' + str(current_run_dict[key]) + ')'
        fileTemplate += param

    #Update properties, create new input file
    workingInputFileName = updateProperties(originalFebFilePath, fileTemplate)

    # TODO: to easily get input files from anywhere, do the shutil move to working directory
    logFile = Results_Folder + '\\' + fileTemplate + '.log'
    RunFEBinFeBio(workingInputFileName, FeBioLocation, logFile)

    # Check for success of the feb run
    if new_check_normal_run(logFile):
        # Post process
        csv_row = []

        # Get the pure file name that just has the material parameters
        csv_filename = date_prefix + '_intermediate.csv'

        # Get the changed material properties
        paren_pattern = re.compile(r'(?<=\().*?(?=\))')  # find digits in parentheses
        prop_result = paren_pattern.findall(fileTemplate)
        prop_final = []
        for prop in prop_result:
            prop = float(prop)
            if prop != 1.0:
                prop_final.append(prop)

        # Get the coordinates for each object in list
        for obj in object_list:
            obj_coords_list.append(gic.extract_coordinates_from_final_step(logFile, workingInputFileName, obj))
            print('Extracting... ' + obj + ' for ' + fileTemplate)

        # Get the PC points for Object2
        pc_points = gic.generate_2d_coords_for_pca(obj_coords_list[0])

        # Begin building the row to be put into the intermediate csv
        csv_row.append(fileTemplate)  # file params
        apex = proc.find_apex(obj_coords_list[1])
        csv_row.append(apex)  # apex FIX
        csv_row.extend(prop_final)
        csv_row.extend(pc_points)  # the 30 pc coordinates

        if first_file_flag:
            with open(Results_Folder + '\\' + csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(csv_row)
        else:
            with open(Results_Folder + '\\' + csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(csv_row)

        # sleep to give the file time to reach directory
        time.sleep(1)
        file_num += 1
        print('Completed Iteration ' + str(file_num) + ": " + fileTemplate)
        obj_coords_list = []

        # Generate the final csv with the 2 pc scores
        pc_df = pd.DataFrame(pc_points)
        # int_df = pd.read_csv("intermediate_pc_data", header=None)
        total_result_PC, pca = PCA_data.PCA_(pc_df)

        PC_scores = total_result_PC[['principal component 1', 'principal component 2']]
        print(PC_scores)
        pc_scores_lst = PC_scores.values.tolist()

        final_row = []
        final_row.append(fileTemplate)
        final_row.append(apex)
        final_row.extend(prop_final)
        final_row.extend(pc_scores_lst)

        if first_file_flag:
            with open(Results_Folder + '\\' + date_prefix + "_features.csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(final_row)
            first_file_flag = False
        else:
            with open(Results_Folder + '\\' + date_prefix + "_features.csv", 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(final_row)

    else:
        os.rename(workingInputFileName, workingInputFileName + '_error')
