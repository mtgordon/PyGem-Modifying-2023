"""
This file is responsible for running FeBio in conjunction with the feb_variables.csv file so that
a run is completed for each row, then is post processed
"""
import csv
import datetime
import math
import os.path
import sys
import subprocess
import xml.etree.ElementTree as ET
import PostProcess_FeBio as proc
import Bottom_Tissue_SA_Final as bts
import pandas as pd
import re
import time
import PCA_data
import ShapeAnalysisVerification as sav
from lib import IOfunctions
import CylinderFunctions
from pygem import RBF
import numpy as np
import CylinderFunctions

# FeBio Variables
#TODO: ENTER IN VAR FILE --> SET PART NAMES FOR ALL MATERIALS --> DONE
dictionary_file = 'feb_variables.csv' #DONE
FeBioLocation = 'C:\\Program Files\\FEBioStudio2\\bin\\febio4.exe'
originalFebFilePath = 'D:\\Gordon\\Automate FEB Runs\\2024_5_9_NewModel\\Base_File\\3 Tissue Model v2 w Muscle.feb'
Results_Folder = 'D:\\Gordon\\Automate FEB Runs\\2024_5_9_NewModel\\MuscleModel_Runs'
# This is for output
object_list = ['Levator Ani Side 2'] #TODO: Get new names for flat, curve, GI Filler --> DONE
# Currently being used to access base object, may need to be changed when looking to generate multiple objects at once
part_list = ['Part1', 'Part3', 'Part7', 'Part10', 'Part11']
cylinder_parts = ['Part3']
ZeroDisplacement = "ZeroDisplacement1"
numCompPCA = 3

# FLAGS
Create_New_Feb_Flag = True
Run_FeBio_File_Flag = False
first_int_file_flag = False
GENERATE_INTERMEDIATE_FLAG = False
Post_Processing_Flag = False

# PLOTTING
plot_points_on_spline = False

#TODO: Input Parameters for Cylinder Creation
num_cylinder_points = 200

#Have the default material variables be 1 (100%) so they do not change if no variable is given
#TODO: Update Everytime you want to change your base file
default_dict = {
    'Part1_E': 1,
    'Part3_E': 1,
    'Pressure': 0,
    'Inner_Radius': 1.25,
    'Outer_Radius': 1.75,
    'Part7_g1': 0,
    'Part7_g2': 0,
    'Part7_g3': 0,
    'Part7_p1': 0,
    'Part7_p2': 0,
    'Part7_Lofl': 0,
    'Part7_smax': 0,
    'Part7_activation': 0
}
default_code_dict = {
    'Part1_E': 'P1_E',
    'Part3_E': 'P3_E',
    'Pressure': 'Pre',
    'Inner_Radius': 'IR',
    'Outer_Radius': 'OR',
    'Part7_g1': 'P7_g1',
    'Part7_g2': 'P7_g2',
    'Part7_g3': 'P7_g3',
    'Part7_p1': 'P7_p1',
    'Part7_p2': 'P7_p2',
    'Part7_Lofl': 'P7_Lofl',
    'Part7_smax': 'P7_smax',
    'Part7_activation': 'P7_activation'
}

'''
Function: RunFEBinFeBio
Takes in the input feb file along with the location of FeBio on the system and runs it via 
the command line. The output log file name can also be given.
'''
def RunFEBinFeBio(inputFileName, FeBioLocation, outputFileName=None):
    CallString = '\"' + FeBioLocation + '\" -i \"' + inputFileName + '\"'
    if outputFileName is not None:
        CallString += ' -o \"' + outputFileName + '\"'
    print("CallString: ", CallString)
    subprocess.call(CallString)

'''
Function: updateProperties
Takes in a specified part name, finds the corresponding material, changes the modulus of the material, 
then saves a new input file with the name relating to the changed part (part, property, new value).

Update 4/29: Can now take in Pressure, Inner & Outer Radius for generating 3D Cylinders 
'''
def updateProperties(origFile, fileTemp):
    new_input_file = Results_Folder + '\\' + fileTemp + '.feb'
    # Parse original FEB file
    tree = ET.parse(origFile)
    root = tree.getroot()

    # Verify log file exists, if not add log file to be 'x;y;z'
    IOfunctions.checkForLogFile(root)

    # Update material property values
    for part_prop in current_run_dict.keys():
        # if it is not above names then it is a part
        if "Part" in part_prop:
            part_name = part_prop.split('_')[0]
            prop_name = part_prop.split('_')[1]

            # Locate the Mesh Domains section to find which part have which materials
            mesh_domains = root.find('MeshDomains')
            for domain in mesh_domains:
                if domain.attrib['name'] == part_name:
                    for mat in tree.find('Material'):
                        if mat.attrib['name'] == domain.attrib['mat']:
                            new_value = current_run_dict[part_prop]
                            mat.find(prop_name).text = str(new_value)

    # Update Pressure Value
    loads = root.find('Loads')
    for surface_load in loads:
        pressure = surface_load.find('pressure')
        pressure.text = str(current_run_dict["Pressure"])

    # Assign inner_radius value from "feb_variables.csv"
    final_inner_radius = float(current_run_dict["Inner_Radius"])

    # Assign outer_radius value from "feb_variables.csv"
    final_outer_radius = float(current_run_dict["Outer_Radius"])

    # Extract points from .feb file and return in array of tuples
    extract_points = CylinderFunctions.get_initial_points_from_parts(root, part_list)

    # Find cylinder height
    global cylinder_height
    cylinder_height = CylinderFunctions.findLargestZ(extract_points)

    # Determine if cylinder radius values are "close" (within 0.001) if so do not morph
    if not math.isclose(float(current_run_dict["Inner_Radius"]), float(default_dict["Inner_Radius"]),
                        abs_tol=0.001) and not math.isclose(float(current_run_dict["Outer_Radius"]),
                                                            float(default_dict["Outer_Radius"]), abs_tol=0.001):

        # Extract only the coordinates for RBF
        initial_coordinates = np.array([coords for coords in extract_points.values()])

        # Assign initial_control_points extract_points
        initial_inner_radius, initial_outer_radius = CylinderFunctions.determineRadiiFromFEB(root, cylinder_parts)
        initial_control_points = CylinderFunctions.generate_annular_cylinder_points(initial_inner_radius,
                                                                                            initial_outer_radius,
                                                                                            cylinder_height,
                                                                                            num_cylinder_points)


        final_control_points = CylinderFunctions.generate_annular_cylinder_points(final_inner_radius, final_outer_radius,
                                                                                          cylinder_height,
                                                                                          num_cylinder_points)

        # Enter the name of surface you would like to get id's from, and it will parse the id's and append the
        # coords from those nodes to initial and final cp for rbf
        zero_displacement = np.array(CylinderFunctions.extractCoordinatesFromSurfaceName(root, ZeroDisplacement))

        initial_control_points = np.concatenate((initial_control_points, zero_displacement))
        final_control_points = np.concatenate((final_control_points, zero_displacement))

        # Call the new morph_points function
        deformed_points = CylinderFunctions.morph_points(initial_control_points, final_control_points,
                                                                 initial_coordinates,
                                                                 extract_points)
        #changes deformed points to be a dictionary
        deformed_points = {item[0]: item[1] for item in deformed_points}

        # Replace coordinates in the original file with the deformed points
        CylinderFunctions.replaceCoordinatesGivenNodeId(root, deformed_points)

    # Write the updated tree to the new FEB file
    tree.write(new_input_file, xml_declaration=True, encoding='ISO-8859-1')

    return new_input_file


# Post Processing Variables
current_date = datetime.datetime.now()
date_prefix = str(current_date.year) + '_' + str(current_date.month)  + '_' + str(current_date.day)
obj_coords_list = []
file_num = 0
csv_filename = Results_Folder + '\\' + date_prefix + '_intermediate.csv'

#Get data from the Run_Variables file
# Newer code (2/14)
run_file = open(dictionary_file)
DOE_dict = csv.DictReader(run_file)


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

current_run_dict = default_dict.copy()

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
    csv_template = ''

    for key in current_run_dict:
        if float(current_run_dict[key]) != float(default_dict[key]):
            param = '' + str(key) + '(' + str(current_run_dict[key]) + ')'
            fileTemplate += param

    # Generate Log CSV File into Results Folder
    IOfunctions.generate_log_csv(current_run_dict, default_code_dict, Results_Folder, fileTemplate + '_log' + '.csv')

    #Update properties, create new input file
    workingInputFileName = updateProperties(originalFebFilePath, fileTemplate)

    # TODO: to easily get input files from anywhere, do the shutil move to working directory
    logFile = Results_Folder + '\\' + fileTemplate + '.log'

    # Print Log file when flag is true
    if Run_FeBio_File_Flag:
        RunFEBinFeBio(workingInputFileName, FeBioLocation, logFile)

        # Check for success of the feb run
        if new_check_normal_run(logFile):
            # Post process
            if GENERATE_INTERMEDIATE_FLAG:
                if first_int_file_flag:

                    edge_elements_dictionary = sav.getCylinderEdgePoints(workingInputFileName, cylinder_parts)

                    # CylinderFunctions.plot_3d_points(edge_elements_dictionary)

                    inner_radius, outer_radius = sav.getRadiiFromEdges(edge_elements_dictionary, cylinder_height, logFile, workingInputFileName, object_list[0])
                    print(inner_radius)
                    print(outer_radius)


                    proc.generate_int_csvs(fileTemplate, object_list, logFile, workingInputFileName, first_int_file_flag,
                                           csv_filename, inner_radius, outer_radius, current_run_dict, plot_points_on_spline)

                    first_int_file_flag = False

                else:

                    edge_elements_dictionary = sav.getCylinderEdgePoints(workingInputFileName, cylinder_parts)

                    inner_radius, outer_radius = sav.getRadiiFromEdges(edge_elements_dictionary, cylinder_height, logFile, workingInputFileName, object_list[0])

                    proc.generate_int_csvs(fileTemplate, object_list, logFile, workingInputFileName, first_int_file_flag,
                                           csv_filename, inner_radius, outer_radius, current_run_dict, plot_points_on_spline)

            file_num += 1
            print('Completed Iteration ' + str(file_num) + ": " + fileTemplate)
            obj_coords_list = []

        else:
            os.rename(workingInputFileName, os.path.splitext(workingInputFileName)[0] + '_error.feb')

if Post_Processing_Flag: # previously called final_csv_flag
    proc.process_features(csv_filename, Results_Folder, date_prefix, numCompPCA)