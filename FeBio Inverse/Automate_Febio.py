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
import generate_pca_points_AVW as gic
import PostProcess_FeBio as proc
import Bottom_Tissue_SA_Final as bts
import pandas as pd
import re
import time
import PCA_data
from lib import IOfunctions

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
    extract_points = IOfunctions.extract_coordinates_list_from_feb(originalFebFilePath, "Object5")


    # Go through each element which is within the csv
    for partProp in current_run_dict.keys():

        if "Part" in partProp:
            partName = partProp.split('_')[0]
            propName = partProp.split('_')[1]

            # Locate the Mesh Domains section to find which parts have which materials
            meshDomains = root.find('MeshDomains')
            for domain in meshDomains:
                if domain.attrib['name'] == partName:
                    for mat in tree.find('Material'):
                        if mat.attrib['name'] == domain.attrib['mat']:
                            newValue = float(mat.find(propName).text) * float(current_run_dict[partProp])
                            mat.find(propName).text = str(newValue)

        elif "Pressure" in partProp:
            loads = root.find('Loads')
            for surface_load in loads:
                pressure = surface_load.find('pressure')
                pressure.text = str(current_run_dict["Pressure"])

        #elif "Inner_Radius" in partProp:
            # Get inner
        #elif "Outer_Radius" in partProp:
            # Get Outer then call inner and outer from PointsExtractionTesting.Py



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
#TODO: IN VAR FILE --> SET PART NAMES FOR ALL MATERIALS --> DONE
dictionary_file = 'feb_variables.csv' #DONE
FeBioLocation = 'C:\\Program Files\\FEBioStudio2\\bin\\febio4.exe'
originalFebFilePath = 'D:\\Gordon\\Automate FEB Runs\\2024_4_29 auto\\Base File\\Basic_Cylinder_Pressure.feb' #DONE

# Post Processing Variables
current_date = datetime.datetime.now()
date_prefix = str(current_date.year) + '_' + str(current_date.month)  + '_' + str(current_date.day)
#TODO: CHANGE PER FILE
object_list = ['Object5'] #TODO: Get new names for flat, curve, GI Filler --> DONE
obj_coords_list = []
file_num = 0
Results_Folder = 'D:\\Gordon\\Automate FEB Runs\\2024_4_29 auto' #DONE
csv_filename = Results_Folder + '\\' + date_prefix + '_intermediate.csv'

# FLAGS
first_int_file_flag = True
final_csv_flag = False
GENERATE_INTERMEDIATE_FLAG = True

#Get data from the Run_Variables file
# Newer code (2/14)
run_file = open(dictionary_file)
DOE_dict = csv.DictReader(run_file)

#Have the default material variables be 1 (100%) so they do not change if no variable is given
#TODO: In FEB-variables file, have the headers increase in number to work with file mover file
#TODO: CHANGE PER FILE
default_dict = {
    'Part5_E': 1,
    'Pressure': 0,
    'Inner_Radius': 1,
    'Outer_Radius': 2
}
#Eventually add outer_radius and pressure

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
    for key in current_run_dict:
        param = '_' + str(key) + '(' + str(current_run_dict[key]) + ')'
        fileTemplate += param

    #Update properties, create new input file
    workingInputFileName = updateProperties(originalFebFilePath, fileTemplate)

    # TODO: to easily get input files from anywhere, do the shutil move to working directory
    logFile = Results_Folder + '\\' + fileTemplate + '.log'


    #TODO: CHANGE NODES FOR CYLINDER HERE

    RunFEBinFeBio(workingInputFileName, FeBioLocation, logFile)

    # Check for success of the feb run
    if new_check_normal_run(logFile):
        # Post process
        if GENERATE_INTERMEDIATE_FLAG:
            if first_int_file_flag:
                proc.generate_int_csvs(fileTemplate, object_list, logFile, workingInputFileName, first_int_file_flag,
                                       csv_filename)
                first_int_file_flag = False
            else:
                proc.generate_int_csvs(fileTemplate, object_list, logFile, workingInputFileName, first_int_file_flag,
                                       csv_filename)

        file_num += 1
        print('Completed Iteration ' + str(file_num) + ": " + fileTemplate)
        obj_coords_list = []

    else:
        os.rename(workingInputFileName, os.path.splitext(workingInputFileName)[0] + '_error.feb')

if final_csv_flag:
    proc.process_features(csv_filename, Results_Folder, date_prefix)
