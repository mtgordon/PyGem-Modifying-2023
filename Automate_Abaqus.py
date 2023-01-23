# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:14:44 2020

@author: mgordon
"""

import os
import shutil
import configparser
import json
from datetime import datetime
import glob
import fileinput
import sys
from lib.Generate_INP import AnalogGenerateINP
#from lib.Test_Curving import AnalogGenerateINP
from lib.RunINPinAbaqus import RunINPinAbaqus
from lib.IOfunctions import findLineNum
# from lib.Post_Processing import Post_Processing
from lib.Post_Processing_Files import Post_Processing_Files
import time
import csv
#import sys


# This function checks the INP file to see if the run was completed
# successfully or not
def analysis_was_successful(file_name):
    # Opens File and loads all lines in to content
    with open(file_name) as f:
        # content is an array of lines
        content = f.readlines()
    
    length = len(content)
    for count in range(0,length):
        
        # index starts from the last line and moves up looking for the 
        # unsuccessful completion
        index = length - count - 1
        line = content[index]
        if "THE ANALYSIS HAS NOT BEEN COMPLETED" in line or "abaqus/Analysis exited with errors" in line or "Abaqus/Explicit Packager exited with an error" in line :
            return False
        
    return True



config = configparser.ConfigParser()
config.sections()

ini_file = "Parameters.ini"

# read from params file
config.read(ini_file)


#################################################################################
##### Default Values: They can be changed according to the Run_Variables file ###
#################################################################################


default_dict = {
        'Apical_Shift' : json.loads(config["AVW"]["avw_shift_default"]),
        'AVW_Width' : json.loads(config["AVW"]["avw_width_default"]),
        'AVW_Length' : json.loads(config["AVW"]["avw_length_default"]),
        'CL_Strain' : json.loads(config["SLACK_STRAIN"]["CL_strain_default"]),
        'US_Strain' : json.loads(config["SLACK_STRAIN"]["US_strain_default"]),
        'Para_Strain' : json.loads(config["SLACK_STRAIN"]["PARA_strain_default"]),
        'Pbody_Material' : json.loads(config["MATERIAL_PROPERTIES"]["PBODY_Default"]),
        'Para_Material' : json.loads(config["MATERIAL_PROPERTIES"]["Para_Default"]),
        'CL_Material' : json.loads(config["MATERIAL_PROPERTIES"]["CL_Default"]),
        'US_Material' : json.loads(config["MATERIAL_PROPERTIES"]["US_Default"]),
        'LA_Material' : json.loads(config["MATERIAL_PROPERTIES"]["LA_Default"]),
        'AVW_Material' : json.loads(config["MATERIAL_PROPERTIES"]["AVW_Default"]),
        'PM_Mid_Material' : json.loads(config["MATERIAL_PROPERTIES"]["PM_Mid_Default"]),
        'Generic_File' : config["FILES"]["GenericINP_default"],
        'Hiatus_Size' : json.loads(config["HIATUS_PROPERTIES"]["Hiatus_Length_default"]),
		'Levator_Plate_PC1' : json.loads(config["SHAPE_ANALYSIS"]["Levator_Plate_PC1"]),
        'Levator_Plate_PC2' : json.loads(config["SHAPE_ANALYSIS"]["Levator_Plate_PC2"]),
        'ICM_PC1' : json.loads(config["SHAPE_ANALYSIS"]["ICM_PC1"]),
        'ICM_PC2' : json.loads(config["SHAPE_ANALYSIS"]["ICM_PC2"])
}


Date = datetime.today().strftime('%Y%m%d')
Output_File_Name =  'Post_Processing_File_' + Date + '.csv'

GenerateINPFile = config.getint("FLAGS","GenerateINPFile") # flag for generating the INP files
RunINPFile = config.getint("FLAGS","RunINPFile") # Flag for running the INP files in Abaqus
GetData = config.getint("FLAGS","get_data") # Flag for using ODB Mechens to extract the data from the odb file
GetNodes = config.getint("FLAGS","get_nodes") # flag for getting the nodal coordinates
Get_Reaction_Forces = config.getint("FLAGS","get_reaction_forces") # flag for getting connective tissue reaction forces
Get_Hiatus_Measurements = config.getint("FLAGS","get_hiatus_measurements") # flag for getting connective tissue reaction forces
AbaqusBatLocation= config["SYSTEM_INFO"]["AbaqusBatLocation"]
vary_loading = config.getint("FLAGS","vary_loading") # Flag for running the INP files in Abaqus
troubleshooting = config.getint("FLAGS","troubleshooting") # Flag for running the INP files in Abaqus

frames = config["POST_ANALYSIS"]["frames"]
frames = list(frames.split(','))

AlternateLoads = json.loads(config["Load"]["AlternateLoads"])
LoadValue = config["Load"]["LoadValue"]
LoadLineSignal = config["Load"]["LoadLineSignal"]

Results_Folder_Location = config["FILES"]["Results_Folder_Location"]


RotationPoint   = json.loads(config["HIATUS_PROPERTIES"]["RotationPoint"])
HiatusPoint     = json.loads(config["HIATUS_PROPERTIES"]["HiatusPoint"])
GIFillerPoint   = json.loads(config["HIATUS_PROPERTIES"]["GIFillerPoint"])

# Changing the CL Density0
DensityFactor = json.loads(config["DENSITY"]["CLDensity"])

FileNamePrefix = config["FILES"]["OutputFilePrefix"]

# Delete all of the temporary files that are written
for fname in os.listdir("."):
    if fname.startswith("Generated"):
        os.remove(os.path.join(".", fname))
try:      
    os.remove("FEA_Results.dat")
except OSError:
    pass

try:
    os.remove("GeneratedINPFile.odb")
except OSError:
    pass

try:
    os.remove("output.txt")        
except OSError:
    pass        


#CLStrain = json.loads(config["SLACK_STRAIN"]["CL"])
#USStrain = json.loads(config["SLACK_STRAIN"]["US"])
#ParaStrain = json.loads(config["SLACK_STRAIN"]["Para"])

#RotationPoint   = json.loads(config["HIATUS_PROPERTIES"]["RotationPoint"])
#HiatusPoint     = json.loads(config["HIATUS_PROPERTIES"]["HiatusPoint"])
#GIFillerPoint   = json.loads(config["HIATUS_PROPERTIES"]["GIFillerPoint"])
#MRIHiatusLength    = json.loads(config["HIATUS_PROPERTIES"]["MRIHiatusLength"])
#AbaqusHiatusLength = MRIHiatusLength


#CLUSValues = json.loads(config["MATERIAL_PROPERTIES"]["CLValues"]) # USValues is in the params file if you want to split cl and us
#ParaValues = json.loads(config["MATERIAL_PROPERTIES"]["ParaValues"])
#LAValues = json.loads(config["MATERIAL_PROPERTIES"]["LAValues"])
#AVWValues = json.loads(config["MATERIAL_PROPERTIES"]["AVWValues"])
#PBODYValues = json.loads(config["MATERIAL_PROPERTIES"]["PBODYValues"])
#PMMIDValues = json.loads(config["MATERIAL_PROPERTIES"]["PMMIDValues"])

## AVW Params
#LengthScale = json.loads(config["AVW"]["LengthScale"])
#WidthScale  = json.loads(config["AVW"]["WidthScale"])
#Shift       = json.loads(config["AVW"]["Shift"])

#GenericINPFile = json.loads(config["FILES"]["GenericINP"])

dictionary_file = 'Run_Variables.csv'


# Newer code (2/14)
run_file = open(dictionary_file)
DOE_dict = csv.DictReader(run_file)

# Flag to indicate it the first time through the loop so that it will create a 
# new file and add headers
first_file = 1


current_run_dict = default_dict

# Run each combination from the rows in Run_Variables
for row in DOE_dict:
    print('Row:', row)
    
    current_run_dict = default_dict
#    print(DOE_dict.fieldnames)
    for key in DOE_dict.fieldnames:
        if key in default_dict.keys():
            current_run_dict[key] = row[key]
        else:
            if key != 'Run Number':
                print(key, 'is an unvalid entry')
                sys.exit()
                
                
    LoadLineNo = findLineNum(current_run_dict['Generic_File'], LoadLineSignal) + 2
    MaterialStartLine = findLineNum(current_run_dict['Generic_File'], "** MATERIALS") + 2
    f = open(current_run_dict['Generic_File'])
    for i in range(0, LoadLineNo-1):
        f.readline()
    OldLoadLine = f.readline()
    f.close()

    LoadLine = OldLoadLine.split(',')
    LoadLine[2] = ' ' + str(LoadValue)
    LoadLine = str.join(',', LoadLine)

    
    # Create a string saying what the tissue values are
    # This will be used to create various filenames
    CurrentParameters = '_CL'+ str(current_run_dict['CL_Material']) + '_PARA' + str(current_run_dict['Para_Material']) + '_PCM' + str(current_run_dict['LA_Material']) + '_AVW' + str(current_run_dict['AVW_Material']) + '_CLSS' + str(int(float(current_run_dict['CL_Strain'])*100)) + '_AVWL' + str(current_run_dict['AVW_Length']) + '_HL' + str(int(float(current_run_dict['Hiatus_Size']))) + '_AVWW' + str(current_run_dict['AVW_Width']) + '_AS' + str(current_run_dict['Apical_Shift']) + '_FLP' + str(current_run_dict['Levator_Plate_PC1']) + '_SLP' + str(current_run_dict['Levator_Plate_PC2']) + '_FICM' + str(current_run_dict['ICM_PC1'])  + '_SICM' + str(current_run_dict['ICM_PC2'])

    Date = datetime.today().strftime('%Y%m%d')
    Gen_File_Code = current_run_dict['Generic_File'][0]
    File_Name_Code = FileNamePrefix + '_D' + Date + '_Gen' + Gen_File_Code + CurrentParameters
    INPOutputFileName = File_Name_Code +'.inp'
    
#   Create the INP file for the tissue combinations    
    if GenerateINPFile == 1:
        shutil.copy(current_run_dict['Generic_File'], INPOutputFileName)
        time.sleep(1)
        material_properties = [float(current_run_dict['CL_Material']), float(current_run_dict['Para_Material']), float(current_run_dict['LA_Material']), float(current_run_dict['AVW_Material']), float(current_run_dict['Pbody_Material']), float(current_run_dict['PM_Mid_Material'])]
        print('mat prop in Automate Abaqus:', material_properties)
        if troubleshooting == 1:
            print('Troubleshooting!!!!!!!!!!!!')
            AnalogGenerateINP(material_properties, MaterialStartLine, LoadLine, LoadLineNo, [float(current_run_dict['CL_Strain']), float(current_run_dict['US_Strain']), float(current_run_dict['Para_Strain'])], DensityFactor[0], current_run_dict['Generic_File'], INPOutputFileName, current_run_dict['AVW_Width'], current_run_dict['AVW_Length'], float(current_run_dict['Apical_Shift']), RotationPoint, HiatusPoint, GIFillerPoint, float(current_run_dict['Hiatus_Size']), float(current_run_dict['Levator_Plate_PC1']), float(current_run_dict['Levator_Plate_PC2']), float(current_run_dict['ICM_PC1']), float(current_run_dict['ICM_PC2']), Results_Folder_Location)
            FinalINPOutputFileName = INPOutputFileName
            INP_error = 0
        else:
            try:
                INP_error = 0
                AnalogGenerateINP(material_properties, MaterialStartLine, LoadLine, LoadLineNo, [float(current_run_dict['CL_Strain']), float(current_run_dict['US_Strain']), float(current_run_dict['Para_Strain'])], DensityFactor[0], current_run_dict['Generic_File'], INPOutputFileName, current_run_dict['AVW_Width'], current_run_dict['AVW_Length'], float(current_run_dict['Apical_Shift']), RotationPoint, HiatusPoint, GIFillerPoint, float(current_run_dict['Hiatus_Size']), float(current_run_dict['Levator_Plate_PC1']), float(current_run_dict['Levator_Plate_PC2']), float(current_run_dict['ICM_PC1']), float(current_run_dict['ICM_PC2']), Results_Folder_Location)
                FinalINPOutputFileName = INPOutputFileName
            except:
                INP_error = 1
                print('ERROR IN GENERATING THE INP FILE')
                FinalINPOutputFileName = File_Name_Code +'_ERROR.inp'
                pass
            
        # Copy the INP file to the results folder
        print(Results_Folder_Location + '\\' + INPOutputFileName)
        try:
            os.remove(Results_Folder_Location + '\\' + FinalINPOutputFileName)
        except:
            pass
        time.sleep(3)

        shutil.copy(INPOutputFileName, Results_Folder_Location + '\\' + FinalINPOutputFileName)
        time.sleep(3)

#    Runs the INP file and puts results in the Results Folder
    if RunINPFile == 1 and INP_error == 0:
        print("Running the INP File in Abaqus")
        RunningPrefix = 'Running_'
#            Creating a shorter filename because Abaqus can't handle long ones
        running_base_filename = 'Running_INP'
        # Create a copy of the INP for running
        if config.getint("FLAGS", "testing") != 0:
            shutil.copy(INPOutputFileName, running_base_filename + '.inp')
        else:
            shutil.copy(INPOutputFileName, RunningPrefix + INPOutputFileName)
        # Move the INP file to the results folder
        shutil.move(INPOutputFileName, Results_Folder_Location + '\\' + INPOutputFileName)
        # Run the INP in Abaqus
        if config.getint("FLAGS", "testing") != 0:
            RunINPinAbaqus(running_base_filename, AbaqusBatLocation)
        else:
            RunINPinAbaqus(RunningPrefix + File_Name_Code, AbaqusBatLocation)
                
        # Test to see if the run completed or not
        if config.getint("FLAGS", "testing") != 0:
            OutputFileName = running_base_filename + '.txt'
        else:
            OutputFileName = RunningPrefix + File_Name_Code + '.txt'
            

        if analysis_was_successful(OutputFileName):
            print("Successful Run")
            RunSuccess = 1
            ODBFile = File_Name_Code + '.odb'
            OutputFileName = File_Name_Code + '.txt'
        else:
            print("Run Not Successfully Completed")
            RunSuccess = 0            
            ODBFile = File_Name_Code + '_ERROR.odb'
            OutputFileName = File_Name_Code + '_output_ERROR.txt'


        if config.getint("FLAGS", "testing") != 0:
            shutil.move(running_base_filename + '.odb', Results_Folder_Location + '\\' + ODBFile)
            shutil.move(running_base_filename + '.txt', Results_Folder_Location + '\\' + OutputFileName)
        else:
            shutil.move(RunningPrefix + File_Name_Code + '.odb', Results_Folder_Location + '\\' + ODBFile)
            shutil.move(RunningPrefix + File_Name_Code + '.txt', Results_Folder_Location + '\\' + OutputFileName)

        for fname in os.listdir("."):
            if config.getint("FLAGS", "testing") != 0:
                if fname.startswith(running_base_filename):
                    try:      
                        os.remove(os.path.join(".", fname))
                    except OSError:
                        pass
            else:
                if fname.startswith(RunningPrefix):
                    try:      
                        os.remove(os.path.join(".", fname))
                    except OSError:
                        pass

    if GetData == 1 and RunSuccess == 1 and INP_error == 0:
        
        error_log_file = Results_Folder_Location + '\Error_Log.txt'
        ODBFile_NoPath = ODBFile
        
        file_to_analyze = Results_Folder_Location + '\Working_' + ODBFile_NoPath
        full_odb_file = Results_Folder_Location + '\\' + ODBFile_NoPath
        shutil.copy(full_odb_file, file_to_analyze)
        INI_File = ini_file

        for frame in frames:
            print("which frame?", frame)
    
##     Lines below can be used if the post processing generates an error and you want it to stop rather than keep going
            if troubleshooting == 1:
                print('Troubleshooting post processing')
                INP_NoPath = os.path.splitext(ODBFile_NoPath)[0] + '.inp'
                full_INP_file_to_analyze = Results_Folder_Location + '\Working_' + INP_NoPath
                print("New INP location: ", full_INP_file_to_analyze)
                shutil.copy(Results_Folder_Location + "\\" + INP_NoPath, full_INP_file_to_analyze)
                full_odb_file_to_analyze = file_to_analyze
                raw_path_base_file_name = os.path.splitext(full_odb_file)[0]
                Post_Processing_Files(full_odb_file_to_analyze, full_INP_file_to_analyze, INI_File, Output_File_Name, first_file, raw_path_base_file_name, frame)
                first_file = 0
            else:
            #    Post process the odb file
                try:
                    INP_NoPath = os.path.splitext(ODBFile_NoPath)[0] + '.inp'
                    full_INP_file_to_analyze = Results_Folder_Location + '\Working_' + INP_NoPath
                    print("New INP location: ", full_INP_file_to_analyze)
                    shutil.copy(Results_Folder_Location + "\\" + INP_NoPath, full_INP_file_to_analyze)
                    full_odb_file_to_analyze = file_to_analyze
                    raw_path_base_file_name = os.path.splitext(full_odb_file)[0]
                    Post_Processing_Files(full_odb_file_to_analyze, full_INP_file_to_analyze, INI_File, Output_File_Name, first_file, raw_path_base_file_name, frame)
                    #   Turn off first file flag so that it will add data to the file next time through
                    first_file = 0
                except:
                    with open(error_log_file, 'a+') as out_file:
                        now = datetime.now()
                        out_file.write(now.strftime("%Y-%m-%d %H:%M:%S") + ' Error in post processing ' + ODBFile_NoPath + '\n')
                    pass
        
        for filename in glob.glob(Results_Folder_Location + "\Working*"):
            os.remove(filename) 

        try:      
            os.remove(ODBFile + '.odb')
        except OSError:
            pass


if vary_loading:

    # Re-run code to vary the loads if the run wasn't completed successfully
    # Loop through the different loads, making INP files, running them, and seeing if they worked
    for LoadPercentage in AlternateLoads:
        ERROR_inp_files = []
        
        # Get all of the files from the folder that didn't run correctly
        for file in glob.glob(Results_Folder_Location + '\*ERROR.odb'):
    #        ERROR Files listed as INP files (_ERROR.odb taken off the end and .inp added)
            ERROR_inp_files.append(os.path.splitext(file)[0] + '.inp')
        
        for Bad_File in ERROR_inp_files:
            File_Name_Code = os.path.split(Bad_File[:-10])[1]
            OriginalINPFile = Bad_File[:-10]+'.inp'
            print(OriginalINPFile)
    #        Get the INPFile's name (no path)
            ResultINPFileName = os.path.split(OriginalINPFile)[1]
            print('INPFile Name = ', ResultINPFileName)
            print('Root Name = ', os.path.splitext(ResultINPFileName)[0])
    #       Need to find out which line in the INP has the load to be changed
    #        Find the line number that corresponds to the LoadLineSignal and add 2 lines
    #       (1 because the numbering starts at 1 and 1 because it is the line after this that we care about)
            LoadLineNo = findLineNum(OriginalINPFile , LoadLineSignal) + 2
            
    #       New INP File name is the same as the previous one but with _Force and the number at the end
            INPFile = os.path.splitext(ResultINPFileName)[0] + '_Force' + str(int(LoadPercentage*100)) + '.inp'
    #       Copy the old file to the new file name
            shutil.copy(OriginalINPFile, INPFile)
    
    #       Loop through each line to read in the line that we want (set to OldLoadLine)
            with open(INPFile) as f:
                for i in range(0, LoadLineNo-1):
                    f.readline()
                OldLoadLine = f.readline()
            
    #       Split the line into an array
            LoadLine = OldLoadLine.split(',')
            print(LoadLine)
    #        print(float(LoadLine[2])*LoadPercentage,round(float(LoadLine[2]) * LoadPercentage,4))
    
    #       Change the 3rd part (the load) to the original load * the Load Percentage
            LoadLine[2] = ' ' + str(round(float(LoadValue) * LoadPercentage,4))
    #       Combine the array again as a string for writnig to the file
            LoadLine = str.join(',', LoadLine)
            print(LoadLine)
     
    #       Go through the file until you hit the correct line and then print the new load line
            for i, line in enumerate(fileinput.input(INPFile, inplace=1)):
                if i == LoadLineNo - 1:
                    print(LoadLine)
                else:
                    sys.stdout.write(line)

            if RunINPFile == 1:
                INPOutputFileName = INPFile
                print(INPOutputFileName)
                
                print("Running the INP File in Abaqus")
                RunningPrefix = 'Running_'
    #            Creating a shorter filename because Abaqus can't handle long ones
                running_base_filename = 'Running_INP'
                # Create a copy of the INP for running
                if config.getint("FLAGS", "testing") != 0:
                    shutil.copy(INPOutputFileName, running_base_filename + '.inp')
                else:
                    shutil.copy(INPOutputFileName, RunningPrefix + INPOutputFileName)
                # Move the INP file to the results folder
                shutil.move(INPOutputFileName, Results_Folder_Location + '\\' + INPOutputFileName)
                # Run the INP in Abaqus
                if config.getint("FLAGS", "testing") != 0:
                    RunINPinAbaqus(running_base_filename, AbaqusBatLocation)
                else:
                    RunINPinAbaqus(RunningPrefix + File_Name_Code, AbaqusBatLocation)
                
                # Test to see if the run completed or not
                if config.getint("FLAGS", "testing") != 0:
                    OutputFileName = running_base_filename + '.txt'
                else:
                    OutputFileName = RunningPrefix + File_Name_Code + '.txt'
                    
                if analysis_was_successful(OutputFileName):
                    print("Successful Run")
                    RunSuccess = 1
                    ODBFile = File_Name_Code + '_Force' + str(int(LoadPercentage*100)) + '.odb'
                    OutputFileName = File_Name_Code + '_Force' + str(int(LoadPercentage*100)) + '.txt'
                    print(os.path.splitext(Bad_File)[0] + '.odb')
                    os.remove(os.path.splitext(Bad_File)[0] + '.odb')
                else:
                    print("Run Not Successfully Completed")
                    RunSuccess = 0
                    ODBFile = File_Name_Code + '_Force' + str(int(LoadPercentage*100)) + '_ERROR.odb'
                    OutputFileName = File_Name_Code + '_Force' + str(int(LoadPercentage*100)) + '_output_ERROR.txt'
    
                if config.getint("FLAGS", "testing") != 0:
                    shutil.move(running_base_filename + '.odb', Results_Folder_Location + '\\' + ODBFile)
                    shutil.move(running_base_filename + '.txt', Results_Folder_Location + '\\' + OutputFileName)
                else:
                    shutil.move(RunningPrefix + File_Name_Code + '.odb', Results_Folder_Location + '\\' + ODBFile)
                    shutil.move(RunningPrefix + File_Name_Code + '.txt', Results_Folder_Location + '\\' + OutputFileName)
                        
                for fname in os.listdir("."):
                    if config.getint("FLAGS", "testing") != 0:
                        if fname.startswith(running_base_filename):
                            try:      
                                os.remove(os.path.join(".", fname))
                            except OSError:
                                pass
                    else:
                        if fname.startswith(RunningPrefix):
                            try:      
                                os.remove(os.path.join(".", fname))
                            except OSError:
                                pass

run_file.close()