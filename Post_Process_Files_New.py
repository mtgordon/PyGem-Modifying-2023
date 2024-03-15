# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:38:02 2019

@author: mgordon

For this file we will
    1) get the file names to be post-processed
    2) pass their names to the Post-processing function


Post Process - File that runs everything (gets the files to be processed and does it)
Post_Processing - File that will take a file and post process it
Post_Processing_Functions - Called by Post_Processing to do the actual processing

"""

from lib.Post_Processing_Files import Post_Processing_Files
import glob
import configparser
import shutil
import subprocess
import os
import time
import datetime

#INI_File = 'Post_Run_Analysis_Input.ini'
INI_File = 'Parameters.ini'
now = datetime.datetime.now()
Output_File_Name = now.strftime("%Y-%m-%d %H-%M-%S") +'_Post_Processing_File.csv'
frame = 'last'


config = configparser.ConfigParser()
config.sections()
config.read(INI_File)

Results_Folder_Location = config["FILES"]["Results_Folder_Location"]
AbaqusBatLocation= config["SYSTEM_INFO"]["AbaqusBatLocation"]
frames = config["POST_ANALYSIS"]["frames"]
error_log_file = Results_Folder_Location + '\Error_Log.txt'
troubleshooting = config["FLAGS"]["troubleshooting"]



frames = list(frames.split(','))
#PMMIDValues = json.loads(config
#print(frames)
    
full_odb_files = []

#print(Results_Folder_Location)


for filename in glob.glob(Results_Folder_Location + "\Working*"):
    os.remove(filename)


# Get all the list of files (with their path) to be post-processed
for file in glob.glob(Results_Folder_Location + '\*.odb'):
    if 'ERROR' in file:
        None
    else:
        full_odb_files.append(file)

# Flag to indicate it the first time through the loop so that it will create a 
# new file and add headers
first_file = 1

#print(full_odb_files)

with open(error_log_file, 'w') as out_file:
    out_file.write('ERROR LOG FILE \n')

# Loop through the files to get the results
for full_odb_file in full_odb_files:
    Upgrade_Necessary = 1
    ODBFile_NoPath = os.path.basename(full_odb_file)
    file_to_analyze = Results_Folder_Location + '\Working_' + ODBFile_NoPath
    if Upgrade_Necessary:
        print('Upgrading the odb file')        
        try:
            os.remove(file_to_analyze.encode('unicode_escape'))
        except:
            pass
        CallString = AbaqusBatLocation + ' -upgrade -job "' + file_to_analyze.replace('\\','\\\\') + '" -odb "' + full_odb_file.replace('\\','\\\\') + '"'
#        print(CallString)
        subprocess.call(CallString)
        print("File was updated")
        print("_______________________________")
    else:
        shutil.copy(full_odb_file, file_to_analyze)

# Pause to make sure the file is created and saved (mostly for using the virtual machine)
    time.sleep(3)


#    INP_NoPath = os.path.splitext(ODBFile_NoPath)[0] + '.inp'
#    full_INP_file_to_analyze = Results_Folder_Location + '\Working_' + INP_NoPath
#    print("New INP location: ", full_INP_file_to_analyze)
#    shutil.copy(Results_Folder_Location + "\\" + INP_NoPath, full_INP_file_to_analyze)
#    full_odb_file_to_analyze = file_to_analyze
#    raw_path_base_file_name = os.path.splitext(full_odb_file)[0]

    for frame in frames:
        print("which frame?", frame)

# Lines below can be used if the post processing generates an error and you want it to stop rather than keep going
        if int(troubleshooting):
            print('TROUBLESHOOTING!!!!')
            INP_NoPath = os.path.splitext(ODBFile_NoPath)[0] + '.inp'
            full_INP_file_to_analyze = Results_Folder_Location + '\Working_' + INP_NoPath
#            print("New INP location: ", full_INP_file_to_analyze)
            shutil.copy(Results_Folder_Location + "\\" + INP_NoPath, full_INP_file_to_analyze)
            full_odb_file_to_analyze = file_to_analyze
            raw_path_base_file_name = os.path.splitext(full_odb_file)[0]
            Post_Processing_Files(full_odb_file_to_analyze, full_INP_file_to_analyze, INI_File, Output_File_Name, first_file, raw_path_base_file_name, frame)
            #   Turn off first file flag so that it will add data to the file next time through
            first_file = 0
        else:
    #        Post process the odb file
            try:
                INP_NoPath = os.path.splitext(ODBFile_NoPath)[0] + '.inp'
                full_INP_file_to_analyze = Results_Folder_Location + '\Working_' + INP_NoPath
#                print("New INP location: ", full_INP_file_to_analyze)
                shutil.copy(Results_Folder_Location + "\\" + INP_NoPath, full_INP_file_to_analyze)
                full_odb_file_to_analyze = file_to_analyze
                raw_path_base_file_name = os.path.splitext(full_odb_file)[0]
                Post_Processing_Files(full_odb_file_to_analyze, full_INP_file_to_analyze, INI_File, Output_File_Name, first_file, raw_path_base_file_name, frame)
                #   Turn off first file flag so that it will add data to the file next time through
                first_file = 0
            except:
                with open(error_log_file, 'a+') as out_file:
                    print('Error in Post Processing', full_odb_file_to_analyze)
                    now = datetime.datetime.now()
                    out_file.write(now.strftime("%Y-%m-%d %H:%M:%S") + ' Error in post processing ' + ODBFile_NoPath + '\n')
                pass
    
        for filename in glob.glob(Results_Folder_Location + "\Working*"):
            os.remove(filename) 