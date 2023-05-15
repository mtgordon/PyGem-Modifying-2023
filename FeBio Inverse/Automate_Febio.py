"""
This file is responsible for running FeBio in conjunction with the feb_variables.csv file so that
a run is completed for each row (Using both an input and optimize feb file)
"""
import csv
import sys
import subprocess
import xml.etree.ElementTree as ET

'''
Function: RunFEBinFeBio
Takes in the geometry feb file and material properties feb file, along with the location of FeBio
on the system and runs it via the command line. The output log file name can also be given.
'''
def RunFEBinFeBio(geometryFileName, propertiesFileName, FeBioLocation, outputFileName=None):
    CallString = FeBioLocation + ' -i ' + geometryFileName + ' -s ' + propertiesFileName
    if outputFileName is not None:
        CallString += ' -o ' + outputFileName
    print(CallString)
    subprocess.call(CallString)


dictionary_file = 'feb_variables.csv'
FeBioLocation = 'C:\\Program Files\\FEBioStudio2\\bin\\febio4.exe'
originalFebFilename = 'op04.feb'

#Get data from the Run_Variables file
# Newer code (2/14)
run_file = open(dictionary_file)
DOE_dict = csv.DictReader(run_file)

# Flag to indicate it the first time through the loop so that it will create a
# new file and add headers
first_file = 1

# Flag to indicate the first time through the while loop for the load search
first_iter = True

#Have the default material variables be 1 (100%) so they do not change if no variable is given
default_dict = {
    'Mat1': 1,
    'Mat2': 1,
    'Mat3': 1
}

current_run_dict = default_dict.copy()

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

    # Parse original FEB file and update coordinates to new FEB file
    tree = ET.parse(originalFebFilename)
    root = tree.getroot()
    for point in root.iter('point'):
        # print(point.text)
        textList = point.text.split(', ')  # 0 is first coord (no change), 1 is second coord (change)
        newCoord = float(textList[1]) * float(current_run_dict['Mat1'])
        point.text = textList[0] + ', ' + str(newCoord)

    # using UTF-8 encoding does not bring up any issues (can be changed if needed)
    workingFebFilename = fileTemplate + '.feb'
    tree.write(workingFebFilename, xml_declaration=True, encoding='UTF-8')

    # TODO: to easily get input files from anywhere, do the shutil move to working directory
    logFile = fileTemplate + '.log'
    RunFEBinFeBio('oi04.feb', workingFebFilename, FeBioLocation, logFile)
