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
Takes in the input feb file along with the location of FeBio on the system and runs it via 
the command line. The output log file name can also be given.
'''
def RunFEBinFeBio(inputFileName, FeBioLocation, outputFileName=None):
    CallString = FeBioLocation + ' -i ' + inputFileName
    if outputFileName is not None:
        CallString += ' -o ' + outputFileName
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
    newInputFile = fileTemp + '.feb'
    tree.write(newInputFile, xml_declaration=True, encoding='ISO-8859-1')

    return newInputFile


dictionary_file = 'feb_variables.csv'
FeBioLocation = 'C:\\Program Files\\FEBioStudio2\\bin\\febio4.exe'
originalFebFilename = 'Curve and Flat and CL and Filler meshed v4.feb'

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
    'Part2_v': 1,
    'Part9_E': 1,
    'Part9_v': 1,
    'Part27_E': 1,
    'Part27_v': 1
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

    print(row['Part2_E'])
    #Update properties, create new input file
    workingInputFileName = updateProperties(originalFebFilename, fileTemplate)

    # TODO: to easily get input files from anywhere, do the shutil move to working directory
    logFile = fileTemplate + '.log'
    RunFEBinFeBio(workingInputFileName, FeBioLocation, logFile)
