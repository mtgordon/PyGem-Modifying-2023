"""
This file is responsible for running FeBio in conjunction with the feb_variables.csv file so that
a run is completed for each row
"""
import csv
import subprocess
import xml.etree.ElementTree as ET

'''
Function: RunFEBinFeBio
Takes in the geometry feb file and material properties feb file, along with the location of FeBio
on the system and runs it via the command line. The output log file name can also be given.
'''
def RunFEBinFeBio(geometryFileName, propertiesFileName, prop2, FeBioLocation, outputFileName=None):
    CallString = FeBioLocation + ' -i ' + geometryFileName + ' -s ' + propertiesFileName + ' ' + prop2
    if outputFileName is not None:
        CallString += ' -o ' + 'ENTER_MAT_PROP_HERE' + '.log'
    print(CallString)
    subprocess.call(CallString)

dictionary_file = 'feb_variables.csv'
FeBioLocation = 'C:\Program Files\FEBioStudio2\\bin\\febio4.exe'

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
    'Material1': 1,
    'Material2': 1,
    'Material3': 1
}

# tree = ET.parse('op04.feb')
# root = tree.getroot()
# for point in root.iter('point'):
#     print(child.tag, child.attrib)

for row in DOE_dict:
    print('Row:', row)


    # TODO: to easily get input files from anywhere, do the shutil move to working directory
    # RunFEBinFeBio('oi04.feb', 'op04.feb', FeBioLocation)

