# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 20:53:39 2019

@author: mtgordon
"""

import subprocess
import os
from Node_Distances import getXClosestNodes

CallString = '"C:\\SIMULIA\\Commands\\abaqus.bat"  CAE noGUI=ODBMechensFunction_v2  -- -odbfilename "I:\\a Pelvic Floor\\Hiatus Runs 2019 12 11\\Working__D20191205_GenB_CL200_PARA100_PCM100_AVW100_CLSS0_AVWL1_CLStrain0.0_HL39_AVWW1" -partname OPAL325_PM_MID-1 -strNodes 19 -var1 COORD -outputfile Temp_Coords -headerflag N -newfileflag Y -frames last'
#CallString = '"C:\\SIMULIA\\Commands\\abaqus.bat"  CAE noGUI=ODBMechensFunction_v2  -- -odbfilename "I:\\a Pelvic Floor\\Hiatus Runs 2019 12 11\\Working__D20191205_GenB_CL200_PARA100_PCM100_AVW100_CLSS0_AVWL1_CLStrain0.0_HL35_AVWW1" -partname OPAL325_PM_MID-1 -strNodes 19 -var1 COORD -outputfile Temp_Coords -headerflag N -newfileflag Y -frames last'

upgraded_inp_file = "I:\\a Pelvic Floor\\Hiatus Runs 2019 12 11\\Working__D20191205_GenB_CL200_PARA100_PCM100_AVW100_CLSS0_AVWL1_CLStrain0.0_HL35_AVWW1"

#path = os.path(upgraded_inp_file)
path = os.path.dirname(os.path.abspath(upgraded_inp_file))
file_name = os.path.basename(upgraded_inp_file)
original_file_name = file_name.replace('Working_','')
print(path)
print(file_name)
print(original_file_name)

#output = subprocess.call(CallString)
#print(output)


try:
    os.environ.pop('PYTHONIOENCODING')
except KeyError:
    pass

JobPath = "I:\\Testing New Code"

cmd = subprocess.Popen(CallString, cwd = JobPath, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, shell=True).communicate()[0]

print(cmd)
print("error" in str(cmd))

if ("error" in str(cmd)):
    
    path = os.path.dirname(os.path.abspath(upgraded_inp_file))
    file_name = os.path.basename(upgraded_inp_file)
    original_file_name = file_name.replace('Working_','')
    file_to_analyse = upgraded_inp_file
    file = path + "\\" + original_file_name
    
    print(file)
    print(file_to_analyze)
    try:
        os.remove(upgraded_inp_file.encode('unicode_escape'))
    except:
        pass
    CallString = AbaqusBatLocation + ' -upgrade -job "' + file_to_analyze.replace('\\','\\\\') + '" -odb "' + file.replace('\\','\\\\') + '"'
    print(CallString)
    subprocess.call(CallString)
#
