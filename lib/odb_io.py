# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:31:51 2020

@author: mgordon
"""
import time
import subprocess
import os
from numpy import genfromtxt
import csv
#import math

'''
Function: get_odb_data
'''
def get_odb_data(material_names, nodes, node_property, frames, AbaqusBatLocation, odb_filename):

    print(material_names)
    DataFileName = 'testing_Temp.csv'
    DataFileNameNoExt = os.path.splitext(DataFileName)[0]

    try:
        os.remove(DataFileName)
    except:
        pass
    header_flag = 'N'
    new_file_flag = 'Y'
#    print("DID I MAKE IT?")
#    CallString = AbaqusBatLocation + '  CAE noGUI=".\lib\Get_Data_From_ODB"  -- -odbfilename "' + odb_filename + '" -partname ' + material_names + ' -strNodes ' + nodes + ' -var1 ' + node_property + ' -outputfile "' + DataFileName + '" -headerflag ' + header_flag+ " -newfileflag " + new_file_flag + " -frames " + frames
    CallString = AbaqusBatLocation + '  CAE noGUI=".\lib\Testing_Get_Data_From_ODB"  -- -odbfilename "' + odb_filename + '" -partname ' + material_names + ' -strNodes ' + nodes + ' -var1 ' + node_property + ' -outputfile "' + DataFileNameNoExt + '" -headerflag ' + header_flag+ " -newfileflag " + new_file_flag + " -frames " + frames
    print(CallString)
    
########################################
    subprocess.call(CallString)
######################################
    
    time.sleep(3)
    
#    raw_data = genfromtxt(DataFileName, delimiter=',')
#    
#    print("raw inside data = ", raw_data)


    with open(DataFileName, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)

#    print('csv.rearder data: ', your_list)
#
    
#    print('lacking a bracket?', your_list[0])
    raw_data = your_list
    
    if node_property == 'COORD':
        grouped_data = list(zip(*[iter(raw_data[0])]*3))
    elif node_property == 'RF':
        grouped_data = []
        for row in raw_data:
#            print('next')
#            print(row)
            try:
#                print(row[0])
                if row[0] == 'Total Force Using Resultants =':
                    grouped_data.append(row[1])
                elif row[0] == 'Total Force Using Components =':
                    grouped_data.append(row[1])
            except:
                pass
#        for i in range(0,int(len(your_list)/3)):
    
#    print("Grouped Inside Data :", grouped_data)
    
    try:
        os.remove(DataFileName)
    except:
        pass
#    print(new_coordinates)
    return grouped_data


#    new_coordinates = []
    
#    print(Coordinates)
#    print(Coordinates[0])
#    print(Coordinates[0,1])
    
#    for i in range(0,len(Coordinates)/3):
#        new_coordinates.append([Coordinates[i,0], Coordinates[i,1], Coordinates[i,2]])
    
#        PM_Mid_top_deformed = Point3D(midsaggital_x, Coordinates[1], Coordinates[2])
