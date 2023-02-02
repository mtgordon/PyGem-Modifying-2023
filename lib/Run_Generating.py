# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:48:18 2020

@author: DeLancey
"""


import csv

'''
Function: level1_level3_combinations
'''
def level1_level3_combinations(Shift, CLStrain, USStrain, ParaStrain, CLUSValues, GenericINPFile, LAValues, MRIHiatusLength):
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    ### Constructing CSV file @@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    level_1_keys = ['Apical_Shift', 'CL_Strain', 'US_Strain', 'Para_Strain', 'CL_Material', 'US_Material']
    level_3_keys = ['Generic_File', 'LA_Material', 'Hiatus_Size']
    level_1_levels = ['Prolapse_Q1', 'Prolapse_Median', 'Prolapse_Q3', 'Control_Q1', 'Control_Median', 'Control_Q3']
    level_3_levels = level_1_levels
    
    level_1_dict = {}
    i = 0
    for level in level_1_levels:
        level_1_dict[level] = {}
        level_1_dict[level]['Apical_Shift'] = Shift[i]
        level_1_dict[level]['CL_Strain'] = CLStrain[i]
        level_1_dict[level]['US_Strain'] = USStrain[i]
        level_1_dict[level]['Para_Strain'] = ParaStrain[i]
        level_1_dict[level]['CL_Material'] = CLUSValues[i]
        level_1_dict[level]['US_Material'] = CLUSValues[i]    
        i += 1
        
    #print(level_3_levels)
    level_3_dict = {}
    i = 0
    for level in level_3_levels:
    #    print(i)
        level_3_dict[level] = {}
        level_3_dict[level]['Generic_File'] = GenericINPFile[i]
        level_3_dict[level]['LA_Material'] = LAValues[i]
        level_3_dict[level]['Hiatus_Size'] = MRIHiatusLength[i]
        i += 1
    
    with open('Run_Variables.csv', mode='w', newline = '') as run_file:
        run_file_writer = csv.writer(run_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        run_file_writer.writerow(level_1_keys + level_3_keys)    
        for level_1 in level_1_levels:
            for level_3 in level_3_levels:
                row = []
                for key in level_1_keys:
                    row.append(level_1_dict[level_1][key])
                for key in level_3_keys:
                    row.append(level_3_dict[level_3][key])
    #            print(row)
                run_file_writer.writerow(row)