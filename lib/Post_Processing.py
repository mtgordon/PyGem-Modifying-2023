# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:54:33 2019

@author: mtgordon
"""


from lib.Test_Post_Processing_Functions import calc_prolapse_size, Calc_Reaction_Forces, calc_exposed_vaginal_length, calc_exposed_vaginal_length2
from lib.Surface_Tools import getBnodes, numOfNodes
from lib.test_Scaling import takeMeasurements_return
from lib.CalculateHiatus import CalculateHiatus_v2
import os
import configparser
import csv
import re
import subprocess
import time


'''
Function: Post_Processing
'''
def Post_Processing(odb_file, INP_File, INI_File, Output_File_Name, first_file_flag, output_base_filename, frame):

    base_file_name = os.path.splitext(os.path.split(odb_file)[1])[0]
    path_base_file_name = os.path.splitext(odb_file)[0]
    print('##################################')
    print(base_file_name)
    print(path_base_file_name)
    print('##################################')  
    Header = ['File Name','Frame']
    Header_Codes = []
    config = configparser.ConfigParser()
    config.sections()
    config.read(INI_File)
    
    Results_Folder_Location = config["FILES"]["Results_Folder_Location"]
    get_prolapse_measurements = config["FLAGS"]["get_prolapse_measurements"]
    get_reaction_forces = config["FLAGS"]["get_reaction_forces"]
    get_nodes = config["FLAGS"]["get_nodes"]
    get_hiatus_measurements = config["FLAGS"]["get_hiatus_measurements"]
    AbaqusBatLocation= config["SYSTEM_INFO"]["AbaqusBatLocation"]
    get_data = config["FLAGS"]["get_data"]
    testing = config["FLAGS"]["testing"]
    print('testing:', testing)
    get_exposed_vaginal_length = config["FLAGS"]["get_exposed_vaginal_length"]
       
    print('Get nodes = ', get_nodes)

#   Loads the different prefixes that are used in parsing the filename
    with open('File_Name_Parsing.csv', mode='r') as infile:
        reader = csv.reader(infile)
        mydict = {rows[0]:(rows[1], rows[2]) for rows in reader}
        i = 0
        infile.seek(0)
        for rows in reader:
            if i:
                Header_Codes.append(rows[0])
            i += 1
    print(Header_Codes)
    ODBFile_NoPath = os.path.basename(odb_file)
    INP_File = odb_file[:-4]

    print(ODBFile_NoPath)
    GenericFileCode = ODBFile_NoPath.split('Gen')[1][0]
    if GenericFileCode == 'U':
        GenericINPFile = 'Unilateral_Generic.inp'
    elif GenericFileCode == 'B':
        GenericINPFile = 'Bilateral_Generic.inp'
    elif GenericFileCode == 'N':
        GenericINPFile = 'Normal_Generic.inp'
    else:
        print('NO GENERIC FILE')
    Output = [ODBFile_NoPath,frame]
    Split_Array = re.split('(\d+\.\d+|-?\d+)|_|-', ODBFile_NoPath)
    Split_Array = [i for i in Split_Array if i]
    print(Split_Array)
    print(Header_Codes)
    for Code in Header_Codes:
        
        ####### Look at using Split_Array = re.split('(\d+)',ODBFile.replace('_',''))
        ######## then look for the header to match something in the list and print the next element
        if Code == 'Gen':
            Output.append(GenericFileCode)
        else:
            Header_Index = Split_Array.index(Code)
            Data_Index = Header_Index + 1
    #        print(Header[j],Header_Index,Data_Index)
            Output.append(Split_Array[Data_Index])
        Header.append(mydict[Code][1])
#        Header.append(Code)
#        print(Code)

    if get_data == '1':
        print('Getting Data')
        
        
                    # SURFACES
        AVW         = "OPAL325_AVW_v6"
        GI_FILLER   = "OPAL325_GIfiller"
        ATFP        = "OPAL325_ATFP"
        ATLA        = "OPAL325_ATLA"
        LA          = "OPAL325_LA"
        PBODY       = "OPAL325_PBody"
        PM_MID      = "OPAL325_PM_mid"
        
        REF_PLANE   = "OPAL325_refPlane_0318_2011" # Old Abaqus File
        # REF_PLANE   = "OPAL325_Refplane_MirrorD_0318 # New Abaqus File
        
        # FIBERS
        CL          = "OPAL325_CL_v6"
        PARA        = "OPAL325_Para_v6"
        US          = "OPAL325_US_v6"
        strains = takeMeasurements_return("Measurements.txt", AVW, [CL, PARA, US], GenericINPFile, INP_File + '.inp')    
        Output.extend(strains)
        Header.extend(['CL Strain', 'Para Strain', 'US Strain'])
        
        
        
        
        
############################################ Get the measurements

        print('Creating Node Coordinate Files')
        material_list = []
        if get_prolapse_measurements == '1' or get_exposed_vaginal_length == '1':
            material_list.append("OPAL325_AVW_v6")
#            if get_exposed_vaginal_length == '1':
#                material_list.append("OPAL325_GIfiller")
        
        print('Getting Nodes')

    #            ML = ["OPAL325_PM_mid", "OPAL325_PBody", "OPAL325_ATFP", "OPAL325_LA", "OPAL325_GIfiller", "OPAL325_ATLA", "OPAL325_AVW_v6"]
    #            MaterialList = ['OPAL325_PM_MID-1', 'OPAL325_PBODY-1', 'OPAL325_ATFP-1', 'OPAL325_LA-1', 'OPAL325_GIFILLER-1', 'OPAL325_ATLA-1','OPAL325_AVW_V6-1']
        MaterialSizeList = []
        for i in range(0,len(material_list)):
            MaterialSizeList.append(numOfNodes(INP_File+'.inp',material_list[i]))

        for p in range (0,len(material_list)):
            if MaterialSizeList[p]+1 > 1750:
                MaterialSizeList[p] = 1750
            nodes = list(range(1,MaterialSizeList[p]+1))
            PassingNodes = ','.join(str(i) for i in nodes)
            Variable1 = "COORD"
            Headerflag = 'N'
            NewFileFlag = 'Y'
            Frames = frame
            MaterialName = material_list[p].upper() + '-1'
            DataFileName = output_base_filename + '_' + MaterialName
            if material_list[p] == "OPAL325_AVW_v6":
                AVW_csv_filename = DataFileName + '.csv'
            print(DataFileName)
#            CallString = AbaqusBatLocation + '  CAE noGUI=ODBMechensFunction_v3  -- -odbfilename "' + path_base_file_name + '" -partname ' + MaterialName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile "' + DataFileName + '" -headerflag ' + Headerflag + " -newfileflag " + NewFileFlag + " -frames " + Frames
            CallString = AbaqusBatLocation + '  CAE noGUI=".\lib\Get_Data_From_ODB"  -- -odbfilename "' + path_base_file_name + '" -partname ' + MaterialName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile "' + DataFileName + '" -headerflag ' + Headerflag + " -newfileflag " + NewFileFlag + " -frames " + Frames
            print(testing)
            
            if testing == '1':
                print('inside testing')
                CallString = AbaqusBatLocation + '  CAE noGUI=".\lib\Get_Data_From_ODB"  -- -odbfilename "' + path_base_file_name + '" -partname ' + MaterialName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile "' + DataFileName + '" -headerflag ' + Headerflag + " -newfileflag " + NewFileFlag + " -frames " + Frames
#                CallString = AbaqusBatLocation + '  CAE noGUI="' + os.getcwd() + '\lib\Get_Data_From_ODB"  -- -odbfilename "' + path_base_file_name + '" -partname ' + MaterialName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile "' + DataFileName + '" -headerflag ' + Headerflag + " -newfileflag " + NewFileFlag + " -frames " + Frames           
            print(CallString)
            
############################## This is the line that gets commented out a lot in testing Post Processing
            subprocess.call(CallString)
############################## This is the line that gets commented out a lot in testing Post Processing
            
            time.sleep(3)
    
    ###########################################################################
        
        if get_prolapse_measurements == '1':
            print('_______________________')
            print('Getting Prolapse Measurements')
            
            
#            calc_prolapse_size2
            
            [max_prolapse, max_prolapse_absolute, max_prolapse_node, nodes] = calc_prolapse_size(GenericINPFile, odb_file, INP_File + '.inp', INI_File, frame, AVW_csv_filename)
#            Output.extend([max_prolapse, max_prolapse_absolute, max_prolapse_node, nodes[0], nodes[1]])
#            Header.extend(['Max Prolapse Deformed', 'Max Prolapse Undeformed', 'Max Prolapse Node', 'Plane Node 1', 'Plane Node 2', 'Plane Node 3'])
#            [max_prolapse, max_prolapse_absolute, max_prolapse_node, nodes] = calc_prolapse_size(GenericINPFile, odb_file, INP_File, INI_File)
##                [max_prolapse, max_prolapse_absolute, max_prolapse_node] = calc_prolapse_size(GenericINPFile, odb_file, INP_File, INI_File)
            Output.extend([max_prolapse, max_prolapse_absolute, max_prolapse_node, nodes[0], nodes[1]])
            Header.extend(['Max Prolapse Deformed', 'Max Prolapse Undeformed', 'Max Prolapse Node', 'Prolapse Node 1', 'Prolapse Node 2'])


            
#            try:
#                [max_prolapse, max_prolapse_absolute, max_prolapse_node, nodes] = calc_prolapse_size(GenericINPFile, odb_file, INP_File, INI_File)
##                [max_prolapse, max_prolapse_absolute, max_prolapse_node] = calc_prolapse_size(GenericINPFile, odb_file, INP_File, INI_File)
#                Output.extend([max_prolapse, max_prolapse_absolute, max_prolapse_node])
#                Header.extend(['Max Prolapse Deformed', 'Max Prolapse Undeformed', 'Max Prolapse Node', 'Plane Nodes'])
##                Header.extend(['Max Prolapse Deformed', 'Max Prolapse Undeformed', 'Max Prolapse Node'])
#            except:
#                print('ERROR CALCULATING PROLAPSE SIZE')
#            else:
#                pass
    #################################################################################
        
                # get results and write them to FEAResults.dat. The different linees
    #            ODBFile = File_Name_Code
    #            shutil.copy(Results_Folder_Location + '\\' + ODBFile + '.odb', ODBFile + '.odb')
    
    ###########################################################################
    ####################### Get the nodes for each material at each step###
    ########### Maybe create a function that you pass the file name with path, the Material List, and some flags and it does everything
        if get_nodes == '1':
            print('Getting Nodes')
            ML = ["OPAL325_AVW_v6"]
    #            ML = ["OPAL325_PM_mid", "OPAL325_PBody", "OPAL325_ATFP", "OPAL325_LA", "OPAL325_GIfiller", "OPAL325_ATLA", "OPAL325_AVW_v6"]
            MaterialList = ['OPAL325_AVW_V6-1']
    #            MaterialList = ['OPAL325_PM_MID-1', 'OPAL325_PBODY-1', 'OPAL325_ATFP-1', 'OPAL325_LA-1', 'OPAL325_GIFILLER-1', 'OPAL325_ATLA-1','OPAL325_AVW_V6-1']
            
    #            MaterialSizeList = [numOfNodes(GenericINPFile, ML[0]), numOfNodes(GenericINPFile, ML[1]), numOfNodes(GenericINPFile, ML[2]), numOfNodes(GenericINPFile, ML[3]), numOfNodes(GenericINPFile, ML[4]), numOfNodes(GenericINPFile, ML[5]),numOfNodes(GenericINPFile, ML[6])]
    #            MaterialSizeList = [numOfNodes(GenericINPFile, ML[0])]
            MaterialSizeList = []
            for i in range(0,len(ML)):
                MaterialSizeList.append(numOfNodes(GenericINPFile,ML[i]))    
    
            for p in range (0,len(MaterialList)):
                nodes = list(range(1,MaterialSizeList[p]+1))
                PassingNodes = ','.join(str(i) for i in nodes)
                Variable1 = "COORD"
                Headerflag = 'N'
                NewFileFlag = 'Y'
#                Frames = 'last'   # This is only true if I am lookinga the data for the end of the run
                Frames = frame
                MaterialName = MaterialList[p]
                DataFileName = output_base_filename + '_' + MaterialName
                print(DataFileName)
                CallString = AbaqusBatLocation + '  CAE noGUI=".\lib\Get_Data_From_ODB"  -- -odbfilename "' + path_base_file_name + '" -partname ' + MaterialName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile "' + DataFileName + '" -headerflag ' + Headerflag + " -newfileflag " + NewFileFlag + " -frames " + Frames
                print(CallString)
                subprocess.call(CallString)
                time.sleep(3)
    
    ###########################################################################
    
    ###########################################################################
    # Next get the Reaction Forces Boundary nodes are passing Nodes
        if get_reaction_forces == '1':
            print('Getting Reaction Forces')
            
            [headers, data] = Calc_Reaction_Forces(path_base_file_name, output_base_filename, GenericINPFile, INI_File, frame)
            Header.extend(headers)
            Output.extend(data)
            
    
    ###########################################################################
    
    ###########################################################################
#   Measure the Exposed Vaginal Length
        if get_exposed_vaginal_length == '1':
            

#            exposed_vaginal_length = calc_exposed_vaginal_length(INI_File, AVW_csv_filename, INP_File+'.inp', odb_file, frame)
            exposed_vaginal_length = calc_exposed_vaginal_length2(GenericINPFile, odb_file, INP_File + '.inp', INI_File, frame, AVW_csv_filename)
            
            Output.extend([exposed_vaginal_length])
            Header.extend(['Exposed Vaginal Length'])

##############################################################################################
##            Getting the Hiatus measurements
        if get_hiatus_measurements == '1':
            print("Getting Hiatus Measurements")
            HiatusMaterial2 = 'OPAL325_PBody'
            HiatusPoint1 = [-3.311,18.9,-22.349]
            HiatusPoint2 = [-3.311,-11.231,-29.325]
            OutputFileName = base_file_name + '_Hiatus'
            deformed_hiatus = CalculateHiatus_v2(AbaqusBatLocation, base_file_name, GenericINPFile, HiatusPoint1, HiatusPoint2, HiatusMaterial2, Results_Folder_Location)
            Output.extend([deformed_hiatus])
            Header.extend(['Deformed Hiatus Length'])
#############################################################################################
#################################################################################
            # Done with the measuremnts, now write the file
        
        

            
        
    if first_file_flag:
        with open(Results_Folder_Location + '\\' + Output_File_Name, 'w', newline = '') as Output_File:
            filewriter = csv.writer(Output_File, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)        
            filewriter.writerow(Header)

    with open(Results_Folder_Location + '\\' + Output_File_Name, 'a', newline = '') as Output_File:
        filewriter = csv.writer(Output_File, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
        filewriter.writerow(Output)