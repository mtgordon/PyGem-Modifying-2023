# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:57:42 2018

@author: mgordon
"""

import subprocess
from numpy import genfromtxt
import ntpath
import os
from lib.workingWith3dDataSets import Point
from lib.Node_Distances import getXClosestNodes
import shutil

'''
Function: CalculateHiatus
'''
def CalculateHiatus(AbaqusBatLocation, ODBFilename, HiatusNode1,HiatusMaterial1,HiatusNode2,HiatusMaterial2,OutputFileName):
    OutputFileName = ntpath.basename(OutputFileName)

    CallString = AbaqusBatLocation + '  CAE noGUI=HiatusNodesFromODB -- -odb ' + ODBFilename + ' -Node1 ' + str(HiatusNode1) + ' -Mat1 ' + HiatusMaterial1 + ' -Node2 ' + str(HiatusNode2) + ' -Mat2 ' + HiatusMaterial2 + ' -Output ' + OutputFileName
    subprocess.call(CallString)
    
    HiatusOutputFile = open(OutputFileName + '.csv','w')    
    HiatusTempFile = OutputFileName + '_temp.csv'
    
    Coordinates = genfromtxt(HiatusTempFile, delimiter=',')
#    print(Deltas)
    for i in range(0,len(Coordinates[:,0])):
        HP1x = Coordinates[i,0]
        HP1y = Coordinates[i,1]
        HP1z = Coordinates[i,2]
        
        HP2x = Coordinates[i,3]
        HP2y = Coordinates[i,4]
        HP2z = Coordinates[i,5]
            
        HiatusDistance = ((HP1x-HP2x)**2+(HP1y-HP2y)**2+(HP1z-HP2z)**2)**.5
        HiatusOutputFile.write('%10.9E, ' % (HiatusDistance))
        HiatusOutputFile.write('\n')
        print(i,HiatusDistance)

'''
Function: CalculateHiatus_v2
'''
def CalculateHiatus_v2(AbaqusBatLocation, ODBFile, GenericINPFile, HiatusPoint1_Array, HiatusPoint2_Array, HiatusMaterial2, Results_Folder_Location):
#    try:      
#        os.remove(DataFileName+'.csv')
#        print(Results_Folder_Location + '\\' +Temp_ODBFile + '.odb')
#        os.remove(Results_Folder_Location + '\\' +Temp_ODBFile + '.odb')
#    except OSError:
#        pass
    HiatusPoint1 = Point(HiatusPoint1_Array[0], HiatusPoint1_Array[1], HiatusPoint1_Array[2])
    HiatusPoint2 = Point(HiatusPoint2_Array[0], HiatusPoint2_Array[1], HiatusPoint2_Array[2])
    
#    Getting the nodes that are the closest to the desired point
#    Currently only getting the closest
#    This was done so that we did not need to change the code if the tissue
#    was remeshed
    nodes, PassingNodeCoordinates = getXClosestNodes(HiatusPoint2, 1, HiatusMaterial2, GenericINPFile)

    distance_x = []
    distance_y = []
    distance_z = []
    Coeff_x = []
    Coeff_y = []
    Coeff_z = []
    Total_Coeff_x = 0
    Total_Coeff_y = 0
    Total_Coeff_z = 0
    power = 50

#    If more than one point was obtained, we need to generate coefficients to
#    know how to weight the points...since we just have 1 point this doesn't matter
    for b in range(0,len(PassingNodeCoordinates)):
        Coeff_x.append(1/((PassingNodeCoordinates[b].x-HiatusPoint2.x)**2)**power)
        Coeff_y.append(1/((PassingNodeCoordinates[b].y-HiatusPoint2.y)**2)**power)
        Coeff_z.append(1/((PassingNodeCoordinates[b].z-HiatusPoint2.z)**2)**power)
        Total_Coeff_x += Coeff_x[b]
        Total_Coeff_y += Coeff_y[b]
        Total_Coeff_z += Coeff_z[b]
    Scaled_Coeff_x=[]
    Scaled_Coeff_y=[]
    Scaled_Coeff_z=[]
    for b in range(0,len(PassingNodeCoordinates)):
        Scaled_Coeff_x.append(Coeff_x[b]/Total_Coeff_x)
        Scaled_Coeff_y.append(Coeff_y[b]/Total_Coeff_y)
        Scaled_Coeff_z.append(Coeff_z[b]/Total_Coeff_z)

#####    Switching from node index to node number (I think)
    for b in range (0,len(nodes)):
        nodes[b] += 1

#    OutputFileName = ntpath.basename(OutputFileName)

    PassingNodes = ','.join(str(i) for i in nodes)
    Variable1 = "COORD"
    Headerflag = 'N'
    NewFileFlag = 'Y'
    Frames = 'all'
    MaterialName = HiatusMaterial2.upper() + '-1'
    Temp_DataFileName = 'Temp_Data_File'
    DataFileName = Temp_DataFileName
    Temp_ODBFile = 'Temp_ODB'
#    Original_ODBFile = ODBFile
    original_odb = Results_Folder_Location + '\\' + ODBFile + '.odb'
    new_file = Results_Folder_Location + '\\' + Temp_ODBFile + '.odb'
    
    Results_Folder_Name = os.path.basename(Results_Folder_Location)
    
    print(os.path.basename(Results_Folder_Location))
    print(Results_Folder_Name)
#    print(original_odb.replace('\\','\\\\'))
#    print(new_file.replace('\\','\\\\'))
##file_to_analyze.replace('\\','\\\\')
#    print(Results_Folder_Location + '\\' + ODBFile + '.odb')
#    print(Results_Folder_Location + '\\' + Temp_ODBFile + '.odb')
    
#    shutil.copy('.\\Results_0129' + '\\' + ODBFile + '.odb', '.\\Results_0129' + '\\' + Temp_ODBFile + '.odb')



#### Not sure why this one isnt' working
#    shutil.copy('.\\' + Results_Folder_Name + '\\' + ODBFile + '.odb', Temp_ODBFile + '.odb')
    print('Results_Folder_Name', Results_Folder_Name)
    print('ODBFile', ODBFile)
    print('Temp_ODBFile', Temp_ODBFile)
    shutil.copy('.\\' + Results_Folder_Name + '\\' + ODBFile + '.odb', Temp_ODBFile + '.odb')




#    shutil.copy(Results_Folder_Location + '\\' + ODBFile + '.odb', Temp_ODBFile + '.odb')    
    ODBFile = Temp_ODBFile
    ODBFile = 'Temp_ODB'
#    CallString = AbaqusBatLocation + '  CAE noGUI=ODBMechensFunction_v2  -- -odbfilename ' + ODBFile + ' -partname ' + MaterialName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile ' + DataFileName + ' -headerflag ' + Headerflag + " -newfileflag " + NewFileFlag + " -frames " + Frames
    CallString = AbaqusBatLocation + '  CAE noGUI=".\lib\Get_Data_From_ODB"  -- -odbfilename "' + ODBFile + '" -partname ' + MaterialName + ' -strNodes ' + PassingNodes + ' -var1 ' + Variable1 + ' -outputfile "' + DataFileName + '" -headerflag ' + Headerflag + " -newfileflag " + NewFileFlag + " -frames " + Frames
    print('------------------------------------------')
    print(CallString)
    print('------------------------------------------')
    subprocess.call(CallString)
#    
#    DataFileName = 'Temp_Data
#    
#    shutil.copy(Temp_DataFileName + '.csv', DataFileName + '.csv')
    
#    HiatusOutputFile = open('.\\Results_0129\\' + OutputFileName + '.csv','w')    
    
    
    
    
    Coordinates = genfromtxt(Temp_DataFileName+'.csv', delimiter=',')
    if len(Coordinates) > 1:
        for i in range(0,len(Coordinates[:,0])):
            print('i:', i)
            HP2x = 0
            HP2y = 0
            HP2z = 0
            for b in range (0,len(nodes)):
                HP2x += Scaled_Coeff_x[b]*Coordinates[i,b*3]
                HP2y += Scaled_Coeff_y[b]*Coordinates[i,b*3+1]
                HP2z += Scaled_Coeff_z[b]*Coordinates[i,b*3+2]
            HiatusDistance = ((HiatusPoint1.x-HP2x)**2+(HiatusPoint1.y-HP2y)**2+(HiatusPoint1.z-HP2z)**2)**.5
#            HiatusOutputFile.write('%10.9E, ' % (HiatusDistance))
#            HiatusOutputFile.write('\n')
    try:      
        os.remove(DataFileName+'.csv')
        print(Results_Folder_Location + '\\' +Temp_ODBFile + '.odb')
        os.remove(Results_Folder_Location + '\\' +Temp_ODBFile + '.odb')
        os.remove(Temp_ODBFile + '.odb')
    except OSError:
        pass
    print(HiatusDistance)
    return(HiatusDistance)