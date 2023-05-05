# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:23:27 2020

@author: DeLancey
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 19:56:24 2017

@author: mgordon
"""


from lib.FiberFunctions import CurveFibersInINP, CurvePARAFibersInINP, getFiberLengths
from lib.INP_Material_Properties import INPAnalogMaterialProperties, RemoveMaterialInsertion
import shutil
import lib.Scaling as Scaling
from lib.workingWith3dDataSets import Point
import lib.IOfunctions as io
from lib.Surface_Tools import optimize_rot_angle, scale_PM_Mid_Width, plot_Dataset
from lib.Node_Distances import setPBodyClosest
import numpy as np
import configparser
import os
import time
import re
import glob

'''
Function: same_geometry_file
'''
def same_geometry_file(OutputINPFile, Results_Folder_Location):

    current_file_name = OutputINPFile
    location = Results_Folder_Location
    
    Split_Array = re.split('(\d+\.\d+|-?\d+)|_|-', current_file_name)
    
    original_file_array = [i for i in Split_Array if i]
    print(original_file_array)
    
    Geometry_Codes = ['CLSS', 'AVWL', 'HL', 'AVWW', 'AS']
    
    Orig_GenericFileCode = current_file_name.split('Gen')[1][0]
    same_geometry_file = ''
    
    for file in glob.glob(location + '\*.inp'):
        try:
            file_name = os.path.split(file)[1]
            new_GenericFileCode = file_name.split('Gen')[1][0]
            Split_Array = re.split('(\d+\.\d+|-?\d+)|_|-', file_name)
            new_file_array = [i for i in Split_Array if i]
                       
            if Orig_GenericFileCode == new_GenericFileCode:
                match = 1
            else:
                match = 0
            for code in Geometry_Codes:
                Header_Index = original_file_array.index(code)
                Data_Index = Header_Index + 1
                original_geometry = original_file_array[Data_Index]
                Header_Index = new_file_array.index(code)
                Data_Index = Header_Index + 1
                new_geometry = new_file_array[Data_Index]
                if new_geometry == original_geometry:
                    pass
                else:
                    match = 0
                    break
            if match == 1:
                same_geometry_file = file
                print('Found the same geometry. Saving Time!')
                break
        except:
            pass    
    return(same_geometry_file)
start = time.process_time()

'''
Function: AnalogGenerateINP
'''
def AnalogGenerateINP(TissueParameters, MaterialStartLine, LoadLine, LoadLineNo, SlackStrain, DensityFactor, GenericINPFile, OutputINPFile, WidthScale, LengthScale, ZSHIFT, RotationPoint, HiatusPoint, GIFillerPoint, HiatusLength, Results_Folder_Location):

#    GenericINPFile = 'Uncurved_AVW.inp'
    
    config = configparser.ConfigParser()
    config.sections()
    config.read("Parameters.ini")

    # SURFACES
    AVW         = "OPAL325_AVW_v6"
    GI_FILLER   = "OPAL325_GIfiller"
    ATFP        = "OPAL325_ATFP"
    ATLA        = "OPAL325_ATLA"
    LA          = "OPAL325_LA"
    PBODY       = "OPAL325_PBody"
    PM_MID      = "OPAL325_PM_mid"
    
    # FIBERS
    CL          = "OPAL325_CL_v6"
    PARA        = "OPAL325_Para_v6"
    US          = "OPAL325_US_v6"
    
    shortcut_file = same_geometry_file(OutputINPFile, Results_Folder_Location)
    
    if shortcut_file != '':
        print("Creating Correct Stress Strain Data Curves")
        # Compensate for the fact that the final fiber lengths aren't necessarily the correct length due to geometrical constraints
        # Instead we will compensate by shifting the stress / strain curve to account for pre-loading
        ApicalSupportTissues = [CL, US, PARA]
    
        # Find the length of the fibers both originally and after the move
        oldFiberLengths = np.array(getFiberLengths(GenericINPFile, ApicalSupportTissues))
        newFiberLengths = np.array(getFiberLengths(shortcut_file, ApicalSupportTissues))
        
        # Calculate the mean of those fiber lengths per tissue
        oldFiberLengthMeans = []
        newFiberLengthMeans = []
        for b, value in enumerate(ApicalSupportTissues):
    
            oldFiberLengthMeans.append(np.mean(oldFiberLengths[b-1]))
            newFiberLengthMeans.append(np.mean(newFiberLengths[b-1]))
    
        # Calculate the length that each tissue should be at by multiplying the original length by 1 + Strain value (probably should have a different name)
        IdealNewFiberLengthMeans = np.multiply(np.array(oldFiberLengthMeans),(1 + np.array(SlackStrain)))
        
        # How much longer is the tissue than it is supposed to be
        PrestretchAmount = newFiberLengthMeans - IdealNewFiberLengthMeans
        
    
        # Create the pre-stretch coefficient for each fiber by doing the pre-stretch / (pre-stretch + length)
        if config.getint("FLAGS", "prestrain_fibers") != 0:
            StretchCoefficients = np.divide(PrestretchAmount,np.array(IdealNewFiberLengthMeans+PrestretchAmount))*-1
        else:
            StretchCoefficients = np.zeros(len(PrestretchAmount))
                    
        # Removing MaterialStartLine and moving it to the Materials file
        shutil.copy(shortcut_file, OutputINPFile)
        RemoveMaterialInsertion(OutputINPFile, GenericINPFile)
        INPAnalogMaterialProperties(TissueParameters, DensityFactor, LoadLine, LoadLineNo, OutputINPFile, StretchCoefficients)
    else:    
 
#        try:
#            os.remove(OutputINPFile)
#        except:
#            pass
#        shutil.copy(GenericINPFile, OutputINPFile)
        
        rot_point =  Point(RotationPoint[0], RotationPoint[1], RotationPoint[2])
        hiatus = Point(HiatusPoint[0], HiatusPoint[1], HiatusPoint[2])
        gifiller = Point(GIFillerPoint[0], GIFillerPoint[1], GIFillerPoint[2])
        
        z_cutoff = hiatus.z  # starting point to optimise where to drop the avw from
    
    
        setPBodyClosest(PBODY, gifiller, GenericINPFile, OutputINPFile)
        
        pm = io.get_dataset_from_file(OutputINPFile, PM_MID)
    
    ################################################################
    ##    Scale the width of the PM_Mid
        connections = io.get_interconnections(GenericINPFile, PM_MID)
    #    pm_new = scale_PM_Mid_Width(pm, connections, 1.5)
    #    plot_Dataset(pm, pm_new)
    ###################################################################
    
    
#    ###############################################################################    
#    # Shift the apex down and lengthen the AVW
#        Scaling.lengthen_and_shift_part(AVW, OutputINPFile, LengthScale, ZSHIFT)
#    ###############################################################################
#        
#    ###############################################################################
#    # Increase the AVW Width
#        Scaling.widen_part(AVW, OutputINPFile, WidthScale)
#    ###############################################################################
        
    ######################################################################################
    # Rotate tissues to obtain the correct hiatus distance
    ######################################################################################
        # Find the angle that it needs to rotate through for the desired hiatus distance
        rotate_angle = optimize_rot_angle(HiatusLength, rot_point, hiatus, gifiller)
        
        # Rotate the tissues the correct amount
        print("Skpping Rotating parts")
        Scaling.rotate_part(AVW, OutputINPFile, rotate_angle, rot_point)
        print('1')
        gi        = Scaling.rotate_part(GI_FILLER, OutputINPFile, rotate_angle, rot_point)
        print('2')
#        aftp      = Scaling.rotate_part(ATFP, OutputINPFile, rotate_angle, rot_point)
#        print('3')
#        atla      = Scaling.rotate_part(ATLA, OutputINPFile, rotate_angle, rot_point)
#        print('4')
#        la        = Scaling.rotate_part(LA, OutputINPFile, rotate_angle, rot_point)
#        print('5')
        pbody     = Scaling.rotate_part(PBODY, OutputINPFile, rotate_angle, rot_point)
        print('6')
        pm_mid    = Scaling.rotate_part(PM_MID, OutputINPFile, rotate_angle, rot_point)
        print('7')
        ######################################################################################
        
    ######################################################################################
    # Droop the AVW (probably pass the generic file and obtain the end points for the droop from there)
        if config.getint("FLAGS", "CurveAVW") != 0:
            avw = Scaling.curve_avw(AVW, OutputINPFile, GenericINPFile, hiatus, z_cutoff, rotate_angle, rot_point, HiatusLength)
    ######################################################################################
            
    ######################################################################################

    # Need to put a wave in the bottom of the AVW so that it doesn't hit the PM_mid tissue
        if config.getint("FLAGS", "distal_AVW") != 0:
            avw = Scaling.narrow_distal_avw(AVW, OutputINPFile, GenericINPFile)
#    #        print(avw)
            
            
        pm = io.get_dataset_from_file(OutputINPFile, PM_MID)
    
        print("Adjusting CL")
    # Adjust the CL tissue
        TempFile = "Temp.inp"
        try:
            os.remove(TempFile)
        except:
            pass
        shutil.copy(OutputINPFile, TempFile)
    # Sets the vector for adding the wave to the fiber to increase length
        dirVector = [1,0,0]
        CurveFibersInINP("OPAL325_AVW_v6", "OPAL325_CL_v6", SlackStrain[0], TempFile, OutputINPFile, dirVector)
        
        print(time.process_time() - start)
        
        print("Adjusting US")
    # Adjust the US tissue
        os.remove(TempFile)
        shutil.copy(OutputINPFile, TempFile)
    # Sets the vector for adding the wave to the fiber to increase length
        dirVector = [1,0,0]
        CurveFibersInINP("OPAL325_AVW_v6", "OPAL325_US_v6", SlackStrain[1], TempFile, OutputINPFile, dirVector)
    
        print(time.process_time() - start)
    
        time.sleep(3)

        
        print("Adjusting Para")
    ## Adjust the Paravaginal tissue
        os.remove(TempFile)
        if config.getint("FLAGS", "curve_PARA") != 0:
            CurvePARAFibersInINP("OPAL325_AVW_v6", "OPAL325_Para_v6", SlackStrain[2], TempFile, OutputINPFile, dirVector, pm, connections)
    #        time.sleep(3)
    #        print("Adjusting Para...Again")
    #        os.remove(TempFile)
    #        time.sleep(3)
    #        shutil.copy(OutputINPFile, TempFile)
    #        time.sleep(3)
    #        CurvePARAFibersInINP("OPAL325_AVW_v6", "OPAL325_Para_v6", 0, TempFile, OutputINPFile, dirVector, pm, connections)

        else:
            shutil.copy(OutputINPFile, TempFile)
            # Sets the vector for adding the wave to the fiber to increase length
            dirVector = [1,0,0]
            CurveFibersInINP("OPAL325_AVW_v6", "OPAL325_Para_v6", SlackStrain[2], TempFile, OutputINPFile, dirVector)
        

        print("Creating Correct Stress Strain Data Curves")
        # Compensate for the fact that the final fiber lengths aren't necessarily the correct length due to geometrical constraints
        # Instead we will compensate by shifting the stress / strain curve to account for pre-loading
        ApicalSupportTissues = [CL, US, PARA]
    
        # Find the length of the fibers both originally and after the move
        oldFiberLengths = np.array(getFiberLengths(GenericINPFile, ApicalSupportTissues))
        newFiberLengths = np.array(getFiberLengths(OutputINPFile, ApicalSupportTissues))
        
        # Calculate the mean of those fiber lengths per tissue
        oldFiberLengthMeans = []
        newFiberLengthMeans = []
        for b, value in enumerate(ApicalSupportTissues):
    
            oldFiberLengthMeans.append(np.mean(oldFiberLengths[b-1]))
            newFiberLengthMeans.append(np.mean(newFiberLengths[b-1]))
            
        # Calculate the length that each tissue should be at by multiplying the original length by 1 + Strain value (probably should have a different name)
        IdealNewFiberLengthMeans = np.multiply(np.array(oldFiberLengthMeans),(1 + np.array(SlackStrain)))
        print('new ideal: ', IdealNewFiberLengthMeans)
        
        # How much longer is the tissue than it is supposed to be
        PrestretchAmount = newFiberLengthMeans - IdealNewFiberLengthMeans
    
            # Create the pre-stretch coefficient for each fiber by doing the pre-stretch / (pre-stretch + length)
        if config.getint("FLAGS", "prestrain_fibers") != 0:
            StretchCoefficients = np.divide(PrestretchAmount,np.array(IdealNewFiberLengthMeans+PrestretchAmount))*-1
        else:
            StretchCoefficients = np.zeros(len(PrestretchAmount))
        
        
        # Removing MaterialStartLine and moving it to the Materials file
        INPAnalogMaterialProperties(TissueParameters, DensityFactor, LoadLine, LoadLineNo, OutputINPFile, StretchCoefficients)
        
        MeasurementsFileName = OutputINPFile + '_Measurements.txt'
        Scaling.takeMeasurements(MeasurementsFileName, AVW, [CL, PARA, US], GenericINPFile, OutputINPFile)
