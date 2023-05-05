# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 19:56:24 2017

@author: mgordon
"""


import sys
import fileinput
import shutil
import numpy as np
from lib.IOfunctions import findLineNum


'''
Function: INPAnalogMaterialProperties

Adding in an array (MaterialPrestretch that will have the additional length (or strain...not sure yet) of the different tissues)
'''
def INPAnalogMaterialProperties(TissueParameters, DensityFactor, LoadLine, LoadLineNo, OutputINPFile, MaterialPrestretchCoefficients):
    
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ OUTPUT FILE NAME")
    print(OutputINPFile)
    
    # Material values when TissueParameters = 100    
    CL_MaterialStrain = np.array([0.033741533, 0.077686123, 0.134919058, 0.209458583, 0.306538014, 0.432973163, 0.597640866, 0.812102213, 1.091414237, 1.455187045])
    US_MaterialStrain = np.array([0.026438298, 0.056788818, 0.091630473, 0.131627841, 0.177543842, 0.230254288, 0.290764591, 0.36022894, 0.439972314, 0.531515757])
#    CLUS_MaterialStrain = np.array([0.00268578, 0.0055405, 0.00857477, 0.0117999, 0.0152279, 0.0188715, 0.0227443, 0.0268606, 0.0312359, 0.0358864])

    # PARAVAG_MaterialStrain taken from CLUS_H100 as per early files
    PARAVAG_MaterialStrain = np.array([0.033741533, 0.077686123, 0.134919058, 0.209458583, 0.306538014, 0.432973163, 0.597640866, 0.812102213, 1.091414237, 1.455187045])
    PCM_Hyper_MaterialStrain = np.array([0.0138501, 0.0318666, 0.055303, 0.0857897, 0.125448, 0.177036, 0.244143, 0.331437, 0.444993, 0.592708])
    AVW_MaterialStrain = np.array([0.0138501, 0.0318666, 0.055303, 0.0857897, 0.125448, 0.177036, 0.244143, 0.331437, 0.444993, 0.592708])
    ATFP_HYPER_MaterialStrain = np.array([0.1, 3.18666, 5.5303, 8.57895, 12.5448, 17.7036, 24.4143, 33.1437, 44.4993, 59.271])
    AWP_dis_HYPER_highdensity_Analog_MaterialStrain = np.array([0.1, 3.18666, 5.5303, 8.57895, 12.5448, 17.7036, 24.4143, 33.1437, 44.4993, 59.271])
    ICM_HYPER_Analog_MaterialStrain = np.array([0.0138501, 0.0318666, 0.055303, 0.0857897, 0.125448, 0.177036, 0.244143, 0.331437, 0.444993, 0.592708])
    PRM_HYPER_Analog_MaterialStrain = ICM_HYPER_Analog_MaterialStrain
    Pbody_Analog_MaterialStrain = np.array([0.0268578, 0.00268578, 0.0055405, 0.00857478, 0.0117999, 0.0152279, 0.0188715, 0.0227443, 0.0268606, 0.0312359])
    PM_mid_Analog_MaterialStrain = Pbody_Analog_MaterialStrain
    PARAVAG_dis_HYPER_Analog_MaterialStrain = ATFP_HYPER_MaterialStrain
    
    
    
    #Define what the tissue parameters should be
    # For now, AWP should be scaled by AVW, Pbody and PM_mid should get their own scales, Parvag gets PARA
    CL_material = TissueParameters[0]
    PARA_material = TissueParameters[1]
    PCM_material = TissueParameters[2] #PCM Hyper Analog
    AVW_material = TissueParameters[3]
    Pbody_material = TissueParameters[4]
    PM_mid_material = TissueParameters[5]

    print("PCM_Material in INP Mat Prop:", PCM_material)


    # if the coefficeint is positive (there is no prestretch), then set the coefficent equal to 0 to do nothing
    CL_Coefficient = min(MaterialPrestretchCoefficients[0],0)
    US_Coefficient = min(MaterialPrestretchCoefficients[1],0)
    PARA_Coefficient = min(MaterialPrestretchCoefficients[2],0)

#    print('************************', PARA_Coefficient)
    
    MaterialStartLine = findLineNum(OutputINPFile, "** MATERIALS") + 2
    
    #Insert the Analog material properties into the INP file under Materials
    for i, line in enumerate(fileinput.input(OutputINPFile, inplace=1)):
        if i == MaterialStartLine-1:
            print('*Material, name=CL_Analog')
            CLDensity = DensityFactor*0.0000009
            print('*Density')
            DensityLine = ' ' + str(CLDensity) + ','
            print(DensityLine)
            print('*Hyperelastic, n=2, test data input')
            print('*Uniaxial Test Data')
            # adding a point corresponding to original 0 stress being 0 strain
            print(str(max(0,(0+CL_Coefficient)*CL_material/100))+', '+ '0.001')
            for k in range(0,10):
                print(str(max(float(k+1)/50,(PARA_Coefficient+(1+PARA_Coefficient)*PARAVAG_MaterialStrain[k])*PARA_material/100))+', '+str(float(k+1)/10))
#                if k > 0:
#                    print(str(max(0.05,float(k+1)/50,(PARA_Coefficient+(1+PARA_Coefficient)*PARAVAG_MaterialStrain[k])*PARA_material/100))+', '+str(float(k+1)/10))
#                else:
#                    print(str(max(float(k+1)/50,(PARA_Coefficient+(1+PARA_Coefficient)*PARAVAG_MaterialStrain[k])*PARA_material/100))+', '+str(float(k+1)/10))
            print('1e-05, -0.1')
            print('1e-05, -0.2')
            print('1e-05, -0.3')
            print('1e-05, -0.4')
            print('1e-05, -0.5')
            print('1e-05, -0.6')
            print('1e-05, -0.7')
            print('1e-05, -0.8')
            print('1e-05, -0.9')
#                print(str(max(float(k+1)/50,(CL_Coefficient+(1+CL_Coefficient)*CL_MaterialStrain[k])*CL_material/100))+', '+str(float(k+1)/10))
            print('*Material, name=US_Analog')
            print('*Density')
            print(' 9e-07,')
            print('*Hyperelastic, n=2, test data input')
            print('*Uniaxial Test Data')
            # adding a point corresponding to original 0 stress being 0 strain
            print(str(max(0,(0+US_Coefficient)*CL_material/100))+', '+ '0.001')
            for k in range(0,10):
                ######################## The line below uses CLmaterail instead of USmaterial
                # I believe because we were changing them at the same time for simplicity's sake
                print(str(max(float(k+1)/50,(US_Coefficient+(1+US_Coefficient)*US_MaterialStrain[k])*CL_material/100))+', '+str(float(k+1)/10))
#            for k in range(0,10):
#                print(str(US_MaterialStrain[k]*CLmaterial/100)+', '+str(float(k+1)/10))
            print('1e-05, -0.1')
            print('1e-05, -0.2')
            print('1e-05, -0.3')
            print('1e-05, -0.4')
            print('1e-05, -0.5')
            print('1e-05, -0.6')
            print('1e-05, -0.7')
            print('1e-05, -0.8')
            print('1e-05, -0.9')
            
            print('*Material, name=PCM_Hyper_Analog')
            print('*Density')
            print(' 1.1e-06,')
            print('*Hyperelastic, n=2, test data input')
            print('*Uniaxial Test Data')
            for k in range(0,10):
                print(str(PCM_Hyper_MaterialStrain[k]*PCM_material/100)+', '+str(float(k+1)/10))
            print('*Material, name=PARAVAG_H_Analog')
            print('*Density')
            print(' 9e-07,')
            print('*Hyperelastic, n=2, test data input')
            print('*Uniaxial Test Data')
            # adding a point corresponding to original 0 stress being 0 strain
            print(str(max(0,(0+PARA_Coefficient)*PARA_material/100))+', '+ '0.001')
#            new_min_used = 0
            for k in range(0,10):
                print(str(max(float(k+1)/50,(PARA_Coefficient+(1+PARA_Coefficient)*PARAVAG_MaterialStrain[k])*PARA_material/100))+', '+str(float(k+1)/10))
#                if k > 3:
#                    print(str(max(0.16,float(k+1)/50,(PARA_Coefficient+(1+PARA_Coefficient)*PARAVAG_MaterialStrain[k])*PARA_material/100))+', '+str(float(k+1)/10))
#                else:
#                    print(str(max(float(k+1)/50,(PARA_Coefficient+(1+PARA_Coefficient)*PARAVAG_MaterialStrain[k])*PARA_material/100))+', '+str(float(k+1)/10))
            print('1e-05, -0.1')
            print('1e-05, -0.2')
            print('1e-05, -0.3')
            print('1e-05, -0.4')
            print('1e-05, -0.5')
            print('1e-05, -0.6')
            print('1e-05, -0.7')
            print('1e-05, -0.8')
            print('1e-05, -0.9')
#                if float(k+1)/50 > (PARA_Coefficient+(1+PARA_Coefficient)*PARAVAG_MaterialStrain[k])*PARA_material/100:
#                    new_min_used = 1
#                    print(str(max(float(k+1)/50,(PARA_Coefficient+(1+PARA_Coefficient)*PARAVAG_MaterialStrain[k])*PARA_material/100))+', '+str(float(k+1)/10))
#                else:
#                    if new_min_used == 1:
#                        print(str(max(float(k+1)/50,((PARA_Coefficient+(1+PARA_Coefficient)*PARAVAG_MaterialStrain[k+1])*PARA_material/100)*2/3))+', '+str(float(k+1)/10))
#                        new_min_used = 0
#                    else:
#                        print(str(max(float(k+1)/50,(PARA_Coefficient+(1+PARA_Coefficient)*PARAVAG_MaterialStrain[k])*PARA_material/100))+', '+str(float(k+1)/10))
#            for k in range(0,10):
#                print(str(PARAVAG_MaterialStrain[k]*PARAmaterial/100)+', '+str(float(k+1)/10))
            print('*Material, name=AVW_HYPER_Analog')
            print('*Density')
            print(' 1.1e-06,')
            print('*Hyperelastic, n=2, test data input')
            print('*Uniaxial Test Data')
            for k in range(0,10):
                print(str(AVW_MaterialStrain[k]*AVW_material/100)+', '+str(float(k+1)/10))
#            print('**')

            print('1e-05, -0.1')
            print('1e-05, -0.2')
            print('1e-05, -0.3')
            print('1e-05, -0.4')
            print('1e-05, -0.5')
            print('1e-05, -0.6')
            print('1e-05, -0.7')
            print('1e-05, -0.8')
            print('1e-05, -0.9')
#################################################################
######### New for more materials#################################
            print('*Material, name=ICM_HYPER_Analog')
            print('*Density')
            print(' 1.1e-06,')
            print('*Hyperelastic, n=2, test data input')
            print('*Uniaxial Test Data')
            for k in range(0,10):
                print(str(ICM_HYPER_Analog_MaterialStrain[k]*PCM_material/100)+', '+str(float(k+1)/10))
            print('*Material, name=PRM_HYPER_Analog')
            print('*Density')
            print(' 1.1e-06,')
            print('*Hyperelastic, n=2, test data input')
            print('*Uniaxial Test Data')
            for k in range(0,10):
                print(str(PRM_HYPER_Analog_MaterialStrain[k]*PCM_material/100)+', '+str(float(k+1)/10))
            print('*Material, name=Pbody_Analog')
            print('*Density')
            print(' 9e-07,')
            print('*Hyperelastic, n=2, test data input')
            print('*Uniaxial Test Data')
            for k in range(0,10):
                print(str(Pbody_Analog_MaterialStrain[k]*Pbody_material/100)+', '+str(float(k+1)/10))
            print('*Material, name=PM_mid_Analog')
            print('*Density')
            print(' 9e-07,')
            print('*Hyperelastic, n=2, test data input')
            print('*Uniaxial Test Data')
            for k in range(0,10):
                print(str(PM_mid_Analog_MaterialStrain[k]*Pbody_material/100)+', '+str(float(k+1)/10))
            print('*Material, name=PARAVAG_dis_HYPER_Analog')
            print('*Density')
            print(' 1.1e-05,')
            print('*Hyperelastic, n=2, test data input')
            print('*Uniaxial Test Data')
            for k in range(0,10):
                print(str(PARAVAG_dis_HYPER_Analog_MaterialStrain[k]*PARA_material/100)+', '+str(float(k+1)/10))            
            print('*Material, name=AWP_dis_HYPER_highdensity_Analog')
            print('*Density')
            print(' 1.1e-04,')
            print('*Hyperelastic, n=2, test data input')
            print('*Uniaxial Test Data')
            for k in range(0,10):
                print(str(AWP_dis_HYPER_highdensity_Analog_MaterialStrain[k]*PARA_material/100)+', '+str(float(k+1)/10))            
###################################################################################################                
                
            sys.stdout.write(line)
        elif i == LoadLineNo - 1:
            print(LoadLine)
        else:
            sys.stdout.write(line)
            
            
'''
Function: RemoveMaterialInsertion
'''
def RemoveMaterialInsertion(OutputINPFile, GenericINPFile):

#    The line where the material that needs to be removed currently starts
    MaterialStartLine = findLineNum(OutputINPFile, "** MATERIALS") + 2

#   The line where the material started in the Generic INP file
#    Find the ** Materials section, move 2 lines down, adjust the number by 1 because it is base 0
    PreviousMaterialStartLine = findLineNum(GenericINPFile, "** MATERIALS") + 2 - 1
    
    print(PreviousMaterialStartLine)
    
    pms_set = [PreviousMaterialStartLine]
    
#    find the text from the start of the material in the Generic INP file
    with open(GenericINPFile) as file:
        first_generic_material = [x for i, x in enumerate(file) if i in pms_set][0]

    print(first_generic_material)
    first_generic_material.rstrip()
    print(first_generic_material)
    
    
    
#    first_generic_material = '*Density'

    PreviousMaterialStartLine = findLineNum(OutputINPFile, first_generic_material.rstrip())
    
    print(PreviousMaterialStartLine)
    
    #Insert the Analog material properties into the INP file under Materials
    for i, line in enumerate(fileinput.input(OutputINPFile, inplace=1)):
#        print(i)
        if MaterialStartLine - 1 <= i <= PreviousMaterialStartLine - 2:
#            print('blank')
            pass
        else:
            sys.stdout.write(line)