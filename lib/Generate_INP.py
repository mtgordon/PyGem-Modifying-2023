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
from scipy.interpolate import Rbf, interp1d
import lib.Surface_Tools_Circle as ss
from lib.Surface_Tools import find_starting_ending_points_for_inside, findInnerNodes
from pygem import RBF, IDW
import matplotlib.pyplot as plt
import lib.Shape_Analysis
import pandas as pd

'''
Function: same_geometry_file
'''
def same_geometry_file(OutputINPFile, Results_Folder_Location):
    #TODO: A temporary flag to switch between using the file rename code and original code
    fileRenameFlag = True

    if fileRenameFlag:
        logFile = Results_Folder_Location + '\\' + os.path.splitext(OutputINPFile)[0] + '_log.csv'
        log_df = pd.read_csv(logFile)
        log_df = log_df.set_index('Property')
        log_df_codes = log_df.set_index('Code')

    current_file_name = OutputINPFile
    location = Results_Folder_Location

    Split_Array = re.split('(\d+\.\d+|-?\d+)|_|-', current_file_name)

    original_file_array = [i for i in Split_Array if i]
    # print(original_file_array)

    Geometry_Codes = ['CLSS', 'AVWL', 'HL', 'AVWW', 'AS', 'FLP', 'SLP', 'FICM', 'SICM']

    if fileRenameFlag:
        Orig_GenericFileCode = log_df.loc['Generic_File', 'Code']
    else:
        Orig_GenericFileCode = current_file_name.split('Gen')[1][0]

    same_geometry_file = ''

    for file in glob.glob(location + '\*.inp'):
        try:
            if fileRenameFlag:
                current_log = os.path.splitext(file)[0] + '_log.csv'
                current_df = pd.read_csv(current_log)
                current_df = current_df.set_index('Property')
                new_GenericFileCode = current_df.loc['Generic_File', 'Code']
                current_df_codes = current_df.set_index('Code')

            else:
                file_name = os.path.split(file)[1]
                new_GenericFileCode = file_name.split('Gen')[1][0]
                Split_Array = re.split('(\d+\.\d+|-?\d+)|_|-', file_name)
                new_file_array = [i for i in Split_Array if i]

            if Orig_GenericFileCode == new_GenericFileCode:
                match = 1
            else:
                match = 0
            for code in Geometry_Codes:
                if fileRenameFlag:
                    original_geometry = float(log_df_codes.loc[code, 'Value'])
                    new_geometry = float(current_df_codes.loc[code, 'Value'])
                else:
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
                print('Found the same geometry. Saving Time! File = ', file)
                break
        except:
            pass
    return(same_geometry_file)
start = time.process_time()

'''
Function: AnalogGenerateINP
'''
def AnalogGenerateINP(TissueParameters, MaterialStartLine, LoadLine, LoadLineNo, SlackStrain, DensityFactor,
                      GenericINPFile, OutputINPFile, WidthScale, LengthScale, ZSHIFT, RotationPoint, HiatusPoint,
                      GIFillerPoint, HiatusLength, levator_plate_PC1, levator_plate_PC2, ICM_PC1, ICM_PC2,
                      positive_CL_point, negative_CL_point, positive_US_point, negative_US_point,
                      positive_PARA_point, negative_PARA_point, positive_CL_remove_percent, negative_remove_percent,
                      mid_x, threshold_x_from_mid, generate_scar_tissue_flag, Results_Folder_Location):

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
        rot_point =  Point(RotationPoint[0], RotationPoint[1], RotationPoint[2])
        hiatus = Point(HiatusPoint[0], HiatusPoint[1], HiatusPoint[2])
        gifiller = Point(GIFillerPoint[0], GIFillerPoint[1], GIFillerPoint[2])

        z_cutoff = hiatus.z  # starting point to optimise where to drop the avw from


        # the GIFillerPoint is a point(set of coordinates) that is defined in the
        # INI file
        # moving the closest node of the PBODY to the exact location of the
        # GI Filler Point
        setPBodyClosest(PBODY, gifiller, GenericINPFile, OutputINPFile)

        pm = io.get_dataset_from_file(OutputINPFile, PM_MID)

    ################################################################
    ##    Scale the width of the PM_Mid
        connections = io.get_interconnections(GenericINPFile, PM_MID)
    #    pm_new = scale_PM_Mid_Width(pm, connections, 1.5)
    #    plot_Dataset(pm, pm_new)
    ###################################################################


    ###############################################################################
    # Shift the apex down and lengthen the AVW
        Scaling.lengthen_and_shift_part(AVW, OutputINPFile, LengthScale, ZSHIFT)
    ###############################################################################

    ###############################################################################
    # Increase the AVW Width
        Scaling.widen_part(AVW, OutputINPFile, WidthScale)
    ###############################################################################

    ######################################################################################
    # Rotate tissues to obtain the correct hiatus distance
    # Switching this to using PyGem
    #
    ######################################################################################
        # Find the angle that it needs to rotate through for the desired hiatus distance
        rotate_angle = optimize_rot_angle(HiatusLength, rot_point, hiatus, gifiller)

        print("Rotating parts")
        # first get the intial control points (CPs)
        # (PM BCs, LA BCs, Hiatus Point on GI Filler, PM inner arch, grid on AVW)

        # PM BCs
        Xs = []
        Ys = []
        Zs = []


        np_points = np.array(io.extractPointsForPartFrom(GenericINPFile, 'OPAL325_PM_mid'))
        Bnodes = ss.getBnodes(GenericINPFile, 'OPAL325_PM_mid')

        for i in Bnodes:
            Xs.append(np_points[i-1,0])
            Ys.append(np_points[i-1,1])
            Zs.append(np_points[i-1,2])

        PM_boundary_CPs = np.c_[Xs,Ys,Zs]

        lib.Shape_Analysis.checkPlot('PM Boundary Initial', PM_boundary_CPs, PM_boundary_CPs)

        # LA BCs
        Xs = []
        Ys = []
        Zs = []

        np_points = np.array(io.extractPointsForPartFrom(GenericINPFile, LA))
        Bnodes = ss.getBnodes(GenericINPFile, LA)

        for i in Bnodes:
            Xs.append(np_points[i-1,0])
            Ys.append(np_points[i-1,1])
            Zs.append(np_points[i-1,2])

        LA_boundary_CPs = np.c_[Xs,Ys,Zs]

        lib.Shape_Analysis.checkPlot('LA Boundary Initial', LA_boundary_CPs, LA_boundary_CPs)

        #TODO: Create ATFP boundary CPs, using raw np_points
        # ATFP BCs
        Xs = []
        Ys = []
        Zs = []

        np_points = np.array(io.extractPointsForPartFrom(GenericINPFile, ATFP))

        for i in range(len(np_points)):
            Xs.append(np_points[i - 1, 0])
            Ys.append(np_points[i - 1, 1])
            Zs.append(np_points[i - 1, 2])

        ATFP_boundary_CPs = np.c_[Xs, Ys, Zs]

        lib.Shape_Analysis.checkPlot('ATFP Boundary Initial', ATFP_boundary_CPs, ATFP_boundary_CPs)


        # Hiatus point on GI Filler
        hiatus_original_CP =  np.array([[gifiller.x, gifiller.y, gifiller.z]])
        rotated_gifiller = ss.rotate_point(gifiller.x, gifiller.y, gifiller.z, rotate_angle, rot_point)
        hiatus_deformed_CP = np.array([[rotated_gifiller.x, rotated_gifiller.y, rotated_gifiller.z]])

        # PM Inner nodes
        PM_MID      = PM_MID
        connections = io.get_interconnections(OutputINPFile, PM_MID)
        PM_Mid = io.get_dataset_from_file(OutputINPFile, PM_MID)
        starting_index, ending_index = find_starting_ending_points_for_inside(PM_Mid)
        innerNodes = findInnerNodes(PM_Mid, connections, starting_index, ending_index)

        Xs = []
        Ys = []
        Zs = []

        for i in innerNodes:
            Xs.append(PM_Mid.node(i).x)
            Ys.append(PM_Mid.node(i).y)
            Zs.append(PM_Mid.node(i).z)

        inner_PM_CPs = np.c_[Xs,Ys,Zs]

        rotated_Xs = []
        rotated_Ys = []
        rotated_Zs = []
        for index in range(len(Xs)):
            rotated_point= ss.rotate_point(Xs[index], Ys[index], Zs[index], rotate_angle, rot_point)
            rotated_Xs.append(rotated_point.x)
            rotated_Ys.append(rotated_point.y)
            rotated_Zs.append(rotated_point.z)
        inner_PM_deformed_CPs = np.c_[rotated_Xs,rotated_Ys,rotated_Zs]

        lib.Shape_Analysis.checkPlot('PM Inner Nodes', inner_PM_CPs, inner_PM_deformed_CPs)


        ######################### AVW_CPs ################################
        # get AVW nodes
        generated_surface = io.get_dataset_from_file(OutputINPFile, AVW).generatedCopy()
        # make interpolation
        generated_interpolator_zx_y = Rbf(generated_surface.zAxis, generated_surface.xAxis, generated_surface.yAxis, function="linear", smooth=5)
        # get max/min z
        z_point_no = 10
        min_z = min(generated_surface.zAxis)
        max_z = max(generated_surface.zAxis)
        spaced_zs = np.linspace(min_z,max_z,z_point_no)
        # linspace between max/min z
        # same with x
        x_point_no = 5
        min_x = min(generated_surface.xAxis)
        max_x = max(generated_surface.xAxis)
        spaced_xs = np.linspace(min_x,max_x,x_point_no)
        # loop through x and z and use interpolation to get y
        init_Xs = []
        init_Ys = []
        init_Zs = []
        mod_Xs = []
        mod_Ys = []
        mod_Zs = []
        for x in spaced_xs:
            for z in spaced_zs:
                y = generated_interpolator_zx_y(z,x)
                # use that for the initial CPs
                init_Xs.append(x)
                init_Ys.append(y)
                init_Zs.append(z)
                # run each point through the rotation function
                rotated_point = ss.rotate_point(x, y, z, rotate_angle, rot_point)
                # use that for the deformed CPs
                mod_Xs.append(rotated_point.x)
                mod_Ys.append(rotated_point.y)
                mod_Zs.append(rotated_point.z)

        AVW_CPs_initial = np.c_[init_Xs,init_Ys,init_Zs]
        AVW_CPs_mod = np.c_[mod_Xs,mod_Ys,mod_Zs]


        # initial_CPs = np.concatenate((PM_boundary_CPs, LA_boundary_CPs, hiatus_original_CP, inner_PM_CPs, AVW_CPs_initial), axis = 0)
        # cust_CPs = np.concatenate((PM_boundary_CPs, LA_boundary_CPs, hiatus_deformed_CP, inner_PM_deformed_CPs, AVW_CPs_mod), axis = 0)
        # # initial_CPs = np.concatenate((PM_boundary_CPs, LA_boundary_CPs, hiatus_original_CP, inner_PM_CPs, AVW_CPs_initial, LP_CPs_initial), axis = 0)
        # # cust_CPs = np.concatenate((PM_boundary_CPs, LA_boundary_CPs, hiatus_deformed_CP, inner_PM_deformed_CPs, AVW_CPs_mod, LP_CPs_mod), axis = 0)


        # fig = plt.figure(2)
        # ax = fig.add_subplot(111, projection='3d')

        # ax.scatter(initial_CPs[:,0],initial_CPs[:,2],initial_CPs[:,1], c = 'b', marker = '+')
        # ax.scatter(cust_CPs[:,0],cust_CPs[:,2],cust_CPs[:,1], c = 'r', marker = '+')

        # # ax.scatter(center_xs,initial_lp_CP_zs,initial_lp_CP_ys, c = 'b', marker = '+')
        # # ax.scatter(center_xs,zs,ys, c = 'r', marker = '+')


        # # print('********************************************')
        # # print('initial:', initial_CPs)
        # # print('cust ones', cust_CPs)

        # rbf = RBF(original_control_points = initial_CPs, deformed_control_points = cust_CPs, func='thin_plate_spline', radius = 10)


        # tissue_list = ['OPAL325_GIfiller', 'OPAL325_LA', 'OPAL325_PBody', 'OPAL325_PM_mid', 'OPAL325_AVW_v6']
        # temp_file = 'morphing_temp.inp'

        # for tissue in tissue_list:
        #     print('hiatus morphing.........', tissue)
        #     # getting the nodes for original levator from Generic file
        #     generic_tissue = io.get_dataset_from_file(OutputINPFile, tissue)











        #     # get the initial tissue nodes into an numpy array and do an intrepolation surface
        #     xs = np.asarray(generic_tissue.xAxis)
        #     ys = np.asarray(generic_tissue.yAxis)
        #     zs = np.asarray(generic_tissue.zAxis)
        #     initial_tissue = np.c_[xs,ys,zs]

        #     new_tissue = rbf(initial_tissue)

        #     shutil.copy(OutputINPFile, temp_file)

        #     for i in range(len(new_tissue)):
        #         generic_tissue.xAxis[i] = new_tissue[i][0]
        #         generic_tissue.yAxis[i] = new_tissue[i][1]
        #         generic_tissue.zAxis[i] = new_tissue[i][2]

        #     io.write_part_to_inp_file(temp_file, tissue, generic_tissue)
        #     # write_part_to_inp_file(file_name, part, data_set):

        #     shutil.copy(temp_file,OutputINPFile)



        #TODO: call from shape_analysis (git test)
        #Call levator plate shape analysis
        initial_lp_CP_ys, initial_lp_CP_zs, LP_CPs_initial \
            = lib.Shape_Analysis.levator_shape_analysis(-1.98, 0.12, False)

        ys, zs, LP_CPs_mod = lib.Shape_Analysis.levator_shape_analysis(levator_plate_PC1, levator_plate_PC2, False)

        #Call ICM shape analysis for INITIAL
        ICM_CPs_initial = lib.Shape_Analysis.ICM_shape_analysis(22, 7.65, initial_lp_CP_ys, initial_lp_CP_zs, False)

        #Call ICM shape analysis for MOD
        ICM_CPs_mod, ICM_CPs_mod_x, ICM_CPs_mod_y, ICM_CPs_mod_z, perp_slope, mid_plate_y, mid_plate_z \
            = lib.Shape_Analysis.ICM_shape_analysis(ICM_PC1, ICM_PC2, ys, zs, True)

        #TODO: IMMEDIATE PLOT OF GENERATED POINTS, I WANT TO COMPARE LP AND ICM POSITION

        fig = plt.figure(9)
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle('boundary initial vs mod (PC1: ' + str(ICM_PC1) + ', PC2: ' + str(ICM_PC2) + ')')
        initial_CPs_ref = np.concatenate((LA_boundary_CPs, LP_CPs_initial), axis=0)
        mod_CPs_ref = np.concatenate((LA_boundary_CPs, LP_CPs_mod), axis=0)

        ax.scatter(initial_CPs_ref[:, 0], initial_CPs_ref[:, 2], initial_CPs_ref[:, 1], c='b', marker='+',
                   label='initial')
        ax.scatter(mod_CPs_ref[:, 0], mod_CPs_ref[:, 2], mod_CPs_ref[:, 1], c='r', marker='+', label='mod')

        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        plt.legend()
        plt.show()

        fig = plt.figure(2)
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle('Lp initial vs Lp mod')

        ax.scatter(LP_CPs_initial[:, 0], LP_CPs_initial[:, 2], LP_CPs_initial[:, 1], c='b', marker='+', label='initial')
        ax.scatter(LP_CPs_mod[:, 0], LP_CPs_mod[:, 2], LP_CPs_mod[:, 1], c='r', marker='+', label='mod')

        plt.legend()
        plt.show()

        fig = plt.figure(2)
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle('ICM initial vs ICM mod (PC1: ' + str(ICM_PC1) + ', PC2: ' + str(ICM_PC2) + ')')

        ax.scatter(ICM_CPs_initial[:, 0], ICM_CPs_initial[:, 2], ICM_CPs_initial[:, 1], c='b', marker='+', label='initial')
        ax.scatter(ICM_CPs_mod[:, 0], ICM_CPs_mod[:, 2], ICM_CPs_mod[:, 1], c='r', marker='+', label='mod')

        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        plt.legend()
        plt.show()

        fig = plt.figure(2)
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle('Combo initial (PC1: ' + str(ICM_PC1) + ', PC2: ' + str(ICM_PC2) + ')')

        ax.scatter(LP_CPs_initial[:, 0], LP_CPs_initial[:, 2], LP_CPs_initial[:, 1], c='b', marker='+',
                   label='LP')
        ax.scatter(ICM_CPs_initial[:, 0], ICM_CPs_initial[:, 2], ICM_CPs_initial[:, 1], c='r', marker='+', label='ICM')

        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        plt.legend()
        plt.show()

        fig = plt.figure(2)
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle('Combo mod (PC1: ' + str(ICM_PC1) + ', PC2: ' + str(ICM_PC2) + ')')

        ax.scatter(LP_CPs_mod[:, 0], LP_CPs_mod[:, 2], LP_CPs_mod[:, 1], c='b', marker='+',
                   label='LP')
        ax.scatter(ICM_CPs_mod[:, 0], ICM_CPs_mod[:, 2], ICM_CPs_mod[:, 1], c='r', marker='+', label='ICM')

        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        plt.legend()
        plt.show()

        initial_CPs_ref = np.concatenate((LP_CPs_initial, ICM_CPs_initial), axis=0)
        mod_CPs_ref = np.concatenate((LP_CPs_mod, ICM_CPs_mod), axis=0)

        fig = plt.figure(2)
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle('Combo initial vs Combo mod (PC1: ' + str(ICM_PC1) + ', PC2: ' + str(ICM_PC2) + ')')

        ax.scatter(initial_CPs_ref[:, 0], initial_CPs_ref[:, 2], initial_CPs_ref[:, 1], c='b', marker='+', label='initial')
        ax.scatter(mod_CPs_ref[:, 0], mod_CPs_ref[:, 2], mod_CPs_ref[:, 1], c='r', marker='+', label='mod')

        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        plt.legend()
        plt.show()


#### Do Hiatus Transformation
        # initial_CPs = np.concatenate((PM_boundary_CPs, LA_boundary_CPs, hiatus_original_CP, inner_PM_CPs, AVW_CPs_initial), axis = 0)
        # cust_CPs = np.concatenate((PM_boundary_CPs, LA_boundary_CPs, hiatus_deformed_CP, inner_PM_deformed_CPs, AVW_CPs_mod), axis = 0)
        initial_CPs = np.concatenate((PM_boundary_CPs, LA_boundary_CPs, hiatus_original_CP, inner_PM_CPs, AVW_CPs_initial, LP_CPs_initial), axis = 0)
        cust_CPs = np.concatenate((PM_boundary_CPs, LA_boundary_CPs, hiatus_deformed_CP, inner_PM_deformed_CPs, AVW_CPs_mod, LP_CPs_initial), axis = 0)

        fig = plt.figure(9)
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(initial_CPs[:,0],initial_CPs[:,2],initial_CPs[:,1], c = 'b', marker = '+')
        ax.scatter(cust_CPs[:,0],cust_CPs[:,2],cust_CPs[:,1], c = 'r', marker = '+')

        plt.show()

        # ax.scatter(center_xs,initial_lp_CP_zs,initial_lp_CP_ys, c = 'b', marker = '+')
        # ax.scatter(center_xs,zs,ys, c = 'r', marker = '+')


        # print('********************************************')
        # print('initial:', initial_CPs)
        # print('cust ones', cust_CPs)

        rbf = RBF(original_control_points = initial_CPs, deformed_control_points = cust_CPs, func='thin_plate_spline', radius = 10)


        tissue_list = ['OPAL325_GIfiller', 'OPAL325_LA', 'OPAL325_PBody', 'OPAL325_PM_mid', 'OPAL325_AVW_v6']
        temp_file = 'morphing_temp.inp'

        for tissue in tissue_list:
            print('hiatus morphing.........', tissue)
            # getting the nodes for original levator from Generic file
            generic_tissue = io.get_dataset_from_file(OutputINPFile, tissue)


            # get the initial tissue nodes into an numpy array and do an intrepolation surface
            xs = np.asarray(generic_tissue.xAxis)
            ys = np.asarray(generic_tissue.yAxis)
            zs = np.asarray(generic_tissue.zAxis)
            initial_tissue = np.c_[xs,ys,zs]

            new_tissue = rbf(initial_tissue)

            shutil.copy(OutputINPFile, temp_file)

            for i in range(len(new_tissue)):
                generic_tissue.xAxis[i] = new_tissue[i][0]
                generic_tissue.yAxis[i] = new_tissue[i][1]
                generic_tissue.zAxis[i] = new_tissue[i][2]

            io.write_part_to_inp_file(temp_file, tissue, generic_tissue)
            # write_part_to_inp_file(file_name, part, data_set):

            shutil.copy(temp_file,OutputINPFile)


        print(hiatus_deformed_CP)
        print(hiatus_deformed_CP[0][1])
#### Do Levator Transformation

        # initial_CPs = LP_CPs_initial
        # cust_CPs = LP_CPs_mod

        #TODO: Removing AVW_CPs_mod from both arrays
        initial_CPs = np.concatenate((PM_boundary_CPs, ATFP_boundary_CPs, LP_CPs_initial, hiatus_deformed_CP, inner_PM_deformed_CPs, ICM_CPs_initial), axis = 0)
        cust_CPs = np.concatenate((PM_boundary_CPs, ATFP_boundary_CPs, LP_CPs_mod, hiatus_deformed_CP, inner_PM_deformed_CPs, ICM_CPs_mod), axis = 0)

        lib.Shape_Analysis.checkPlot('Levator Transformation Init v. Mod (Includes ATFP Boundary)', initial_CPs, cust_CPs)

        # TODO: This is the radius changed one
        rbf = RBF(original_control_points = initial_CPs, deformed_control_points = cust_CPs, func='thin_plate_spline', radius = 40)


        tissue_list = ['OPAL325_GIfiller', 'OPAL325_LA', 'OPAL325_PBody', 'OPAL325_PM_mid', 'OPAL325_AVW_v6']
        temp_file = 'morphing_temp.inp'

        for tissue in tissue_list:
            print('hiatus morphing.........', tissue)
            # getting the nodes for original levator from Generic file
            generic_tissue = io.get_dataset_from_file(OutputINPFile, tissue)


            # get the initial tissue nodes into an numpy array and do an intrepolation surface
            xs = np.asarray(generic_tissue.xAxis)
            ys = np.asarray(generic_tissue.yAxis)
            zs = np.asarray(generic_tissue.zAxis)
            initial_tissue = np.c_[xs,ys,zs]

            new_tissue = rbf(initial_tissue)

            shutil.copy(OutputINPFile, temp_file)

            for i in range(len(new_tissue)):
                generic_tissue.xAxis[i] = new_tissue[i][0]
                generic_tissue.yAxis[i] = new_tissue[i][1]
                generic_tissue.zAxis[i] = new_tissue[i][2]

            io.write_part_to_inp_file(temp_file, tissue, generic_tissue)
            # write_part_to_inp_file(file_name, part, data_set):

            shutil.copy(temp_file,OutputINPFile)
            if tissue == 'OPAL325_LA':
                xs_storage_pre = xs
                zs_storage_pre = zs
                ys_storage_pre = ys
                # print(new_tissue)
                # print(type(new_tissue))
                xs_storage = new_tissue[:,0]
                zs_storage = new_tissue[:,2]
                ys_storage = new_tissue[:,1]



############# I believe below is just for graphing to check things
        center = -4
        x_threshold = 2
        midline_xs = []
        midline_ys = []
        midline_zs = []
        for index, x in enumerate(xs_storage):
            if abs(x-center) < x_threshold:
                midline_xs.append(x)
                midline_ys.append(ys_storage[index])
                midline_zs.append(zs_storage[index])

        initial_midline_xs = []
        initial_midline_ys = []
        initial_midline_zs = []
        for index, x in enumerate(xs_storage_pre):
            if abs(x-center) < x_threshold:
                initial_midline_xs.append(x)
                initial_midline_ys.append(ys_storage_pre[index])
                initial_midline_zs.append(zs_storage_pre[index])

        fig = plt.figure(2)
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(xs_storage,zs_storage,ys_storage, c = 'b', marker = '+')
        ax.scatter(LP_CPs_mod[:,0],LP_CPs_mod[:,2],LP_CPs_mod[:,1], c = 'r', marker = '+')

        plt.show()

        fig = plt.figure(4)
        ax = fig.add_subplot(111)
        ax.scatter(zs_storage,ys_storage, c = 'b', marker = '+')
        ax.scatter(LP_CPs_mod[:,2],LP_CPs_mod[:,1], c = 'r', marker = '+')

        plt.show()

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.scatter(zs_storage_pre,ys_storage_pre, c = 'b', marker = '+')
        ax.scatter(LP_CPs_initial[:,2],LP_CPs_initial[:,1], c = 'r', marker = '+')

        plt.show()

        fig = plt.figure(3)
        ax = fig.add_subplot(111)
        ax.scatter(midline_zs,midline_ys, c = 'b', marker = '+')
        ax.scatter(LP_CPs_mod[:,2],LP_CPs_mod[:,1], c = 'r', marker = '+')

        plt.show()

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.scatter(initial_midline_zs,initial_midline_ys, c = 'b', marker = '+')
        ax.scatter(LP_CPs_initial[:,2],LP_CPs_initial[:,1], c = 'r', marker = '+')

        plt.show()


        graphing_tissue_list = ['OPAL325_GIfiller', 'OPAL325_LA', 'OPAL325_PBody', 'OPAL325_PM_mid', 'OPAL325_AVW_v6']
        temp_file = 'morphing_temp.inp'

        fig = plt.figure(2)
        ax = fig.add_subplot(111, projection='3d')
        for tissue in graphing_tissue_list:
            # print('hiatus morphing.........', tissue)
            # getting the nodes for original levator from Generic file
            generic_tissue = io.get_dataset_from_file(OutputINPFile, tissue)


            # get the initial tissue nodes into an numpy array and do an intrepolation surface
            # ax.scatter(xs = np.asarray(generic_tissue.xAxis)


            # ax.scatter(xs_storage,zs_storage,ys_storage, c = 'b', marker = '+')
            ax.scatter(generic_tissue.xAxis,generic_tissue.zAxis,generic_tissue.yAxis, c = 'r', marker = '+')

        plt.show()

# #### Do ICM Transformation
# #### Do ICM Transformation
# #### Do ICM Transformation
# #### Do ICM Transformation
#TODO: From here to "droop" is the commented out stuff

#         # First I need to create Control Points for the current levator ICM
#
#
#         # find the "y" intercept for that line
#         perp_intercept = mid_plate_y - perp_slope*mid_plate_z
#
#         # setting up p1 and p2 to find the distance between that line and
#         # p1 = np.array([mid_plate_z,interp_init(center, mid_plate_z)])
#         p1 = np.array([mid_plate_z,mid_plate_y])
#         p2_z = -20
#         p2 = np.array([p2_z,perp_slope*p2_z + perp_intercept])
#
#
#         # getting the nodes for original levator from Generic file
#         levator = 'OPAL325_LA'
#         generic_LA = io.get_dataset_from_file(temp_file, levator)
#
#         # # getting the nodes for the modified levator
#         # customized_LA = io.get_dataset_from_file(customized_INP, levator)
#
#         # get the initial levator nodes into an numpy array and do an intrepolation surface
#         xs = np.asarray(generic_LA.xAxis)
#         ys = np.asarray(generic_LA.yAxis)
#         zs = np.asarray(generic_LA.zAxis)
#         initial_LA = np.c_[xs,ys,zs]
#
#         # how many mm a point can be off of the ICM line and still be part of the ICM
#         ICM_threshold = 1
#
#         # creating the ICM Control points
#         ICM_Nodes = np.array([[0,0,0]])
#         for node in initial_LA:
#             # z,y
#             p3 = np.array([node[2],node[1]])
#             if abs((np.abs(np.cross(p2-p1, p1-p3))) / np.linalg.norm(p2-p1)) < ICM_threshold:
#                 ICM_Nodes = np.concatenate((ICM_Nodes, [node]), axis = 0)
#
#         # remove the dummy first element
#         ICM_Nodes = ICM_Nodes[1:]
#
#
#         # reformatting the control points
#         ICM_xs = []
#         ICM_ys = []
#         ICM_zs = []
#         for coordinates in ICM_Nodes:
#             ICM_zs.append(coordinates[2])
#             ICM_ys.append(coordinates[1])
#             ICM_xs.append(coordinates[0])
#
#
#         ordered_ICM_zs = [x for _, x in sorted(zip(ICM_xs, ICM_zs))]
#         ordered_ICM_xs = sorted(ICM_xs)
#         ordered_ICM_ys = [x for _, x in sorted(zip(ICM_xs, ICM_ys))]
#
#         # go through and find the distance between each point,
#         # keeping track of the distance where x is closest to the center
#         tot_dist = 0
#         tot_dist_arr = []
#         last_z = ordered_ICM_zs[0]
#         last_y = ordered_ICM_ys[0]
#         last_x = ordered_ICM_xs[0]
#         for index, z in enumerate(ordered_ICM_zs):
#             dist = ((last_z - z)**2+(last_y - ordered_ICM_ys[index])**2 + ((last_x - ordered_ICM_xs[index])**2))**.5
#             last_z = ordered_ICM_zs[index]
#             last_y = ordered_ICM_ys[index]
#             last_x = ordered_ICM_xs[index]
#             tot_dist += dist
#             tot_dist_arr.append(tot_dist)
#
#
#
#         curve_x = interp1d(tot_dist_arr, ordered_ICM_xs)
#         # curve_y = UnivariateSpline(tot_dist_arr, y_sorted, k = 5)
#         curve_y = interp1d(tot_dist_arr, ordered_ICM_ys)
#         # curve_z = UnivariateSpline(tot_dist_arr, z_sorted, k = 5)
#         curve_z = interp1d(tot_dist_arr, ordered_ICM_zs)
#         # curve_x = interp1d(tot_dist_arr, x_sorted)
#
#         # the points are then createdto make the spacing for the points equal
#         spaced_distance_array = np.linspace(0,tot_dist_arr[-1],100)
#
#         new_distance_array  = [0]
#         # previous_z = curve_z(0)
#         # previous_y = curve_y(0)
#         # previous_x = curve_x(0)
#         new_zs = [curve_z(0)]
#         new_ys = [curve_y(0)]
#         new_xs = [curve_x(0)]
#         center_x_check = 99999999
#         for i in range (0,len(spaced_distance_array)):
#             new_ys.append(float(curve_y(spaced_distance_array[i])))
#             new_zs.append(float(curve_z(spaced_distance_array[i])))
#             new_xs.append(float(curve_x(spaced_distance_array[i])))
#             new_distance_array.append(((new_ys[-1] - new_ys[-2])**2 + (new_zs[-1]-new_zs[-2])**2 + (new_xs[-1] - new_xs[-2])**2)**0.5 + new_distance_array[-1])
#             if abs(new_xs[-1] - (-2)) < center_x_check:
#                 center_x_check = abs(new_xs[-1] - (-2))
#                 init_center_dist = new_distance_array[-1]
#             # print('x,y,z,dist:', new_xs[-1], new_ys[-1], new_zs[-1], new_distance_array[-1])
#
#
#         init_percent_dist_arr = new_distance_array/new_distance_array[-1]
#         init_center_percent_dist = init_center_dist/new_distance_array[-1]
#
#         init_new_curve_y = interp1d(init_percent_dist_arr, new_ys)
#         init_new_curve_z = interp1d(init_percent_dist_arr, new_zs)
#         init_new_curve_x = interp1d(init_percent_dist_arr, new_xs)
#
#         #### Same thing using the modified CPs instead of FEA coordinates
#             #TODO: Here is where the separate x, y, z is used from ICM shape analysis
#         # go through and find the distance between each point,
#         # keeping track of the distance where x is closest to the center
#         ordered_ICM_xs = ICM_CPs_mod_x
#         ordered_ICM_ys = ICM_CPs_mod_y
#         ordered_ICM_zs = ICM_CPs_mod_z
#
#         tot_dist = 0
#         tot_dist_arr = []
#         last_z = ordered_ICM_zs[0]
#         last_y = ordered_ICM_ys[0]
#         last_x = ordered_ICM_xs[0]
#         for index, z in enumerate(ordered_ICM_zs):
#             dist = ((last_z - z)**2+(last_y - ordered_ICM_ys[index])**2 + ((last_x - ordered_ICM_xs[index])**2))**.5
#             last_z = ordered_ICM_zs[index]
#             last_y = ordered_ICM_ys[index]
#             last_x = ordered_ICM_xs[index]
#             tot_dist += dist
#             tot_dist_arr.append(tot_dist)
#
#         mod_point_distance_percentages = tot_dist_arr/tot_dist_arr[-1]
#
#         curve_x = interp1d(tot_dist_arr, ordered_ICM_xs)
#         # curve_y = UnivariateSpline(tot_dist_arr, y_sorted, k = 5)
#         curve_y = interp1d(tot_dist_arr, ordered_ICM_ys)
#         # curve_z = UnivariateSpline(tot_dist_arr, z_sorted, k = 5)
#         curve_z = interp1d(tot_dist_arr, ordered_ICM_zs)
#         # curve_x = interp1d(tot_dist_arr, x_sorted)
#
#         # the points are then createdto make the spacing for the points equal
#         spaced_distance_array = np.linspace(0,tot_dist_arr[-1],100)
#
#         new_distance_array  = [0]
#         # previous_z = curve_z(0)
#         # previous_y = curve_y(0)
#         # previous_x = curve_x(0)
#         new_zs = [curve_z(0)]
#         new_ys = [curve_y(0)]
#         new_xs = [curve_x(0)]
#         center_x_check = 999999
#         for i in range (0,len(spaced_distance_array)):
#             new_ys.append(float(curve_y(spaced_distance_array[i])))
#             new_zs.append(float(curve_z(spaced_distance_array[i])))
#             new_xs.append(float(curve_x(spaced_distance_array[i])))
#             new_distance_array.append(((new_ys[-1] - new_ys[-2])**2 + (new_zs[-1]-new_zs[-2])**2 + (new_xs[-1] - new_xs[-2])**2)**0.5 + new_distance_array[-1])
#             if abs(new_xs[-1] - (-2)) < center_x_check:
#                 center_x_check = abs(new_xs[-1] - (-2))
#                 mod_center_dist = new_distance_array[-1]
#             # print('x,y,z,dist:', new_xs[-1], new_ys[-1], new_zs[-1], new_distance_array[-1])
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#         mod_percent_dist_arr = new_distance_array/new_distance_array[-1]
#         mod_center_percent_dist = mod_center_dist/new_distance_array[-1]
#
#         mod_new_curve_y = interp1d(mod_percent_dist_arr, new_ys)
#         mod_new_curve_z = interp1d(mod_percent_dist_arr, new_zs)
#         mod_new_curve_x = interp1d(mod_percent_dist_arr, new_xs)
#
#
#
#
#
#
#         init_CP_xs = init_new_curve_x(mod_point_distance_percentages)
#         init_CP_ys = init_new_curve_y(mod_point_distance_percentages)
#         init_CP_zs = init_new_curve_z(mod_point_distance_percentages)
#
#         # initial_CPs = LP_CPs_initial
#         # cust_CPs = LP_CPs_mod
#
#         ICM_CPs_initial = np.c_[init_CP_xs,init_CP_ys,init_CP_zs]
#
#         #TODO: First ICM Plot
#         fig = plt.figure(1)
#         ax = fig.add_subplot(111)
#         # ax.scatter(ICM_CPs_mod[:,0], ICM_CPs_mod[:,1], c = 'b', marker = '+')
#         # ax.scatter(ICM_CPs_initial[:,0],ICM_CPs_initial[:,1], c = 'r', marker = '+')
#         ax.scatter(ICM_CPs_mod[:,1], ICM_CPs_mod[:,0], c = 'b', marker = '+')
#         ax.scatter(ICM_CPs_initial[:,1],ICM_CPs_initial[:,0], c = 'r', marker = '+')
#
#
#         plt.show()
#
#         #TODO: Second ICM Plot
#         fig = plt.figure(1)
#         ax = fig.add_subplot(111)
#         ax.scatter(ICM_CPs_mod[:,2], ICM_CPs_mod[:,1], c = 'b', marker = '+')
#         ax.scatter(ICM_CPs_initial[:,2],ICM_CPs_initial[:,1], c = 'r', marker = '+')
#
#         plt.show()
#
# #################### I might need to do the levator transformation without
#         ### the AVW points and then tranform the GI Filler and AVW with the AVW points
#         ### Otherwise the AVW might not line up with the GI Filler
#
#         #TODO: This is the plot i care about, there is no plot
#         initial_CPs = np.concatenate((PM_boundary_CPs, AVW_CPs_mod, LP_CPs_mod, ICM_CPs_initial), axis = 0)
#         cust_CPs = np.concatenate((PM_boundary_CPs, AVW_CPs_mod, LP_CPs_mod, ICM_CPs_mod), axis = 0)
#
#
#         rbf = RBF(original_control_points = initial_CPs, deformed_control_points = cust_CPs, func='thin_plate_spline', radius = 10)
#
#         #TODO: aight lets add a plot then
#         fig = plt.figure(7)
#         ax = fig.add_subplot(111, projection='3d')
#
#         ax.scatter(initial_CPs[:, 0], initial_CPs[:, 2], initial_CPs[:, 1], c='b', marker='+')
#         ax.scatter(cust_CPs[:, 0], cust_CPs[:, 2], cust_CPs[:, 1], c='r', marker='+')
#
#         plt.show()
#
#         tissue_list = ['OPAL325_GIfiller', 'OPAL325_LA', 'OPAL325_PBody', 'OPAL325_PM_mid', 'OPAL325_AVW_v6']
#         temp_file = 'morphing_temp.inp'
#
#         for tissue in tissue_list:
#             print('hiatus morphing.........', tissue)
#             # getting the nodes for original levator from Generic file
#             generic_tissue = io.get_dataset_from_file(OutputINPFile, tissue)
#
#
#             # get the initial tissue nodes into an numpy array and do an intrepolation surface
#             xs = np.asarray(generic_tissue.xAxis)
#             ys = np.asarray(generic_tissue.yAxis)
#             zs = np.asarray(generic_tissue.zAxis)
#             initial_tissue = np.c_[xs,ys,zs]
#
#             new_tissue = rbf(initial_tissue)
#
#             shutil.copy(OutputINPFile, temp_file)
#
#             for i in range(len(new_tissue)):
#                 generic_tissue.xAxis[i] = new_tissue[i][0]
#                 generic_tissue.yAxis[i] = new_tissue[i][1]
#                 generic_tissue.zAxis[i] = new_tissue[i][2]
#
#             io.write_part_to_inp_file(temp_file, tissue, generic_tissue)
#             # write_part_to_inp_file(file_name, part, data_set):
#
#             shutil.copy(temp_file,OutputINPFile)
#             if tissue == 'OPAL325_LA':
#                 xs_storage_pre = xs
#                 zs_storage_pre = zs
#                 ys_storage_pre = ys
#                 print(new_tissue)
#                 print(type(new_tissue))
#                 xs_storage = new_tissue[:,0]
#                 zs_storage = new_tissue[:,2]
#                 ys_storage = new_tissue[:,1]
#
#
#
#
#
#
#
#
#
#         # General things
#         # x coordinate of the center
#         center = -2
#
#         # how many mm a point can be off of the ICM line and still be part of the ICM
#         ICM_threshold = 1
#
#         # Generic file to use for morphing
#         Generic_INP = GenericINPFile
#
#         # Levator Plate
#         # name of the tissue that we are morphing
#         levator = LA
#
#         # getting the nodes for original levator from Generic file
#         generic_LA = io.get_dataset_from_file(Generic_INP, levator)
#
#         # get the initial levator nodes into an numpy array and do an intrepolation surface
#         xs = np.asarray(generic_LA.xAxis)
#         ys = np.asarray(generic_LA.yAxis)
#         zs = np.asarray(generic_LA.zAxis)
#         initial_LA = np.c_[xs,ys,zs]
#         interp_init = Rbf(xs, zs, ys, function = 'thin_plate')  # radial basis function interpolator instance
#
#         # number of points for the levator plate line
#         midline_points = 5
#
#         # get midline isocurve by choosing z and x data and interpolating the ys
#         z = np.linspace(min(zs),max(zs)-10,midline_points)
#         x = np.linspace(center,center,len(z))
#
#         # add in a point in the middle of the levator plate to correspond with the
#         # ICM band
#         # find the middle of the levator plate
#         mid_plate_z = (max(zs) + min(zs))/2
#         z = np.append(z,mid_plate_z)
#         x = np.append(x,center)
#         interp_ys = interp_init(x,z)
#
#         # put the levator plate CP coordinates into np arrays
#         midline_initial_CPs = np.c_[x,interp_ys,z]
#
#
#         # Levator Iso
#         # number of points for the ICM line
#         iso_points = 10
#
#         # (everything here is looking from the side and ignorning the xcoordinates)
#         mid_plate_y = interp_init(center,mid_plate_z)
#         z_diff_min = float('inf')
#         z_diff_min_2 = float('inf')
#         z_1 = float('inf')
#         z_2 = float('inf')
#         y_1 = float('inf')
#         y_2 = float('inf')
#         for index, node_z in enumerate(z):
#             # print('***',z_diff_min_2)
#             if index < len(z) - 1:
#                 if abs(node_z - mid_plate_z) < z_diff_min_2:
#                     if abs(node_z-mid_plate_z) < z_diff_min:
#                         z_diff_min_2 = z_diff_min
#                         z_2 = z_1
#                         y_2 = y_1
#                         z_diff_min = abs(node_z-mid_plate_z)
#                         z_1 = node_z
#                         y_1 = interp_ys[index]
#                     else:
#                         z_diff_min_2 = abs(node_z-mid_plate_z)
#                         z_2 = node_z
#                         y_2 = interp_ys[index]
#
#         # find the slope at the middle of the levator plate (where the ICM is)
#         slope = (y_2-y_1)/(z_2-z_1)
#
#         ####### Equation #########################
#         # The equation is in the z-y plane       #
#         # give it a z and it will predict the y  #
#         ##########################################
#
#         # find perpendicular slope which is the slope of the ICM line
#         perp_slope = -1/slope
#         # find the "y" intercept for that line
#         perp_intercept = mid_plate_y - perp_slope*mid_plate_z
#
#         # setting up p1 and p2 to find the distance between that line and
#         # p1 = np.array([mid_plate_z,interp_init(center, mid_plate_z)])
#         p1 = np.array([mid_plate_z,mid_plate_y])
#         p2_z = -20
#         p2 = np.array([p2_z,perp_slope*p2_z + perp_intercept])
#
#
#         # creating the ICM Control points
#         ICM_Nodes = np.array([[0,0,0]])
#         for node in initial_LA:
#             # z,y
#             p3 = np.array([node[2],node[1]])
#             if abs((np.abs(np.cross(p2-p1, p1-p3))) / np.linalg.norm(p2-p1)) < ICM_threshold:
#                 ICM_Nodes = np.concatenate((ICM_Nodes, [node]), axis = 0)
#
#         # remove the dummy first element
#         ICM_Nodes = ICM_Nodes[1:]
#
#         ICM_Nodes_old = ICM_Nodes
#         # print(ICM_Nodes)
#
#         # reformatting the control points
#         ICM_xs = []
#         ICM_ys = []
#         ICM_zs = []
#         for coordinates in ICM_Nodes:
#             ICM_zs.append(coordinates[2])
#             ICM_ys.append(coordinates[1])
#             ICM_xs.append(coordinates[0])
#
#
#         ordered_ICM_zs = [x for _, x in sorted(zip(ICM_xs, ICM_zs))]
#         ordered_ICM_xs = sorted(ICM_xs)
#
#         # create interpolation of the ICM nodes to
#         # the interpolation takes an x and gives the z
#         ICM_interp = interp1d(ordered_ICM_xs,ordered_ICM_zs)
#         ICM_interp = interp1d(ICM_xs,ICM_zs)
#         # get evenly spaced x coordiantes with the range of the ICM_xs
#         ICM_CP_xs = np.linspace(min(ICM_xs),max(ICM_xs),iso_points)
#         # get the zs corresponding to the xs using the interpolation
#         ICM_CP_zs = ICM_interp(ICM_CP_xs)
#
#         # Given the z (from the interpolation) get the y (from the line equation)
#         ICM_CP_ys = perp_slope*ICM_CP_zs + perp_intercept
#
#         ICM_CP_xs_original = ICM_CP_xs
#         ICM_CP_ys_original = ICM_CP_ys
#         ICM_CP_zs_original = ICM_CP_zs
#
#         # putting the ICM curve at 1/4, 1/2, and 3/4
#         full_ICM_CP_xs = []
#         full_ICM_CP_ys = []
#         full_ICM_CP_zs = []
#         mid_plate_z = (max(zs) + min(zs))/2
#         z_list = [(max(zs) - min(zs))/4 + min(z), (max(zs) - min(zs))*2/4 + min(z), (max(zs) - min(zs))*3/4 + min(z)]
#         for i in range(len(z_list)):
#             # I need the z shift and the y shift
#             z_shift = z_list[i] - mid_plate_z
#             y_shift = interp_init(center, z_list[i]) - mid_plate_y
#             full_ICM_CP_xs = np.concatenate((full_ICM_CP_xs,ICM_CP_xs),axis = 0)
#             full_ICM_CP_ys = np.concatenate((full_ICM_CP_ys,ICM_CP_ys+y_shift),axis = 0)
#             full_ICM_CP_zs = np.concatenate((full_ICM_CP_zs,ICM_CP_zs+z_shift),axis = 0)
#
#         # put the ICM CPs in a format for PyGeM
#         ICM_CPs = np.column_stack((full_ICM_CP_xs,full_ICM_CP_ys,full_ICM_CP_zs))
#
#
#         # PM Inner nodes
#         PM_MID      = PM_MID
#         connections = io.get_interconnections(OutputINPFile, PM_MID)
#         PM_Mid = io.get_dataset_from_file(OutputINPFile, PM_MID)
#         starting_index, ending_index = find_starting_ending_points_for_inside(PM_Mid)
#         innerNodes = findInnerNodes(PM_Mid, connections, starting_index, ending_index)
#
#         Xs = []
#         Ys = []
#         Zs = []
#
#         for i in innerNodes:
#             Xs.append(PM_Mid.node(i).x)
#             Ys.append(PM_Mid.node(i).y)
#             Zs.append(PM_Mid.node(i).z)
#
#         inner_PM_CPs = np.c_[Xs,Ys,Zs]
#
#
#
#         # print(hiatus_original_CP)
#         # print(type(hiatus_original_CP))
#         # print(hiatus_original_CP.shape)
#         initial_CPs = np.concatenate((midline_initial_CPs, ICM_CPs, PM_boundary_CPs, inner_PM_CPs, hiatus_original_CP), axis = 0)
#         cust_CPs = np.concatenate((midline_initial_CPs, ICM_CPs, PM_boundary_CPs, inner_PM_CPs, hiatus_deformed_CP ), axis = 0)
#
#
#
#         rbf = RBF(original_control_points = initial_CPs, deformed_control_points = cust_CPs, func='thin_plate_spline', radius = 10)
#
#
#         fig = plt.figure(2)
#         ax = fig.add_subplot(111, projection='3d')
#
#         ax.scatter(initial_CPs[:,0],initial_CPs[:,2],initial_CPs[:,1], c = 'b', marker = '+')
#         ax.scatter(cust_CPs[:,0],cust_CPs[:,2],cust_CPs[:,1], c = 'r', marker = '+')
#
#         plt.show()
#
#         tissue_list = ['OPAL325_GIfiller', 'OPAL325_LA', 'OPAL325_PBody', 'OPAL325_PM_mid', 'OPAL325_AVW_v6']
#         temp_file = 'morphing_temp.inp'
#
#         for tissue in tissue_list:
#             print('morphing.........', tissue)
#             # getting the nodes for original levator from Generic file
#             generic_tissue = io.get_dataset_from_file(Generic_INP, tissue)
#
#
#             # get the initial levator nodes into an numpy array and do an intrepolation surface
#             xs = np.asarray(generic_tissue.xAxis)
#             ys = np.asarray(generic_tissue.yAxis)
#             zs = np.asarray(generic_tissue.zAxis)
#             initial_tissue = np.c_[xs,ys,zs]
#
#             new_tissue = rbf(initial_tissue)
#
#             shutil.copy(OutputINPFile, temp_file)
#
#             for i in range(len(new_tissue)):
#                 generic_tissue.xAxis[i] = new_tissue[i][0]
#                 generic_tissue.yAxis[i] = new_tissue[i][1]
#                 generic_tissue.zAxis[i] = new_tissue[i][2]
#
#
#             io.write_part_to_inp_file(temp_file, tissue, generic_tissue)
#             # write_part_to_inp_file(file_name, part, data_set):
#             shutil.copy(temp_file,OutputINPFile)
#
#
#         Scaling.rotate_part(AVW, OutputINPFile, rotate_angle, rot_point)
#         print('1')
#         gi        = Scaling.rotate_part(GI_FILLER, OutputINPFile, rotate_angle, rot_point)
#         print('2')
#         aftp      = Scaling.rotate_part(ATFP, OutputINPFile, rotate_angle, rot_point)Added
#         print('3')
#         atla      = Scaling.rotate_part(ATLA, OutputINPFile, rotate_angle, rot_point)
#         print('4')
#         la        = Scaling.rotate_part(LA, OutputINPFile, rotate_angle, rot_point)
#         print('5')
#         pbody     = Scaling.rotate_part(PBODY, OutputINPFile, rotate_angle, rot_point)
#         print('6')
#         pm_mid    = Scaling.rotate_part(PM_MID, OutputINPFile, rotate_angle, rot_point)
#         print('7')
#         ######################################################################################

        #ADDING THE SCAR TISSUE FUNCTION
        if generate_scar_tissue_flag != 0:
            io.write_points_x_from_mid_excluded_to_inp(OutputINPFile, AVW, mid_x, threshold_x_from_mid, 22, 'AWP', 'SCAR_TISSUE')

    ######################################################################################
        #TODO: Commented out droop
        # Droop the AVW (probably pass the generic file and obtain the end points for the droop from there)
        if config.getint("FLAGS", "CurveAVW") != 0:
            avw = Scaling.curve_avw(AVW, OutputINPFile, GenericINPFile, hiatus, z_cutoff, rotate_angle, rot_point, HiatusLength)
    ######################################################################################

    ######################################################################################


# ##### End of Sabbatical Code Below
#     # Need to put a wave in the bottom of the AVW so that it doesn't hit the PM_mid tissue
#         if config.getint("FLAGS", "distal_AVW") != 0:
#             avw = Scaling.narrow_distal_avw(AVW, OutputINPFile, GenericINPFile)
# #    #        print(avw)
# ##### End of Sabbatical Code Above

        ######################################################################################
####### New Curving Code Below
# Need to put a wave in the bottom of the AVW so that it doesn't hit the PM_mid tissue

        pm_mid = io.get_dataset_from_file(OutputINPFile, PM_MID)
        if config.getint("FLAGS", "testing") != 0:
            AVW_connections = io.get_interconnections(GenericINPFile, AVW)
            avw = Scaling.narrow_distal_avw_narrow(AVW, OutputINPFile, GenericINPFile, pm_mid, connections, AVW_connections)
#        print(avw)
        # avw = Scaling.narrow_distal_avw_narrow_and_curve(AVW, OutputINPFile, GenericINPFile, pm_mid, connections, AVW_connections)
####### New Curving Code Above





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
        CurveFibersInINP("OPAL325_AVW_v6", "OPAL325_CL_v6", SlackStrain[0], TempFile, OutputINPFile, dirVector,
                         positive_CL_point, negative_CL_point, positive_CL_remove_percent, negative_remove_percent)

        print(time.process_time() - start)

        print("Adjusting US")
    # Adjust the US tissue
        os.remove(TempFile)
        shutil.copy(OutputINPFile, TempFile)
    # Sets the vector for adding the wave to the fiber to increase length
        dirVector = [1,0,0]
        CurveFibersInINP("OPAL325_AVW_v6", "OPAL325_US_v6", SlackStrain[1], TempFile, OutputINPFile, dirVector,
                         positive_US_point, negative_US_point, positive_CL_remove_percent, negative_remove_percent)

        print(time.process_time() - start)

        time.sleep(3)


        print("Adjusting Para")
    ## Adjust the Paravaginal tissue
        os.remove(TempFile)
        shutil.copy(OutputINPFile, TempFile)
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
            CurveFibersInINP("OPAL325_AVW_v6", "OPAL325_Para_v6", SlackStrain[2], TempFile, OutputINPFile, dirVector,
                             positive_PARA_point, negative_PARA_point, positive_CL_remove_percent, negative_remove_percent)


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
        print('SlackStrain', SlackStrain)
        print('new ideal: ', IdealNewFiberLengthMeans)
        print('old fiber means:', oldFiberLengthMeans)
        print('new fiber means:', newFiberLengthMeans)

        # How much longer is the tissue than it is supposed to be
        PrestretchAmount = newFiberLengthMeans - IdealNewFiberLengthMeans

            # Create the pre-stretch coefficient for each fiber by doing the pre-stretch / (pre-stretch + length)
        # Currently if it is pre-strained it has a NEGATIVE stretch coefficient
        if config.getint("FLAGS", "prestrain_fibers") != 0:
            StretchCoefficients = np.divide(PrestretchAmount,np.array(IdealNewFiberLengthMeans+PrestretchAmount))*-1
        else:
            StretchCoefficients = np.zeros(len(PrestretchAmount))

        print('Stretch Coefficients: ', StretchCoefficients)

        # Removing MaterialStartLine and moving it to the Materials file
        print("TissueParameters in Generate INP:", TissueParameters)
        INPAnalogMaterialProperties(TissueParameters, DensityFactor, LoadLine, LoadLineNo, OutputINPFile, StretchCoefficients)

        MeasurementsFileName = Results_Folder_Location + '\\' + OutputINPFile + '_Measurements.txt'
        Scaling.takeMeasurements(MeasurementsFileName, AVW, [CL, PARA, US], GenericINPFile, OutputINPFile)

