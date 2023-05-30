"""
About: Decomposition of Shape Analysis Code Section
The following functions were originally extracted from
Generate_INP.py, in which all pertained to shape analysis control
points of different parts. The common code between these sections
in the Generate_INP.py file were moved here and broken down into
functions and helper functions to streamline the codebase.
"""
import pandas
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

'''
Function: generate_data_points
Helper function for the <levator_shape_analysis> and <ICM_shape_analysis>
functions, and acts as the common point between the 2 sections
'''
def generate_data_points(part, PCA_1, PCA_2, filename):
    # PCA_1_SD = 17.89743185504574
    # PCA_1_coefficient = 1
    # PCA_1_score = PCA_1_SD*PCA_1_coefficient
    PCA_1_score = PCA_1

    # PCA_2_SD = 15.963434817775179
    # PCA_2_coefficient = 2
    # PCA_2_score = PCA_2_SD * PCA_2_coefficient
    PCA_2_score = PCA_2

    # TODO: these variables need to make it out of the function
    df = pandas.read_csv(filename)
    print(df)

    index = df.index

    z_Or_xs = [] #could be zs or xs based on which part
    ys = []
    initial_lp_CP_ys = []
    initial_lp_CP_zORxs = [] #could be zs or xs based on which part
    center_xs = []

    # Generate the data points in the MRI coordinate system
    for i in range(1, len(df)):
        #TODO: Remove the str cast after getting rid of string line in csv
        condition = df["point_number"] == i
        row_num = index[condition]
        col_num = df.columns.get_loc(part + '_PC1_x_coefficient')
        PC1_m_x = float(df.iloc[row_num, col_num])
        col_num = df.columns.get_loc(part + '_PC2_x_coefficient')
        PC2_m_x = float(df.iloc[row_num, col_num])
        col_num = df.columns.get_loc(part + '_x_intercept')
        b_x = float(df.iloc[row_num, col_num])
        print(PC1_m_x, PCA_1_score, PC2_m_x, PCA_2_score)
        if part == 'LP':
            ys.append(PC1_m_x * PCA_1_score + PC2_m_x * PCA_2_score + b_x)
        else:
            z_Or_xs.append(PC1_m_x * PCA_1_score + PC2_m_x * PCA_2_score + b_x)

        col_num = df.columns.get_loc(part + '_PC1_y_coefficient')
        PC1_m_y = float(df.iloc[row_num, col_num])
        col_num = df.columns.get_loc(part + '_PC2_y_coefficient')
        PC2_m_y = float(df.iloc[row_num, col_num])
        col_num = df.columns.get_loc(part + '_y_intercept')
        b_y = float(df.iloc[row_num, col_num])

        center_xs.append(-2)

        if part == 'LP':
            z_Or_xs.append(PC1_m_y * PCA_1_score + PC2_m_y * PCA_2_score + b_y)
        else:
            ys.append(PC1_m_y * PCA_1_score + PC2_m_y * PCA_2_score + b_y)

        #TODO: This uses MRI data, so if i am doing ICM just reuse the generic, do not generate new
        # if part == 'LP':
        #     col_num = df.columns.get_loc(part + '_FEA_z')
        #     initial_lp_CP_zORxs.append(float(df.iloc[row_num, col_num]))
        #     col_num = df.columns.get_loc(part + '_FEA_y')
        #     initial_lp_CP_ys.append(float(df.iloc[row_num, col_num]))


    if part == 'LP':
        return df, index, z_Or_xs, ys, center_xs
    else:
        return df, index, z_Or_xs, ys


'''
Function: levator_shape_analysis
This function takes in the PCA scores given in the Generate_INP.py file 
'''
def levator_shape_analysis(PCA_1, PCA_2, isInitTest):
    filename = 'opal325_PCA_shape_data.csv'

    # # scale and angle for OPALX
    # scale = 1.0075
    # angle = -0.040693832

    # scale and angle for Aging
    scale = 0.91
    angle = -0.088

    #Call helper function
    df, index, zs, ys, center_xs \
        = generate_data_points('LP', PCA_1, PCA_2, filename)

    if isInitTest:
        LP_CPs = np.c_[center_xs, ys, zs]
        return ys, zs, LP_CPs

    zs = np.array(zs)
    ys = np.array(ys)

    zs = zs*-1
    ys = ys*-1

    print('flipped')
    print(zs)
    print(ys)

    for index, (z,y) in enumerate(zip(zs,ys)):
        # print(index)
        # print(z*np.cos(angle)-y*np.sin(angle))

        zs[index] = z*np.cos(angle)-y*np.sin(angle)
        ys[index] = z*np.sin(angle)+y*np.cos(angle)
        # print(index, z, y)

    print('rotated')
    print(zs)
    print(ys)

    index = df.index
    #TODO: remove str cast
    condition = df["point_number"] == 8
    row_num = index[condition]
    col_num = df.columns.get_loc('LP_FEA_z')
    FEA_z = float(df.iloc[row_num,col_num])
    col_num = df.columns.get_loc('LP_FEA_y')
    FEA_y = float(df.iloc[row_num,col_num])

    horiz_shift = FEA_z - zs[-1]
    vert_shift = FEA_y - ys[-1]

    print('shifted amount')
    print(FEA_z)
    print(FEA_y)
    print(horiz_shift)
    print(vert_shift)

    zs = zs + horiz_shift
    ys = ys + vert_shift

    print('shifted')
    print(zs)
    print(ys)

    scaling_center_z = zs[-1]
    scaling_center_y = ys[-1]

    print('scaling center')
    print(scaling_center_z)
    print(scaling_center_y)

    zs = (zs-scaling_center_z)*scale + scaling_center_z
    ys = (ys-scaling_center_y)*scale + scaling_center_y

    print('scaled')
    print(zs)
    print(ys)

    # LP_CPs_initial = np.c_[center_xs,initial_lp_CP_ys,initial_lp_CP_zs]
    LP_CPs = np.c_[center_xs,ys,zs]

    # print("%%%%%%%%%%%%%%%%%%", LP_CPs_mod)

    return ys, zs, LP_CPs


'''
Function: ICM_shape_analysis
The control center for ICM point creation. This function will generate two sets of ICM control points, initial and mod
The information required for each of these sets are listed as follows:
INITIAL:
- PC scores: These are currently hardcoded
- LP points: utilize the initial_lp_CP_ys and initial_lp_CP_zs from <levator_shape_analysis>
- isMod: false, so things like the x,y,z mod points and returning slope, etc will not happen

MOD:
- PC scores: These are the passed along ones that originate from the Run_Variables.csv
- LP points: utilize the ys and zs from <levator_shape_analysis>
- isMod: true, so the x,y,z of the mod CPs as well as info like the slope will be returned

'''
def ICM_shape_analysis(PCA_1, PCA_2, ys, zs, isMod):
    #TODO: ICM START

    filename = 'ICM_PCA_shape_data.csv'

    #TODO: Later change this to get these from LP_CPs_mod below where it is needed
    y_sorted = ys
    z_sorted = zs

    scale = 1.0075
    angle = -0.040693832

    #Call helper function
    df, index, xs, ys \
        = generate_data_points('ICM', PCA_1, PCA_2, filename)

    xs = np.array(xs)
    ys = np.array(ys)
    ys = ys * -1

    #TODO: Plot directly from generate_data_points
    fig = plt.figure(10)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.plot(xs, ys, c='b', marker='+')

    # print('ICM shape analysis points')
    # print(xs)
    # print(ys)


#######################################
    # Working below on finding the center of LP and slope at that point
###########################################

    #TODO: Remove the last point from the sorted arrays (Changed to start from 2, was 0 before)
    # z_sorted = z_sorted[2:-1]
    # y_sorted = y_sorted[2:-1]
    tot_dist = 0
    tot_dist_arr = [0]
    last_z = z_sorted[0]
    last_y = y_sorted[0]
    for i in range(0, len(z_sorted)-1):
        dist = ((z_sorted[i] - z_sorted[i+1])**2+(y_sorted[i] - y_sorted[i+1])**2)**.5
        tot_dist += dist
        tot_dist_arr.append(tot_dist)

#TODO: Should these use the one less point list?
    # curve_y = UnivariateSpline(tot_dist_arr, y_sorted, k = 5)
    curve_y = interp1d(tot_dist_arr, y_sorted, kind='quadratic')
    # kind_y = interp1d(tot_dist_arr, y_sorted, kind='quadratic')
    # curve_z = UnivariateSpline(tot_dist_arr, z_sorted, k = 5)
    curve_z = interp1d(tot_dist_arr, z_sorted, kind='quadratic')
    # kind_z = interp1d(tot_dist_arr, z_sorted, kind='cubic')
    # curve_x = interp1d(tot_dist_arr, x_sorted)

    # the points are then createdto make the spacing for the points equal
    spaced_distance_array = np.linspace(0,tot_dist_arr[-1],100)
    # granular_array = np.arange(0, tot_dist_arr[-1], 0.1)

    new_distance_array  = [0]
    previous_z = curve_z(0)
    previous_y = curve_y(0)
    # previous_x = curve_x(0)
    new_zs = [curve_z(0)]
    new_ys = [curve_y(0)]
    # new_xs = [curve_x(0)]
    for i in range (0,len(spaced_distance_array)):
        new_ys.append(float(curve_y(spaced_distance_array[i])))
        new_zs.append(float(curve_z(spaced_distance_array[i])))
        # new_xs.append(float(curve_x(spaced_distance_array[i])))
        new_distance_array.append(((new_ys[-1] - new_ys[-2])**2 + (new_zs[-1]-new_zs[-2])**2)**0.5 + new_distance_array[-1])
        # print('x,y,z,dist:', new_xs[-1], new_ys[-1], new_zs[-1], new_distance_array[-1])

    new_curve_z = interp1d(new_distance_array, new_zs)
    new_curve_y = interp1d(new_distance_array, new_ys)

    # TODO: remove later once the curve and midpoint are confirmed correct
    # splineZ = [kind_z(0)]
    # splineY = [kind_y(0)]
    # test_distance_array = [0]
    # for i in range(0,len(granular_array)):
    #     splineZ.append(kind_z(granular_array[i]))
    #     splineY.append(kind_y(granular_array[i]))
    #     test_distance_array.append(
    #         ((splineY[-1] - splineY[-2]) ** 2 + (splineZ[-1] - splineZ[-2]) ** 2) ** 0.5 + test_distance_array[-1])


    fig = plt.figure(2)
    fig.suptitle('spline test, (curve vs sorted)')
    ax = fig.add_subplot(111)
    plt.plot(new_zs, new_ys)
    plt.plot(z_sorted, y_sorted)
    ax.set_xlabel('z')
    ax.set_ylabel('y')
    plt.show()

    # test_half = test_distance_array[-1]
    half_distance = new_distance_array[-1] / 2
    print('___________________ new distance array last value:' + str(new_distance_array[-1]))

    # The value below may need to be changed. It is currently where the
    # levator plate is located in the x coordinates
    # middle_x = -2
    mid_plate_y = float(new_curve_y(half_distance))
    mid_plate_z = float(new_curve_z(half_distance))

    dist_for_slope = 4

    y_before = float(new_curve_y(half_distance - dist_for_slope))
    z_before = float(new_curve_z(half_distance - dist_for_slope))

    y_after = float(new_curve_y(half_distance + dist_for_slope))
    z_after = float(new_curve_z(half_distance + dist_for_slope))

    # # find the slope at the middle of the levator plate (where the ICM is)
    slope = (y_after-y_before)/(z_after-z_before)

    # ICM slope is negative and inverse
    # # find perpendicular slope which is the slope of the ICM line
    perp_slope = -1/slope

    angle = np.arctan(1/perp_slope)




#######################################
    # Working above on finding the center of LP and slope at that point
###########################################


    zs = np.zeros(len(ys))
    for index, (z,y) in enumerate(zip(zs,ys)):
        # print(index)
        # print(z*np.cos(angle)-y*np.sin(angle))

        zs[index] = y*np.sin(angle)
        ys[index] = y*np.cos(angle)
        # print(index, z, y)

    # print('ICM rotated')
    # print(xs)
    # print(ys)
    # print(zs)




    # index = df.index
    # condition = df["point_number"] == str(8)
    # row_num = index[condition]
    # col_num = df.columns.get_loc('ICM_FEA_z')
    # FEA_z = float(df.iloc[row_num,col_num])
    # col_num = df.columns.get_loc('ICM_FEA_y')
    # FEA_y = float(df.iloc[row_num,col_num])

    ### edit: what the middle node is for the ICM
    #TODO: Take number of points in file - 1 then divide by 2
    bottom_node_position = int((len(df) - 1) / 2)
    horiz_shift = mid_plate_z - zs[bottom_node_position]
    vert_shift = mid_plate_y - ys[bottom_node_position]

    # print('ICM shifted amount')
    # print(FEA_z)
    # print(FEA_y)
    # print(horiz_shift)
    # print(vert_shift)

    zs = zs + horiz_shift
    ys = ys + vert_shift

    # print('ICM shifted')
    # print(zs)
    # print(ys)

    scaling_center_z = zs[bottom_node_position]
    scaling_center_y = ys[bottom_node_position]

    # print('ICM scaling center')
    # print(scaling_center_z)
    # print(scaling_center_y)

    zs = (zs-scaling_center_z)*scale + scaling_center_z
    ys = (ys-scaling_center_y)*scale + scaling_center_y

    # print('ICM scaled')
    # print(zs)
    # print(ys)

    ICM_CPs = np.c_[xs,ys,zs]

    if isMod:
        ICM_CPs_mod_x = xs
        ICM_CPs_mod_y = ys
        ICM_CPs_mod_z = zs
        return ICM_CPs, ICM_CPs_mod_x, ICM_CPs_mod_y, ICM_CPs_mod_z, perp_slope, mid_plate_y, mid_plate_z
    else:
        return ICM_CPs

    # print("ICM %%%%%%%%%%%%%%%%%%", LP_CPs_mod)


def checkPlot(title, initialPoints, modPoints):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(title)
    ax.scatter(initialPoints[:, 0], initialPoints[:, 2], initialPoints[:, 1], c='b', marker='+',
               label='initial')
    ax.scatter(modPoints[:, 0], modPoints[:, 2], modPoints[:, 1], c='r', marker='+', label='mod')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    plt.legend()
    plt.show()
