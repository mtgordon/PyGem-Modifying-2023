# Import PyGeM's RBF and other necessary libraries
import pygem as pg
import numpy as np
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import numpy as np
from lib import IOfunctions
import numpy as np
import math
from pygem import RBF


febio_file_name = "D:\\Gordon\\Automate FEB Runs\\2024_4_29 auto\\Base File\\Basic_Cylinder_Pressure.feb"
node_name = "Object5"
extract_points = IOfunctions.extract_coordinates_list_from_feb(febio_file_name, node_name)
#TODO: Input Parameters for Cylinder Creation
num_points = 200

def findLargestZ(): # Uses extract_points within function
    maxz = 0
    for tuple in extract_points:
        maxz = max(maxz, tuple[2])

    return maxz

height = findLargestZ()

#plot_3d_points(extract_points)

'''
   This function utilizes the "extract_coordinates_dic_from_feb" function which returns a dictionary of x,y,z coordinates
   To make these coords more useful this function sorts the dictionary into 3 arrays containing all x-values, y-values, & z-values.
'''
def separate_xyz_coords(point_dict):
   # Initialize empty arrays to hold x, y, and z coordinates
   x_values = []
   y_values = []
   z_values = []


   # Loop through the dictionary items
   for key, coords in point_dict.items():
       # Ensure the coordinates have length 3
       if len(coords) == 3:
           x_values.append(coords[0])
           y_values.append(coords[1])
           z_values.append(coords[2])


   return x_values, y_values, z_values


# x_coords, y_coords, z_coords = separate_xyz_coords(extract_points)

def generate_annular_cylinder_points(inner_radius, outer_radius, height, num_points):
  x = []
  y = []
  z = []
  for i in np.arange(0, height, .2):
     for j in range(num_points):
        x.append(outer_radius * (np.cos(j * 2 * np.pi / num_points)))
        y.append(outer_radius * (np.sin(j * 2 * np.pi / num_points)))
        z.append(i)

        x.append(inner_radius * (np.cos(j * 2 * np.pi / num_points)))
        y.append(inner_radius * (np.sin(j * 2 * np.pi / num_points)))
        z.append(i)

  # Combine x, y, and z coordinates
  points = np.column_stack((x, y, z))


  return points



def plot_3d_points(points, fig):
   #fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')


   # Plot points
   ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')


   # Set labels
   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')


   # Set aspect ratio
   ax.set_box_aspect([1, 1, 1])


   plt.show()


#plot_3d_points(cylinder1points)


"""
Function: determineRadiiFromFEB
This method utilizes the .feb files extracted points which are parsed from the .feb file previously using
IOfunctions.extract_coordinates_list_from_feb, We then convert this to a np array and take the first
two nodes x-values which are the inner and outer radius correspondingly.

TEST INPUT: [(x1, y1, z1), (x2, y2, z2)]
TEST OUTPUT: [Inner_Radius, Outer_Radius]
"""
def determineRadiiFromFEB(extracted_points):
    # convert array containing tuples to np array
    extract_points = np.array(extracted_points)

    for i in range(2):  # Loop over the range from 0 to 1 (inclusive)
        if i == 0:
            inner_radius = extract_points[i][0]

        if i == 1:
            outer_radius = extract_points[i][0]

    # create cylinder using our found inner & outer radius
    cylinderpoints = generate_annular_cylinder_points(inner_radius, outer_radius, height, num_points)

    return cylinderpoints

cylinder1points = determineRadiiFromFEB(extract_points)

# Cylinder which we are morphing. This is the Cylinder that MATTERS!!
cylinder2points = generate_annular_cylinder_points(2,3,height,num_points)
#plot_3d_points(cylinder2points)

#TODO: This uses RBF Interpolator from the SciPy Library, Currently unused, because we are using PyGem

# rbf = RBF(original_control_points = cylinder1points, deformed_control_points = cylinder2points, func='thin_plate_spline', radius = 10)
# rbf(cylinder1points)
# Initialize the RBF with the original control points and their deformations
#rbf = RBFInterpolator(cylinder1points, deformations, kernel='thin_plate_spline')



#TODO: This uses Pygem, only works on COBS for rn
rbf = RBF(cylinder1points, cylinder2points, func='thin_plate_spline')

# Apply the RBF transformation to the first set of points
extract_points = np.array(extract_points)
deformed_points = rbf(extract_points)


# Plot the deformed points
#plot_3d_points(deformed_points)
deformed_points_list = []
for tuple in deformed_points:
    deformed_points_list.append(list(tuple))

# print(deformed_points_list)
# print(extract_points)
IOfunctions.replace_node_in_new_feb_file(febio_file_name, node_name, "extract_cylinder.feb", deformed_points_list)
