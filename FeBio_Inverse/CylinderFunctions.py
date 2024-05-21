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
import xml.etree.ElementTree as ET


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


def plot_3d_points(pointslist):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Create subplot only once

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    markers = ['o', 's', '^', 'v', '<', '>', '1', '2']  # Define markers

    for idx, points in enumerate(pointslist):
        # Plot points with a unique color and marker
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors[idx], marker=markers[idx])

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set aspect ratio
    ax.set_box_aspect([1, 1, 1])

    plt.show()




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
    #TODO: Determine the radii for use later
    cylinderpoints = generate_annular_cylinder_points(inner_radius, outer_radius, height, num_points)

    return cylinderpoints


def extractCoordinatesFromPart(root, partname, deformed_points_list):
    """
        Extracts coordinates of nodes belonging to a specific part from an XML representation.

        Parameters:
            root (ElementTree.Element): The root element of the XML representing the finite element model.
            partname (str): The name of the part from which coordinates need to be extracted.
            deformed_points_list (list): A list of tuples [(node_id, coordinates), ...] containing node IDs and their
            corresponding coordinates.

        Returns:
            list: A list of lists containing node IDs and their corresponding coordinates, e.g.,
                                                            [[node_id1, (x1, y1, z1)], [node_id2, (x2, y2, z2)], ...].

        Description:
            This function searches for the elements representing hexahedral finite elements of the specified 'partname'
            in the XML structure.
            It then iterates through each element to extract the node IDs associated with them.
            After that, it looks for nodes in the XML structure and checks if their IDs match with the extracted node
            IDs.
            If a match is found, it retrieves the coordinates from the 'deformed_points_list' and adds them to the
            output list.
            The function finally returns a list containing the node IDs and their corresponding coordinates for
            the specified part.

        Note:
            - The function assumes the XML structure contains elements representing finite elements with
             hexahedral (hex8) type.
            - It also assumes that the XML structure includes node elements with IDs corresponding to those specified
             in the element definitions.
            - The 'deformed_points_list' should contain tuples of node IDs and their coordinates.
        """

    elements = root.find('.//Elements[@type="hex8"][@name="{}"]'.format(partname))
    elem_ids_set = set()
    coordinatesarray = []

    if elements is not None:  # Check if the Elements tag is found
        # Find all elem elements within the Elements
        elem_elements = elements.findall('elem')

        # Iterate over each elem element
        for elem in elem_elements:
            # Extract the text content containing the numbers
            numbers_text = elem.text.strip()
            # Convert text to list of integers
            elem_ids = [int(num.strip()) for num in numbers_text.split(',')]
            # Add the elem IDs to the set
            elem_ids_set.update(elem_ids)

    for node in root.findall('.//Nodes/node'):
        # Get the ID of the current node
        current_node_id = int(node.get('id'))

        # Check if the current node ID is in the set of quad IDs
        if current_node_id in elem_ids_set:
            current = []
            inner_text = node.text.strip()

            # Find the coordinates from deformed_points_list using current_node_id
            for item in deformed_points_list:
                if item[0] == current_node_id:
                    coordinates = item[1]  # Extract coordinates from deformed_points_list

                    # Append coordinates to current
                    current.append(current_node_id)
                    current.append(coordinates)
                    coordinatesarray.append(current)

    return coordinatesarray



def replaceCoordinatesGivenNodeId(root, coordinates):
    """
        Replaces the coordinates of a specific 'Nodes' element in an FEBio file with new coordinates.

        Parameters:
            file_name (str): The name of the FEBio file.
            nodes_name (str): The name of the 'Nodes' element to replace.
            coordinates_list (list): A list of coordinate tuples [(x1, y1, z1), (x2, y2, z2), ...] for the new coordinates.

        Returns:
            None

        Description:
            This function reads the FEBio file specified by 'file_name' and finds the 'Mesh' element.
            It searches for the 'Nodes' element with the specified 'name' attribute and removes it if found.
            Then, a new 'Nodes' element is created with the same 'name' attribute, and 'node' elements are added with the new coordinates.
            The updated XML tree is then written back to the file.

            If the 'Mesh' element or the 'Nodes' element with the specified 'name' attribute is not found, an appropriate message is printed.

        Example:
            >>> file_name = "input.feb"
            >>> node_name = "MyNodes"
            >>> coordinates_list = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
            >>> replace_node_in_feb_file(file_name, nodes_name, coordinates_list)

        Note:
            - This function assumes that the FEBio file exists and is formatted correctly as an XML file.
            - The 'node_name' should match the 'name' attribute of the 'Nodes' element to be replaced.
            - The 'coordinates_list' should contain coordinate tuples in the order [(x1, y1, z1), (x2, y2, z2), ...].
            - The function modifies the FEBio file in-place and does not create a new file.
        """

    updates_dict = {node_id: coords for node_id, coords in coordinates}
    # Iterate over all node elements in the XML
    for node in root.findall('.//Nodes/node'):
        # Get the id of the current node
        node_id = int(node.get('id'))

        # Check if this node id is in the updates
        if node_id in updates_dict:
            # Get the new coordinates for this node
            new_coords = updates_dict[node_id]

            # Update the text of the node with the new coordinates
            node.text = ','.join(map(str, new_coords))



def extractCoordinatesFromSurfaceName(root, surface_name):
    """
        Extracts coordinates of nodes belonging to a specific surface from an XML representation.

        Parameters:
            root (ElementTree.Element): The root element of the XML representing the finite element model.
            surface_name (str): The name of the surface from which coordinates need to be extracted.

        Returns:
            list: A list of lists containing coordinates of nodes belonging to the specified surface.

        Description:
            This function searches for the specified 'surface_name' within the XML structure.
            It then iterates through each 'quad4' element within the surface to extract the node IDs associated with them.
            After that, it looks for nodes in the XML structure and checks if their IDs match with the extracted node IDs.
            If a match is found, it retrieves the coordinates and adds them to the output list.
            The function finally returns a list containing the coordinates of nodes belonging to the specified surface.

        Note:
            - The function assumes the XML structure contains 'Surface' tags with 'quad4' elements representing surface elements.
            - It also assumes that the XML structure includes node elements with IDs corresponding to those specified in the 'quad4' elements.
        """

    # Set to store quad IDs
    quad_ids_set = set()
    coordinatesarray = []

    # Find the specific Surface tag within the root
    surface = root.find('.//Surface[@name="{}"]'.format(surface_name))

    if surface is not None:  # Check if the Surface tag is found
        # Find all quad4 elements within the Surface
        quad4_elements = surface.findall('.//quad4')

        # Iterate over each quad4 element
        for quad in quad4_elements:
            # Extract the text content containing the numbers
            numbers_text = quad.text.strip()

            # Convert text to list of integers
            quad_ids = [int(num.strip()) for num in numbers_text.split(',')]

            # Add the quad IDs to the set
            quad_ids_set.update(quad_ids)

        # Iterate over all nodes in the root
        for node in root.findall('.//Nodes/node'):
            # Get the ID of the current node
            current_node_id = int(node.get('id'))

            # Check if the current node ID is in the set of quad IDs
            if current_node_id in quad_ids_set:
                # Extract the inner text of the current node containing coordinates
                inner_text = node.text.strip()

                # Convert coordinates text to list of floats
                coordinates = [float(coord) for coord in inner_text.split(',')]

                # Append coordinates to initial and final control points arrays
                coordinatesarray.append(coordinates)

        return coordinatesarray

def get_inital_points_from_parts(root, part_list):
    control_points = []

    for part in part_list:
        elements = root.find('.//Elements[@type="hex8"][@name="{}"]'.format(part))
        elem_ids_set = set()
        if elements is not None:
            elem_elements = elements.findall('elem')
            for elem in elem_elements:
                numbers_text = elem.text.strip()
                elem_ids = [int(num.strip()) for num in numbers_text.split(',')]
                elem_ids_set.update(elem_ids)
        for node in root.findall('.//Nodes/node'):
            current_node_id = int(node.get('id'))
            if current_node_id in elem_ids_set:
                coordinates = [float(coordinate) for coordinate in node.text.split(',')]
                control_points.append([current_node_id, coordinates])

    return control_points





# Define the new function for point morphing
def morph_points(initial_controlpoints, final_controlpoints, initial_coordinates, extract_points_dict):

    # Use RBF to find differences between both cylinders
    rbf = RBF(initial_controlpoints, final_controlpoints, func='thin_plate_spline')

    # Call rbf to return deformed points given extract_points
    deformed_coordinates = rbf(initial_coordinates)

    deformed_points_dict = {key: deformed_coordinates[i] for i, key in enumerate(extract_points_dict.keys())}
    deformed_points = [[key, value] for key, value in deformed_points_dict.items()]

    return deformed_points