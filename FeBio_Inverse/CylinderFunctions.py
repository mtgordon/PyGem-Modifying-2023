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

def findLargestZ(extract_points_dict):
    """
    Finds the largest z-coordinate value from a dictionary of points.

    Parameters:
        extract_points_dict (dict): A dictionary where each key is a node ID and the value is a list of coordinates, e.g.,
                                    {node_id1: [x1, y1, z1], node_id2: [x2, y2, z2], ...}.

    Returns:
        float: The largest z-coordinate value found in the dictionary of points.
    """
    maxz = float('-inf')  # Initialize to negative infinity to handle all possible z-values
    for coords in extract_points_dict.values():
        maxz = max(maxz, coords[2])

    return maxz

def separate_xyz_coords(point_dict):
    """
        Separates the x, y, and z coordinates from a dictionary of coordinates.

        Parameters:
            point_dict (dict): A dictionary containing node IDs as keys and their corresponding coordinates as values,
                               e.g., {node_id1: (x1, y1, z1), node_id2: (x2, y2, z2), ...}.

        Returns:
            list: An array containing all x-values.
            list: An array containing all y-values.
            list: An array containing all z-values.

        Description:
            This function takes a dictionary of coordinates where the keys are node IDs and the values are tuples of
            (x, y, z) coordinates. It separates these coordinates into three arrays containing all x-values, y-values, and
            z-values, respectively.
    """
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
    """
        Generates points to create an annular cylinder mesh.

        Parameters:
            inner_radius (float): The inner radius of the annular cylinder.
            outer_radius (float): The outer radius of the annular cylinder.
            height (float): The height of the annular cylinder.
            num_points (int): The number of points to generate around the circumference of the cylinder.

        Returns:
            numpy.ndarray: An array containing the generated points with shape (N, 3), where N is the total number of points
                           and each row represents the (x, y, z) coordinates of a point.

        Description:
            This function generates points to create an annular cylinder mesh. It iterates over the height of the cylinder
            and generates points at each height level around the circumference. For each height level, it generates
            'num_points' points evenly spaced around the circumference at both the inner and outer radii.
    """
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


def plot_3d_points(points_dict):
    """
        Plots 3D points.

        Parameters:
            pointslist (list): A list of numpy arrays where each array contains 3D points with shape (N, 3),
                               representing (x, y, z) coordinates.

        Returns:
            None

        Description:
            This function plots 3D points using matplotlib. It takes a list of numpy arrays, where each array
            represents a set of 3D points. It plots each set of points with a unique color and marker.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Create subplot only once

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    markers = ['o', 's', '^', 'v', '<', '>', '1', '2']  # Define markers

    for idx, (id, points) in enumerate(points_dict.items()):

        # Plot points with a unique color and marker
        ax.scatter(points[0], points[1], points[2], c=colors[0],
                   marker=markers[0], label=f'ID {id}')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set aspect ratio
    ax.set_box_aspect([1, 1, 1])

    plt.show()


def plot_pc_points(coords_array, z_value=0, color='b', marker='o', label='Points'):
    """
    Plots 3D points.

    Parameters:
        coords_array (list): A list of 2D points, where each point is a list or tuple [x, y].
        z_value (float): The z-coordinate value to be used for all points (default is 0).
        color (str): Color of the points.
        marker (str): Marker style for the points.
        label (str): Label for the points in the plot legend.

    Returns:
        None

    Description:
        This function plots 3D points using matplotlib. It takes a list of 2D points (x and y coordinates).
        It assigns the specified z-coordinate value to all points and plots them.
    """
    points = np.array(coords_array)

    x, y = points[:, 0], points[:, 1]
    z = np.full_like(x, z_value)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=color, marker=marker, label=label)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_box_aspect([1, 1, 1])

    plt.legend()
    plt.show()


def pair_points(points):
    # Convert all elements to floats
    cleaned_points = [float(item) if isinstance(item, np.ndarray) else item for item in points]

    # Determine the midpoint for splitting into x and y values
    midpoint = len(cleaned_points) // 2

    # Split the list into x and y values
    x_values = cleaned_points[:midpoint]
    y_values = cleaned_points[midpoint:]

    # Pair the x and y values
    paired_points = list(zip(x_values, y_values))
    print('paired_points: ', paired_points)
    print('\n')
    return paired_points

def determineRadiiFromFEB(root, cylinder_part):
    """
    Determines the inner and outer radii of a cylinder from a point cloud.

    Parameters:
        root (ElementTree.Element): The root element of the XML representing the finite element model.
        cylinder_part (list): A list containing the names of parts representing the cylinder.

    Returns:
        float: The inner radius of the cylinder.
        float: The outer radius of the cylinder.

    Description:
        This function extracts the coordinates of nodes belonging to a specific part of the cylinder from the XML
        representation. It then calculates the midline of the cylinder by averaging the x and y coordinates of all
        points. Using this midline, it calculates the Euclidean distance of each point from the midline and identifies
        the smallest and largest distances, which correspond to the inner and outer radii of the cylinder, respectively.
    """
    # Extract coordinates from the part that is a cylinder
    cylinder_dict = get_initial_points_from_parts(root, cylinder_part)

    # Extract coordinates from the dictionary
    coordinates = np.array(list(cylinder_dict.values()))

    # Calculate the midline of the cylinder (assuming cylinder axis along z-axis)
    mid_x = np.mean(coordinates[:, 0])
    mid_y = np.mean(coordinates[:, 1])

    # Calculate distances from each point to the midline
    distances = np.array([distance_to_midline(x, y, mid_x, mid_y) for x, y, z in coordinates])

    # Find the smallest and largest distances
    inner_radius = np.min(distances)
    outer_radius = np.max(distances)

    return inner_radius, outer_radius


def distance_to_midline(x, y, mid_x, mid_y):
    """
       Helper function to compute the Euclidean distance from a point to the midline (cylinder axis).

       Parameters:
           x (float): The x-coordinate of the point.
           y (float): The y-coordinate of the point.
           mid_x (float): The x-coordinate of the midline.
           mid_y (float): The y-coordinate of the midline.

       Returns:
           float: The Euclidean distance from the point to the midline.

       Description:
           This function calculates the Euclidean distance from a given point (x, y) to the midline of the cylinder,
           defined by the average x and y coordinates (mid_x, mid_y).
       """
    return np.sqrt((x - mid_x) ** 2 + (y - mid_y) ** 2)


def extractCoordinatesFromPart(root, part_name, deformed_points_list):
    """
    Extracts coordinates of nodes belonging to a specific part from an XML representation.

    Parameters:
        root (ElementTree.Element): The root element of the XML representing the finite element model.
        part_name (str): The name of the part from which coordinates need to be extracted.
        deformed_points_list (list): A list of tuples [(node_id, coordinates), ...] containing node IDs and their
        corresponding coordinates.

    Returns:
        dict: A dictionary containing node IDs as keys and their corresponding coordinates as values, e.g.,
              {node_id1: (x1, y1, z1), node_id2: (x2, y2, z2), ...}.

    Description:
        This function searches for the elements representing hexahedral finite elements of the specified 'partname'
        in the XML structure.
        It then iterates through each element to extract the node IDs associated with them.
        After that, it looks for nodes in the XML structure and checks if their IDs match with the extracted node
        IDs.
        If a match is found, it retrieves the coordinates from the 'deformed_points_list' and adds them to the
        output dictionary.
        The function finally returns a dictionary containing the node IDs and their corresponding coordinates for
        the specified part.
    """

    elements = root.find('.//Elements[@type="hex8"][@name="{}"]'.format(part_name))
    elem_ids_set = set()
    coordinates_dict = {}

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

        # Check if the current node ID is in the set of elem IDs
        if current_node_id in elem_ids_set:
            # Find the coordinates from deformed_points_list using current_node_id
            for item in deformed_points_list:
                if item[0] == current_node_id:
                    coordinates = item[1]  # Extract coordinates from deformed_points_list

                    # Add to dictionary
                    coordinates_dict[current_node_id] = coordinates

    return coordinates_dict


def replaceCoordinatesGivenNodeId(root, coordinates_dict):
    """
    Replaces the coordinates of nodes in an FEBio file based on node IDs with new coordinates.

    Parameters:
        root (ElementTree.Element): The root element of the XML representing the finite element model.
        coordinates_dict (dict): A dictionary containing node IDs as keys and their corresponding coordinates as values, e.g.,
                                 {node_id1: (x1, y1, z1), node_id2: (x2, y2, z2), ...}.

    Returns:
        None (The feb file is updated)

    Description:
        This function iterates through all 'node' elements within the 'Nodes' section of the provided XML root.
        It updates the coordinates of each node based on the provided dictionary of coordinates.
        Each key in the dictionary is a node ID, and its value is a tuple of new coordinates.
        The function finds the 'node' elements matching the node IDs and updates their text with the new coordinates.

    Example:
        >>> root = ...  # Parse your XML root element
        >>> coordinates_dict = {1: (1.0, 2.0, 3.0), 2: (4.0, 5.0, 6.0), 3: (7.0, 8.0, 9.0)}
        >>> replaceCoordinatesGivenNodeId(root, coordinates_dict)
    """
    # Iterate over all node elements in the XML
    for node in root.findall('.//Nodes/node'):
        # Get the id of the current node
        node_id = int(node.get('id'))

        # Check if this node id is in the updates
        if node_id in coordinates_dict:
            # Get the new coordinates for this node
            new_coords = coordinates_dict[node_id]

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
    coordinates_array = []

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
                coordinates_array.append(coordinates)

        return coordinates_array


def get_initial_points_from_parts(root, part_list):
    """
    Extracts initial coordinates of nodes belonging to specified parts from an XML representation.

    Parameters:
        root (ElementTree.Element): The root element of the XML representing the finite element model.
        part_list (list): A list of part names from which coordinates need to be extracted.

    Returns:
        dict: A dictionary containing node IDs as keys and their corresponding coordinates as values, e.g.,
              {node_id1: [x1, y1, z1], node_id2: [x2, y2, z2], ...}.

    Description:
        This function searches for the elements representing hexahedral finite elements of the specified parts
        in the XML structure.
        It then iterates through each element to extract the node IDs associated with them.
        After that, it looks for nodes in the XML structure and checks if their IDs match with the extracted node IDs.
        If a match is found, it retrieves the coordinates from the node elements and adds them to the output dictionary.
        The function finally returns a dictionary containing the node IDs and their corresponding coordinates for
        the specified parts.

    Example:
        >>> root = ...  # Parse your XML root element
        >>> part_list = ['part1', 'part2']
        >>> initial_points_dict = get_initial_points_from_parts(root, part_list)

    Note:
        - This function assumes that the XML structure includes 'node' elements with 'id' attributes within a 'Nodes' section.
        - The function modifies the XML tree in-place.
    """

    control_points = {}

    for part in part_list:
        # Find the Elements tag for the current part
        elements = root.find('.//Elements[@type="hex8"][@name="{}"]'.format(part))
        elem_ids_set = set()

        if elements is not None:
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

        # Iterate over all node elements in the XML
        for node in root.findall('.//Nodes/node'):
            # Get the ID of the current node
            current_node_id = int(node.get('id'))

            # Check if the current node ID is in the set of elem IDs
            if current_node_id in elem_ids_set:
                # Extract coordinates from the node text
                coordinates = [float(coordinate) for coordinate in node.text.split(',')]
                # Add the node ID and coordinates to the dictionary
                control_points[current_node_id] = coordinates

    return control_points


def morph_points(initial_control_points, final_control_points, initial_coordinates, extract_points_dict):
    """
        Morphs points from the initial state to the final state using radial basis function (RBF) interpolation.

        Parameters:
            initial_control_points (numpy.ndarray)
            final_control_points (numpy.ndarray)
            initial_coordinates (numpy.ndarray)
            extract_points_dict (dict)

        Returns:
            list: A list of lists containing node IDs and their corresponding deformed coordinates after morphing,
                  e.g., [[node_id1, [x1', y1', z1']], [node_id2, [x2', y2', z2']], ...].

        Description:
            This function performs point morphing from the initial state to the final state using radial basis function
            (RBF) interpolation. It uses the initial and final control points to define the transformation between the
            two states. The RBF interpolation is applied to all initial coordinates to obtain their deformed positions
            in the final state.
    """
    # Use RBF to find differences between both cylinders
    rbf = RBF(initial_control_points, final_control_points, func='thin_plate_spline')

    # Call rbf to return deformed points given extract_points
    deformed_coordinates = rbf(initial_coordinates)

    deformed_points_dict = {key: deformed_coordinates[i] for i, key in enumerate(extract_points_dict.keys())}
    deformed_points = [[key, value] for key, value in deformed_points_dict.items()]

    return deformed_points
