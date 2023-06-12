import numpy as np


# Author: Yutian Yang
# Created: 5/26/2023
# Description: This is a file of all the interaction functions with points that treat a scar
# Version: 1.0
# Email: yyt542985333@gmail.com

def get_points_x_from_mid(file_name, part_name, mid_x, threshold_x_from_mid):
    from lib.IOfunctions import extractPointsForPartFrom

    """
    Get the points within a specified range of x values from the mid_x point in a given file and part.

    Parameters:
        file_name (str): The name of the file containing the points.
        part_name (str): The name of the part containing the points.
        mid_x (float): The x-coordinate of the mid-point.
        threshold_x_from_mid (float): The threshold distance from the mid-point in the x-direction.

    Returns:
        filtered_points (list): The filtered points within the specified range.
        filtered_count (int): The count of filtered points.
        total_count (int): The total count of all points in the part.

    Description:
        This function reads the points from the specified file and part, and filters out the points that fall within
        the specified range of x values from the mid_x point. The range is determined by adding/subtracting the
        threshold_x_from_mid value from the mid_x value. The function returns the filtered points, the count of filtered
        points, and the total count of all points in the part.

    Example:
        >>> file_name = "points_file.csv"
        >>> part_name = "Part1"
        >>> mid_x = 5.0
        >>> threshold_x_from_mid = 2.0
        >>> filtered_points, filtered_count, total_count = get_points_x_from_mid(file_name, part_name, mid_x, threshold_x_from_mid)
        >>> print(filtered_points)
        [2, 3, 4, 5, 6]
        >>> print(filtered_count)
        5
        >>> print(total_count)
        10

    Notes:
        - The function assumes that the points file is in a format compatible with the `extractPointsForPartFrom()` function.
        - The part name (part_name) should match the exact name used in the file.
        - The mid_x value should correspond to the desired x-coordinate of the midpoint.
        - The threshold_x_from_mid value determines the range of x values to be considered around the midpoint.
        - The filtered_points list contains the indices of the filtered points, not the actual point coordinates.
    """
    # Generate an array including all the points from the given file with part
    all_points = np.array(extractPointsForPartFrom(file_name, part_name))
    # print(all_points)

    print("mid x: ", mid_x)

    # Calculate threshold z value
    threshold_left = mid_x - threshold_x_from_mid
    threshold_right = mid_x + threshold_x_from_mid

    # Get the original sort number of each filtered point
    filtered_points = []
    for i, point in enumerate(all_points):
        if threshold_left < point[0] < threshold_right:
            filtered_points.append(i + 1)

    # Return the filtered points, filtered count, and total count
    return filtered_points, len(filtered_points), all_points.shape[0]

'''
Function: get_points_below
'''


def get_points_below(file_name, part_name, threshold_percentage):
    from lib.IOfunctions import extractPointsForPartFrom

    """
    Retrieves the points below a specified threshold percentage along the z-axis from a given file and part.

    Parameters:
        file_name (str): The name of the file to extract points from.
        part_name (str): The name of the part to extract points from.
        threshold_percentage (float): The threshold percentage below which points are considered.

    Returns:
        tuple: A tuple containing the following elements:
            - filtered_points (list): A list of the original sort numbers of the filtered points.
            - filtered_count (int): The number of points below the threshold.
            - total_count (int): The total number of points in the part.

    Description:
        This function reads the content of the specified file and extracts the points belonging to the specified part.
        It then determines the minimum and maximum z-values among all the points. Based on the threshold percentage,
        a threshold z-value is calculated. Points whose z-coordinate is below the threshold are considered as filtered points.

        The function returns a tuple containing the original sort numbers of the filtered points, the count of filtered points,
        and the total count of points in the part.

    Example:
        >>> file_name = "input.inp"
        >>> part_name = "GivenPart"
        >>> threshold_percentage = 30
        >>> filtered_points, filtered_count, total_count = get_points_below(file_name, part_name, threshold_percentage)
        >>> print(filtered_points)
        [1, 3, 4, 6, 8, 10]
        >>> print(filtered_count)
        6
        >>> print(total_count)
        10
    """
    # Generate an array including all the points from the given file with part
    all_points = np.array(extractPointsForPartFrom(file_name, part_name))

    # Find the minimum and maximum z values
    min_z = np.min(all_points[:, 2])
    max_z = np.max(all_points[:, 2])

    # Calculate threshold z value
    threshold = threshold_percentage * 0.01 * (max_z - min_z) + min_z

    # Get the original sort number of each filtered point
    filtered_points = []
    for i, point in enumerate(all_points):
        if point[2] < threshold:
            filtered_points.append(i + 1)

    # Return the filtered points, filtered count, and total count
    return filtered_points, len(filtered_points), all_points.shape[0]
