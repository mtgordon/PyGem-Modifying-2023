import IOfunctions_Inp as io
import numpy as np


# file name and part name are subjected to change.
# file_name = "../../Normal_Generic.inp"
# file_name = "../Normal_Generic_copy.inp"
# file_name = "../test_writting.inp"

# ## Test the writing funciton
# io.write_Nset_to_inp_file(file_name, part_name, 999, "AWP_dis", ranked_filtered_points)
#
def get_points_below(file_name, part_name, threshold_percentage):
    # Generate an array including all the points from the given file with part
    all_points = np.array(io.extractPointsForPartFrom(file_name, part_name))

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



# Example usage
file_name = "../Normal_Generic_copy.inp"
part_name = "OPAL325_AVW_v6"
# part_name = "OPAL325_CL_v6"


# threshold_percentage = 37
#
# filtered_points, filtered_count, total_count = get_points_below(file_name, part_name, threshold_percentage)
# print("Filtered points number: ", filtered_points)
# print("Filtered count: ", filtered_count)
# print("Total count: ", total_count)

# Get connection TRUE test
# points_list, dic = io.extractPointsForPartFrom(file_name, part_name, True)
#
# for coordinates in points_list:
#     print(coordinates)
#
# for key, value in dic.items():
#     print(f"Node {key}: {value}")

exclude_list = io.get_dis_Nset_points_list(file_name,part_name,"AWP")
print(exclude_list)