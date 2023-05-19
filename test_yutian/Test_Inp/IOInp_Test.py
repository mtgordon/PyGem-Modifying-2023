from test_yutian.functions import IOfunctions_Inp as io

file_name = "../Normal_Generic_copy.inp"
part_name = "OPAL325_AVW_v6"

# nodes_list = [1,2,5]

# points_list = [
#     (11, 2, 3),
#     (4, 5, 16),
#     (7, 8, 9),
#     (10, 11, 12)
# ]
# element_list = [
#     1, 2, 3, 4, 5, 31, 31, 31, 31, 3, 23, 123, 213, 213, 231, 3, 211, 3, 213, 213, 11, 3, 3, 132, 13, 13, 13, 13, 13, 3,
#     131
# ]
#
# Nset_list = [
#     1, 2, 3, 4, 5, 31, 31, 31, 31, 3, 23, 123, 213, 213, 231, 3, 211, 3, 213, 213, 11, 3, 3, 132, 13, 13, 13, 13, 13, 3,
#     131
# ]

"""
TEST: io.write_Nset_to_inp_file
"""
# io.write_Nset_to_inp_file(file_name, "OPAL325_ATFP", 999, "AWP_dis", Nset_list)

"""
TEST: io.write_new_part_to_inp_file
"""
# io.write_new_part_to_inp_file(file_name, part_name, "element", points_list, element_list)

"""
TEST: io.addToVals
"""
# connections = [[], [], [], []]
# nums = [1, 2, 3, 4]
# updated_connections = io.addToVals(connections, nums)
# print(updated_connections)


"""
TEST: io.get_interconnections
"""
connection_list = io.get_interconnections(file_name, part_name)
for connection in connection_list:
    print(connection)


"""
TEST: io.write_points_below_excluded_to_inp
"""
# filtered_points, filtered_count, total_count = io.get_points_below(file_name, part_name, 37)
# print("Filtered points number: ", filtered_points)
# print("Filtered count: ", filtered_count)
# print("Total count: ", total_count)
#
# io.write_points_below_excluded_to_inp(file_name, part_name, 37, 999, "AWP")