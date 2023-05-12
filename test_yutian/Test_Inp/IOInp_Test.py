from test_yutian.functions import IOfunctions_Inp as io

file_name = "../Normal_Generic_copy.inp"
part_name = "OPAL325_AVW_v6"

nodes_list = [1,2,5]

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

# Test the writing funciton
# io.write_Nset_to_inp_file(file_name, "OPAL325_ATFP", 999, "AWP_dis", Nset_list)

## Test the writting fucntion
# io.write_new_part_to_inp_file(file_name, part_name, "element", points_list, element_list)

print(io.extractNodesFromINP(file_name, part_name, nodes_list))