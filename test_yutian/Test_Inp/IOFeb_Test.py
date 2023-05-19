from test_yutian.functions import IOfunctions_Feb as io

file_name = "../First CFD_test.feb"
node_name = "Object1"
new_file_name = "../FEB_TEST.feb"

"""
TEST: io.extract_coordinates_dic_from_feb
"""
# nodes_dictionary = io.extract_coordinates_dic_from_feb(file_name, node_name)
# print("The dictionary of all the points: ", nodes_dictionary)
#
"""
TEST: io.extract_coordinates_list_from_feb
"""
# points_list = io.extract_coordinates_list_from_feb(file_name, node_name)
# print("The list of all the point: ")
# for item in points_list:
#     print(item)
#
# print(nodes_dictionary.get(1))
# print(nodes_dictionary.get(3))
# print(nodes_dictionary.get(4))

"""
TEST: io.extract_node_id_list_from_feb
"""
# node_id_list = io.extract_node_id_list_from_feb(file_name, node_name)
# print("The list of ids of all the nodes: ", node_id_list)
#
coordinates_list = [(1, 11, 1), (1, 11, 1), (1, 11, 1), (1, 11, 1), (1, 1, 1)]

"""
TEST: io.replace_node_in_feb_file
"""
# io.replace_node_in_feb_file(file_name, node_name, coordinates_list)


"""
TEST: extract_coordinates_list_from_feb
Get connection TRUE test:
"""

# points_list, connection_dic = io.extract_coordinates_list_from_feb(file_name, node_name, True)
# for coordinates in points_list:
#     print(coordinates)
#
# for key, value in connection_dic.items():
#     print(f"Node {key}: {value}")
#
# print(io.extractNodesFromFEB(file_name, node_name, [1, 3, 5]))
#
# print(points_list[0], points_list[2], points_list[4])

"""
TEST: extract_coordinates_list_from_feb2
Get connection TRUE test:
"""
# points_list, connection_list = io.extract_coordinates_list_from_feb2(file_name, node_name, True)
# for connections in connection_list:
#     print(connections)

"""
TEST: replace_node_in_new_feb_file
"""
# io.replace_node_in_new_feb_file(file_name, node_name, new_file_name, coordinates_list)

"""
TEST: get_dataset_from_feb_file
"""
print(io.get_dataset_from_feb_file(file_name, node_name))


