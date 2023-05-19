import xml.etree.ElementTree as ET


def fetch_node_id(file_name, node_name, get_connections=False):
    """
    This function extracts the IDs of nodes from an FEB file and returns them as a list.

    Parameters:
    file_name (str): The name of the FEB file to extract node IDs from.
    node_name (str): The name of the node element in the FEB file.
    get_connections (bool, optional): Whether to also extract the connections between nodes. Default is False.

    Returns:
    node_id_list (list): A list containing the IDs of nodes.
    """
    tree = ET.parse(file_name)
    root = tree.getroot()

    node_id_list = []  # an empty list

    # Find the 'Nodes' element with the target_name attribute
    nodes_element = root.find(f".//Nodes[@name='{node_name}']")

    # Check if the 'Nodes' element is found
    if nodes_element is not None:
        # Iterate over each 'node' element in the found 'Nodes' element
        # for node in nodes_element.iter('node'):
        #     node_id_list.append(tuple(float(coordinate) for coordinate in node.text.split(',')))

        node_id_list = [node.attrib['id'] for node in nodes_element.findall('node')]
    else:
        print(f"No 'Nodes' element with the name '{node_name}' found.")
    return node_id_list


def extract_coordinates_from_final_step(log_file_path, feb_file_path, object_value):
    """
    This method will read a text file ideally the log file parsed as a parameter and
    will extract the nodes from the final step of the run program and store in a 2D
    list the node number(as first element of the 2D list), and the coordinates in a list
    (as the second element of the 2D list)
    """
    index_list = fetch_node_id(feb_file_path, object_value)
    with open(log_file_path) as file:
        lines = file.readlines()

    # Find the index of the last step
    last_step_index = len(lines) - 1
    while last_step_index >= 0 and not lines[last_step_index].startswith('Step'):
        last_step_index -= 1
    if last_step_index < 0:
        raise ValueError('No Step found in the FEBio log file')

    # Extract the x, y, and z coordinates for the last step
    last_step_coords = []
    for line in lines[last_step_index+3:]:
        line = line.strip()
        if not line:
            break
        cols = line.split()
        node_index = (cols[0])
        if node_index in index_list:
            coords = [float(c) for c in cols[1:]]
            last_step_coords.append([node_index, coords])
    return last_step_coords
