import numpy as np
from lib.workingWith3dDataSets import GeneratedDataSet, DataSet3d
from math import floor

def openFile(file_name):
    with open(file_name) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


def extractNodesFromINP(file_name, part, nodes):
    TissueCoordinates = np.array(extractPointsForPartFrom(file_name, part, get_connections=False))
#    print(TissueCoordinates)
    NodeCoordinates = []
    for i in nodes:
        NodeCoordinates.append(TissueCoordinates[i-1])
    return NodeCoordinates

def extractPointsForPartFrom(file_name, part, get_connections=False):
    # print(file_name)
    content = openFile(file_name)
    found = False
    line_index = 1
    part = "*Part, name="+ part
    line_vals = []
    #connection_vals = []
    conVal = {}
    started = False
    started_connections = False
    for line in content:
        if not found and part not in line:
            continue # go to next line
        elif found == False:
            found = True
        else:
            
            data = line.split(",")
            try:
                # if line contains the next expected node number
                if (int)(data[0]) == line_index:
                
                    # append data
                    started = True
                    if started_connections and get_connections:
                        temp = list(map(int, data[1:]))
                        #connection_vals.append(temp)
                        conVal[temp[0]] = temp[1:]
                    else:

                        line_vals.append(data)
                        
                    line_index += 1
                    
                # if we have started parsing data and the next line is not our expected node number    
                elif started:
                    # we have collected al the data
                    break
            # occurs at "*Node" after we have found part name
            except ValueError:
                if data[0] == "*Element" and get_connections:
                    started_connections = True
                    line_index = 1
                    continue
                else:
                    continue

    xyz_vals = []
    for data_point in line_vals:
        # the node values are strings and need to be conerted to floats
        xyz_vals.append([float(data_point[1]), float(data_point[2]), float(data_point[3])])
      


    if get_connections:
        return xyz_vals, conVal
    else: 
        return xyz_vals
    
def extractPointsForPartFrom2(file_name, part, get_connections=False):
    content = openFile(file_name)
    found = False
    line_index = 1
    part = "*Part, name="+ part
    line_vals = []
    connection_vals = []
    started = False
    started_connections = False
    for line in content:
        if not found and part not in line:
            continue # go to next line
        elif found == False:
            found = True
        else:
            
            data = line.split(",")
            try:
                # if line contains the next expected node number
                if (int)(data[0]) == line_index:
                
                    # append data
                    started = True
                    if started_connections and get_connections:
                        temp = list(map(int, data[1:]))
                        connection_vals.append(temp)


                    else:

                        line_vals.append(data)
                        
                    line_index += 1
                    
                # if we have started parsing data and the next line is not our expected node number    
                elif started:
                    # we have collected al the data
                    break
            # occurs at "*Node" after we have found part name
            except ValueError:
                if data[0] == "*Element" and get_connections:
                    started_connections = True
                    line_index = 1
                    continue
                else:
                    continue

    xyz_vals = []
    for data_point in line_vals:
        # the node values are strings and need to be conerted to floats
        xyz_vals.append([float(data_point[1]), float(data_point[2]), float(data_point[3])])
      


    if get_connections:
        return xyz_vals, connection_vals
    else: 
        return xyz_vals


def write_new_inp_file(file_name, part, new_file_name, data_set):
    node_pad = 7  # how big of a space to leave for the data
    data_pad = 13 # this insures that of a number is 3 digits long it will be padded with spaces to the specified justifaction to maintian the given length
    
    content = openFile(file_name)
    new_file = open(new_file_name, 'w')
    write = False
    found = False
    part = "*Part, name="+ part
    index = 0
    max_index = data_set.number_of_points - 1
    for line in content:
        if write and max_index >= index:
            """
            The  node number is Justified right with a length of 7
            """
            line = (str(index + 1).rjust(node_pad) + ',' + str(data_set.xAxis[index]).rjust(data_pad) + ',' + str(data_set.yAxis[index]).rjust(data_pad)+ ', '+ str(data_set.zAxis[index]).rjust(data_pad))
            new_file.write(line + "\n")
            index += 1
        else:
            new_file.write(line + "\n")
            
        if found and not write:
            write = True
            
        if part in line:
            found = True
            
    new_file.close()

def write_part_to_inp_file(file_name, part, data_set):
    node_pad = 7  # how big of a space to leave for the data
    data_pad = 13 # this insures that of a number is 3 digits long it will be padded with spaces to the specified justifaction to maintian the given length
#    print(file_name)    
    content = openFile(file_name)
    f = open(file_name, 'w')
    write = False
    found = False
    part = "*Part, name="+ part
    index = 0
    max_index = data_set.number_of_points - 1
    for line in content:
        if write and max_index >= index:
            """
            The  node number is Justified right with a length of 7
            """
            line = (str(index + 1).rjust(node_pad) + ',' + str(data_set.xAxis[index]).rjust(data_pad) + ',' + str(data_set.yAxis[index]).rjust(data_pad)+ ', '+ str(data_set.zAxis[index]).rjust(data_pad))
            f.write(line + "\n")
            index += 1
        else:
            f.write(line + "\n")
            
        if found and not write:
            write = True
            
        if part in line:
            found = True
    
    f.close()

#Extracts points but puts it into DataSet3d class
def get_dataset_from_file(file_name, part_name):
#    print(file_name, part_name)
    np_points = np.array(extractPointsForPartFrom(file_name, part_name))
#    print(np_points)
    return DataSet3d(list(np_points[:, 0]), list(np_points[:, 1]), list(np_points[:, 2]))

def get_interconnections(file_name, part_name): #connections-between-material

    connections = []
    number_of_nodes = len(extractPointsForPartFrom(file_name, part_name))

    #creates a list of empty nodes
    for i in range(0, number_of_nodes):
        connections.append([])

    findLine = "*Part, name=" + part_name

    with open(file_name, 'r') as f:
        while(True):
            line = f.readline()
            if line.startswith(findLine):
                break
        while(True):
            line = f.readline()
            if line.startswith("*Element"):
                break
        while (True):
            if line.startswith("*Element"): #we don't care
                line = f.readline()
            elif line.startswith("*"): #done getting connections
                break
            else:
                strings = line.replace(" ", "").strip().split(",")
                nums = []
                for num in strings:
                    nums.append(int(num))

                connections = addToVals(connections, nums[1:])
                line = f.readline()
        
    return connections

def addToVals(connections, nums):
    
    if len(nums) < 4:
        for num in nums:
        #goto connection[num], add all unique in nums
            for neighbor in nums:
                if neighbor not in connections[num-1] and neighbor != num:
                    connections[num-1].append(neighbor)

    elif len(nums) == 4:
        for i, num in enumerate(nums):
            for j, neighbor in enumerate(nums):
                if neighbor not in connections[num-1] and (i + j) % 2 == 1: #indices 0,1,2,3 and each is the corner of a square; do not want diagonal or self.
                #all the ones we want to add are odd+even indices
                #odd+odd or even+even only occur when adding diagonals or self
                    connections[num-1].append(neighbor)

    else:
        raise(RuntimeError("Error retrieving connections; can only handle sets of 2, 3, or 4 values"))
    return connections

#Returns the line number (not index) that a given string occurs in
def findLineNum(file_location, string):
    lineNum = 1
    f = open(file_location, 'r')
    for line in f:
        if string == line.strip():
            return lineNum
        lineNum += 1
    raise EOFError("String not found", file_location, string)



############### functions for reading csv file once it has been created with nodal coordinates
def getFEAData(FileName,nodes):
    csv = np.genfromtxt (FileName, delimiter=",")
#    print(csv)
#    [rows,columns] = csv.shape
#    print(rows)
#    print(columns)
    x=np.zeros(nodes)
    y=np.zeros(nodes)
    z=np.zeros(nodes)
    for i in range(0,nodes):
        x[i]=csv[i*3]
        y[i]=csv[i*3+1]
        z[i]=csv[i*3+2]
    return [x,y,z]


def getFEADataCoordinates(FileName):
    csv = np.genfromtxt (FileName, delimiter=",")

# If there are multiple rows, get the last one
    try:
        data = csv[-1,:]
    except IndexError:
        data = csv

    nodes = floor((len(data)/3))
    x=np.zeros(nodes)
    y=np.zeros(nodes)
    z=np.zeros(nodes)
    for i in range(0,nodes):
        x[i]=data[i*3]
        y[i]=data[i*3+1]
        z[i]=data[i*3+2]
    return [x,y,z]

def getInitialPositions(FileName,nodes):
    csv = np.genfromtxt (FileName, delimiter=",")
    x = np.array(csv[:,1])
    y = np.array(csv[:,2])
    z = np.array(csv[:,3])
    return [x,y,z]


################# Functions that are no longer used #################

