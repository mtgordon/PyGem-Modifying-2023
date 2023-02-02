# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:55:21 2019

@author: mgordon
"""

#import shutil

'''
Function: remove_connections
'''
def remove_connections(part, INP_file):

    with open(INP_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    
    StartTrigger = '*Element, type=CONN3D2'
    EndTrigger = '*Connector Section'
    RemovedConnections = 0
    Connections = 0
    with open(INP_file,'w') as new_file:
        for line in content:
            if Connections:
                if EndTrigger in line:
                    Connections = 0
                    new_file.write(line + "\n")
                elif part in line:
                    RemovedConnections += 1
    #                new_line = line
    #                print(line[0])
    #                print(int(line[0]))
    #                print(int(line[0])-RemovedConnections)
    #                print(str(int(line[0])-RemovedConnections))
    #                new_line = str(int(line[0]) - RemovedConnections) + line[1:]
    #                print(new_line)
    #                new_file.write(new_line + "\n")
                else:
                    print(line)
                    new_line = str(int(line.split(',')[0]) - RemovedConnections) + "," +  ",".join(line.split(',')[1:])
                    new_file.write(new_line + "\n")
            elif StartTrigger in line:
                Connections = 1
                new_file.write(line + "\n")
            else:
                new_file.write(line + "\n")