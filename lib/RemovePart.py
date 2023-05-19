# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:55:21 2019

@author: mgordon
"""

#import shutil
import re

'''
Function: remove_part
'''
def remove_part(part, INP_file):
    with open(INP_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    
    Triggers = ['*Part','*Elset','*Instance','*Nset']
    SetTriggers = ['Elset','Nset']
    SetNames = [part]
    
    DeletingPart = 0
    FirstLineTrigger = 0
    
    with open(INP_file,'w') as new_file:
    # Go through each line
        for line in content:
    #        If the line has both one of the triggers and the correct material then it is the first line of a section to be removed
    #        if part in line and any(Trigger in line for Trigger in Triggers):
            if any(name in line for name in SetNames) and any(Trigger in line for Trigger in Triggers):
                FirstLineTrigger = 1
    #       If it is the first line of a section or if we are in the middle of delting a part...
            if FirstLineTrigger or DeletingPart:
    #            Turn the flag on that we're deleting a part
                DeletingPart = 1
                if FirstLineTrigger:
    #                Turn off the first line trigger
                    FirstLineTrigger = 0
                    if '*Part' in line:
                        EndTriggers = ['*End Part']
                    else:
                        EndTriggers = Triggers
                    pass
                else:
    #                If not the first line and the line contains a trigger, write the line and turn off the deleting part flag
                    if any(Trigger in line for Trigger in EndTriggers):
                        new_file.write(line + "\n")
                        DeletingPart = 0
                    else:
                        if any(Trigger in line for Trigger in SetTriggers):
                            SetNames.append(line.split('=')[1].split(',')[0])                            
    #                    Otherwise don't write anything and just move on to the next line
                        pass
            else:
    #            If we're not deleting a part, just write the line
                new_file.write(line + "\n")

'''
Function: remove_connections
'''
def remove_connections(nodes, part, INP_file):
    if nodes == ['*']:
        nodes = ['']
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
#                for node in nodes:
#                    print(part + '-1.' + str(node))
                if EndTrigger in line:
                    Connections = 0
                    new_file.write(line + "\n")
                    #part + '-1.' + str(node) in line
                elif any(bool(re.search(rf"\b{part + '-1.' + str(node)}\b", line)) for node in nodes):
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