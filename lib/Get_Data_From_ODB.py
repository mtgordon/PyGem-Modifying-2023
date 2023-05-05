import odbAccess 
import sys
from lib.clParse2 import clParse

'''
Script: Get_Data_From_ODB.py
'''

# % /* Author: Javier Palacio Torralba
# %  * Mechens
# %  * E-mail contact: javier.palacio@mechens.org
# %  *
# %  * This code is free software: you can redistribute it and/or
# %  * modify it under the terms of the GNU Affero General Public License as
# %  * published by the Free Software Foundation, either version 3 of the
# %  * License, or (at your option) any later version.
# %  *
# %  * The function is distributed in the hope that it will be useful,
# %  * but WITHOUT ANY WARRANTY; without even the implied warranty of
# %  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# %  * GNU Affero General Public License for more details.
# %  *
# %  * You should have received a copy of the GNU Affero General Public License
# %  * along with this program.  If not, see <http://www.gnu.org/licenses/>.
# %  **/

# %  * The objective of this Python Script is to read information from Abaqus output database (.odb) files and write it into a file.
# %  * This script reads the reaction forces of a certain node -nodeNumber- and writes it each time step for all parts in the assembly.
# %  * To launch this script you should write in the command line, in the same folder as the odb file is:
# %  * abaqus cae nogui=ReadODB
# %  * The option nogui prevents the graphical user interface of opening therefore accelerating the results.
##############################################
#VARIABLE DECLARATION
#
##############################################

#def main():




OdbFilename = clParse(sys.argv, "-odbfilename")
PartName = clParse(sys.argv, "-partname")
StringNodes = clParse(sys.argv, "-strNodes")
NodesOfInterest = StringNodes.split(",")
Variable1 = clParse(sys.argv, "-var1")
OutputFile = clParse(sys.argv, "-outputfile")

#Alternative: import flagInArgs from clParse
#(would have to change) if statements below
HeaderFlag = clParse(sys.argv, "-headerflag")
NewFileFlag = clParse(sys.argv, "-newfileflag")

Frames = clParse(sys.argv, "-frames")
#print('Frames = ', Frames)
######################################################
## Only able to get the last frame
#OdbFilename = sys.argv[-7]
#print("ODBFile: ",OdbFilename)
#PartName = sys.argv[-6]
#print("PartName: ",PartName)
#StringNodes  = sys.argv[-5]
#NodesOfInterest=StringNodes.split(",")
#print("NodesOfInterest: ",NodesOfInterest)
#Variable1 = sys.argv[-4]
#print("Variable 1: ",Variable1)
#OutputFile = sys.argv[-3]
#print("OutputFile: ",OutputFile)
#HeaderFlag = sys.argv[-2]
#print("HeaderFlag: ",HeaderFlag)
#NewFileFlag = sys.argv[-1]
#print("NewFileFlag: ",NewFileFlag)
######################################################



#names = ['PVM100_ParaV50_CL100_US100-01042016']  #Insert the names of the .odb files without the extension
names = ['GeneratedINPFile']  #Insert the names of the .odb files without the extension

#Test 7: Nodes (comment out below)
#NodesOfInterest = list(range(1,310))		    #Insert the nodes of interest

PreferredExtension = '.csv' #Insert the extension for the output file

StepName = 'Step-1'         #Name of the step of interest

# Test 6: Material Name (just comment out below as PartName is defined above)    
#PartName = 'OPAL325_AVW_V6-1'       #Name of the Part

#Variable1 = 'U'             #Name of the Variable 1

Variable2 = 'RF'            #Name of the Variable 2

#print(ODBFile)

NameOfFile = OutputFile + PreferredExtension
#NameOfFile = 'FEA_Results'+PreferredExtension
# Test 9: Testing New File Flaga
#FileResultsX = open(NameOfFile,'w')    



for y in range(len(names)):
    if NewFileFlag == 'Y':
        FileResultsX = open(NameOfFile,'w')
    else:
        FileResultsX = open(NameOfFile,'a')

##    	Name = OdbFilename +'.odb'
#    FileResultsX.write('Nodes =')
#    FileResultsX.write(NodesOfInterest)
#    FileResultsX.write('\n')

#    NodesOfInterest = list(range(1,10))		    #Insert the nodes of interest

#    FileResultsX.write('Nodes =')
#    FileResultsX.write(str(NodesOfInterest))
#    FileResultsX.write('\n')
#Test 8: Header Flag

    if HeaderFlag == 'Y':
        FileResultsX.write(PartName)
        FileResultsX.write('\n')            
#    for y in range(len(names)):
        

#Test 4: Passing ODBfile
#	Name = names[y]+'.odb'
    Name = OdbFilename +'.odb'  
    myOdb = odbAccess.openOdb(path=Name)
    lastStep = myOdb.steps[StepName]
    
    if Frames == 'last':
        first_frame = len(lastStep.frames)-1
        last_frame = len(lastStep.frames)
    elif Frames == 'all':
        first_frame = 0
        last_frame = len(lastStep.frames)
    else:
        first_frame = int(Frames)
        last_frame = int(Frames)+1

#    FirstFrame = len(lastStep.frames)-1

#    FirstFrame = 0
#    FileResultsX.write('%10.9E,' % (len(lastStep.frames)))
#    FileResultsX.write('\n')		
    for z in range (first_frame,last_frame):
        current_frame = myOdb.steps[StepName].frames[z]
#        tiempo=current_frame.frameValue
        
        TotalForce = 0
        TotalXForce = 0
        TotalYForce = 0
        TotalZForce = 0
#        FileResultsX.write('%10.9E,' % (z))
#        FileResultsX.write('\n')		
        for i in NodesOfInterest:
            node = myOdb.rootAssembly.instances[PartName].getNodeFromLabel(int(i))
            VariableData = current_frame.fieldOutputs[Variable1].getSubset(region=node)
    #			ReactionForce = current_frame.fieldOutputs[Variable2].getSubset(region=node)
            for val in VariableData.values:
                if Variable1 == 'COORD':
                    FileResultsX.write('%10.9E, %10.9E, %10.9E,' % (val.data[0],val.data[1],val.data[2]))
                elif Variable1 == 'RF':
                    ForceMagnitude = (val.data[0]**2+val.data[1]**2+val.data[2]**2)**0.5
                    FileResultsX.write('%10.9E, %10.9E, %10.9E, %10.9E,' % (val.data[0],val.data[1],val.data[2],ForceMagnitude))
#                    FileResultsX.write('%10.9E, %10.9E, %10.9E, %10.9E, %10.9E, %10.9E, %10.9E,' % (val.data[0],val.data[1],val.data[2],ForceMagnitude,TotalXForce,TotalYForce,TotalZForce))
                    FileResultsX.write('\n')	
                    TotalXForce += val.data[0]
                    TotalYForce += val.data[1]
                    TotalZForce += val.data[2]
                    TotalForce += ForceMagnitude
        if Variable1 == 'RF':
            FileResultsX.write('Total Force Using Resultants =,')                            
            FileResultsX.write('%10.9E,' % (TotalForce)) 
            FileResultsX.write('\n')		                           
            FileResultsX.write('Total Force Using Components =,')                            
            TotalComponentForce = (TotalXForce**2 + TotalYForce**2 + TotalZForce**2)**0.5
            FileResultsX.write('%10.9E,' % (TotalComponentForce))             
    #                       FileResultsX.write('\n')	
    #                       FileResultsX.write('%10.9E\t %10.9E\t' % (val.data[0],val.data[1]))
    #                       FileResultsX.write('%10.9E\t %10.9E\t' % (val.data[0],val.data[1],val.data[2]))
    
    #			for val in ReactionForce.values:
    #				FileResultsX.write('%10.9E\t %10.9E\t' % (val.data[2],tiempo))
    
    			

        FileResultsX.write('\n')		
