"""
This file is designed to use the IO functions relating to FeBio, which can entail feb files that are XML input files,
and log files that are txt output files. The goal is to replicate the same process as the post-processing currently
implemented for the INP files with Abaqus.
"""

import test_yutian.functions.IOfunctions_Feb as IO
import load_testing.Functions.NodeIDs

def find_apex(inputFile):
    coordList = IO.extract_coordinates_list_from_feb2(inputFile, 'Object8')
    min_y = coordList[0][1]
    count = 0
    for coord in coordList:
        if coord[1] < min_y:
            min_y = coord[1]

    return min_y
