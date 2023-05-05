# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 19:56:24 2017

@author: mgordon

This runs the given INP file in Abaqus using VMWare at the University of Michigan
"""

import subprocess

'''
Function: RunINPinAbaqus
'''
def RunINPinAbaqus(FileName, AbaqusBatLocation):
    CallString = AbaqusBatLocation + ' job=' + FileName + ' interactive cpus=4 ask_delete=OFF >> ' + FileName + '.txt'
    print(CallString)
    subprocess.call(CallString)
