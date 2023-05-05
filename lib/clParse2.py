import sys

'''
Function: clParse

CONVERT TO DESIRED TYPE IN CODE CALLING THIS
arguments returned will always be string type
args : (give it args)
flag : (a string, "-o" for example)
numOfArgs : number of arguments after the flag to return
'''

def clParse(args, flag, numOfArgs = 1):
	for i, arg in enumerate(args):
		if flag == arg:
			vals = []
			for j in range(1, numOfArgs+1):
				vals.append(args[i + j])
			if numOfArgs == 1:
				return vals[0]
			return vals
	raise KeyError(flag + " not found in args")


'''
Function: flagInArgs

Returns boolean value if flag exists in file
'''
def flagInArgs(args, flag):
	for arg in args:
		if flag == arg:
			return True
	return False