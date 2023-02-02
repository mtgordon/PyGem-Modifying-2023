#TODO: move this function to an appropiate file

from lib.workingWith3dDataSets import Point, DataSet3d, GeneratedDataSet
from lib.IOfunctions import extractPointsForPartFrom, write_part_to_inp_file

'''
Function: getSomeDistance
'''
def getSomeDistance(referencePoint, desiredPoint, material, filename, HiatusLength):
	#Load Material from filename (find that function)
	#materialPoints
	
	number_of_points_to_check = 10
	matPoints = []
	closestPoints = []
	index = []

	XYZs = extractPointsForPartFrom(filename, material)
	for k in XYZs:
		matPoints.append(Point(k[0], k[1], k[2]))

	#insert some initially
	for i in range(0, number_of_points_to_check):
		closestPoints.append(matPoints[i])
		index.append(i)

	for j in range(0, len(matPoints)):
		p = matPoints[j]
		for i in range(0, number_of_points_to_check):
			if p.distance(desiredPoint) < closestPoints[i].distance(desiredPoint):
				del closestPoints[i]
				del index[i]
				index.append(j)
				closestPoints.append(p)
				break

	closest = closestPoints[0]
	closestIndex = index[0]
	
	for i in range(0, len(closestPoints)):
		p = closestPoints[i]
		if abs(HiatusLength - p.distance(referencePoint)) < \
        abs(HiatusLength - closest.distance(referencePoint)):
			closest = p[0]
			closestIndex = index[i]

	#+1 is for node number, not index
	return [closest.distance(referencePoint), closestIndex + 1]

'''
Function: getXClosestNodes
'''
def getXClosestNodes(desiredPoint, X, material, filename):
	#Load Material from filename (find that function)
	#materialPoints
	
	number_of_points_to_check = X
	matPoints = []
	closestPoints = []
	index = []

	XYZs = extractPointsForPartFrom(filename, material)

	for k in XYZs:
		matPoints.append(Point(k[0], k[1], k[2]))


	#insert some initially
	for i in range(0, number_of_points_to_check):
		closestPoints.append(matPoints[i])
		index.append(i)

	for j in range(0, len(matPoints)):
		p = matPoints[j]
		for i in range(0, number_of_points_to_check):
			if p.distance(desiredPoint) < closestPoints[i].distance(desiredPoint):
				del closestPoints[i]
				del index[i]
				index.append(j)
				closestPoints.append(p)
				break

	#Enable the following for [[x1,y1,z1],[x2,y2,z2]] instead of [p1,p2,p3]
	"""
	temp = closestPoints
	closestPoints = []
	for p in temp:
		closestPoints.append([p.x,p.y,p.z])
	"""


	return index, closestPoints
#	return closestPoints

'''
Function: setPBodyClosest
'''
def setPBodyClosest(PBody, GIPoint, input_file_name, output_file_name):

    points = []
    Xs = []
    Ys = []
    Zs = []

    XYZs = extractPointsForPartFrom(input_file_name, PBody)
    for k in XYZs:
        points.append(Point(k[0], k[1], k[2]))
        Xs.append(k[0])
        Ys.append(k[1])
        Zs.append(k[2])

    GDS = GeneratedDataSet(Xs, Ys, Zs)

    index, unused_variable = getXClosestNodes(GIPoint, 1, PBody,input_file_name)
    index = index[0]
    p = points[index]
    new_point = Point(GIPoint.x, GIPoint.y, p.z)
    GDS.modify_point(new_point, index)

    GDS.node(index)

    write_part_to_inp_file(output_file_name, PBody, GDS)
    return

"""
material = "OPAL325_GIfiller"
filename = "GenericINPFile_Analog_4_v2.inp"
HiatusLength = 33.36
referencePoint = Point(-6.698, 17.145, -22.5788)
desiredPoint = Point(-5.5655, -15.746196, -28.043)
print(getSomeDistance(referencePoint, desiredPoint, material, filename, HiatusLength))
print(getXClosestNodes(desiredPoint, 10, material, filename))"""