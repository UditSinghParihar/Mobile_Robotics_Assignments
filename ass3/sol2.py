import open3d as o3d
from sys import argv
import math
from math import cos, sin
import cv2
import autograd.numpy as np
from autograd import jacobian
import sys


def read(file):
	pcd = o3d.io.read_point_cloud(file)
	pcdX = np.asarray(pcd.points)
	pcdColor = np.asarray(pcd.colors).T

	return pcdX, pcdColor, pcd


# All projected pixels have +ve coordinates
xCam = 2.25; yCam = 0.15; zCam = 0.5; tCam = math.radians(-20)


def show(pcd):
	axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

	mesh_sphere = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
	global xCam; global yCam; global zCam; global tCam
	trans = np.array([[cos(tCam), -sin(tCam), 0, xCam], [sin(tCam), cos(tCam), 0, yCam], [0, 0, 1, zCam], [0, 0, 0, 1]])
	mesh_sphere.transform(trans)

	o3d.visualization.draw_geometries([pcd, axis, mesh_sphere])


def showImg(img):
	cv2.imshow("image", img)
	cv2.waitKey(0)


def getP():
	K = np.array([[525, 0, 319.5], [0, 525, 239.5], [0, 0, 1]])

	global xCam; global yCam; global zCam; global tCam

	# Camera wrt world = RT_c_w
	RT_c_w = np.array([[cos(tCam), -sin(tCam), 0, xCam], [sin(tCam), cos(tCam), 0, yCam], [0, 0, 1, zCam], [0, 0, 0, 1]])
	RT_w_c = np.linalg.inv(RT_c_w)
	P = np.dot(K, RT_w_c[0:3, :])

	P = P/P[2, 3]
	return P, K, RT_w_c[0:3, :]


def showFrob(P1, P2, str1="mat1", str2="mat2"):
	np.set_printoptions(suppress=True)
	print("Frobenius norm between %s and %s is: %f" % (str1, str2, np.linalg.norm(P1 - P2, 'fro')))


def getRT(P, K):
	RT = np.dot(np.linalg.pinv(K), P)

	return RT


def getNoiseP(POrig, K):
	np.random.seed(23)
	limit = 0.5

	PNoise = np.zeros(POrig.shape)

	for r in range(POrig.shape[0]):
		for c in range(POrig.shape[1]):
			PNoise[r, c] = POrig[r, c] + np.random.uniform(0, limit)
	PNoise[2, 3] = 1

	return PNoise, getRT(PNoise, K)


def getImg(pxh, pcdColor):
	# Based on range of x and y coordinates
	# print("x, y max range: ", np.max(pxh, 1), "x,y min range: ", np.min(pxh, 1))
	height = 670; width = 630

	img = np.zeros((height, width, 3), np.uint8)

	for i in range(pxh.shape[1]):
		r = int(pxh[1, i]); c = int(pxh[0, i])
		if(r<height and c<width and r>0 and c>0):
			red = 255*pcdColor[0, i]; green = 255*pcdColor[1, i]; blue = 255*pcdColor[2, i]
			img[r, c] = (blue, green, red)
			
	return img


def getPx(POrig, pcdX):
	ones = np.ones((pcdX.shape[0], 1))
	pcdXh = np.hstack((pcdX, ones)).T
	
	pxh = np.dot(POrig, pcdXh)
	# print("Max depth: %f and Min depth: %f" % (np.max(pxh[2, :]), np.min(pxh[2, :])))

	pxh[0, :] = pxh[0, :]/pxh[2, :]
	pxh[1, :] = pxh[1, :]/pxh[2, :]
	pxh[2, :] = pxh[2, :]/pxh[2, :]

	return pxh[0:2, :]


def getCorres(pxh, pcdX):
	pcdX = pcdX.T
	# 15 random corrspondences
	# return pxh[:, 50:65], pcdX[:, 50:65]	
	return pxh[:, 150:165], pcdX[:, 150:165]


def getFunc(P):
	f0 = []
	global pxC; global pXC

	for i in range(pXC.shape[1]):
		xi = pxC[0, i]; yi = pxC[1, i]
		Xi = pXC[0, i]; Yi = pXC[1, i]; Zi = pXC[2, i]

		fxi = xi - ((P[0]*Xi + P[1]*Yi + P[2]*Zi + P[3])/(P[8]*Xi + P[9]*Yi + P[10]*Zi + P[11]))
		fyi = yi - ((P[4]*Xi + P[5]*Yi + P[6]*Zi + P[7])/(P[8]*Xi + P[9]*Yi + P[10]*Zi + P[11]))

		f0.append(fxi); f0.append(fyi)

	f0 = np.array(f0)

	return f0


def numJac(P):
	# P = P.reshape(P.shape[0]*P.shape[1])
		
	J = jacobian(getFunc)
	JP = J(P)

	return JP


def analyJac(P):
	P = P.reshape(3, 4)
	global pxC; global pXC

	J = []

	for i in range(pxC.shape[1]):
		x = pxC[0, i]; y = pxC[1, i]
		X = pXC[0, i]; Y = pXC[1, i]; Z = pXC[2, i]

		nx = P[0,0]*X + P[0,1]*Y + P[0,2]*Z + P[0,3]; d = P[2,0]*X + P[2,1]*Y + P[2,2]*Z + P[2,3]

		dfxp11 = -X/d  
		dfxp12 = -Y/d
		dfxp13 = -Z/d		
		dfxp14 = -1/d		
		dfxp21 = 0	
		dfxp22 = 0		
		dfxp23 = 0
		dfxp24 = 0
		dfxp31 = nx*X/d**2
		dfxp32 = nx*Y/d**2
		dfxp33 = nx*Z/d**2
		dfxp34 = nx/d**2

		dfxp = [dfxp11, dfxp12, dfxp13, dfxp14, dfxp21, dfxp22, dfxp23, dfxp24, dfxp31, dfxp32, dfxp33, dfxp34]

		J.append(dfxp)

		ny = P[1,0]*X + P[1,1]*Y + P[1,2]*Z + P[1,3]; d = P[2,0]*X + P[2,1]*Y + P[2,2]*Z + P[2,3]

		dfyp11 = 0
		dfyp12 = 0
		dfyp13 = 0
		dfyp14 = 0
		dfyp21 = -X/d
		dfyp22 = -Y/d
		dfyp23 = -Z/d
		dfyp24 = -1/d
		dfyp31 = ny*X/d**2
		dfyp32 = ny*Y/d**2
		dfyp33 = ny*Z/d**2
		dfyp34 = ny/d**2

		dfyp = [dfyp11, dfyp12, dfyp13, dfyp14, dfyp21, dfyp22, dfyp23, dfyp24, dfyp31, dfyp32, dfyp33, dfyp34]

		J.append(dfyp)

	J = np.array(J)

	return J


def pseudoInv(J):
	cov = np.dot(J.T, J)
	inv = np.linalg.pinv(cov)

	return np.dot(inv, J.T)


def gaussNewton(P0):
	pCur = P0.reshape(P0.shape[0]*P0.shape[1])

	fCur = getFunc(pCur)
	normCur = np.linalg.norm(fCur)
	threshNorm = 1e-1

	while(normCur > threshNorm):
		print("Reprojection error: %f" % normCur)
		
		# JCur = numJac(pCur)
		JCur = analyJac(pCur)

		pInv = pseudoInv(JCur)
		pNext = pCur - np.dot(pInv, fCur)

		pCur = pNext
		fCur = getFunc(pCur)
		normCur = np.linalg.norm(fCur)

	np.set_printoptions(precision=2, suppress=True)
	POpt = pCur.reshape(3, 4)
	POpt = POpt/POpt[2, 3]

	print("Reprojection error: %f" % normCur)
	
	return POpt


if __name__ == '__main__':
	file = str(argv[1])
	pcdX, pcdColor, pcd = read(file)
	show(pcd)

	POrig, K, RTOrig = getP()
	np.set_printoptions(precision=2, suppress=True)
	print("Original P: "); print(POrig)

	pxh = getPx(POrig, pcdX)

	img = getImg(pxh, pcdColor)
	showImg(img)

	pxC, pXC = getCorres(pxh, pcdX)
	
	PNoise, RTNoise = getNoiseP(POrig, K)
	print("Noisy P: "); print(PNoise); showFrob(POrig, PNoise, "POrig", "PNoise")
	# showFrob(RTOrig, RTNoise, "RTOrig", "RTNoise")
	
	PNoiseVec = PNoise.reshape(PNoise.shape[0]*PNoise.shape[1])
	JNum = numJac(PNoiseVec)
	JAnaly = analyJac(PNoiseVec)
	showFrob(JNum, JAnaly, "JNum", "JAnaly")

	POpt = gaussNewton(PNoise)
	print("Optimized P: "); print(POpt); showFrob(POrig, POpt, "POrig", "POpt")

	RTOpt = getRT(POpt, K)
	showFrob(RTOrig, RTOpt, "RTOrig", "RTOpt")
	print("Original RT: "); print(RTOrig); print("Optimized RT: "); print(RTOpt)
