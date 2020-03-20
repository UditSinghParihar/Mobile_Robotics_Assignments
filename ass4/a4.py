# Usage: python a4.py

# Note 1: We only used the closest landmark to update the robot position as it is giving good trajectory comparison to using all landmarks for updating.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import sin as s, cos as c, atan2
from sys import exit


def drawTraj(X, Y, THETA, L):
	plt.plot(X, Y, 'bo', markersize=2)
	plt.plot(L[:, 0], L[:, 1], 'g*', markersize=8)

	plt.show()


def poseFromOdom(X0, Y0, THETA0, v, v_var, om, om_var):
	X = []; Y = []; THETA = []
	X.append(X0); Y.append(Y0); THETA.append(THETA0)

	dt = 0.1
	for i in range(1, len(v)):
		nv = np.random.normal(0, v_var); nom = np.random.normal(0, om_var)

		x = X[i-1] + dt * c(THETA[i-1]) * (v[i] + nv)
		y = Y[i-1] + dt * s(THETA[i-1]) * (v[i] + nv)
		theta = THETA[i-1] + dt * (om[i] + nom)

		X.append(x); Y.append(y); THETA.append(theta)

	X = np.asarray(X); Y = np.asarray(Y); THETA = np.asarray(THETA)

	return (X, Y, THETA)


def drawTwo(X1, Y1, THETA1, X2, Y2, THETA2, L):
	plt.plot(X1, Y1, 'ro', markersize=2)
	plt.plot(X2, Y2, 'bo', markersize=2)
	plt.plot(L[:, 0], L[:, 1], 'g*', markersize=8)

	plt.show()


def ekf(PPre, xPre, yPre, tPre, vCur, omCur, v_var, om_var, rCur, bCur, r_var, b_var, l, d):
	dt = 0.1
	# 1. Prediction based on Measurement
	FPrev = np.array([[1, 0, -dt * vCur * s(tPre)],
						[0, 1, dt * vCur * c(tPre)],
						[0, 0, 1]])

	QCur = np.array([[(dt * c(tPre))**2 * v_var, dt**2 * s(tPre) * c(tPre) * v_var, 0], 
						[dt**2 * s(tPre) * c(tPre) * v_var, (dt * s(tPre))**2 * v_var, 0],
						[0, 0, om_var]])

	PPred = FPrev @ PPre @ FPrev.T + QCur
	
	xPred = xPre + dt * c(tPre) * vCur
	yPred = yPre + dt * s(tPre) * vCur
	tPred = tPre + dt * omCur

	RCur = np.array([[r_var, 0], [0, b_var]])
	

	# 2. Finding Kalman Gain
	Kcur = np.zeros((3, 2)); GCur = np.zeros((2, 3))
	sensPred = np.zeros((2, 1)); sensMeas = np.zeros((2, 1))

	# Updating based on sensor information only when atleast one landmark is found. i.e. rCur has atleast one nonzero value
	if(np.any(rCur)):
		minInd = -1
		for i, (xl, yl) in enumerate(l):
			# Updating based on nearest landmark only. i.e. Finding the index of the least nonzero element in rCur
			minInd = int(np.where(rCur == np.min(rCur[np.nonzero(rCur)]))[0][0])

		if(np.any(rCur)):
			xl = l[minInd, 0]; yl = l[minInd, 1] 
			alpha = float(yl - yPred - (d * s(tPred)))
			beta = float(xl - xPred - (d * c(tPred)))
			dist = float((alpha**2 + beta**2)**0.5)

			GCur = np.array([[-beta/dist, -alpha/dist, (beta * d * s(tPred) - alpha * d * c(tPred))/dist], 
								[alpha/dist**2, -beta/dist**2, -(beta * d * c(tPred) + alpha * d * s(tPred) + alpha**2 + beta**2)/dist**2]])

			Kcur = PPred @ GCur.T @ np.linalg.pinv(GCur @ PPred @ GCur.T + RCur)

			sensPred = np.array([[dist], [atan2(alpha, beta) - float(tPred)]])
			sensMeas = np.array([[rCur[minInd]], [bCur[minInd]]])       

	# 3. Correction based on Sensor Measurement
	PCur = (np.eye(3) - Kcur @ GCur) @ PPred
	xCur = xPred + (Kcur @ (sensMeas - sensPred))[0]
	yCur = yPred + (Kcur @ (sensMeas - sensPred))[1]
	tCur = tPred + (Kcur @ (sensMeas - sensPred))[2]

	return (xCur, yCur, tCur, PCur)


def getOpt(r, b, r_var, b_var, v, om, v_var, om_var, l, d, X0, Y0, THETA0):
	X = []; Y = []; THETA = []; Pk = np.diag(np.array([1, 1, 0.1]))
	X.append(X0); Y.append(Y0); THETA.append(THETA0)

	for i in range(1, len(v)):
		xCur, yCur, tCur, Pk = ekf(Pk, X[-1], Y[-1], THETA[-1], 
									float(v[i]), float(om[i]), float(v_var), float(om_var),
									r[i, :], b[i, :], float(r_var), float(b_var),
									l, float(d))

		X.append(xCur); Y.append(yCur); THETA.append(tCur)

	# print(len(X), len(Y), len(THETA))
	return (X, Y, THETA)



if __name__ == '__main__':
	np.set_printoptions(precision=5, suppress=True)

	dataset = np.load('dataset.npz')

	t, x_true, y_true, th_true, l, r, r_var, b, b_var, v, v_var, om, om_var, d  = dataset['t'], dataset['x_true'], dataset['y_true'], dataset['th_true'], dataset['l'], dataset['r'], dataset['r_var'], dataset['b'], dataset['b_var'], dataset['v'], dataset['v_var'], dataset['om'], dataset['om_var'], dataset['d']

	# drawTraj(x_true, y_true, th_true, l)
	
	(XOdom, YOdom, THETAOdom) = poseFromOdom(x_true[0], y_true[0], th_true[0], v, v_var, om, om_var)
	print("Now seeing: Ground truth trajectory (RED) X Pure odometry estimated trajectory without EKF (BLUE):")
	drawTwo(x_true, y_true, th_true, XOdom, YOdom, THETAOdom, l)

	# Step 2: Current estimated pose given previous pose and control.
	XOpt, YOpt, THETAOpt = getOpt(r, b, r_var, b_var, v, om, v_var, om_var, l, d, x_true[0], y_true[0], th_true[0])
	print("Now seeing: EKF Estimated trajectory (BLUE):")
	drawTraj(XOpt, YOpt, THETAOpt, l)
	print("Now seeing: Ground truth trajectory (RED) X EKF estimated trajectory without EKF (BLUE):")
	drawTwo(x_true, y_true, th_true, XOpt, YOpt, THETAOpt, l)
