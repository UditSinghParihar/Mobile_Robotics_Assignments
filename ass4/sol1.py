# Usage : python sol1.py

import numpy as np
import matplotlib.pyplot as plt
from math import sin as s, cos as c, atan2
from sys import exit


def drawTraj(X, Y, THETA, L):
	plt.plot(X, Y, 'ro', markersize=2)
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
	
	sum1 = np.zeros((3, 3)); sum2 = np.zeros((3, 1))
	for i, (xl, yl) in enumerate(l):
		# Looping only when atleast one landmark is found. i.e. rCur has atleast one nonzero value
		if(np.any(rCur)):
			# Updating based on nearest landmark only. i.e. Finding the index of the least nonzero element in rCur
			minInd = int(np.where(rCur == np.min(rCur[np.nonzero(rCur)]))[0][0])
			if(i == minInd):
				alpha = float(yl - yPred - (d * s(tPred)))
				beta = float(xl - xPred - (d * c(tPred)))
				dist = float((alpha**2 + beta**2)**0.5)

				GCur = np.array([[-beta/dist, -alpha/dist, (beta * d * s(tPred) - alpha * d * c(tPred))/dist], 
									[alpha/dist, -beta/dist, -(beta * d * c(tPred) + alpha * d * s(tPred) + alpha**2 + beta**2)/dist]])

				Kcur = PPred @ GCur.T @ np.linalg.pinv(GCur @ PPred @ GCur.T + RCur)

				sum1 = sum1 + Kcur @ GCur

				sensPred = np.array([[dist], [atan2(alpha, beta) - float(tPred)]])
				sensMeas = np.array([[rCur[i]], [bCur[i]]])		
				sum2 = sum2 + Kcur @ (sensMeas - sensPred)

	PCur = (np.eye(3) - sum1) @ PPred
	xCur = xPred + sum2[0]
	yCur = yPred + sum2[1]
	tCur = tPred + sum2[2]

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

	return (X, Y, THETA)


if __name__ == '__main__':
	np.set_printoptions(precision=5, suppress=True)

	dts = np.load('dataset.npz')

	for key, items in dts.items():
		if(key == "t"): t = items
		if(key == "x_true"): x_true = items
		if(key == "y_true"): y_true = items
		if(key == "th_true"): th_true = items
		if(key == "r"): r = items
		if(key == "r_var"): r_var = items
		if(key == "b"): b = items
		if(key == "b_var"): b_var = items
		if(key == "l"): l = items
		if(key == "v"): v = items
		if(key == "v_var"): v_var = items
		if(key == "om"): om = items
		if(key == "om_var"): om_var = items
		if(key == "d"): d = items

	# drawTraj(x_true, y_true, th_true, l)

	(XOdom, YOdom, THETAOdom) = poseFromOdom(x_true[0], y_true[0], th_true[0], v, v_var, om, om_var)

	# drawTraj(XOdom, YOdom, THETAOdom, l)

	drawTwo(x_true, y_true, th_true, XOdom, YOdom, THETAOdom, l)

	XOpt, YOpt, THETAOpt = getOpt(r, b, r_var, b_var, v, om, v_var, om_var, l, d, x_true[0], y_true[0], th_true[0])
	drawTraj(XOpt, YOpt, THETAOpt, l)
	drawTwo(x_true, y_true, th_true, XOpt, YOpt, THETAOpt, l)