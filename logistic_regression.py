#!/usr/bin/env python3
import numpy as np
from scipy.special import expit
from sklearn import linear_model

def logistic_regression(x, y):
	N, d = x.shape
	w = np.zeros(d)

	for i in range(100000):
#		grad = np.array([sum([expit(-y[i]*w.dot(x[i]))*(-y[i]*x[i][j]) for i in range(N)])/N for j in range(d)])
		grad = np.dot(-y*expit(-y*np.dot(w, x.T)), x)/N
		norm = np.linalg.norm(grad)
		print(norm)
		if norm < 0.0001:
			break
		else:
			w = w - 0.1*grad
	print(i)

	return w

if __name__ == '__main__':
	import cProfile
	import random

	N = 100000
	wf = np.array([0.3, 0.6, 0.6, 0.3])
	x_list = -10 + 20*np.random.rand(N, len(wf))
	y_list = np.array([np.random.choice([1, -1], p=[expit(wf.dot(x)), 1 - expit(wf.dot(x))]) for x in x_list])
	cProfile.run('w = logistic_regression(x_list, y_list)')
	print(w)
