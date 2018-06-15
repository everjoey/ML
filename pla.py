#!/usr/bin/env python3
import numpy as np
import random
import cProfile

def perceptron(x_list, y_list):
	w = np.zeros(4)
	print(w)
	while True:
		for i in range(len(x_list)):
			if np.sign(w.dot(x_list[i])) != y_list[i] and y_list[i] != 0:
				w = w + y_list[i]*x_list[i]
				print(w)
				break
		else:
			break
		continue
	return w

if __name__ == '__main__':
	wf = np.array([1,1,1,1])
	x_list = [np.array([
	random.randrange(-10,10)+random.random(),
	random.randrange(-10,10)+random.random(),
	random.randrange(-10,10)+random.random(),
	1,
	]) for i in range(10000)]
	print(x_list)
	y_list = [np.sign(wf.dot(x)) for x in x_list]
	print(y_list)

	cProfile.run('w = perceptron(x_list, y_list)')
	print(w)
