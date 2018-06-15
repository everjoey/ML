#!/usr/bin/env python3
import numpy as np
import random
import cProfile

def pocket(x_list, y_list, limit):
	wp = np.zeros(4)
	print(wp)
	wp_err_count = sum([np.sign(wp.dot(x_list[i])) != y_list[i] for i in range(len(x_list))])
	print(wp_err_count)
	for i in range(limit):
		j = random.randrange(len(x_list))
		x = x_list[j]
		y = y_list[j]
		if np.sign(wp.dot(x)) != y and y != 0:
			w = wp + y*x
			print(w)
			w_err_count = sum([np.sign(w.dot(x_list[i])) != y_list[i] for i in range(len(x_list))])
			print(w_err_count)
			wp = w if w_err_count < wp_err_count else wp
	return wp

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

	cProfile.run('w = pocket(x_list, y_list, 40000)')
	print(w)
