#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree

def cal_impurity(y):
		return np.mean((y - np.mean(y))**2)

class DecisionTreeRegressor(object):
	def __init__(self, max_depth , min_impurity_split):
		self.min_impurity_split = min_impurity_split
		self.max_depth = max_depth

	def _build_tree(self, X, y):
		#self._depth += 1
		#print(self._depth)

		is_leaf = (
		cal_impurity(y) <= self.min_impurity_split or
		self._depth >= self.max_depth
		)

		if is_leaf:
			return lambda x: np.mean(y)

		self._depth += 1
		print(self._depth)

		min_impurity = np.inf
		for j in range(X.shape[1]):
			#index = np.argsort(X[:,j])
			#X[:,j] = X[index,j]
			#y[:] = y[index]
			for i in range(X.shape[0] - 1):
				left = y[0:i+1]
				right = y[i+1:]
				impurity = (len(left)/len(y))*cal_impurity(left) + (len(right)/len(y))*cal_impurity(right)
				if impurity < min_impurity:
					min_impurity = impurity
					threshold = (X[i,j] + X[i+1,j])/2
					index = i
		print(cal_impurity(y[0:index+1]))
		print(cal_impurity(y[index+1:]))
		print(threshold)

		left_tree = self._build_tree(X[0:index+1], y[0:index+1])
		#self._depth -= 1
		#print(self._depth)
		right_tree = self._build_tree(X[index+1:], y[index+1:])
		self._depth -= 1
		print(self._depth)

		return lambda x: (x[0] < threshold)*left_tree(x) + (x[0] > threshold)*right_tree(x)

	def fit(self, X, y):
		self._depth = 0
		self.tree = self._build_tree(X, y)
		#print(self._depth)

	def predict(self, X):
		return [self.tree(X[i]) for i in range(len(X))]



if __name__ == '__main__':
	x = np.linspace(-3, 3, 100).reshape(-1, 1)
	y = np.sin(x)# + np.random.rand(len(x))
	index = np.arange(len(x))
	np.random.shuffle(index)
	#print(x)
	x = x[index]
	y = y[index]
	plt.scatter(x, y)
	#print(x)


	t1 = DecisionTreeRegressor(4, 0.001)
	t1.fit(x, y)
	plt.scatter(x, t1.predict(x))

	t2 = tree.DecisionTreeRegressor(max_depth=4, min_impurity_split=0.001)
	t2.fit(x, y)
	plt.scatter(x, t2.predict(x))

	plt.show()
