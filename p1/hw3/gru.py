import torch
import torch.nn as nn
import numpy as np
import itertools


class Sigmoid:
	"""docstring for Sigmoid"""
	def __init__(self):
		pass

	def forward(self, x):
		self.res = 1/(1+np.exp(-x))
		return self.res

	def backward(self):
		return self.res * (1-self.res)

	def __call__(self, x):
		return self.forward(x)


class Tanh:
	def __init__(self):
		pass

	def forward(self, x):
		self.res = np.tanh(x)
		return self.res

	def backward(self):
		return 1 - (self.res**2)

	def __call__(self, x):
		return self.forward(x)


class GRU_Cell:
	"""docstring for GRU_Cell"""
	def __init__(self, in_dim, hidden_dim):
		self.d = in_dim
		self.h = hidden_dim
		h = self.h
		d = self.d

		self.Wzh = np.random.randn(h, h)
		self.Wrh = np.random.randn(h, h)
		self.Wh  = np.random.randn(h, h)

		self.Wzx = np.random.randn(h, d)
		self.Wrx = np.random.randn(h, d)
		self.Wx  = np.random.randn(h, d)

		self.dWzh = np.zeros((h, h))
		self.dWrh = np.zeros((h, h))
		self.dWh  = np.zeros((h, h))

		self.dWzx = np.zeros((h, d))
		self.dWrx = np.zeros((h, d))
		self.dWx  = np.zeros((h, d))

		self.z_act = Sigmoid()
		self.r_act = Sigmoid()
		self.h_act = Tanh()
		
	def forward(self, x, h):
		# input:
		# 	- x: shape(input dim),  observation at current time-step
		# 	- h: shape(hidden dim), hidden-state at previous time-step
		# 
		# output:
		# 	- h_t: hidden state at current time-step
		self.x = x
		self.h_t_prev = h
		self.z_t = self.z_act(np.matmul(self.Wzh, h) + np.matmul(self.Wzx, x))
		self.r_t = self.r_act(np.matmul(self.Wrh, h) + np.matmul(self.Wrx, x))

		self.z_7 = np.multiply(self.r_t, h)
		self.h_t_tilt = self.h_act(np.matmul(self.Wh, self.z_7) + np.matmul(self.Wx, x))

		self.z_11 = 1 - self.z_t
		self.h_t = np.multiply(self.z_11, h) + np.multiply(self.z_t, self.h_t_tilt)
		return self.h_t

	def backward(self, delta):
		# input:
		# 	- delta: 	shape(hidden dim), summation of derivative wrt loss from next layer at 
		# 			same time-step and derivative wrt loss from same layer at
		# 			next time-step
		#
		# output:
		# 	- dx: 	Derivative of loss wrt the input x
		# 	- dh: 	Derivative of loss wrt the input hidden h
		print("delta shape:", delta.shape)
		print("in_dim:", self.d)
		print("hidden_dim:", self.h)
		dz_12 = dz_13 = delta
		dz_t = np.multiply(dz_13, self.h_t_tilt.T)
		dh_t_tilt = np.multiply(dz_13, self.z_t.T)
		# print(dh_t_tilt.shape)
		dz_11 = np.multiply(dz_12, self.h_t_prev.T)
		dh_t = np.multiply(dz_12, self.z_11.T)
		dz_t += -dz_11

		dz_10 = np.multiply(dh_t_tilt, self.h_act.backward().T)
		# print(dz_10.shape)
		dz_8 = dz_9 = dz_10
		self.dWx = np.matmul(self.x.reshape(-1, 1), dz_9).T
		dx = np.dot(dz_9, self.Wx)
		self.dWh = np.dot(self.z_7.reshape(-1, 1), dz_8).T
		dz_7 = np.dot(dz_8, self.Wh)
		dr_t = np.multiply(dz_7, self.h_t_prev.T)
		dh_t += np.multiply(dz_7, self.r_t.T)

		dz_6 = np.multiply(dr_t, self.r_act.backward().T)
		dz_4 = dz_5 = dz_6
		self.dWrx = np.dot(self.x.reshape(-1, 1), dz_5).T
		dx += np.dot(dz_5, self.dWrx)
		self.dWrh = np.dot(self.h_t_prev.reshape(-1, 1), dz_4).T
		dh_t += np.dot(dz_4, self.Wrh)

		dz_3 = np.multiply(dz_t, self.z_act.backward().T)
		dz_2 = dz_1 = dz_3
		self.dWzx = np.dot(self.x.reshape(-1, 1), dz_2).T
		dx += np.dot(dz_2, self.Wzx)
		self.dWzh = np.dot(self.h_t_prev.reshape(-1, 1), dz_1).T
		dh_t += np.dot(dz_1, self.Wzh)

		return dx, dh_t


def test():
	c = GRU_Cell(in_dim=5, hidden_dim=3)
	x = np.array([1, 2, 3, 4, 5])
	h = np.array([1, 2, 3])
	# delta = np.array([6, 7, 8])
	delta = np.array([[1, 2, 3]])
	print(c.forward(x, h))
	print(c.backward(delta))


if __name__ == '__main__':
	test()









