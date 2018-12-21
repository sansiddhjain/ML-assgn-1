from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt 

#Read and Reshape Data
X = np.genfromtxt('weightedX.csv', dtype=float)
y = np.genfromtxt('weightedY.csv', dtype=float)
y = y.reshape((len(y), 1))
ones = np.ones((len(X), 2))
ones[:, 1] = X
X = ones

#Normalisation
norm_mean_x = np.mean(X[:, 1])
norm_std_x = np.std(X[:, 1])
X[:, 1] = (X[:, 1] - norm_mean_x)/norm_std_x

#------Part (a)------

#Linear Regression Fit
Z = np.dot(X.T, X)
Z = np.linalg.inv(Z)
theta = np.dot(Z, np.dot(X.T, y))

#Linear Regression Plot
plt.figure()
plt.plot(X[:, 1], y, 'bo')
plt.plot(X[:, 1], np.dot(X, theta), 'g-')
plt.title('Linear Regression Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('2_a_linear.png')
plt.show()

#------Part (b)------

#Weighted Linear Regression
x_plt = np.linspace(np.amin(X[:, 1]), np.amax(X[:, 1]), num=1000)

#For a given array input of x, and a given tau, function outputs corresponding y
def create_weighted(X, y, x_plt, tau):
	y_plt = np.zeros(1000)
	W = np.zeros((X.shape[0], X.shape[0]))
	
	for i in range(1000):
		x = x_plt[i]
		arr = ((X[:, 1] - x)*(X[:, 1] - x))/(2*tau*tau)
		arr = np.exp(-arr)
		# print arr
		W = np.diag(arr)

		Z = np.dot(X.T, np.dot(W, X))
		Z = np.linalg.inv(Z)
		theta_w = np.dot(Z, np.dot(X.T, np.dot(W, y)))

		x_inp = [1, x]
		x_inp = np.asarray(x_inp)
		y_plt[i] = np.dot(theta_w.T, x_inp)

	return y_plt

#Weighted Regression Plot
plt.figure()
plt.plot(X[:, 1], y, 'bo')
plt.plot(x_plt, create_weighted(X, y, x_plt, 0.8), 'g-', label='Tau-0.8')
plt.title('Weighted Regression Plot - Tau-0.8')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('2_b_weighted_08.png')
plt.show()

#------Part (c)------

plt.figure()
plt.plot(X[:, 1], y, 'bo')
plt.plot(x_plt, create_weighted(X, y, x_plt, 0.1), 'g-', label='Tau-0.1')
plt.title('Weighted Regression Plot - Tau-0.1')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('2_c_weighted_01.png')
plt.show()

plt.figure()
plt.plot(X[:, 1], y, 'bo')
plt.plot(x_plt, create_weighted(X, y, x_plt, 0.3), 'g-', label='Tau-0.3')
plt.title('Weighted Regression Plot - Tau-0.3')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('2_c_weighted_03.png')
plt.show()

plt.figure()
plt.plot(X[:, 1], y, 'bo')
plt.plot(x_plt, create_weighted(X, y, x_plt, 2), 'g-', label='Tau-2')
plt.title('Weighted Regression Plot - Tau-2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('2_c_weighted_2.png')
plt.show()

plt.figure()
plt.plot(X[:, 1], y, 'bo')
plt.plot(x_plt, create_weighted(X, y, x_plt, 10), 'g-', label='Tau-10')
plt.title('Weighted Regression Plot - Tau-10')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('2_c_weighted_10.png')
plt.show()

plt.legend()