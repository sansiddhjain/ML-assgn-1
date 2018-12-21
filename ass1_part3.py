from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt 

#------Part (a)------

#Relevant functions
def sigmoid(z):
	return 1/(1+np.exp(-z))

def hypothesis(theta, X):
	return sigmoid(np.dot(X, theta))

def CostF(y, X, theta):
	A = np.multiply(y, np.log(hypothesis(theta, X)))
	B = np.multiply(1-y, np.log(1-hypothesis(theta, X)))
	return np.sum(A+B)

def delCostF(y, X, theta):
	arr = y - hypothesis(theta, X)
	arr = arr.reshape(len(arr))
	dim = len(theta)
	delJ = np.zeros(dim)
	for j in range(dim):
		vec = np.multiply(arr, X[:, j])
		delJ[j] = np.sum(vec)

	delJ = delJ.reshape((len(delJ), 1))
	return delJ


def Hessian(X, theta):
	a = hypothesis(theta, X)
	b = 1-hypothesis(theta, X)
	a = a.reshape(len(a))
	b = b.reshape(len(b))
	arr = a*b

	dim = len(theta)
	H = np.zeros((dim, dim))
	for j in range(dim):
		for k in range(dim):
			vec = np.multiply(arr, np.multiply(X[:, j], X[:, k]))
			H[j][k] = -np.sum(vec)

	return H

#Reading Data, Reshaping X
X = np.genfromtxt('logisticX.csv', dtype=float, delimiter=',')
y = np.genfromtxt('logisticY.csv', dtype=int)
l = X.shape[1]
ones = np.ones((len(X), l+1))
ones[:, 1:] = X
X = ones

#Normalisation
norm_mean = np.mean(X, axis=0)
norm_std = np.std(X, axis=0)

for i in range(1, X.shape[1]):
	X[:, i] = (X[:, i] - norm_mean[i])/norm_std[i]

#Reshaping y
X0 = X[y==0]
X1 = X[y==1]
y = y.reshape((len(y), 1))

#init
theta = np.zeros((l+1, 1))
n_iter = 0
err_lim = 1e-15
err_val = 1

#Newton's Method
while (err_val > err_lim):
	costf_old = CostF(y, X, theta)
	# print 'Cost Function - ' + str(costf_old)
	H = Hessian(X, theta)
	H_inv = np.linalg.inv(H)
	delJ = delCostF(y, X, theta)
	theta = theta - np.dot(H_inv, delJ)
	n_iter += 1
	costf_new = CostF(y, X, theta)
	# print costf_new
	err_val = abs(costf_new - costf_old)
	# print err_val

print 'Number of iterations - ' + str(n_iter)
print 'Final Cost Function - ' + str(CostF(y, X, theta))
print 'Final theta - ' + str(theta)

x1 = np.linspace(np.amin(X[:, 1]), np.amax(X[:, 1]), num=100)
x2 = -(theta[0]+theta[1]*x1)/(theta[2])

#------Part (b)------

#Plotting of Learnt Hypothesis
plt.figure()
plt.plot(X0[:, 1], X0[:, 2], 'bx', label='y = 0')
plt.plot(X1[:, 1], X1[:, 2], 'ro', label='y = 1')
plt.plot(x1, x2, 'g-', label='Decision Boundary')
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.savefig('3_logistic.png')
plt.show()
