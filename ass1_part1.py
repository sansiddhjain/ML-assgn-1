from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import math

#Relevant Functions
def hypothesis(theta, X):
	return np.dot(X, theta)

def CostF(y, X, theta):
	h = hypothesis(theta, X)
	J = 0.5*np.multiply(y-h, y-h)
	m = X.shape[0]
	J = J/m
	return np.sum(J)

def delCostF(y, X, theta):
	A = hypothesis(theta, X) - y
	delJ = np.dot(X.T, A)
	m = X.shape[0]
	delJ = delJ/m
	return delJ


#Input data
X = np.genfromtxt('linearX.csv', dtype=float)
y = np.genfromtxt('linearY.csv', dtype=float)

#Reshaping Data
y = y.reshape((len(y), 1))
ones = np.ones((len(X), 2))
ones[:, 1] = X
X = ones

#Normalisation
norm_mean_x = np.mean(X[:, 1])
norm_std_x = np.std(X[:, 1])
X[:, 1] = (X[:, 1] - norm_mean_x)/norm_std_x

#analytical solution
Z = np.dot(X.T, X)
Z = np.linalg.inv(Z)
W = np.dot(X.T, y)
analyticalSol = np.dot(Z, W)

#init
theta = np.zeros((2, 1))

#hyper-param
alpha = 0.025
epsilon = 1e-8

#Error Function
err = np.sqrt(np.sum(np.multiply(theta-analyticalSol, theta-analyticalSol)))

theta1_list = [theta[0]]
theta2_list = [theta[1]]
J_list = [CostF(y, X, theta)]

#Batch Gradient Descent Implementation
n_iter = 0
while (err > epsilon):
	J = CostF(y, X, theta)
	delJ = delCostF(y, X, theta)
	theta = np.subtract(theta, np.multiply(alpha, delJ))
	theta1_list.append(theta[0])
	theta2_list.append(theta[1])
	J_list.append(CostF(y, X, theta))
	n_iter = n_iter + 1
	err = np.sqrt(np.sum(np.multiply(theta-analyticalSol, theta-analyticalSol)))

print 'Number of iterations - ' + str(n_iter)

print 'Learnt Line Theta - \n'+str(theta)
print 'Analytical Line Theta- \n'+str(analyticalSol)
print 'Cost Function for Gradient Sol - ' + str(CostF(y, X, theta))
print 'Cost Function for Analytical Sol - ' + str(CostF(y, X, analyticalSol))

plt.figure()
plt.plot(X[:, 1], y, 'bo', label='Data')
plt.plot(X[:, 1], hypothesis(theta, X), 'r-', label='Gradient Descent Solution')
plt.title('Learnt Hypothesis')
plt.xlabel('Acidity of Wine (X)')
plt.ylabel('Density of Wine (y)')
plt.legend()
plt.savefig('1_linear_plot.png')
plt.show()

#3d Plot of Cost Function
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(theta1_list, theta2_list, J_list, label='Cost Function')
ax.legend()
plt.title('J(Theta0, Theta1)')
ax.set_zlabel('J')
ax.set_xlabel('Theta0')
ax.set_ylabel('Theta1')
plt.show()

theta1_list = np.asarray(theta1_list)
theta1_list = theta1_list.reshape(len(theta1_list))
theta2_list = np.asarray(theta2_list)
theta2_list = theta2_list.reshape(len(theta2_list))
J_list = np.asarray(J_list)
J_list = J_list.reshape(len(J_list))

#Contour Plot
m = X.shape[0]
y_tmp = y.reshape(len(y))

theta1 = np.linspace(-1.1, 1.1, num=100)
theta2 = np.linspace(-0.2, 0.2, num=100)
TH1, TH2 = np.meshgrid(theta1, theta2)

J = np.zeros(TH1.shape)
for i in range(TH1.shape[0]):
	for j in range(TH1.shape[1]):
		theta_for_contour = np.zeros(2)
		theta_for_contour[0] = TH1[i][j]
		theta_for_contour[1] = TH2[i][j]
		theta_for_contour.reshape((2,1))
		J[i][j] = CostF(y, X, theta_for_contour)

levels = J_list[:200]

plt.figure()
plt.axis([0.8, 1.2, -0.1, 0.1])
CS = plt.contour(TH1, TH2, J, levels)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Contour Plot for Part(e) (First 200 iters) (Eta = 0.025)')
plt.xlabel('Theta0')
plt.ylabel('Theta1')
plt.savefig('1_contour_parte_eta0025.png')
plt.show()