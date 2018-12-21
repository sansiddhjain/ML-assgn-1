from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt 

#Returns solution to equation of form a*x^2+b*x+c
def quadratic_sol(a, b, c):
	D = np.sqrt(b*b - 4*a*c)
	sol = [(-b-D)/(2*a), (-b+D)/(2*a)]
	return sol

X = np.genfromtxt('q4x.dat', dtype=float)
y = np.genfromtxt('q4y.dat', dtype=str)

# Alaska represented by 0, Canada by 1
y[y=='Alaska'] = 0
y[y=='Canada'] = 1
y = y.astype(int)

#------Part (a)------

#Getting estimate of phi for p(y)
unique, counts = np.unique(y, return_counts=True)
phi = counts[0]/(counts[0] + counts[1])

#Normalisation
norm_mean = np.mean(X, axis=0)
norm_std = np.std(X, axis=0)

for i in range(X.shape[1]):
	X[:, i] = (X[:, i] - norm_mean[i])/norm_std[i]

#Estimation of Parameters - same sigma
X0 = X[y==0]
#mu0 = (sigma{y(i) == 0}x(i))/(sigma{y(i) == 0})
mu0 = np.mean(X0, axis=0)

X1 = X[y==1]
#mu1 = (sigma{y(i) == 1}x(i))/(sigma{y(i) == 1})
mu1 = np.mean(X1, axis=0)

#CoVar = (sigma{y(i) == 0}[x(i) - mu]*[x(i) - mu].T)/(sigma{y(i) == 0})
dim = X.shape
sigma = np.zeros((dim[1], dim[1]))
for i in range(dim[0]):
	t = X[i] - mu0
	sigma += np.dot(t.reshape(dim[1], 1), t.reshape(1, dim[1]))
sigma = sigma/dim[0]

print 'mu0 - '+str(mu0)
print 'mu1 - '+str(mu1)
print 'sigma - \n'+str(sigma)

#------Part (b)------

plt.figure()
plt.plot(X0[:, 0], X0[:, 1], 'rx', label='Alaska')
plt.plot(X1[:, 0], X1[:, 1], 'bx', label='Canada')
plt.title('Q4 GDA - Data Plot')
plt.xlabel('Growth Ring Diameters - Fresh Water (x1)')
plt.ylabel('Growth Ring Diameters - Marine Water (x2)')
plt.legend()
plt.savefig('4_gda_data_plot.png')
plt.show()

#------Part (c)------

#Decision Boundary Plotting - Linear
sigma_inv = np.linalg.inv(sigma)
A = np.dot(mu0.T, np.dot(sigma_inv, mu0))
B = np.dot(mu1.T, np.dot(sigma_inv, mu1))
c = np.subtract(A, B)

a = 2*np.dot((mu0-mu1).T, sigma_inv)

linear_line = (c - a[0]*X[:, 0])/a[1]

plt.figure()
plt.plot(X0[:, 0], X0[:, 1], 'rx', label='Alaska')
plt.plot(X1[:, 0], X1[:, 1], 'bx', label='Canada')
plt.plot(mu0[0], mu0[1], 'ro')
plt.plot(mu1[0], mu1[1], 'bo')
plt.plot(X[:, 0], linear_line, 'g-', label='Linear Decision Boundary')
plt.title('Q4 GDA - Linear Decision Boundary')
plt.xlabel('Growth Ring Diameters - Fresh Water (x1)')
plt.ylabel('Growth Ring Diameters - Marine Water (x2)')
plt.legend()
plt.savefig('4_gda_lin_dec_bound.png')
plt.show()

#------Part (d)------

#Estimation of Parameters - different sigma
X0 = X[y==0]
#mu0 = (sigma{y(i) == 0}x(i))/(sigma{y(i) == 0})
mu0 = np.mean(X0, axis=0)
dim = X0.shape
#sigma0 = (sigma{y(i) == 0}[x(i) - mu]*[x(i) - mu].T)/(sigma{y(i) == 0})
sigma0 = np.zeros((dim[1], dim[1]))
for i in range(dim[0]):
	t = X0[i] - mu0
	sigma0 += np.dot(t.reshape(dim[1], 1), t.reshape(1, dim[1]))
sigma0 = sigma0/dim[0]

X1 = X[y==1]
#mu1 = (sigma{y(i) == 1}x(i))/(sigma{y(i) == 1})
mu1 = np.mean(X1, axis=0)
dim = X1.shape
#sigma1 = (sigma{y(i) == 1}[x(i) - mu]*[x(i) - mu].T)/(sigma{y(i) == 1})
sigma1 = np.zeros((dim[1], dim[1]))
for i in range(dim[0]):
	t = X1[i] - mu1
	sigma1 += np.dot(t.reshape(dim[1], 1), t.reshape(1, dim[1]))
sigma1 = sigma1/dim[0]

print 'mu0 - '+str(mu0)
print 'mu1 - '+str(mu1)
print 'sigma0 - \n'+str(sigma0)
print 'sigma1 - \n'+str(sigma1)

#------Part (e)------

#Decision Boundary Plotting - Quadratic
sigma0_inv = np.linalg.inv(sigma0)
sigma1_inv = np.linalg.inv(sigma1)

M = sigma1_inv - sigma0_inv

lA = 2*np.dot(mu0.T, sigma0_inv)
lB = 2*np.dot(mu1.T, sigma1_inv)
l = lA - lB

cA = np.dot(mu1.T, np.dot(sigma1_inv, mu1))
cB = np.dot(mu0.T, np.dot(sigma0_inv, mu0))
cC = np.log(np.linalg.det(sigma1)/np.linalg.det(sigma0))
C = cA - cB + cC

x1 = np.linspace(np.amin(X[:, 0]), np.amax(X[:, 0]), num=100)
x2_1 = np.zeros(100)
x2_2 = np.zeros(100)
for i in range(100):
	a = M[1][1]
	b = 2*M[0][1]*x1[i]+l[1]
	c = M[0][0]*x1[i]*x1[i] + l[0]*x1[i] + C
	x2_1[i], x2_2[i] = quadratic_sol(a, b, c)

plt.figure()
axes = plt.gca()
axes.set_xlim([-10, 10])
plt.plot(X0[:, 0], X0[:, 1], 'rx', label='Alaska')
plt.plot(X1[:, 0], X1[:, 1], 'bx', label='Canada')
plt.plot(mu0[0], mu0[1], 'ro')
plt.plot(mu1[0], mu1[1], 'bo')
plt.plot(X[:, 0], linear_line, 'g-', label='Linear')
plt.plot(x1, x2_2, 'y-', label='Quadratic')
plt.title('Q4 GDA - Quadratic Decision Boundary')
plt.xlabel('Growth Ring Diameters - Fresh Water (x1)')
plt.ylabel('Growth Ring Diameters - Marine Water (x2)')
plt.legend()
plt.savefig('4_gda_quad_dec_bound.png')
plt.show()