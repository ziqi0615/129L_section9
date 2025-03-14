#!/usr/bin/env python

from numpy import sqrt, dot, linspace, array, zeros
from matplotlib import pyplot as plt

# A
# H = (p1^2 + p2^2)/2 - 1/sqrt(q1^2 + q2^2)
def Hq(q, p):
	return q / dot(q, q)**1.5
def Hp(q, p):
	return p

def euler(q0, p0, T, n):
	t = linspace(0, T, n+1)
	dt = T / n
	q = zeros((n+1, *q0.shape))
	p = zeros((n+1, *p0.shape))
	q[0], p[0] = q0, p0
	for i in range(n):
		q[i+1] = q[i] + Hp(q[i], p[i]) * dt
		p[i+1] = p[i] - Hq(q[i], p[i]) * dt
	return t, q, p

e = 0.6
q0 = array([1-e, 0])
p0 = array([0, sqrt((1+e)/(1-e))])
t1, q1, p1 = euler(q0, p0, 200, 100000)

# B
def symplectic_euler(q0, p0, T, n):
	t = linspace(0, T, n+1)
	dt = T / n
	q = zeros((n+1, *q0.shape))
	p = zeros((n+1, *p0.shape))
	q[0], p[0] = q0, p0
	for i in range(n):
		p[i+1] = p[i] - Hq(q[i], p[i]) * dt
		q[i+1] = q[i] + Hp(q[i], p[i+1]) * dt
	return t, q, p

t2, q2, p2 = symplectic_euler(q0, p0, 200, 400000)

plt.plot(*q1.T, label='Euler')
plt.plot(*q2.T, label='Symplectic Euler')
plt.xlabel('$q_1$')
plt.ylabel('$q_2$')
plt.legend()
plt.show()