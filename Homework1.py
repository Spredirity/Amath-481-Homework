from math import sin
import numpy as np
import matplotlib.pyplot as plt

## Q1

#Part 1
x = np.array([-1.6]) # initial guess
np.set_printoptions(precision = 15)

for j in range(1000):
    x_new = x[j] - ((x[j] * np.sin(3 * x[j]) - np.exp(x[j])) 
                  / (np.sin(3 * x[j]) + 3 * x[j] * np.cos(3 * x[j]) - np.exp(x[j])))
    x = np.append( x, x_new)
    fc = x[j] * np.sin(3 * x[j]) - np.exp(x[j])
    
    if abs(fc) < 1e-6:
        break

A1 = []
A1 = x
newton_step = len(A1) - 1
print("Final value:", x_new)
print("Number of iterations: ",  newton_step)
print("vector of x-values: ", A1)

# Part 2
A2 = []
xr = -0.4; xl = -0.7

dx = 0.1
x2 = np.arange(-10, 10+dx, dx)
y = x2 * np.sin(3 * x2) - np.exp(x2)
plt.plot(x2, y)
plt.axhline(0, color='red', linestyle='--')
plt.axis([-1, 1, -1, 1])
# plt.show() 
# # the curve is downward

for j in range(0, 1000):
    xc = (xr + xl) / 2
    fc = xc * np.sin(3 * xc) - np.exp(xc)
    A2.append(xc)
    if ( fc > 0 ):
        xl = xc
    else:
        xr = xc

    if( abs(fc) < 1e-6 ):
        break
bisec_step = j + 1
print("fianl mid point value:", xc)
print("Number of iterations: ",  bisec_step)
print("fianl mid point values:", A2)

A3 = [newton_step, bisec_step]
# Show the final solution
print("A1: ", A1)
print("A2: ", A2)
print("A3: ", A3)

## Q2
A = np.array([[1, 2],
              [-1, 1]])

B = np.array([[2, 0],
              [0, 2]])

C = np.array([[2, 0, -3],
             [0, 0, -1]])

D = np.array([[1, 2],
              [2, 3],
              [-1, 0]])

x = np.array([1, 0])

y = np.array([0, 1])

z = np.array([1, 2, -1])

A4 = A + B

A5 = 3 * x - 4 * y

A6 = np.dot(A, x) 

x_minus_y = x - y
A7 = np.dot(B,  x_minus_y)

A8 = np.dot(D, x)

A9 = np.dot(D , y) + z

A10 = np.dot(A, B)

A11 = np.dot(B, C)

A12 = np.dot(C, D)

print("(a) A4: ", A4)
print("(b) A5: ", A5)
print("(c) A6: ", A6)
print("(d) A7: ", A7)
print("(e) A8: ", A8)
print("(f) A9: ", A9)
print("(g) A10: ", A10)
print("(h) A11: ", A11)
print("(i) A12: ", A12)