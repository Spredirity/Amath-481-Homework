import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

tol = 1e-6  # define a tolerance level
col = ['r', 'b', 'g', 'c', 'm']  # eigenfunc colors
L = 4  # define the length of the domain
dx = 0.1  # define the step size
xshoot = np.arange(-L, L + dx, dx)  # define the x range
beta_start = 0.1  # beginning value of beta

def shoot2(phi, x, beta):
   return [phi[1], (1 * x**2 - beta) * phi[0]]

A1 = np.zeros((len(xshoot), 5)) 
A2 = np.zeros(5)

for modes in range(1, 6):  # begin mode loop
    beta = beta_start  # initial value of eigenvalue beta
    dbeta = 0.2  # default step size in beta

    for _ in range(1000):  # begin convergence loop for beta
        sd = np.sqrt(L**2 - beta)
        x0 = [1, sd]
        y = odeint(shoot2, x0, xshoot, args=(beta, ))

        if abs(y[-1, 1] + sd * y[-1,0]) < tol:  # check for convergence
            break  # get out of convergence loop

        if ((-1) ** (modes + 1) * (y[-1, 1] + sd * y[-1,0])) > 0:
            beta += dbeta
        else:
            beta -= dbeta
            dbeta /= 2

    A2[modes - 1] = beta  # store eigenvalue
    beta_start = beta + 0.1  # after finding eigenvalue, pick new start
    norm = np.trapezoid(y[:, 0]**2, xshoot) # calculate the normalization
    eigenfuction = abs(y[:, 0] / np.sqrt(norm)) # normalize eigenfunction
    A1[:, modes - 1] = eigenfuction  # store eigenfunction
    plt.plot(xshoot, eigenfuction, col[modes - 1])  # plot modes

print(A1)
print(A2)
plt.show()  # show the plot       