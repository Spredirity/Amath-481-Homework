import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
from scipy.integrate import solve_ivp, simpson
import math


# HW 2 part a

def shoot2(x, phi, beta):
   return [phi[1], (1 * x**2 - beta) * phi[0]]


tol = 1e-6  # define a tolerance level
col = ['r', 'b', 'g', 'c', 'm']  # eigenfunc colors
L = 4  # define the length of the domain
dx = 0.1  # define the step size
xshoot = np.arange(-L, L + dx, dx)  # define the x range
beta_start = 0.1  # beginning value of beta

A1 = np.zeros((len(xshoot), 5)) 
A2 = np.zeros(5)

for modes in range(1, 6):  # begin mode loop
    beta = beta_start  # initial value of eigenvalue beta
    dbeta = 0.2  # default step size in beta

    for _ in range(1000):  # begin convergence loop for beta
        sd = np.sqrt(L**2 - beta)
        y0 = [1, sd]
        sol = solve_ivp(shoot2, [xshoot[0], xshoot[-1]], y0, args=(beta, ), t_eval=xshoot)

        if abs(sol.y[1, -1] + sd * sol.y[0,-1]) < tol:  # check for convergence
            break  # get out of convergence loop

        if ((-1) ** (modes + 1) * (sol.y[1, -1] + sd * sol.y[0, -1])) > 0:
            beta += dbeta
        else:
            beta -= dbeta
            dbeta /= 2

    A2[modes - 1] = beta  # store eigenvalue
    beta_start = beta + 0.1  # after finding eigenvalue, pick new start
    norm = np.trapezoid(sol.y[0]**2, xshoot) # calculate the normalization
    eigenfuction = abs(sol.y[0] / np.sqrt(norm)) # normalize eigenfunction
    A1[:, modes - 1] = eigenfuction  # store eigenfunction
    plt.plot(xshoot, eigenfuction, col[modes - 1])  # plot modes

print(A1)
print(A2)
#plt.show()  # show the plot

# HW 3 part b
L = 4  # Range limit
dx = 0.1  # X step size
x = np.arange(-L, L + dx, dx)  # X range with 81 points
N = len(x) - 2  # Number of interior points excluding boundaries

B = np.zeros((N, N))  # Initialize the matrix B

B1 = np.zeros((N, N))
for i in range(N):
    B1[i, i] = -2 - (x[i + 1] ** 2) * dx ** 2  # Diagonal elements
for i in range(N - 1):
    B1[i, i + 1] = 1  # Upper diagonal elements
    B1[i + 1, i] = 1  # Lower diagonal elements

B2 = np.zeros((N, N))
B2[0,0] = 4 / 3
B2[0,1] = -1 / 3
B2[N - 1, N - 2] = - 1 / 3
B2[N - 1, N - 1] = 4 / 3

B = B1 + B2 
B = B / dx ** 2

D, V = eigs(- B, k = 5, which = 'SM')  # Solve the eigenvalue problem

phi_0 = (4 / 3) * V[0, :] - (1 / 3) * V[1, :] # Boundary condition at x = -L
phi_n = - (1 / 3) * V[-2, :] + (4 / 3) * V[-1, :] # Boundary condition at x = L

V = np.vstack((phi_0, V, phi_n))  # Add the boundary conditions to the eigenvectors

for i in range(5):
    norm = np.trapezoid(V[:, i] ** 2, x)  # Calculate the normalization
    V[:, i] = abs(V[:, i] / np.sqrt(norm))  # Normalize the eigenvectors
    plt.plot(x, V[:, i])  # Plot the eigenvectors

plt.legend(['phi 1', 'phi 2', 'phi 3', 'phi 4', 'phi 5'], loc = 'lower left')  # Add a legend to the plot

A3 = V  # Store the eigenvectors
A4 = D  # Store the eigenvalues

print(A3)
print(A4)
#plt.show()  # Show the plot

# HW 3 part c
def shoot_c(x, phi, epsilon, gamma):
    return[phi[1], gamma * phi[0] ** 2 + x ** 2 - epsilon * phi[0]]


tol = 1e-6  # Tolerance level
L = 2  # Range limit
dx = 0.1  # X step size
xshoot = np.arange(-L, L + dx, dx)  # X range with 41 points
gvalue = [0.05, -0.05] # Initial values of gamma

A5, A7 = np.zeros((len(xshoot), 2)), np.zeros((len(xshoot), 2))  
A6, A8 = np.zeros(2), np.zeros(2)  # Initialize the results for eigenvalues

for gamma in gvalue:
    epsilon0 = 0.1  # Initial value of epsilon
    A  = 1e-6  # Initial step size

    for modes in range(1, 3):
        dA = 0.01

        for j in range(100):
            epsilon = epsilon0
            depsilon = 0.2

            for i in range(100):
                phi0  = [A, np.sqrt(L ** 2 - epsilon) * A]
                
                solu = solve_ivp(shoot_c, [xshoot[0], xshoot[-1]], 
                                 phi0, args = (epsilon, gamma), t_eval = xshoot)
                phi_sol = solu.y.T
                x_sol = solu.t

                # Check for convergence
                diff = phi_sol[-1, 1]+ np.sqrt(L ** 2 - epsilon) * phi_sol[-1, 0] 
                if abs(diff) < tol:
                    break

                # Update epsilon
                if ((-1) ** (modes + 1) * diff) > 0:
                    epsilon += depsilon
                else:
                    epsilon -= depsilon
                    depsilon /= 2

            integral = simpson(phi_sol[:, 0] ** 2, x = x_sol)  # Calculate the normalization
            if abs(integral - 1) < tol:
                break

            # Update A
            if integral - 1 < 0:
                A += dA
            
            else:
                A -= dA
                dA /= 2

        #update epsilon0
        epsilon0 = epsilon + 0.2

        if gamma > 0:
            A5[:, modes - 1] = np.abs(phi_sol[:, 0])  # Store the eigenfunctions
            A6[modes - 1] = epsilon # Store the eigenvalues
        else:
            A7[:, modes - 1] = np.abs(phi_sol[:, 0]) # Store the eigenfunctions
            A8[modes - 1] = epsilon # Store the eigenvalues

plt.plot(xshoot, A5)
plt.plot(xshoot, A7)

plt.legend(["$\\phi_1$", "$\\phi_2$"], loc="upper right")
print(A6)
print(A8)
# plt.show()

# HW 3 part d
def defd(x, phi, epsilon):
    return [phi[1], (x ** 2 - epsilon) * phi[0]]


L = 2  # Range limit
x_span = [-L, L]  # X range
epsilon = 1  # Initial value of epsilon
A = 1  # Initial value of A
phi0 = [A, np.sqrt(L ** 2 - epsilon) * A]  # Initial values of phi
tols = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]  # Tolerance levels

dt45, dt23, dtRadua, dtBDF = [], [], [], []

for tol in tols:
    options = {'rtol': tol, 'atol': tol}

    sol45 = solve_ivp(defd, x_span, phi0, method='RK45', args=(epsilon,), **options)
    sol23 = solve_ivp(defd, x_span, phi0, method='RK23', args=(epsilon,), **options)
    solRadua = solve_ivp(defd, x_span, phi0, method='Radau', args=(epsilon,), **options)
    solBDF = solve_ivp(defd, x_span, phi0, method='BDF', args=(epsilon,), **options)

    # calculate average time steps
    dt45.append(np.mean(np.diff(sol45.t)))
    dt23.append(np.mean(np.diff(sol23.t)))
    dtRadua.append(np.mean(np.diff(solRadua.t)))
    dtBDF.append(np.mean(np.diff(solBDF.t)))

# Fit the data
fit45 = np.polyfit(np.log(dt45), np.log(tols), 1)
fit23 = np.polyfit(np.log(dt23), np.log(tols), 1)
fitRadua = np.polyfit(np.log(dtRadua), np.log(tols), 1)
fitBDF = np.polyfit(np.log(dtBDF), np.log(tols), 1)

slope45 = fit45[0]
slope23 = fit23[0]
slopeRadua = fitRadua[0]
slopeBDF = fitBDF[0]

A9 = np.array([slope45, slope23, slopeRadua, slopeBDF])  # Store the slopes

print(A9)  # Print the slopes

# HW 3 part e
# part e
L = 4
dx = 0.1
x = np.arange(-L, L + dx, dx)
n = len(x)

h = np.array([np.ones_like(x),
              2 * x,
              4 * (x ** 2) - 2,
              8 * (x ** 3) - 12 * x,
              16 * (x ** 4) - 48 * (x ** 2) + 12])

phi = np.zeros((n, 5))
for j in range(5):
    phi[:, j] = (np.exp(- x**2 / 2) * h[j, :] /
                 (np.sqrt(math.factorial(j) * 2**j * np.sqrt(np.pi)))).T

Erpsi_a = np.zeros(5)
Erpsi_b = np.zeros(5)
Er_a = np.zeros(5)
Er_b = np.zeros(5)

array = [1, 3, 5, 7, 9]
for j in range(5):
    Erpsi_a[j] = simpson(((np.abs(A1[:, j])) - np.abs(phi[:, j]))**2, x)
    Erpsi_b[j] = simpson(((np.abs(A3[:, j])) - np.abs(phi[:, j]))**2, x)

    Er_a[j] = 100 * (abs(A2[j] - (array[j])) / (array[j]))
    Er_b[j] = 100 * (abs(A4[j] - (array[j])) / (array[j]))

A10 = Erpsi_a
A12 = Erpsi_b

print(A10)
print(A12)

A11 = Er_a
A13 = Er_b

print(A11)
print(A13)
