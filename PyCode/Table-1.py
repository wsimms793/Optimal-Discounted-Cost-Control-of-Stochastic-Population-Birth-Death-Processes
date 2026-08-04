##    Importing Relavent Packages   ##
import numpy as np
import matplotlib.pyplot as plt

#################################################################################
#Parameters
parameters = [
    # lam, mu, alpha, P, C, n, s, s_plot
    [0.5, 1.0, 1.0, 200, 100, 100000, 500, 100],
    [0.4, 1.2, 0.8, 100, 200, 100000, 500, 100],
    [0.2, 0.9, 0.5, 30, 80, 100000, 500, 100],
    [0.8, 1.1, 0.4, 140, 150, 100000, 500, 100],
    [1.0, 0.7, 0.6, 250, 300, 100000, 500, 100],
    [1.2, 0.9, 0.5, 30, 250, 100000, 500, 100],
    [0.3, 0.6, 0.2, 85, 120, 100000, 500, 100],
    [0.9, 1.5, 0.3, 67.5, 75, 100000, 500, 100],
    [1.4, 1.0, 0.7, 100, 200, 100000, 500, 100],
    [0.6, 0.95, 0.25, 75, 180, 100000, 500, 100],
]

n,s,s_plot = 100000,500,100
#################################################################################
#Constructing arrays to recrod results.

#Matrix of costs across states and iterations, Vector for optimal descions,
#vector for optimal costs, Vector for states.
V_matrix = np.zeros((s, n))
D_opt = np.zeros(s)
V_opt = np.zeros(s)
states = np.arange(0, s_plot)

#################################################################################

#Function for producing generator matrix components.
def rates(i, a, lam, mu):
    #Birth rate.
    forw = lam * i + a

    #Death rate.
    back = mu * i

    #Diagonal entry.
    same = -(forw + back)
    return forw, back, same


#################################################################################

#Function for computing the Bellman equation.

def bellman(i, alpha, C, P, V_back, V_forw, lam, mu):

    # Cost when spontatnous infection is prevented.
    a = 0
    forw, back, diag = rates(i, a, lam, mu)
    K = (P * i + C * (1 - a)) / (alpha - diag)
    K += (V_forw * forw + V_back * back) / (alpha - diag)

    # Cost when spontatnous infection is allowed.
    a = 1
    forw, back, diag = rates(i, a, lam, mu)

    L = (P * i + C * (1 - a)) / (alpha - diag)
    L += (V_forw * forw + V_back * back) / (alpha - diag)

    #Compute the minimum cost and record the assosiated policy.
    if K <= L:
        return K, 0
    else:
        return L, 1


#################################################################################

# Main loop for finding optimal discounted cost and associated policy.
for k, p in enumerate(parameters):
    lam, mu, alpha, P, C, n, s, s_plot = p

    # Reset arrays for the current parameter set.
    V_matrix = np.zeros((s, n))
    D_opt = np.zeros(s, dtype=int)
    V_opt = np.zeros(s)

    for j in range(1, n):
        for i in range(s):

            # Lower boundary state.
            if i == 0:
                V_matrix[i, j], d = bellman(
                    i,
                    alpha,
                    C,
                    P,
                    0,
                    V_matrix[i + 1, j - 1],
                    lam,
                    mu,
                )

            # Upper boundary state.
            elif i == s - 1:
                V_forw = V_matrix[i, j - 1] + (P / alpha)

                V_matrix[i, j], d = bellman(
                    i,
                    alpha,
                    C,
                    P,
                    V_matrix[i - 1, j - 1],
                    V_forw,
                    lam,
                    mu,
                )

            # Intermediate states.
            else:
                V_matrix[i, j], d = bellman(
                    i,
                    alpha,
                    C,
                    P,
                    V_matrix[i - 1, j - 1],
                    V_matrix[i + 1, j - 1],
                    lam,
                    mu,
                )

            # Record optimal values at the final iteration.
            if j == n - 1:
                V_opt[i] = V_matrix[i, j]
                D_opt[i] = d

    condition_value = C * (alpha + mu - lam)

    if P < condition_value:
        prediction = "f*(i) = 1 for all i"
    elif P > condition_value:
        prediction = "f*(i) = 0 for all i"
    else:
        prediction = "f*(i) = 0 or 1 for all i"

    print(
        f"lambda={lam}, mu={mu}, alpha={alpha}, P={P}, C={C}, "
        f"C(alpha + mu - lambda)={condition_value}, "
        f"Prediction: {prediction}, "
        f"Optimal policy: {D_opt}"
    )
