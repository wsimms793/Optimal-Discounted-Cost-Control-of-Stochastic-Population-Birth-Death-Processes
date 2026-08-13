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
    [1.4, 1.0, 0.7, 300, 200, 100000, 500, 100],
    [0.6,0.95,0.25,75,280,100000,500,100]
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

# Function for producing generator matrix components.
def rates(i, d, lam, mu):

    #Birth rate.
    forw =  lam * i

    if i == 0:
        #Death rate at zeroth state.
        back = 0
        detect = 0

        #Diagonal entry for the zeroth row.
        same = -forw

    else:

        #Death rate for a general state.
        back = mu * i
        detect = d

        #Diagonal entry for a general state.
        same = -(forw + back + detect)

    return forw, back, detect, same


#################################################################################


#Function for computing the Bellman equation.
def bellman(
    i,
    alpha,
    C,
    P,
    V_back,
    V_forw,
    V_zero,
    lam,
    mu,
):

    # Cost when detection is not applied.
    d = 0

    forw, back, detect, diag = rates(
        i,
        d,
        lam,
        mu
    )

    K = (P * i + C * d) / (alpha - diag)

    K += (
        V_forw * forw
        + V_back * back
        + V_zero * detect
    ) / (alpha - diag)


    # Cost when detection is applied.
    d = 1

    forw, back, detect, diag = rates(
        i,
        d,
        lam,
        mu
    )

    L = (P * i + C * d) / (alpha - diag)

    L += (
        V_forw * forw
        + V_back * back
        + V_zero * detect
    ) / (alpha - diag)


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
                    V_matrix[0, j - 1],
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
                    V_matrix[0, j - 1],
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
                    V_matrix[0, j - 1],
                    lam,
                    mu,
                )

            # Record optimal values at the final iteration.
            if j == n - 1:
                V_opt[i] = V_matrix[i, j]
                D_opt[i] = d

    lower_bound = max(
        np.ceil(C * (alpha + mu - lam) / P),
        np.ceil(
            C * (alpha + mu - lam + 1) * (alpha - 1)
            / (alpha * P)
        )
    )

    upper_bound = (
        np.floor(C * (alpha + mu - lam + 1) / P) + 1
    )

    detecting_states = np.where(D_opt == 1)[0]

    if len(detecting_states) == 0:
        numerical_m = f">{s - 1}"
    else:
        numerical_m = detecting_states[0]

    prediction = (
        f"{int(lower_bound)} <= m <= {int(upper_bound)}"
    )

    print(
        f"lambda={lam}, mu={mu}, alpha={alpha}, P={P}, C={C}, "
        f"Lower bound={int(lower_bound)}, "
        f"Upper bound={int(upper_bound)}, "
        f"Prediction: {prediction}, "
        f"Numerical m: {numerical_m}"
    )
