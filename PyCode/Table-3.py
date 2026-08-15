##    Importing Relavent Packages   ##
import numpy as np
import matplotlib.pyplot as plt

#################################################################################

parameters = [
    # beta, gamma, N, alpha, P, C
    [4.0, 4.5, 100, 0.30, 700,  100],
    [1.0, 1.2, 100, 0.20, 700,  200],
    [1.5, 1.8, 100, 0.10, 500,  300],
    [4.0, 4.5, 100, 0.40, 200,  400],
    [2.0, 2.4, 100, 0.45, 100,  500],
    [0.5, 0.8, 100, 0.20, 350,  600],
    [1.0, 1.2, 100, 0.10,  20,  700],
    [3.0, 3.4, 100, 0.40, 100,  800],
    [5.0, 5.5, 100, 0.45, 200,  900],
    [2.0, 2.3, 100, 0.45, 500, 1000],
]

#################################################################################
#Function for calculating rates


def rates(i, beta, gamma, N, d):

    #State i = 0
    if i == 0:
        forw = 0
        back = 0
        same = 0
        return forw, back, same

    # State i = N
    if i == N:
        forw = 0
        back = gamma * i
        same = -back - d
        return forw, back, same

    # General state
    forw = beta * i * (N - i) / N
    back = gamma * i
    same = -forw - back - d
    return forw, back, same

#################################################################################
#Construct generator matrix

def generator(beta, gamma, N, f):

    matrix = np.zeros((N + 1, N + 1))

    for i in range(N + 1):
        forw, back, same = rates(
            i,
            beta,
            gamma,
            N,
            f[i]
        )

        for j in range(N + 1):

            if i == j:
                matrix[i, j] = same

            elif j == i - 1:
                if i == 1:
                    matrix[i,j] = back + f[i]
                else:
                    matrix[i, j] = back

            elif j == i + 1:
                matrix[i, j] = forw

            elif j == 0:
                if i != 0 and i != 1:
                    matrix[i,j] = f[i]

    return matrix


#################################################################################
#Cost array under a policy


def cost(P, C, f, N):

    cost_array = np.zeros(N + 1)

    for i in range(N + 1):
        cost_array[i] = i * P + C * (f[i])

    return cost_array

#################################################################################
#Discounted cost


def J(alpha, f, beta, N, gamma, P, C):

    Q = generator(
        beta,
        gamma,
        N,
        f
    )

    I = np.identity(N + 1)

    A = alpha * I - Q
    B = cost(P, C, f, N)

    x = np.linalg.solve(A, B)

    return x




#################################################################################
# Proposal discounted cost at state i


def proposal_cost(
    i,
    a,
    f,
    beta,
    N,
    gamma,
    P,
    C,
    J_values
):

    # The rates must correspond to the proposed action a,
    # rather than the current action f[i].
    forw, back, same = rates(
        i,
        beta,
        gamma,
        N,
        a
    )

    # The cost must also correspond to the proposed action a.
    c = i * P + C * a

    # State i = 0
    if i == 0:

        D = (
            c
            + J_values[i + 1] * forw
            + J_values[i] * same
        )

        return D

    # State i = 1
    if i == 1:

        D = (
            c
            + J_values[i + 1] * forw
            + J_values[0] * (back + a)
            + J_values[i] * same
        )

        return D

    # State i = N
    if i == N:

        D = (
            c
            + J_values[i - 1] * back
            + J_values[0] * a
            + J_values[i] * same
        )

        return D

    # General state, 2 <= i <= N - 1
    D = (
        c
        + J_values[i + 1] * forw
        + J_values[i - 1] * back
        + J_values[0] * a
        + J_values[i] * same
    )

    return D


#################################################################################
# Run policy iteration for each parameter set


for beta, gamma, N, alpha, P, C in parameters:

#############################################################################
    # Initial policy

    f = np.zeros(N + 1)

    # Action at state zero is fixed as zero
    f[0] = 0


#############################################################################
    # Initial discounted cost

    J_f = J(
        alpha,
        f,
        beta,
        N,
        gamma,
        P,
        C
    )


#############################################################################
    # Store policies and discounted costs

    F = [f.copy()]
    K = [J_f.copy()]


#############################################################################
    # Policy iteration

    while True:

        f_new = f.copy()

        # Propose the alternative action at states 1,...,N
        for i in range(1, N + 1):

            if f_new[i] == 0:
                f_new[i] = 1

            else:
                f_new[i] = 0

        # Action at state zero is fixed
        f_new[0] = 0


        # Check whether the proposed action is an improvement
        for i in range(1, N + 1):

            D = proposal_cost(
                i,
                f_new[i],
                f,
                beta,
                N,
                gamma,
                P,
                C,
                J_f
            )

            current_value = J_f[i]

            if D >= alpha * current_value:
                f_new[i] = f[i]

        # Action at state zero remains fixed
        f_new[0] = 0


        # Stopping condition
        if np.array_equal(f_new, f):
            break


        # Update policy
        f = f_new.copy()


        # Recalculate discounted cost
        J_f = J(
            alpha,
            f,
            beta,
            N,
            gamma,
            P,
            C
        )


        # Store policy and discounted cost
        F.append(f.copy())
        K.append(J_f.copy())


    #############################################################################
    # Numerical threshold

    intervention_states = np.where(f[1:] == 1)[0] + 1

    if len(intervention_states) == 0:
        numerical_m = "No intervention"

    else:
        numerical_m = int(intervention_states[0])

#############################################################################
    # Approximate threshold

    approximate_m = min(
        int(
            np.ceil(
                (N * (gamma - beta) / beta)
                * (
                    np.exp(C * beta / (P * N)) - 1
                )
            )
        ),
        N
    )

#############################################################################
    # Print results

    print("Beta =", beta)
    print("Gamma =", gamma)
    print("N =", N)
    print("Alpha =", alpha)
    print("P =", P)
    print("C =", C)
    print("Numerical m =", numerical_m)
    print("Approximate m =", approximate_m)
    print("Final f =", f)
    print()
