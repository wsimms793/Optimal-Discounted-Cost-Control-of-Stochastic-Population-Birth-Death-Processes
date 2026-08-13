##    Importing Relavent Packages   ##
import numpy as np
import matplotlib.pyplot as plt

#################################################################################

#Parameters


beta, gamma, N, alpha, b, P, C = [
    5,
    4,
    100,
    0.8,
    0.3,
    600,
    1200
]
#################################################################################

#Function for calculating rates


def rates(i, beta, gamma, N, b, a):

    #State i = 0
    if i == 0:
        forw = a
        back = 0
        same = -a
        return forw, back, same

    # State i = N
    if i == N:
        forw = 0
        back = (b + gamma) * i
        same = -back
        return forw, back, same

    # General state
    forw = beta * i * (N - i) / N + a
    back = (b + gamma) * i
    same = -forw - back

    return forw, back, same


#################################################################################
#Construct generator matrix


def generator(beta, gamma, N, b, f):

    matrix = np.zeros((N + 1, N + 1))

    for i in range(N + 1):

        forw, back, same = rates(
            i,
            beta,
            gamma,
            N,
            b,
            f[i]
        )

        for j in range(N + 1):

            if i == j:
                matrix[i, j] = same

            elif j == i - 1:
                matrix[i, j] = back

            elif j == i + 1:
                matrix[i, j] = forw

    return matrix


#################################################################################
#Cost array under a policy


def cost(P, C, f, N):

    cost_array = np.zeros(N + 1)

    for i in range(N + 1):
        cost_array[i] = i * P + C * (1 - f[i])

    return cost_array


#################################################################################
#Discounted cost


def J(alpha, f, beta, b, N, gamma, P, C):

    Q = generator(
        beta,
        gamma,
        N,
        b,
        f
    )

    I = np.identity(N + 1)

    A = alpha * I - Q
    B = cost(P, C, f, N)

    x = np.linalg.solve(A, B)

    return x


#################################################################################
# Proposal discounted cost at state i


def d(i, a, f, beta, b, N, gamma, P, C, J_values):

    forw, back, same = rates(
        i,
        beta,
        gamma,
        N,
        b,
        a
    )

    c = i * P + C * (1 - a)

    #State i = 0
    if i == 0:

        D = (
            c
            + J_values[i + 1] * forw
            + J_values[i] * same
        )

        return D

    # State i = N
    if i == N:

        D = (
            c
            + J_values[i - 1] * back
            + J_values[i] * same
        )

        return D

    # General state
    D = (
        c
        + J_values[i + 1] * forw
        + J_values[i - 1] * back
        + J_values[i] * same
    )

    return D


#################################################################################
# Initial policy


f = np.zeros(N + 1)


# Action at state N is fixed as zero
f[N] = 0


#################################################################################
# Initial discounted cost


J_f = J(
    alpha,
    f,
    beta,
    b,
    N,
    gamma,
    P,
    C
)


#################################################################################
#Store policies and discounted costs


F = [f.copy()]
K = [J_f.copy()]


#################################################################################
#Policy iteration

while True:


    f_new = f.copy()

    # Propose the alternative action for states 0,...,N-1
    for i in range(N):

        if f_new[i] == 0:
            f_new[i] = 1

        else:
            f_new[i] = 0

    # Action at state N is fixed as zero
    f_new[N] = 0

    #Check whether each proposed action is an improvement
    for i in range(N):

        D = d(
            i,
            f_new[i],
            f,
            beta,
            b,
            N,
            gamma,
            P,
            C,
            J_f
        )

        current_value = J_f[i]

        if D >= alpha * current_value:
            f_new[i] = f[i]

    # Final action remains fixed as zero
    f_new[N] = 0

    #Stopping condition
    if np.array_equal(f_new, f):
        break

    #Update policy
    f = f_new.copy()

    #Recalculate discounted cost
    J_f = J(
        alpha,
        f,
        beta,
        b,
        N,
        gamma,
        P,
        C
    )

    #Append complete policy and discounted cost
    F.append(f.copy())
    K.append(J_f.copy())


#################################################################################
#Results

print()
print("Optimal policy:")
print(f)

print()
print("Optimal discounted cost:")
print(J_f)

#################################################################################
#Constructing polcies and discounted costs for comparsion.

#Threshold tested, difference between polciy and optimal, and states.
space = np.arange(0,  N+1 , 1)
diff = np.zeros((len(space), N+1))
states = np.arange(0, N+1, 1)

#Loop over all polcies to be tested.
for k in range(len(space)):

  #Threshold for policy.
  m = space[k]

  #Proposed policy.
  F_pol = np.zeros(N+1)
  F_pol[m:] = 1
  F_pol[N] = 0

  J_pol = J(
      alpha,
      F_pol,
      beta,
      b,
      N,
      gamma,
      P,
      C
  )
  diff[k,:] = J_pol - J_f


#################################################################################
#Plotting
plt.contourf(states, space, diff, levels=250, cmap = 'cividis')
plt.colorbar(label="Excess cost over the optimal policy")
plt.xlabel("State, i")
plt.ylabel(r"Threshold $m$: $a = 1$ for states after $m \leq N-1$")
plt.show()
