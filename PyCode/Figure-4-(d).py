##    Importing Relavent Packages   ##
import numpy as np
import matplotlib.pyplot as plt

#################################################################################

#Parameters

beta, gamma, N, alpha, b, P, C = [
    20, 4, 100, 5.0, 0.3, 6, 80
]


#################################################################################
#Function for calculating rates


def rates(i, beta, gamma, N, b,d):

    #State i = 0
    if i == 0:
        forw = 0
        back = 0
        same = 0
        return forw, back, same
    # State i = N
    if i == N:
        forw = 0
        back = (b + gamma) * i
        same = -back - d
        return forw, back, same

    # General state
    forw = beta * i * (N - i) / N
    back = (b + gamma) * i
    same = -forw - back - d
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


def proposal_cost(
    i,
    a,
    f,
    beta,
    b,
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
        b,
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
# Initial policy

f = np.zeros(N + 1)

#################################################################################
# Initial discounted cost

J_f = J(alpha, f, beta, b, N, gamma, P, C)


#################################################################################
#Policy iteration

while True:


    f_new = f.copy()

    # Propose the alternative action for states 0,...,N-1
    for i in range(N+1):

        if f_new[i] == 0:
            f_new[i] = 1

        else:
            f_new[i] = 0


    #Check whether each proposed action is an improvement
    for i in range(N+1):

        D = proposal_cost(
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
plt.contourf(states, space, diff, levels=250, cmap = 'plasma')
plt.colorbar(label="Excess cost over the optimal policy")
plt.xlabel("State, i")
plt.ylabel("Threshold m: d = 1 from state m onward")
plt.show()
