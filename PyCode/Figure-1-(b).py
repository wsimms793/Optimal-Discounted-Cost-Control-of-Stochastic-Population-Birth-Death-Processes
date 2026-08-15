##    Importing Relavent Packages   ##
import numpy as np
import matplotlib.pyplot as plt

#################################################################################

#Parameters.

#Define [Birth rate, Death rate,
#Discount factor, Cost constant for remaining in a state,
#Cost constant for remaining in a state, Number of iterations used for optimization,
#Number of considered states, Number of plotted states]


lam, mu, alpha, P, C, n, s, s_plot = [
    0.4,
    1.2,
    0.8,
    100,
    200,
    5000,
    250,
    100,
]

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

#Main Loop for finding optimal discounted cost and assosiated polciy.

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



#################################################################################

#Function for computing cost under a arbitiary policy.
def cost_policy(D, P, C, alpha, s, n, lam, mu):

    # Initialize cost matrix for the policy.
    V = np.zeros((s, n))

    # Iterate through time steps.
    for j in range(1, n):

        # Iterate through all states.
        for i in range(s):

            # Get action from the policy vector.
            a = D[i]
            # Compute corresponding transition rates.
            forw, back, diag = rates(i, a, lam, mu)

            # Lower boundary condition for policy cost.
            if i == 0:
                V[i, j] = (P * i + C * (1 - a)) / (alpha - diag)
                V[i, j] += V[i + 1, j - 1] * forw / (alpha - diag)

            # Upper boundary condition for policy cost.
            elif i == s - 1:
                V_forw = V[i, j - 1] + (P / alpha)

                V[i, j] = (P * i + C * (1 - a)) / (alpha - diag)
                V[i, j] += (
                    V[i - 1, j - 1] * back + V_forw * forw
                ) / (alpha - diag)

            # Intermediate states for policy cost.
            else:
                V[i, j] = (P * i + C * (1 - a)) / (alpha - diag)
                V[i, j] += (
                    V[i - 1, j - 1] * back + V[i + 1, j - 1] * forw
                ) / (alpha - diag)

    # Return the converged cost vector.
    return V[:, n - 1]



#################################################################################

#Constructing and filling arrays and matrices for plotting.

# Slice optimal results for plotting.
V_opt_plot = V_opt[:s_plot]
D_opt_plot = D_opt[:s_plot]

# Threshold values to be tested.
spacing = np.arange(1, s_plot, 10)

# Difference between policy cost and optimal cost.
diff = np.zeros(
    (len(spacing), s_plot)
)

# Loop over all threshold policies to be evaluated.
for k in range(len(spacing)):

    # Threshold for the policy we are testing.
    m = spacing[k]

    # Policy:
    # allow immigration for i < m
    # prevent immigration for i >= m
    D_pol = np.ones(s)
    D_pol[m:] = 0

    # Cost corresponding to this policy.
    V_pol = cost_policy(D_pol, P, C, alpha, s, n, lam, mu)
    V_pol_plot = V_pol[:s_plot]

    # Calculate difference from the optimal cost.
    diff[k, :] = V_pol_plot - V_opt_plot


#################################################################################

# Contour plot of excess costs across thresholds and states.
plt.contourf(states, spacing, diff, levels=50, cmap= "inferno")
plt.colorbar(label="Excess cost over the optimal policy")
plt.xlabel("State, i")
plt.ylabel(r"Threshold $m$: $a= 0$ from state $m$ onward")
plt.show()

# Print optimal decisions for the plotted states.
print(D_opt_plot)

##################################################################################

#Plots of Anayltic Vs Numeric.

# Analytic solution when f_0(i) = 0 is optimal.
J_f1_plot = (
    P * (states + 1 / (lam - mu)) / (alpha + mu - lam)
    - (P * 1) / (alpha * (lam - mu)))

plt.plot(
    states,
    V_opt_plot,
    color="red",
    linestyle="--",
    linewidth=2.5,
    label="Numerical $V^*$",
)

plt.plot(
    states,
    J_f1_plot,
    color="blue",
    linestyle="-",
    label=r"$J_\alpha(i,f_0)$",
)


plt.xlabel("State, $i$")
plt.ylabel("Discounted cost")
plt.legend()
plt.show()

#################################################################################

# Absolute difference between numerical and analytic values.
analytic_diff = np.abs(V_opt_plot - J_f1_plot)

# Maximum absolute difference.
max_diff = np.max(analytic_diff)
print("Maximum absolute difference:", max_diff)
