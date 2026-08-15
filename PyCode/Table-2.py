##    Importing Relavent Packages   ## 
import numpy as np 
import matplotlib.pyplot as plt 
  
################################################################################# 
 
parameters = [ 
    # beta, gamma, N, alpha, P, C 
    [5.0, 4.3, 100, 0.30, 700,  100], 
    [5.0, 1.1, 100, 0.20, 700,  200], 
    [3.0, 1.3, 100, 0.10, 500,  300], 
    [5.0, 4.3, 100, 0.40, 200,  400], 
    [3.0, 2.2, 100, 0.45, 100,  500], 
    [5.0, 0.8, 100, 0.20, 350,  600], 
    [2.0, 1.1, 100, 0.10,  20,  700], 
    [5.0, 3.3, 100, 0.40, 100,  800], 
    [6.0, 5.3, 100, 0.45, 200,  900], 
    [3.0, 2.2, 100, 0.45, 500, 1000], 
] 
################################################################################# 
 
#Function for calculating rates 
 
def rates(i, beta, gamma, N, a): 
 
    #State i = 0 
    if i == 0: 
        forw = a 
        back = 0 
        same = -a 
        return forw, back, same 
 
    # State i = N 
    if i == N: 
        forw = 0 
        back = gamma * i 
        same = -back 
        return forw, back, same 
 
    # General state 
    forw = beta * i * (N - i) / N + a 
    back = gamma * i 
    same = -forw - back 
 
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
 
 
def d(i, a, f, beta, N, gamma, P, C, J_values): 
 
    forw, back, same = rates( 
        i, 
        beta, 
        gamma, 
        N, 
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
 
 
 
for beta, gamma, N, alpha, P, C in parameters: 
 
##################################################################################### 
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
        N, 
        gamma, 
        P, 
        C 
    ) 
 
################################################################################# 
    # Store policies and discounted costs 
 
    F = [f.copy()] 
    K = [J_f.copy()] 
 
################################################################################# 
    # Policy iteration 
 
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
 
        # Check whether each proposed action is an improvement 
        for i in range(N): 
 
            D = d( 
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
 
        # Final action remains fixed as zero 
        f_new[N] = 0 
 
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
 
        # Append complete policy and discounted cost 
        F.append(f.copy()) 
        K.append(J_f.copy()) 
 
    ################################################################################# 
    # Results for this parameter set 
 
    intervention_states = np.where(f[:N] == 1)[0] 
 
    if len(intervention_states) == 0: 
        numerical_m = N 
    else: 
        numerical_m = intervention_states[0] 
 
    approximate_m = min( 
        (np.ceil( 
            1 / (np.exp(C * beta / (P * N)) - 1) 
        )), 
        N - 1 
    ) 
 
 
    print("Beta =", beta) 
    print("Gamma =", gamma) 
    print("N =", N) 
    print("Alpha =", alpha) 
    print("P =", P) 
    print("C =", C) 
    print("Approximate m =", approximate_m) 
    print("Numerical m =", numerical_m) 
    print("Final policy =", f) 
    print()
