import pulp
import numpy as np
import pandas as pd

import streamlit as st


# Target values
ytarget = np.array([2482, 2466, 2448, 2436, 2374, 2356, 2338, 2310])  # N_indep_Freqx1 vector
# ytarget = np.array([2482, 2466, 2448, 2437, 2374, 2356, 2335, 2310])  # N_indep_Freqx1 vector KB233

# ytarget = np.array([2482, 2472, 2444, 2375, 2358, 2336, 2310])  # N_indep_Freqx1 vector %KB161

st.write("Target Frequency")
st.write(ytarget)

# Initial value
# x0 = 2482
x0 = ytarget[0]

# Fixed first modifier value

# Separate ytarget into two groups: 24xx and 23xx
group1 = [y for y in ytarget if y >= 2400]
group2 = [y for y in ytarget if y < 2400]

# Ensure both groups are non-empty
if not group1 or not group2:
    raise ValueError("ytarget must contain both 24xx and 23xx values.")

# Calculate M1 as the difference between the largest of each group
M1 = max(group2) - max(group1)





def print_multiple_separator_lines(char, length, lines):
    for _ in range(lines):
        print(char * length)



# Define the problem as a minimization (we'll minimize number of modifiers used)
prob = pulp.LpProblem("Minimize_Modifier_Usage", pulp.LpMinimize)

M_lowBound = min(ytarget)-max(ytarget)
M_lowBound = -110
M_upBound = -1

N_indep_Freq = len(ytarget)
N_MF = 6

# Variables
A = [[pulp.LpVariable(f"A_{i}_{j}", cat="Binary") for j in range(N_MF)] for i in range(N_indep_Freq)]
M = [pulp.LpVariable(f"M_{j}", lowBound=M_lowBound, upBound=M_upBound, cat="Integer") for j in range(N_MF)]
AM = [[pulp.LpVariable(f"AM_{i}_{j}", lowBound=M_lowBound, upBound=0, cat="Integer") for j in range(N_MF)] for i in range(N_indep_Freq)]

# Fix M[0]
prob += (M[0] == M1)

# Constraints to match each ytarget[i] exactly
for i in range(N_indep_Freq):
    prob += (x0 + sum(AM[i][j] for j in range(N_MF)) == ytarget[i])

# Force first row of A to be zero
for j in range(N_MF):
    prob += (A[0][j] == 0)

# Big-M constraints to enforce AM[i][j] = A[i][j] * M[j]
M_min, M_max = M_lowBound, M_upBound
for i in range(N_indep_Freq):
    for j in range(N_MF):
        prob += AM[i][j] >= M_min * A[i][j]
        prob += AM[i][j] <= M_max * A[i][j]
        prob += AM[i][j] <= M[j] + (1 - A[i][j]) * (-M_min)
        prob += AM[i][j] >= M[j] - (1 - A[i][j]) * (-M_min)

# Optional: allow at most one modifier active per row (comment out if not needed)
# for i in range(N_indep_Freq):
#     prob += pulp.lpSum(A[i][j] for j in range(N_MF)) <= 1

# Objective: minimize number of modifiers used
prob += pulp.lpSum(A[i][j] for i in range(N_indep_Freq) for j in range(N_MF))

# Solve the problem
prob.solve()

# Output results
print("Solver status:", pulp.LpStatus[prob.status])


# Print A matrix
A_values = [[int(A[i][j].varValue) for j in range(N_MF)] for i in range(N_indep_Freq)]
print("Computed A matrix:\n", np.array(A_values))

# Print M values
M_values = [int(M[j].varValue) for j in range(N_MF)]
print("Computed M vector:", M_values)

# Optional: Reconstructed y values
yfinal = [x0 + sum(A_values[i][j] * M_values[j] for j in range(N_MF)) for i in range(N_indep_Freq)]
print("Reconstructed yfinal:", yfinal)
print("Target ytarget:     ", ytarget.tolist())



# Print error vector
errors = [int(ytarget[i] - yfinal[i]) for i in range(N_indep_Freq)]
print("Errors (ytarget - yfinal):", errors)




# Convert to NumPy arrays
A_matrix = np.array(A_values)
M_vector = np.array(M_values)




# Multiply A by M
# A_dot_M_Matrix = np.dot(A_matrix, M_matrix)



# Element-wise multiplication of each row in A with M
A_dot_M_Matrix = A_matrix * M_vector



# Reshape ytarget to be a column vector (N_indep_Freqx1)
ytarget = ytarget.reshape(-1, 1)

# Concatenate the vector and matrix A
table = np.hstack((ytarget, A_dot_M_Matrix))

# Create a DataFrame with headers
headers = ['Target'] + [f'MF {i+1}' for i in range(A_matrix.shape[1])]
df = pd.DataFrame(table, columns=headers)

print_multiple_separator_lines('#', 60, 3)
# Print the DataFrame
print(df)
print_multiple_separator_lines('#', 60, 3)



st.write("Optimization complete")

