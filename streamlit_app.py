import pulp
import numpy as np
import pandas as pd

import streamlit as st



st.set_page_config(layout="wide")

# st.image("Hussein Hussein_Headshot_1.png", width=200,  caption="Sunrise by the mountains")


st.link_button("@ Hussein Hussein", "https://www.linkedin.com/in/husseinmehussein/")
# Target values
# ytarget = np.array([2482, 2466, 2448, 2436, 2374, 2356, 2338, 2310])  # N_indep_Freqx1 vector
# ytarget = np.array([2482, 2466, 2448, 2437, 2374, 2356, 2335, 2310])  # N_indep_Freqx1 vector KB233

# ytarget = np.array([2482, 2472, 2444, 2375, 2358, 2336, 2310])  # N_indep_Freqx1 vector %KB161


st.title("MF Table Generator  ")
st.write("version 1.0.0")
st.write("© 2025 Hussein Hussein @ Skyworks Solutions")


text = "Please note that NPI approval is still needed!"
highlight_color =  "#ffff00"

highlighted_text = f"<span style='background-color:{highlight_color};'>{text}</span>"
st.markdown(highlighted_text, unsafe_allow_html=True)


suggest_MF_upper_limit = 0


col1top, col2top = st.columns([0.6,0.4])

with col1top:
    st.write("## Enter Frequency Info:")

    col1, col2, col3, spacer1 = st.columns([0.1,0.1,0.1, 0.10])
    N_Freq = col1.number_input(
        label=" No. Frequencies",
        min_value=1 ,
        value=8,
        help="e.g. F1, F2, ...etc"
    )

    N_TF = col2.number_input(
        label="No. TF groups ",
        min_value=1 ,
        value=6,
        help="e.g. TF1, TF2, ...etc"
    )

    N_MF = N_TF - 1

    max_shunt_fs = col3.number_input(
        label="Max shunt fs",
        min_value=1 ,
        value=2374,
        help="i.e. the largest fs of all the shunt resonators"
    )


with col2top:
    st.write("### Advanced settings:")

    advcol1, spacer1, advcol2, spacer2 = st.columns([0.4,0.01,0.4,0.01])

    with advcol1:
        MF1_uppler_limit = advcol1.number_input(
            label="MF upper limit",
            max_value=-1 ,
            value=-110,
            help="i.e. the absolute max value of any given MF"
        )
        
    with advcol2:
        advcol2.write("All shunts share MF1?")

        toggle_MF1_shared = st.toggle("",
                                    value=True, 
                                    help="Turn this on to force all the shunt resonators to have the same MF1")


default_TF_List = [2482, 2466, 2448, 2436, 2374, 2356, 2338, 2310, 0, 0 ]


TF_values = []
labels = []



# Display side by side
Main_col1, spacer1, Main_col2, spacer2, Main_col3 = st.columns([0.2,0.1, 0.1,0.1,0.4])

with Main_col1:
    st.write("## Enter Frequency Values")

    for i in range(1, N_Freq + 1):
        label_col, input_col = st.columns([0.2, 0.2],
                                            vertical_alignment="center",
                                                border=True) # Adjust ratio as needed
        label_col.markdown(f"**F{i} Value:**")

        temp_value = input_col.number_input(
            label="",
            min_value=0 ,
            value=default_TF_List[i-1],
            # key=f"TF_input_{i}",# Unique key for each input
            label_visibility="collapsed" # hides the label space
        ) # to define many TF variables without repeating code lines

        TF_values.append(temp_value)
        labels.append(f"TF{i}")


ytarget = np.array(TF_values)
df = pd.DataFrame({" Target Frequency": labels, "Value (MHz)": ytarget})
df.index = range(1, len(df) + 1) # adds "1" to the table index to make it start from 1 instead of 0

with Main_col2:
    st.write("### Entered Target Frequency ")
    st.table(df)

################################################################
# Initial value
# x0 = 2482
x0 = ytarget[0]

# Fixed first modifier value

# max_shunt_fs = 2400
# max_shunt_fs = 2374

# Separate ytarget into two groups: 24xx and 23xx
group_series_reonsators = [y for y in ytarget if y > max_shunt_fs]
group_shunt_reonsators = [y for y in ytarget if y <= max_shunt_fs]

# Ensure both groups are non-empty
if not group_series_reonsators or not group_shunt_reonsators:
    raise ValueError("ytarget must contain both series and shunt values.")

# Calculate M1 as the difference between the largest of each group
M1 = max(group_shunt_reonsators) - max(group_series_reonsators)

suggest_MF_upper_limit = M1

with advcol1:
    st.markdown(f"""
    **Suggested MF upper limit = {suggest_MF_upper_limit}**
    <span title="This value is calculated as: (largest shunt fs - largest series fs)" style="cursor: help;"> ℹ️</span>
    """, unsafe_allow_html=True)


def print_multiple_separator_lines(char, length, lines):
    for _ in range(lines):
        print(char * length)



# Define the problem as a minimization (we'll minimize number of modifiers used)
prob = pulp.LpProblem("Minimize_Modifier_Usage", pulp.LpMinimize)

M_lowBound = max_shunt_fs-max(ytarget)
M_lowBound = MF1_uppler_limit
M_upBound = -1

N_indep_Freq = len(ytarget)


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


### Optional Feature:


if toggle_MF1_shared:
    # Force first column of A for the shunt resonator target to be 1
    for i in range(N_indep_Freq):
        if ytarget[i] <= max(group_shunt_reonsators):
            prob += (A[i][0] == 1)



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

yfinal = np.array(yfinal)

# Multiply A by M
# A_dot_M_Matrix = np.dot(A_matrix, M_matrix)



# Element-wise multiplication of each row in A with M
A_dot_M_Matrix = A_matrix * M_vector



# Reshape ytarget to be a column vector (N_indep_Freqx1)
ytarget = ytarget.reshape(-1, 1)
yfinal = yfinal.reshape(-1, 1)

# Concatenate the vector and matrix A
table = np.hstack((ytarget, yfinal, A_dot_M_Matrix))

# Create a DataFrame with headers
headers = ['Target'] + ['Final'] + [f'MF {i+1}' for i in range(A_matrix.shape[1])]
df = pd.DataFrame(table, columns=headers)
df.index = range(1, len(df) + 1) # adds "1" to the table index to make it start from 1 instead of 0

# Style the header: bold + navy blue
styled_df = df.style.set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold'), ('color', 'navy')]}
])



print_multiple_separator_lines('#', 60, 3)
# Print the DataFrame
print(df)
print_multiple_separator_lines('#', 60, 3)


with Main_col3:
    if all(e == 0 for e in errors):
        st.success('feasible solution found!', icon="✅")
    else:
        st.error('**No feasible solution found!**', icon="🚨")

    st.write("# Generated MF Table")
    st.table(styled_df)




st.image("skyworks-solutions-inc-vector-logo.svg", width=200,  caption="")



