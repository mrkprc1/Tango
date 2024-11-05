import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.linalg import block_diag
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap

# Load the Font Awesome font for the Moon and Sun symbols.
prop = FontProperties(fname='Font Awesome 6 Free-Regular-400.otf')
plt.rcParams['font.family'] = prop.get_family()

def plot_tango(tango_array, tango_array_given, cross_constraints=None, equal_constraints=None):
    """
    Plots the Tango game board with given and solved cells, and constraints.

    Parameters:
    tango_array (numpy.ndarray): The solved Tango game board.
    tango_array_given (numpy.ndarray): The Tango game board with given cells.
    cross_constraints (list): List of tuples with the indices of the cells that are constrained to be different.
    equal_constraints (list): List of tuples with the indices of the cells that are constrained to be equal.

    Returns:
    None
    """

    # Create an empty array with the same shape as the Tango array.
    N = tango_array.shape[1]
    fig, ax = plt.subplots(figsize=(5,5))
    blank_array = np.empty(tango_array.shape)
    blank_array[:] = np.nan

    # Set the cells that are given in the Tango array to zero, this changes the color of the cell.
    blank_array[np.where(~np.isnan(tango_array_given))] = 0
    cmap = ListedColormap(['lightgrey'])
    ax.matshow(blank_array, cmap=cmap)

    # Define the symbols for the Moon and Sun.
    symbol_mapping = {
    0: '\uf186', # Moon
    1: '\uf185'} # Sun

    # Draw gridlines.
    for i in range(N+1):
        ax.axhline(y=i-0.5, color='black', linestyle='-', linewidth=2)
        ax.axvline(x=i-0.5, color='black', linestyle='-', linewidth=2)

    # Add 'X' and '=' constraints.
    if cross_constraints is None:
        cross_constraints = []
    for con in cross_constraints:
        if con[1] == con[0]+1:
            y = con[0] // N
            x = con[0] % N + 0.5
        else:
            y = con[0] // N + 0.5
            x = con[0] % N
        ax.scatter(x, y, marker='x',color='dimgrey', s=100)
    if equal_constraints is None:
        equal_constraints = []
    for con in equal_constraints:
        if con[1] == con[0]+1:
            y = (con[0] // N)
            x = (con[0] % N) + 0.5
            ax.annotate('=', [x, y], ha='center', va='center', color='dimgrey', size=20)
        else:
            y = (con[0] // N) + 0.45
            x = (con[0] % N)
            ax.annotate('=', [x, y], ha='center', va='center', color='dimgrey', size=20, rotation=90)

    # Add the Moon and Sun symbols to the cells.
    for i, txt in np.ndenumerate(tango_array):
        if np.isnan(txt):
            continue
        elif txt == 0:
            ax.annotate(symbol_mapping[0], [i[1], i[0]], color='tab:blue', ha='center', va='center', size=30, fontproperties=prop)
        elif txt == 1:
            ax.annotate(symbol_mapping[1], [i[1], i[0]], color='tab:orange', ha='center', va='center', size=30, fontproperties=prop)    
    plt.axis('off')
    plt.show()

# Define the Tango game.
N = 6
num_bits = N**2
tango_array_given = np.array([
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [np.nan,      0, np.nan, np.nan, np.nan, np.nan],
    [np.nan,      0, np.nan, np.nan, np.nan, np.nan],
    [np.nan, np.nan, np.nan, np.nan,      1, np.nan],
    [np.nan, np.nan, np.nan, np.nan,      1, np.nan],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
cross_constraints = [
    [(2,4), (2,5)],
    [(3,1), (4,1)],
    [(4,2), (4,3)],
]
equal_constraints = [
    [(3,4), (3,5)],
    [(3,6), (4,6)],
    [(5,2), (5,3)]
]
idx_cross = [((con[0][0]-1)*N+con[0][1]-1, (con[1][0]-1)*N+con[1][1]-1) for con in cross_constraints]
idx_equal = [((con[0][0]-1)*N+con[0][1]-1, (con[1][0]-1)*N+con[1][1]-1) for con in equal_constraints]

# Plot the Tango game.
plot_tango(
    tango_array=tango_array_given, 
    tango_array_given=tango_array_given, 
    cross_constraints=idx_cross, 
    equal_constraints=idx_equal)

# Define constraints that define the Tango game.
id_N = np.identity(N)
T = np.array([np.pad(np.ones(3, dtype=int), (i, N-3-i)) for i in range(N-2)])
A_row_col = np.concatenate([
    np.kron(id_N, np.ones((1, N))),
    np.kron(np.ones((1, N)), id_N)
])
A_triples = np.concatenate([
    np.kron(id_N, T),
    np.kron(T, id_N)
])
A_cross = np.array([np.identity(num_bits, dtype=int)[con[0], :] + np.identity(num_bits, dtype=int)[con[1], :] for con in idx_cross])
b_cross = np.ones(A_cross.shape[0], dtype=int)
A_equal = np.array([np.identity(num_bits, dtype=int)[con[0], :] - np.identity(num_bits, dtype=int)[con[1], :] for con in idx_equal])
b_equal = np.zeros(A_equal.shape[0], dtype=int)
A_cross_equal = np.concatenate([A_cross, A_equal])
b_cross_equal = np.concatenate([b_cross, b_equal])

# Set the objective as zero, since we are only interested in finding a feasible solution.
c = np.zeros(num_bits, dtype=np.int64)

# Set the bounds as lb=0, ub=1 for all variables to ensure they are binary.
lb = np.zeros(num_bits, dtype=np.int64)
ub = np.ones(num_bits, dtype=np.int64)

# Update the bounds for the variables that are already set, i.e. are given cells.
tango_vec = tango_array_given.ravel()
idx_init = np.where(1-np.isnan(tango_array_given).ravel())[0]
for i in idx_init:
    lb[i] = tango_vec[i]
    ub[i] = tango_vec[i]

# Define the linear constraints and bounds objects.
lin_con_row_col = LinearConstraint(A_row_col, (N/2)*np.ones(A_row_col.shape[0], dtype=np.int64), (N/2)*np.ones(A_row_col.shape[0], dtype=np.int64))
lin_con_triples = LinearConstraint(A_triples, np.ones(A_triples.shape[0], dtype=np.int64), 2*np.ones(A_triples.shape[0], dtype=np.int64))
lin_con_cross_equal = LinearConstraint(A_cross_equal, b_cross_equal, b_cross_equal)
constraints = [lin_con_row_col, lin_con_triples, lin_con_cross_equal]
bounds = Bounds(lb, ub)

# Solve the MILP.
res = milp(c, integrality=np.ones(num_bits), constraints=constraints, bounds=bounds)

# Plot the result.
plot_tango(
    tango_array=res['x'].reshape((N, N)), 
    tango_array_given=tango_array_given, 
    cross_constraints=idx_cross, 
    equal_constraints=idx_equal)