import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from IPython.display import display, Math
from scipy.optimize import root

'''
# Define the system of equations (Case 1)
def system_of_equations1(vars, random_numbers):
    x, y, z = vars
    f1 = random_numbers[0]*(x**2)*(z**2) - random_numbers[1]*(x*y) - random_numbers[2]*(y**2) - x*y*(z**2)
    f2 = (x**2)*z - x*(z**2) - random_numbers[3]*z + random_numbers[4]*x
    f3 = (y**2)*z + y*(z**2) - random_numbers[5]*y - random_numbers[6]*z
    return [f1, f2, f3]

# Define the system of equations (Case 2)
def system_of_equations2(vars, random_numbers):
    x, y, z = vars
    f1 = (x-y)*(y+x*z**2)
    f2 = (x-z)*(x*z + 1)
    f3 = (y+z)*(y*z - 1)
    return [f1, f2, f3]

# Define the system of equations (Case 3)
def system_of_equations3(vars, random_numbers):
    x, y, z = vars
    f1 = x*z + y**2 - (x**2)*(z**2) - x*y*(z**2)
    f2 = (x-z)*(x*z + 1)
    f3 = (y+z)*(y*z - 1)
    return [f1, f2, f3]



# Generate random coefficients
np.random.seed(42)  # For reproducibility
random_numbers = np.random.randint(-10, 10, size=7)

# Create a denser grid of initial guesses
x_vals = np.linspace(-5, 5, 50)
y_vals = np.linspace(-5, 5, 50)
z_vals = np.linspace(-5, 5, 50)
initial_guesses = np.array(np.meshgrid(x_vals, y_vals, z_vals)).T.reshape(-1, 3)

# Solve the system for each initial guess
solutions = []
for guess in initial_guesses:
    sol = root(system_of_equations, guess, args=(random_numbers,), method='hybr')
    if sol.success:
        solutions.append(sol.x)

# Remove duplicates with a tolerance
unique_solutions = []
tolerance = 1e-6
for sol in solutions:
    if not any(np.allclose(sol, usol, atol=tolerance) for usol in unique_solutions):
        unique_solutions.append(sol)

# Convert to a NumPy array
unique_solutions = np.array(unique_solutions)

# Classify solutions
real_solutions = unique_solutions[np.isclose(unique_solutions.imag, 0, atol=tolerance).all(axis=1)]
complex_solutions = unique_solutions[~np.isclose(unique_solutions.imag, 0, atol=tolerance).all(axis=1)]

# Print counts
print(f"Total unique solutions: {len(unique_solutions)}")
print(f"Real solutions: {len(real_solutions)}")
print(f"Complex solutions: {len(complex_solutions)}")

# Visualize the real solutions
fig = go.Figure()

# Plot real parts
fig.add_trace(go.Scatter3d(
    x=real_solutions[:, 0].real,
    y=real_solutions[:, 1].real,
    z=real_solutions[:, 2].real,
    mode='markers',
    marker=dict(size=3, color='blue'),
    name='Real Solutions'
))

# Update layout
fig.update_layout(
    scene=dict(
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        zaxis_title='Z-axis'
    ),
    title="Visualization of Real Solutions"
)

# Show the plot
fig.show()


# Indicate the number of complex solutions
if len(complex_solutions) > 0:
    print(f"Complex solutions detected! Total: {len(complex_solutions)}")
else:
    print("No complex solutions found.")
'''


random.seed(42)
# Generate 7 random integers within a specified range
def generate_random_integers(start, end, count=7):
    return [random.randint(start, end) for _ in range(count)]

random_numbers = generate_random_integers(-10, 10) 

# Generic case 1: (a_i random)
def poly1(x, y, z):
    return random_numbers[0]*(x**2)*(z**2) - random_numbers[1]*(x*y) - random_numbers[2]*(y**2) - x*y*(z**2)  

def poly2(x, y, z):
    return (x**2)*z - x*(z**2) - random_numbers[3]*z + random_numbers[4]*x

def poly3(x, y, z):
    return (y**2)*z + y*(z**2) - random_numbers[5]*y - random_numbers[6]*z


# Non-Generic case 2: Reducibles (a1=a3=1; a2=-1; a4=a5=1; a6=a7=1)
def p1(x, y, z):
    return (x-y)*(y+x*z**2)
def irred_p1_1(x, y, z):
    return (x-y)
def irred_p1_2(x, y, z):
    return (y+x*z**2)


def p2(x, y, z):
    return (x-z)*(x*z + 1)
def irred_p2_1(x, y, z):
    return (x-z)
def irred_p2_2(x, y, z):
    return (x*z + 1)


def p3(x, y, z):
    return (y+z)*(y*z - 1)
def irred_p3_1(x, y, z):
    return (y+z)
def irred_p3_2(x, y, z):
    return (y*z - 1)

# Non-generic case 3: Almost all areducibles (a1=!1, a2=-a3=-1)
def p1_irred(x, y, z):
    return x*z + y**2 - (x**2)*(z**2) - x*y*(z**2)


# Set up the range and resolution for x, y, and z
x_vals = np.linspace(-5, 5, 350)
y_vals = np.linspace(-5, 5, 350)
z_vals = np.linspace(-5, 5, 350)


x_vals2 = np.linspace(-5, 5, 100)
y_vals2 = np.linspace(-5, 5, 100)
z_vals2 = np.linspace(-5, 5, 100)

# Create a 3D meshgrid
X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)
X2, Y2, Z2 = np.meshgrid(x_vals2, y_vals2, z_vals2)


# Evaluate each polynomial on the grid
## CASE 1
poly1_vals = poly1(X, Y, Z)
poly2_vals = poly2(X, Y, Z)
poly3_vals = poly3(X, Y, Z)

poly1_vals2 = poly1(X2, Y2, Z2)
poly2_vals2 = poly2(X2, Y2, Z2)
poly3_vals2 = poly3(X2, Y2, Z2)


## CASE 2
p1_vals = p1(X, Y, Z)
p1_vals2 = p1(X2, Y2, Z2)
p1_1_vals = irred_p1_1(X, Y, Z)
p1_2_vals = irred_p1_2(X, Y, Z)

p2_vals = p2(X, Y, Z)
p2_vals2 = p2(X2, Y2, Z2)
p2_1_vals = irred_p2_1(X, Y, Z)
p2_2_vals = irred_p2_2(X, Y, Z)

p3_vals = p3(X, Y, Z)
p3_vals2 = p3(X2, Y2, Z2)
p3_1_vals = irred_p3_1(X, Y, Z)
p3_2_vals = irred_p3_2(X, Y, Z)

## CASE 3
p1_irred_vals = p1_irred(X, Y, Z)
p1_irred_vals2 = p1_irred(X2, Y2, Z2)

## INTERSECTIONS (CASE 1)
# Find intersections of p1 and p2
tolerance = 0.1
intersection_mask_12 = (np.abs(poly1_vals) < tolerance) & (np.abs(poly2_vals) < tolerance)
intersection_points_12 = np.vstack((X[intersection_mask_12], Y[intersection_mask_12], Z[intersection_mask_12])).T

# Find intersections of p1 and p3
intersection_mask_13 = (np.abs(poly1_vals) < tolerance) & (np.abs(poly3_vals) < tolerance)
intersection_points_13 = np.vstack((X[intersection_mask_13], Y[intersection_mask_13], Z[intersection_mask_13])).T

# Find intersections of p3 and p2
intersection_mask_32 = (np.abs(poly3_vals) < tolerance) & (np.abs(poly2_vals) < tolerance)
intersection_points_32 = np.vstack((X[intersection_mask_32], Y[intersection_mask_32], Z[intersection_mask_32])).T

# Find full variety
intersection_mask_full = (np.abs(poly1_vals) < tolerance) & (np.abs(poly2_vals) < tolerance) & (np.abs(poly3_vals) < tolerance)
intersection_pts = np.vstack((X[intersection_mask_full], Y[intersection_mask_full], Z[intersection_mask_full])).T

## INTERSECTIONS (CASE 2)
tolerance2 = 0.1

# Irreducibles of p1 with irreducibles of p3
intersection_mask_1131 = (np.abs(p1_1_vals) < tolerance2) & (np.abs(p3_1_vals) < tolerance2)
intersectionpoints_1131 = np.vstack((X[intersection_mask_1131], Y[intersection_mask_1131], Z[intersection_mask_1131])).T
intersection_mask_1232 = (np.abs(p1_2_vals) < tolerance2) & (np.abs(p3_2_vals) < tolerance2)
intersectionpoints_1232 = np.vstack((X[intersection_mask_1232], Y[intersection_mask_1232], Z[intersection_mask_1232])).T
intersection_mask_1231 = (np.abs(p1_2_vals) < tolerance2) & (np.abs(p3_1_vals) < tolerance2)
intersectionpoints_1231 = np.vstack((X[intersection_mask_1231], Y[intersection_mask_1231], Z[intersection_mask_1231])).T
intersection_mask_1132 = (np.abs(p1_1_vals) < tolerance2) & (np.abs(p3_2_vals) < tolerance2)
intersectionpoints_1132 = np.vstack((X[intersection_mask_1132], Y[intersection_mask_1132], Z[intersection_mask_1132])).T

# Irreducibles of p1 with irreducibles of p2
intersection_mask_1121 = (np.abs(p1_1_vals) < tolerance2) & (np.abs(p2_1_vals) < tolerance2)
intersectionpoints_1121 = np.vstack((X[intersection_mask_1121], Y[intersection_mask_1121], Z[intersection_mask_1121])).T
intersection_mask_1222 = (np.abs(p1_2_vals) < tolerance2) & (np.abs(p2_2_vals) < tolerance2)
intersectionpoints_1222 = np.vstack((X[intersection_mask_1222], Y[intersection_mask_1222], Z[intersection_mask_1222])).T
intersection_mask_1122 = (np.abs(p1_1_vals) < tolerance2) & (np.abs(p2_2_vals) < tolerance2)
intersectionpoints_1122 = np.vstack((X[intersection_mask_1122], Y[intersection_mask_1122], Z[intersection_mask_1122])).T
intersection_mask_1221 = (np.abs(p1_2_vals) < tolerance2) & (np.abs(p2_1_vals) < tolerance2)
intersectionpoints_1221 = np.vstack((X[intersection_mask_1221], Y[intersection_mask_1221], Z[intersection_mask_1221])).T

# Irreducibles of p2 with irreducibles of p3
intersection_mask_2131 = (np.abs(p2_1_vals) < tolerance2) & (np.abs(p3_1_vals) < tolerance2)
intersectionpoints_2131 = np.vstack((X[intersection_mask_2131], Y[intersection_mask_2131], Z[intersection_mask_2131])).T
intersection_mask_2232 = (np.abs(p2_2_vals) < tolerance2) & (np.abs(p3_2_vals) < tolerance2)
intersectionpoints_2232 = np.vstack((X[intersection_mask_2232], Y[intersection_mask_2232], Z[intersection_mask_2232])).T
intersection_mask_2132 = (np.abs(p2_1_vals) < tolerance2) & (np.abs(p3_2_vals) < tolerance2)
intersectionpoints_2132 = np.vstack((X[intersection_mask_2132], Y[intersection_mask_2132], Z[intersection_mask_2132])).T
intersection_mask_2231 = (np.abs(p2_2_vals) < tolerance2) & (np.abs(p3_1_vals) < tolerance2)
intersectionpoints_2231 = np.vstack((X[intersection_mask_2231], Y[intersection_mask_2231], Z[intersection_mask_2231])).T

# Irreducibles triple intersection p1 1
intersection_mask_112131 = (np.abs(p1_1_vals) < tolerance2) & (np.abs(p2_1_vals) < tolerance2) & (np.abs(p3_1_vals) < tolerance2)
intersectionpoints_112131 = np.vstack((X[intersection_mask_112131], Y[intersection_mask_112131], Z[intersection_mask_112131])).T
intersection_mask_112132 = (np.abs(p1_1_vals) < tolerance2) & (np.abs(p2_1_vals) < tolerance2) & (np.abs(p3_2_vals) < tolerance2)
intersectionpoints_112132 = np.vstack((X[intersection_mask_112132], Y[intersection_mask_112132], Z[intersection_mask_112132])).T
intersection_mask_112231 = (np.abs(p1_1_vals) < tolerance2) & (np.abs(p2_2_vals) < tolerance2) & (np.abs(p3_1_vals) < tolerance2)
intersectionpoints_112231 = np.vstack((X[intersection_mask_112231], Y[intersection_mask_112231], Z[intersection_mask_112231])).T
intersection_mask_112232 = (np.abs(p1_1_vals) < tolerance2) & (np.abs(p2_2_vals) < tolerance2) & (np.abs(p3_2_vals) < tolerance2)
intersectionpoints_112232 = np.vstack((X[intersection_mask_112232], Y[intersection_mask_112232], Z[intersection_mask_112232])).T

# Irreducibles triple intersection p1 2
intersection_mask_122131 = (np.abs(p1_2_vals) < tolerance2) & (np.abs(p2_1_vals) < tolerance2) & (np.abs(p3_1_vals) < tolerance2)
intersectionpoints_122131 = np.vstack((X[intersection_mask_122131], Y[intersection_mask_122131], Z[intersection_mask_122131])).T
intersection_mask_122132 = (np.abs(p1_2_vals) < tolerance2) & (np.abs(p2_1_vals) < tolerance2) & (np.abs(p3_2_vals) < tolerance2)
intersectionpoints_122132 = np.vstack((X[intersection_mask_122132], Y[intersection_mask_122132], Z[intersection_mask_122132])).T
intersection_mask_122231 = (np.abs(p1_2_vals) < tolerance2) & (np.abs(p2_2_vals) < tolerance2) & (np.abs(p3_1_vals) < tolerance2)
intersectionpoints_122231 = np.vstack((X[intersection_mask_122231], Y[intersection_mask_122231], Z[intersection_mask_122231])).T
intersection_mask_122232 = (np.abs(p1_2_vals) < tolerance2) & (np.abs(p2_2_vals) < tolerance2) & (np.abs(p3_2_vals) < tolerance2)
intersectionpoints_122232 = np.vstack((X[intersection_mask_122232], Y[intersection_mask_122232], Z[intersection_mask_122232])).T


# Full variety
#im_full = (np.abs(p1_1_vals) < 1) & (np.abs(p1_2_vals) < 1) & (np.abs(p2_1_vals) < 1) & (np.abs(p2_2_vals) < 1) & (np.abs(p3_1_vals) < 1) & (np.abs(p3_2_vals) < 1)
im_full = (np.abs(p1_vals) < tolerance2) & (np.abs(p2_vals) < tolerance2) & (np.abs(p3_vals) < tolerance2)
intersectionpts = np.vstack((X[im_full], Y[im_full], Z[im_full])).T

## INTERSECTIONS (CASE 3)

# Irreducibles triple intersection
intersection_mask_12131 = (np.abs(p1_irred_vals) < tolerance2) & (np.abs(p2_1_vals) < tolerance2) & (np.abs(p3_1_vals) < tolerance2)
intersectionpoints_12131 = np.vstack((X[intersection_mask_12131], Y[intersection_mask_12131], Z[intersection_mask_12131])).T
intersection_mask_12132 = (np.abs(p1_irred_vals) < tolerance2) & (np.abs(p2_1_vals) < tolerance2) & (np.abs(p3_2_vals) < tolerance2)
intersectionpoints_12132 = np.vstack((X[intersection_mask_12132], Y[intersection_mask_12132], Z[intersection_mask_12132])).T
intersection_mask_12231 = (np.abs(p1_irred_vals) < tolerance2) & (np.abs(p2_2_vals) < tolerance2) & (np.abs(p3_1_vals) < tolerance2)
intersectionpoints_12231 = np.vstack((X[intersection_mask_12231], Y[intersection_mask_12231], Z[intersection_mask_12231])).T
intersection_mask_12232 = (np.abs(p1_irred_vals) < tolerance2) & (np.abs(p2_2_vals) < tolerance2) & (np.abs(p3_2_vals) < tolerance2)
intersectionpoints_12232 = np.vstack((X[intersection_mask_12232], Y[intersection_mask_12232], Z[intersection_mask_12232])).T

# Irreducibles pairwise intersections
intersection_mask_131 = (np.abs(p1_irred_vals) < tolerance2) & (np.abs(p3_1_vals) < tolerance2)
intersectionpoints_131 = np.vstack((X[intersection_mask_1131], Y[intersection_mask_1131], Z[intersection_mask_1131])).T
intersection_mask_132 = (np.abs(p1_irred_vals) < tolerance2) & (np.abs(p3_2_vals) < tolerance2)
intersectionpoints_132 = np.vstack((X[intersection_mask_1232], Y[intersection_mask_1232], Z[intersection_mask_1232])).T
intersection_mask_121 = (np.abs(p1_irred_vals) < tolerance2) & (np.abs(p2_1_vals) < tolerance2)
intersectionpoints_121 = np.vstack((X[intersection_mask_1121], Y[intersection_mask_1121], Z[intersection_mask_1121])).T
intersection_mask_122 = (np.abs(p1_irred_vals) < tolerance2) & (np.abs(p2_2_vals) < tolerance2)
intersectionpoints_122 = np.vstack((X[intersection_mask_1222], Y[intersection_mask_1222], Z[intersection_mask_1222])).T


## CASE 1 PLOTS
# Create subplots for each polynomial surface
fig1 = make_subplots(
    rows=1, cols=3,
    specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
    subplot_titles=[f"$({random_numbers[0]})x^2z^2 - ({random_numbers[1]})xy - ({random_numbers[2]})y^2 - xyz^2 = 0$",
    f"$(x^2)z - xz^2 - ({random_numbers[3]})z + ({random_numbers[4]})x = 0$",
    f"$(y^2)z + yz^2 - ({random_numbers[5]})y - ({random_numbers[6]})z = 0$"]
)

# Add poly1 surface to the first subplot
fig1.add_trace(go.Isosurface(
    x=X2.flatten(),
    y=Y2.flatten(),
    z=Z2.flatten(),
    value=poly1_vals2.flatten(),
    isomin=-0.1,
    isomax=0.1,
    surface_count=2,
    colorscale="Blues",
    opacity=0.5,
    name='$f_1$'
), row=1, col=1)

# Add poly2 surface to the second subplot
fig1.add_trace(go.Isosurface(
    x=X2.flatten(),
    y=Y2.flatten(),
    z=Z2.flatten(),
    value=poly2_vals2.flatten(),
    isomin=-0.1,
    isomax=0.1,
    surface_count=2,
    colorscale="Greens",
    opacity=0.5,
    name='$f_2$'
), row=1, col=2)

# Add poly3 surface to the third subplot
fig1.add_trace(go.Isosurface(
    x=X2.flatten(),
    y=Y2.flatten(),
    z=Z2.flatten(),
    value=poly3_vals2.flatten(),
    isomin=-0.1,
    isomax=0.1,
    surface_count=2,
    colorscale="Reds",
    opacity=0.5,
    name='$f_3$'
), row=1, col=3)

# Update layout for clarity
fig1.update_layout(
    scene=dict(
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        zaxis_title='Z-axis',
    ),
    title="Locus of individual polynomials",
)
#fig1.show()

# Create subplots for each intersection of polynomial surfaces
fig2 = make_subplots(
    rows=1, cols=3,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    subplot_titles=["$V(f_1,f_2)$", "$V(f_1,f_3)$", "$V(f_2,f_3)$"]
)

# Add p1 and p2 intersection
fig2.add_trace(go.Scatter3d(
    x=intersection_points_12[:, 0],
    y=intersection_points_12[:, 1],
    z=intersection_points_12[:, 2],
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="$V(f_1,f_2)$"
), row=1, col=1)

# Add p1 and p3 intersection
fig2.add_trace(go.Scatter3d(
    x=intersection_points_13[:, 0],
    y=intersection_points_13[:, 1],
    z=intersection_points_13[:, 2],
    mode='markers',
    marker=dict(size=3, color='blue'),
    name="$V(f_1,f_3)$"
), row=1, col=2)

# Add p2 and p3 intersection
fig2.add_trace(go.Scatter3d(
    x=intersection_points_32[:, 0],
    y=intersection_points_32[:, 1],
    z=intersection_points_32[:, 2],
    mode='markers',
    marker=dict(size=3, color='green'),
    name="$V(f_2,f_3)$"
), row=1, col=3)

# Update layout for clarity
fig2.update_layout(
    title="Intersection Curves of Polynomial Surfaces",
    scene=dict(
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        zaxis_title='Z-axis',
    )
)
#fig2.show()


# Plot the full variety (intersection of the three polynomials)
fig3 = go.Figure()
fig3.add_trace(go.Scatter3d(
    x=intersection_pts[:, 0],
    y=intersection_pts[:, 1],
    z=intersection_pts[:, 2], 
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="$V$"
))

# Update layout for clarity
fig3.update_layout(
    scene=dict(
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        zaxis_title='Z-axis',
    ),
    title="Full Variety"
)
#fig3.show()



## CASE 2 PLOTS

## 1
# Create subplots for each polynomial surface
fig1_v2 = make_subplots(
    rows=1, cols=3,
    specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
    subplot_titles=["$(x - y)(y + xz^2) = 0$", "$(x - z)(xz + 1) = 0$", "$(y + z)(yz - 1) = 0$"]
)

# Add p1 surface to the first subplot
fig1_v2.add_trace(go.Isosurface(
    x=X2.flatten(),
    y=Y2.flatten(),
    z=Z2.flatten(),
    value=p1_vals2.flatten(),
    isomin=-0.1,
    isomax=0.1,
    surface_count=2,
    colorscale="Blues",
    opacity=0.5,
    name='$(x - y)(y + xz^2) = 0$'
), row=1, col=1)

# Add p2 surface to the second subplot
fig1_v2.add_trace(go.Isosurface(
    x=X2.flatten(),
    y=Y2.flatten(),
    z=Z2.flatten(),
    value=p2_vals2.flatten(),
    isomin=-0.1,
    isomax=0.1,
    surface_count=2,
    colorscale="Greens",
    opacity=0.5,
    name='$(x - z)(xz + 1) = 0$'
), row=1, col=2)

# Add p3 surface to the third subplot
fig1_v2.add_trace(go.Isosurface(
    x=X2.flatten(),
    y=Y2.flatten(),
    z=Z2.flatten(),
    value=p3_vals2.flatten(),
    isomin=-0.1,
    isomax=0.1,
    surface_count=2,
    colorscale="Reds",
    opacity=0.5,
    name='$(y + z)(yz - 1) = 0$'
), row=1, col=3)

# Update layout for clarity
fig1_v2.update_layout(
    scene=dict(
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        zaxis_title='Z-axis',
    ),
    title="Locus of individual polynomials",
)

#fig1_v2.show()

## 2
# Create plot for each individual irreducible component
fig2_v2 = make_subplots(
    rows=2, cols=3,
    specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}], [{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
    subplot_titles=["$(x - y) = 0$", "$(y + xz^2) = 0$", "$(x - z) = 0$","$(xz + 1) = 0$", "$(y + z) = 0$", "$(yz - 1) = 0$"]
)


fig2_v2.add_trace(go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=p1_1_vals.flatten(),
    surface_count=2,
    isomin=-0.1,
    isomax=0.1,
    colorscale="Blues",
    opacity=0.5,
    name='$(x - y) = 0$'
), row=1, col=1)


fig2_v2.add_trace(go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=p1_2_vals.flatten(),
    surface_count=2,
    isomin=-0.1,
    isomax=0.1,
    colorscale="Blues",
    opacity=0.5,
    name='$(y + xz^2) = 0$'
), row=1, col=2)


fig2_v2.add_trace(go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=p2_1_vals.flatten(),
    surface_count=2,
    isomin=-0.1,
    isomax=0.1,
    colorscale="Greens",
    opacity=0.5,
    name='$(x - z) = 0$'
), row=1, col=3)


fig2_v2.add_trace(go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=p2_2_vals.flatten(),
    surface_count=2,
    isomin=-0.1,
    isomax=0.1,
    colorscale="Greens",
    opacity=0.5,
    name='$(xz + 1) = 0$'
), row=2, col=1)


fig2_v2.add_trace(go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=p3_1_vals.flatten(),
    surface_count=2,
    isomin=-0.1,
    isomax=0.1,
    colorscale="Reds",
    opacity=0.5,
    name='$(y + z) = 0$'
), row=2, col=2)


fig2_v2.add_trace(go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=p3_2_vals.flatten(),
    isomin=-0.1,
    isomax=0.1,
    surface_count=2,
    colorscale="Reds",
    opacity=0.5,
    name='$(yz - 1) = 0$'
), row=2, col=3)


fig2_v2.update_layout(
    scene=dict(
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        zaxis_title='Z-axis',
    ),
    title="Locus of irreducible components",
)

#fig2_v2.show()

## 3
# Create plot for intersection of irreducibles
fig3_v2 = make_subplots(
    rows=3, cols=4,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}],
           [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}],
           [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    subplot_titles=["$V((x - y),(y + z))$", "$V((x - y),(yz - 1))$", "$V((y + xz^2),(y+z))$", "$V((y + xz^2),(yz - 1))$",
                    "$V((x - y),(x - z))$", "$V((x - y),(xz + 1))$", "$V((y + xz^2),(x-z))$", "$V((y + xz^2),(xz + 1))$",
                    "$V((y+xz^2),(y + z))$", "$V((x - z),(yz - 1))$", "$V((xz + 1),(y+z))$", "$V((xz + 1),(yz - 1))$",
                    ]
)

# Add 1131
fig3_v2.add_trace(go.Scatter3d(
    x=intersectionpoints_1131[:, 0],
    y=intersectionpoints_1131[:, 1],
    z=intersectionpoints_1131[:, 2],
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="1131"
), row=1, col=1)

# Add 1132
fig3_v2.add_trace(go.Scatter3d(
    x=intersectionpoints_1132[:, 0],
    y=intersectionpoints_1132[:, 1],
    z=intersectionpoints_1132[:, 2],
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="1132"
), row=1, col=2)

# Add 1231
fig3_v2.add_trace(go.Scatter3d(
    x=intersectionpoints_1231[:, 0],
    y=intersectionpoints_1231[:, 1],
    z=intersectionpoints_1231[:, 2],
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="1231"
), row=1, col=3)

# Add 1232
fig3_v2.add_trace(go.Scatter3d(
    x=intersectionpoints_1232[:, 0],
    y=intersectionpoints_1232[:, 1],
    z=intersectionpoints_1232[:, 2],
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="1232"
), row=1, col=4)

# 1121
fig3_v2.add_trace(go.Scatter3d(
    x=intersectionpoints_1121[:, 0],
    y=intersectionpoints_1121[:, 1],
    z=intersectionpoints_1121[:, 2],
    mode='markers',
    marker=dict(size=3, color='blue'),
    name="1121"
), row=2, col=1)

# Add 1122
fig3_v2.add_trace(go.Scatter3d(
    x=intersectionpoints_1122[:, 0],
    y=intersectionpoints_1122[:, 1],
    z=intersectionpoints_1122[:, 2],
    mode='markers',
    marker=dict(size=3, color='blue'),
    name="1122"
), row=2, col=2)

# Add 1221
fig3_v2.add_trace(go.Scatter3d(
    x=intersectionpoints_1221[:, 0],
    y=intersectionpoints_1221[:, 1],
    z=intersectionpoints_1221[:, 2],
    mode='markers',
    marker=dict(size=3, color='blue'),
    name="1221"
), row=2, col=3)

# Add 1222
fig3_v2.add_trace(go.Scatter3d(
    x=intersectionpoints_1222[:, 0],
    y=intersectionpoints_1222[:, 1],
    z=intersectionpoints_1222[:, 2],
    mode='markers',
    marker=dict(size=3, color='blue'),
    name="1222"
), row=2, col=4)

# Add 2131
fig3_v2.add_trace(go.Scatter3d(
    x=intersectionpoints_2131[:, 0],
    y=intersectionpoints_2131[:, 1],
    z=intersectionpoints_2131[:, 2],
    mode='markers',
    marker=dict(size=3, color='green'),
    name="2131"
), row=3, col=1)

# Add 2132
fig3_v2.add_trace(go.Scatter3d(
    x=intersectionpoints_2132[:, 0],
    y=intersectionpoints_2132[:, 1],
    z=intersectionpoints_2132[:, 2],
    mode='markers',
    marker=dict(size=3, color='green'),
    name="2132"
), row=3, col=2)

# Add 2231
fig3_v2.add_trace(go.Scatter3d(
    x=intersectionpoints_2231[:, 0],
    y=intersectionpoints_2231[:, 1],
    z=intersectionpoints_2231[:, 2],
    mode='markers',
    marker=dict(size=3, color='green'),
    name="2231"
), row=3, col=3)

# Add 2232
fig3_v2.add_trace(go.Scatter3d(
    x=intersectionpoints_2232[:, 0],
    y=intersectionpoints_2232[:, 1],
    z=intersectionpoints_2232[:, 2],
    mode='markers',
    marker=dict(size=3, color='green'),
    name="2232"
), row=3, col=4)

# Update layout for clarity
fig3_v2.update_layout(
    title="Pairwise Intersection Curves of Irreducible Polynomial Surfaces",
    scene=dict(
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        zaxis_title='Z-axis',
    )
)
fig3_v2.show()

## 4
# Plot the full variety (intersection of the three polynomials)
fig4_v2 = go.Figure()
fig4_v2.add_trace(go.Scatter3d(
    x=intersectionpts[:, 0],
    y=intersectionpts[:, 1],
    z=intersectionpts[:, 2], 
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="Intersection Curve"
))

# Update layout for clarity
fig4_v2.update_layout(
    scene=dict(
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        zaxis_title='Z-axis',
    ),
    title="$V"
)
#fig4_v2.show()

## 5
fig5_v2 = make_subplots(
    rows=2, cols=4,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}],
           [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    subplot_titles=["$V((x - y),(y + z),(x - z))$", "$V((x - y),(y + z),(yz - 1))$", "$V((x - y),(y + z),(xz + 1))$", "$V((x - y),(xz + 1),(yz - 1))$",
                    "$V((y + xz^2),(y + z),(x - z))$", "$V((y + xz^2),(y + z),(yz - 1))$", "$V((y + xz^2),(y + z),(xz + 1))$", "$V((y + xz^2),(xz + 1),(yz - 1))$",
                    ]
)

# Add 112131
fig5_v2.add_trace(go.Scatter3d(
    x=intersectionpoints_112131[:, 0],
    y=intersectionpoints_112131[:, 1],
    z=intersectionpoints_112131[:, 2],
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="112131"
), row=1, col=1)

# Add 112132
fig5_v2.add_trace(go.Scatter3d(
    x=intersectionpoints_112132[:, 0],
    y=intersectionpoints_112132[:, 1],
    z=intersectionpoints_112132[:, 2],
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="112132"
), row=1, col=2)

# Add 112231
fig5_v2.add_trace(go.Scatter3d(
    x=intersectionpoints_112231[:, 0],
    y=intersectionpoints_112231[:, 1],
    z=intersectionpoints_112231[:, 2],
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="112231"
), row=1, col=3)

# Add 112232
fig5_v2.add_trace(go.Scatter3d(
    x=intersectionpoints_112232[:, 0],
    y=intersectionpoints_112232[:, 1],
    z=intersectionpoints_112232[:, 2],
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="112232"
), row=1, col=4)

# Add 122131
fig5_v2.add_trace(go.Scatter3d(
    x=intersectionpoints_122131[:, 0],
    y=intersectionpoints_122131[:, 1],
    z=intersectionpoints_122131[:, 2],
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="122131"
), row=2, col=1)

# Add 122132
fig5_v2.add_trace(go.Scatter3d(
    x=intersectionpoints_122132[:, 0],
    y=intersectionpoints_122132[:, 1],
    z=intersectionpoints_122132[:, 2],
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="122132"
), row=2, col=2)

# Add 122231
fig5_v2.add_trace(go.Scatter3d(
    x=intersectionpoints_122231[:, 0],
    y=intersectionpoints_122231[:, 1],
    z=intersectionpoints_122231[:, 2],
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="122231"
), row=2, col=3)

# Add 122232
fig5_v2.add_trace(go.Scatter3d(
    x=intersectionpoints_122232[:, 0],
    y=intersectionpoints_122232[:, 1],
    z=intersectionpoints_122232[:, 2],
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="122232"
), row=2, col=4)

# Update layout for clarity
fig5_v2.update_layout(
    scene=dict(
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        zaxis_title='Z-axis',
    ),
    title="$Triple Intersection of Irreducibles$"
)

#fig5_v2.show()

## CASE 3 PLOTS

## 1
# Create subplots for each polynomial surface
fig1_v3 = make_subplots(
    rows=1, cols=1,
    specs=[[{'type': 'surface'}]],
    subplot_titles=["$xz + y^2 - x^2z^2 - xyz^2 = 0$"]
)

# Add p1 surface to the first subplot
fig1_v3.add_trace(go.Isosurface(
    x=X2.flatten(),
    y=Y2.flatten(),
    z=Z2.flatten(),
    value=p1_irred_vals2.flatten(),
    isomin=-0.1,
    isomax=0.1,
    surface_count=2,
    colorscale="Blues",
    opacity=0.5,
    name='$(x - y)(y + xz^2) = 0$'
), row=1, col=1)

# Update layout for clarity
fig1_v3.update_layout(
    scene=dict(
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        zaxis_title='Z-axis',
    ),
    title="Locus of irreducible polynomial",
)

#fig1_v3.show()

## 2 Triple intersections
fig2_v3 = make_subplots(
    rows=2, cols=2,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}], [{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    subplot_titles=["$V(f_1, (x-z), (y+z))$", "$V(f_1, (xz+1), (y+z))$",
                    "$V(f_1, (x-z), (yz-1)$", "$V(f_1,(xz + 1),(yz - 1))$"
                    ]
)

# Add 12131
fig2_v3.add_trace(go.Scatter3d(
    x=intersectionpoints_12131[:, 0],
    y=intersectionpoints_12131[:, 1],
    z=intersectionpoints_12131[:, 2],
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="12131"
), row=1, col=1)

# Add 12231
fig2_v3.add_trace(go.Scatter3d(
    x=intersectionpoints_12231[:, 0],
    y=intersectionpoints_12231[:, 1],
    z=intersectionpoints_12231[:, 2],
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="12231"
), row=1, col=2)

# Add 12132
fig2_v3.add_trace(go.Scatter3d(
    x=intersectionpoints_12132[:, 0],
    y=intersectionpoints_12132[:, 1],
    z=intersectionpoints_12132[:, 2],
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="12132"
), row=2, col=1)


# Add 12232
fig2_v3.add_trace(go.Scatter3d(
    x=intersectionpoints_12232[:, 0],
    y=intersectionpoints_12232[:, 1],
    z=intersectionpoints_12232[:, 2],
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="12232"
), row=2, col=2)

# Update layout for clarity
fig2_v3.update_layout(
    scene=dict(
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        zaxis_title='Z-axis',
    ),
    title="Triple Intersection of Irreducibles"
)

#fig2_v3.show()

## 3
# Create plot for pairwise intersection of irreducibles
fig3_v3 = make_subplots(
    rows=2, cols=4,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}],
           [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    subplot_titles=["$V(f_1, (y+z))$", "$V(f_1, (yz-1))$", "$V(f_1, (x-z))$", "$V(f_1, (xz+1))$",
                    "$V((x - z),(y + z))$", "$V((x - z),(yz - 1))$", "$V((xz + 1),(y+z))$", "$V((xz + 1),(yz - 1))$"
                    ]
)

# Add 131
fig3_v3.add_trace(go.Scatter3d(
    x=intersectionpoints_131[:, 0],
    y=intersectionpoints_131[:, 1],
    z=intersectionpoints_131[:, 2],
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="131"
), row=1, col=1)

# Add 132
fig3_v3.add_trace(go.Scatter3d(
    x=intersectionpoints_132[:, 0],
    y=intersectionpoints_132[:, 1],
    z=intersectionpoints_132[:, 2],
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="132"
), row=1, col=2)

# Add 121
fig3_v3.add_trace(go.Scatter3d(
    x=intersectionpoints_121[:, 0],
    y=intersectionpoints_121[:, 1],
    z=intersectionpoints_121[:, 2],
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="121"
), row=1, col=3)

# Add 122
fig3_v3.add_trace(go.Scatter3d(
    x=intersectionpoints_122[:, 0],
    y=intersectionpoints_122[:, 1],
    z=intersectionpoints_122[:, 2],
    mode='markers',
    marker=dict(size=3, color='purple'),
    name="122"
), row=1, col=4)


# Add 2131
fig3_v3.add_trace(go.Scatter3d(
    x=intersectionpoints_2131[:, 0],
    y=intersectionpoints_2131[:, 1],
    z=intersectionpoints_2131[:, 2],
    mode='markers',
    marker=dict(size=3, color='green'),
    name="2131"
), row=2, col=1)

# Add 2132
fig3_v3.add_trace(go.Scatter3d(
    x=intersectionpoints_2132[:, 0],
    y=intersectionpoints_2132[:, 1],
    z=intersectionpoints_2132[:, 2],
    mode='markers',
    marker=dict(size=3, color='green'),
    name="2132"
), row=2, col=2)

# Add 2231
fig3_v3.add_trace(go.Scatter3d(
    x=intersectionpoints_2231[:, 0],
    y=intersectionpoints_2231[:, 1],
    z=intersectionpoints_2231[:, 2],
    mode='markers',
    marker=dict(size=3, color='green'),
    name="2231"
), row=2, col=3)

# Add 2232
fig3_v3.add_trace(go.Scatter3d(
    x=intersectionpoints_2232[:, 0],
    y=intersectionpoints_2232[:, 1],
    z=intersectionpoints_2232[:, 2],
    mode='markers',
    marker=dict(size=3, color='green'),
    name="2232"
), row=2, col=4)


# Update layout for clarity
fig3_v3.update_layout(
    title="Pairwise Intersection Curves of Irreducible Polynomial Surfaces",
    scene=dict(
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        zaxis_title='Z-axis',
    )
)
#fig3_v3.show()