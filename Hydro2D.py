# Final project in computational physics.
# In This project two dimensional hydrodynamic system for incompressible flow with an obstacle is solved.
# Equations are derived from continuum equation and Navier-Stokes equation.
# The solution is based on Gauss-Seidel iterative method for 2 dimensional lattice.


import numpy as np
import matplotlib.pyplot as plt
from copy import copy


# Dimension of 2D lattice.
LATTICE_X = 120
LATTICE_Y = 40

# Obstacle position
OBSTACLE_Y = 8
OBSTACLE_X_LEFT = 36
OBSTACLE_X_RIGHT = 44

# Stream and vorticity indices.
STREAM = 0
VORTICITY = 1

# Relaxation Parameter. Used in Gauss-Seidel iteration for data interpolation
W = 0.1

# Reynold error for its critical estimation
REYNOLDS_ERR = 0.1


# Initiate a 3D lattice array using the given spacing and initial velocity in x direction.
# 2D for physical dimension and 1D for stream and vorticity functions.
# h - spacing between cells
# v_0 - initial velocity in x direction
def init_lattice(h):
    # Set number of spaces. y is for rows and x is for columns.
    x_spaces = int(LATTICE_X / h) + 1
    y_spaces = int(LATTICE_Y / h) + 1
    x = np.linspace(0, LATTICE_X, x_spaces)
    y = np.linspace(0, LATTICE_Y, y_spaces)
    # Setting 3D array. 2D shape for positioning, 1D for stream and vorticity functions.
    lattice = np.zeros(shape=(y_spaces, x_spaces, 2))
    # Velocity initial condition
    lattice[:, :, STREAM] = np.meshgrid(x, y)[1]
    # Change coordinates
    obstacle_x_right = int(OBSTACLE_X_RIGHT / h)
    obstacle_x_left = int(OBSTACLE_X_LEFT / h)
    obstacle_y = int(OBSTACLE_Y / h)
    # Zero the obstacle's stream function (including edges)
    lattice[:obstacle_y + 1, obstacle_x_left:obstacle_x_right + 1, STREAM] = 0

    # Setting boundary conditions to A-H:
    # If not set here, boundary condition already holds

    # B
    lattice[1:obstacle_y, obstacle_x_right, VORTICITY] = 2 * lattice[1:obstacle_y, obstacle_x_right + 1, STREAM]
    # D
    lattice[1:obstacle_y, obstacle_x_left, VORTICITY] = 2 * lattice[1:obstacle_y, obstacle_x_left - 1, STREAM]
    # C
    lattice[obstacle_y, obstacle_x_left + 1:obstacle_x_right, VORTICITY] = \
        2 * lattice[obstacle_y + 1, obstacle_x_left + 1:obstacle_x_right, STREAM]

    # B-C point
    lattice[obstacle_y, obstacle_x_right, VORTICITY] = 0.5 * (lattice[obstacle_y + 1, obstacle_x_right, VORTICITY]
                                                              + lattice[obstacle_y, obstacle_x_right + 1, VORTICITY])
    # D-C point
    lattice[obstacle_y, obstacle_x_left, VORTICITY] = 0.5 * (lattice[obstacle_y + 1, obstacle_x_left, VORTICITY]
                                                             + lattice[obstacle_y, obstacle_x_left - 1, VORTICITY])

    return lattice


# Set boundary conditions along the grid.
# lattice - 3D numpy array. 2D for physical dimension, 1D for stream and vorticity functions
# reynolds - reynolds number
# conditions - Neumann's boundary condition for stream function over G line.
def set_boundary(lattice, h, g_cond):
    # Applying Dirichlet boundary conditions
    # B
    obstacle_y = int(OBSTACLE_Y / h)
    obstacle_x_left = int(OBSTACLE_X_LEFT / h)
    obstacle_x_right = int(OBSTACLE_X_RIGHT / h)
    vorticity_b = 2 * lattice[1:obstacle_y, obstacle_x_right, STREAM]
    lattice[1:obstacle_y, obstacle_x_right, VORTICITY] = \
        W * vorticity_b + (1-W) * lattice[1:obstacle_y, obstacle_x_right, VORTICITY]
    # C
    vorticity_c = 2 * lattice[obstacle_y + 1, obstacle_x_left + 1:obstacle_x_right, STREAM]
    lattice[obstacle_y, obstacle_x_left + 1:obstacle_x_right, VORTICITY] = \
        W * vorticity_c + (1-W) * lattice[obstacle_y, obstacle_x_left + 1:obstacle_x_right, VORTICITY]
    # D
    vorticity_d = 2 * lattice[1:obstacle_y, obstacle_x_left - 1, STREAM]
    lattice[1:obstacle_y, obstacle_x_left, VORTICITY] = \
        W * vorticity_d + (1-W) * lattice[1:obstacle_y, obstacle_x_left, VORTICITY]

    # Obstacle Corners:
    # B-C
    bc_corner = 0.5 * (lattice[obstacle_y, obstacle_x_right + 1, STREAM] + lattice[obstacle_y + 1, obstacle_x_right, STREAM])
    lattice[obstacle_y, obstacle_x_right, VORTICITY] = W * bc_corner + (1 - W) * lattice[obstacle_y, obstacle_x_right, VORTICITY]
    # C-D
    cd_corner = 0.5 * (lattice[obstacle_y, obstacle_x_left + 1, STREAM] + lattice[obstacle_y + 1, obstacle_x_left, STREAM])
    lattice[obstacle_y, obstacle_x_left, VORTICITY] = W * cd_corner + (1 - W) * lattice[obstacle_y, obstacle_x_left, VORTICITY]

    # Applying Neumann boundary conditions
    # F
    stream_f = 0.25 * (2 * lattice[1:-1, 1, STREAM] + lattice[2:, 0, STREAM] + lattice[:-2, 0, STREAM] - lattice[1:-1, 0, VORTICITY])
    lattice[1:-1, 0, STREAM] = W * stream_f + (1 - W) * lattice[1:-1, 0, STREAM]
    # G
    stream_g = 0.25 * (2 * lattice[-2, 1:-1, STREAM] + lattice[-1, 2:, STREAM] + lattice[-1, :-2, STREAM]
                       - lattice[-1, 1:-1, VORTICITY] + 2 * g_cond)
    lattice[-1, 1:-1, STREAM] = W * stream_g + (1 - W) * lattice[-1, 1:-1, STREAM]
    # H
    stream_h = 0.25 * (2 * lattice[1:-1, -2, STREAM] + lattice[2:, -1, STREAM] + lattice[:-2, -1, STREAM] - lattice[1:-1, -1, VORTICITY])
    lattice[1:-1, -1, STREAM] = W * stream_h + (1 - W) * lattice[1:-1, -1, STREAM]

    vorticity_h = 0.25 * (lattice[2:, -1, VORTICITY] - lattice[:-2, -1, VORTICITY])
    lattice[1:-1, -1, VORTICITY] = W * vorticity_h + (1 - W) * lattice[1:-1, -1, VORTICITY]


# Update i,j value of stream function
# lattice - 3D numpy array. 2D for physical dimension, 1D for stream and vorticity functions
# i - Row index
# j - Column index
def stream_step(lattice, i, j):
    stream_ij = 0.25 * (lattice[i + 1, j, STREAM] + lattice[i - 1, j, STREAM] + lattice[i, j + 1, STREAM] + lattice[i, j - 1, STREAM]
                        - lattice[i, j, VORTICITY])
    # Interpolate
    lattice[i, j, STREAM] = W * stream_ij + (1 - W) * lattice[i, j, STREAM]


# Update i,j value of vorticity function
# lattice - 3D numpy array. 2D for physical dimension, 1D for stream and vorticity functions
# i - Row index
# j - Column index
def vorticity_step(lattice, i, j, reynolds):
    vorticity_ij = 0.25 * (lattice[i + 1, j, VORTICITY] + lattice[i - 1, j, VORTICITY] + lattice[i, j + 1, VORTICITY]
                           + lattice[i, j - 1, VORTICITY])
    temp_1 = (lattice[i + 1, j, STREAM] - lattice[i - 1, j, STREAM]) * (lattice[i, j + 1, VORTICITY] - lattice[i, j - 1, VORTICITY])
    temp_2 = (lattice[i, j + 1, STREAM] - lattice[i, j - 1, STREAM]) * (lattice[i + 1, j, VORTICITY] - lattice[i - 1, j, VORTICITY])
    vorticity_ij -= (reynolds / 16) * (temp_1 - temp_2)
    # Interpolate
    lattice[i, j, VORTICITY] = W * vorticity_ij + (1 - W) * lattice[i, j, VORTICITY]


# gauus-Seidel iterative step.
# Updates stream and vorticity functions including edges of lattice and obstacle.
# lattice - 3D numpy array. 2D for physical dimension, 1D for stream and vorticity functions
# reynolds - reynolds number.
# g_cond - derivative condition over G line, in units of v_0.
def gauss_seidel_iteration(lattice, reynolds, h, g_cond):
    obstacle_y = int(OBSTACLE_Y / h)
    obstacle_x_left = int(OBSTACLE_X_RIGHT / h)
    obstacle_x_right = int(OBSTACLE_X_RIGHT / h)
    y_edge, x_edge, func = lattice.shape
    for f in range(func):
        for i in range(1, y_edge-1):
            for j in range(1, x_edge-1):
                # Skipping obstacle and boundaries.
                if 0 <= i <= obstacle_y and obstacle_x_left <= j <= obstacle_x_right:
                    continue
                if f == STREAM:
                    stream_step(lattice, i, j)
                else:  # VORTICITY
                    vorticity_step(lattice, i, j, reynolds)

    # Set boundaries of obstacle and lattice.
    set_boundary(lattice, h, g_cond)


# Plot stream function
# lattice - 3D numpy array. 2D for physical dimension, 1D for stream and vorticity functions
# title - title of the figure
def plot_stream(lattice, h, title):
    # Coordinates
    x = np.linspace(0, LATTICE_X, int(LATTICE_X / h) + 1)
    y = np.linspace(0, LATTICE_Y, int(LATTICE_Y / h) + 1)
    x_pos, y_pos = np.meshgrid(x, y)
    # Velocities by definition of stream function.
    U, V = np.gradient(lattice[:, :, STREAM])
    V *= -1

    fig, ax = plt.subplots(figsize=(8, 4))
    # Mask for obstacle drawing
    mask_1 = np.ones(V.shape, dtype=bool)
    mask_1[int(LATTICE_Y-OBSTACLE_Y/h):, int(OBSTACLE_X_LEFT/h):int(OBSTACLE_X_RIGHT/h) + 1] = False
    # Mask for stream lines
    mask_2 = np.zeros(V.shape, dtype=bool)
    mask_2[:int(OBSTACLE_Y/h)+1, int(OBSTACLE_X_LEFT/h):int(OBSTACLE_X_RIGHT/h)+1] = True
    V = np.ma.array(V, mask=mask_2)
    U = np.ma.array(U, mask=mask_2)
    U[0, :] = np.nan
    # Plot
    ax.imshow(mask_1, extent=(0, LATTICE_X, 0, LATTICE_Y), cmap='gray', aspect='auto')
    ax.set_title(title)
    # strm = ax.contour(x_pos, y_pos, lattice[:, :, 0], cmap='jet')
    # fig.colorbar(strm)
    strm = ax.streamplot(x_pos, y_pos, U, V, arrowstyle='-', linewidth=2, color=U, cmap='viridis', density=[5, 1])
    fig.colorbar(strm.lines)
    plt.tight_layout()
    plt.show()


# Solve 2D Navier Stokes equation in cartesian coordinates with rectangle obstacle.
# Solution is based on discretization of 2D lattice with equal spacing and Gauus-Seidel iterative method.
# spacing - lattice spacing.
# reynolds - reynolds number.
# eps - error tolerance for convergence. Default value is 10**-3
# Return - a tuple of 2. (lattice, number of iterations)
def hydro_2D(spacing, reynolds, eps=10**-3):
    lattice = init_lattice(spacing)
    # Neumann boundary condition for stream function over G in units of v_0.
    g_cond = 1
    # Number of iterations
    it = 0
    # Arbitrary values to suffice loop's first condition
    norm_1 = 2
    norm_2 = 1
    while abs(norm_2 / norm_1) < 1-eps:
        if it % 10 == 0:
            lattice_1 = copy(lattice)
            gauss_seidel_iteration(lattice, reynolds, spacing, g_cond)
            lattice_2 = copy(lattice)
            gauss_seidel_iteration(lattice, reynolds, spacing, g_cond)
            norm_2 = np.linalg.norm(lattice - lattice_2)
            norm_1 = np.linalg.norm(lattice_2 - lattice_1)
            it += 2
            continue

        gauss_seidel_iteration(lattice, reynolds, spacing, g_cond)
        it += 1

    return lattice, it


# Solve and plot stream function with spacing=1 and reynolds number = 0.01.
def task_1():
    spacing = 1
    reynolds = 0.01
    lattice, it = hydro_2D(spacing, reynolds)
    print("Number of iterations for convergence: ", it)
    plot_stream(lattice, spacing, "Spacing=1 and Reynolds=0.01")


# Solve and plot stream function with spacing=1 and reynolds number = 4.
def task_2():
    spacing = 1
    reynolds = 4
    lattice, it = hydro_2D(spacing, reynolds)
    print("Number of iterations for convergence: ", it)
    plot_stream(lattice, spacing, "Spacing=1 and Reynolds=4")


# Finding critical reynolds number for spacing=1
def task_3():
    spacing = 1
    reynolds_l = 0.01
    reynolds_r = 4
    reynolds_mid = (reynolds_l + reynolds_r) / 2
    while reynolds_l - REYNOLDS_ERR < reynolds_mid < reynolds_r - REYNOLDS_ERR:
        lattice, it = hydro_2D(spacing, reynolds_mid)
        U, V = np.gradient(lattice[:, :, STREAM])
        V *= -1
        # When there are velocities in left direction, a vortex is starting to form.
        not_critical = np.all(U[1:int((OBSTACLE_Y - 1) / spacing), int((OBSTACLE_X_RIGHT + 1) / spacing):int((OBSTACLE_X_RIGHT + 5) / spacing)] >= 0)
        if not_critical:
            reynolds_l = reynolds_mid
        else:
            reynolds_r = reynolds_mid

        reynolds_mid = (reynolds_l + reynolds_r) / 2

    print("Critical reynolds number is: ", reynolds_mid)


# Finding critical reynolds number for spacing=0.5
def task_4():
    spacing = 0.5
    reynolds_l = 0.01
    reynolds_r = 4
    reynolds_mid = (reynolds_l + reynolds_r) / 2
    while reynolds_l - REYNOLDS_ERR < reynolds_mid < reynolds_r - REYNOLDS_ERR:
        lattice, it = hydro_2D(spacing, reynolds_mid)
        U, V = np.gradient(lattice[:, :, STREAM])
        V *= -1
        # When there are velocities in left direction, a vortex is starting to form.
        not_critical = np.all(U[1:int((OBSTACLE_Y - 1)/spacing), int((OBSTACLE_X_RIGHT + 1)/spacing):int((OBSTACLE_X_RIGHT + 5)/spacing)] >= 0)
        plot_stream(lattice, spacing, reynolds_mid)
        if not_critical:
            reynolds_l = reynolds_mid
        else:
            reynolds_r = reynolds_mid

        reynolds_mid = (reynolds_l + reynolds_r) / 2


# Solve and plot stream function with spacing=1 and reynolds number=15.
def task_5():
    spacing = 1
    reynolds = 15
    lattice, it = hydro_2D(spacing, reynolds)
    print("Number of iterations for convergence: ", it)
    plot_stream(lattice, spacing, "Spacing=1 and Reynolds=15")


# Solve and plot stream function with spacing=0.5 and reynolds number=15.
def task_6():
    spacing = 0.5
    reynolds = 15
    lattice, it = hydro_2D(spacing, reynolds)
    print("Number of iterations for convergence: ", it)
    plot_stream(lattice, spacing, "Spacing=1 and Reynolds=15")

if __name__ == '__main__':
    # task_1()
    # task_2()
    # task_3()
    # task_4()
    # task_5()
    # task_6
