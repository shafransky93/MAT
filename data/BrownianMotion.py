import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# Coulomb potential function
def coulomb_potential(r, a):
    return -1 / np.sqrt(r**2 + a**2)

# Define simulation parameters
a = 1
N = int(input("Enter the number of particles (default 100): ") or "100")
N_grid = 51
L = 10
threshold = 0.1

# Initialize particle positions
x_values = np.zeros(N)
y_values = np.zeros(N)
z_values = np.zeros(N)

# Initialize wavefunction
x, y, z = np.meshgrid(np.linspace(-L, L, N_grid), np.linspace(-L, L, N_grid), np.linspace(-L, L, N_grid))
psi_values = np.exp(-(x**2 + y**2 + z**2)/(4*a**2))

# Create figure and axes
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
plt.subplots_adjust(bottom=0.2)

# Plot wavefunction
surf = ax.plot_surface(x[:,:,25], y[:,:,25], psi_values[:,:,25], cmap='jet', alpha=0.5)
ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.set_zlim(-L, L)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=30, azim=30)

# Define the animation function
def animate(i, ax, surf, x_values, y_values, z_values, N, N_grid, L, threshold):
    # Calculate new position for each particle
    x_values += np.random.normal(0, 1, N)
    y_values += np.random.normal(0, 1, N)
    z_values += np.random.normal(0, 1, N)

    # Update the wavefunction
    psi_values_new = np.exp(-(x**2 + y**2 + z**2)/(4*a**2))
    for i in range(N):
        psi_values_new += np.exp(-((x-x_values[i])**2 + (y-y_values[i])**2 + (z-z_values[i])**2)/(4*a**2))

    # Reshape psi_values into a cube of size N_grid x N_grid x N_grid
    try:
        psi_values_new = np.reshape(psi_values_new, (N_grid, N_grid, N_grid))
    except ValueError:
        print("Error: cannot reshape array of size {} into shape ({}, {}, {})".format(
            psi_values_new.size, N_grid, N_grid, N_grid))
        return

    # Update the surface plot
    verts = []
    for j in range(N_grid-1):
        for k in range(N_grid-1):
            verts.append([(x[j,k,25], y[j,k,25], psi_values_new[j,k,25]),
                          (x[j+1,k,25], y[j+1,k,25], psi_values_new[j+1,k,25]),
                          (x[j+1,k+1,25], y[j+1,k+1,25], psi_values_new[j+1,k+1,25]),
                          (x[j,k+1,25], y[j,k+1,25], psi_values_new[j,k+1,25])])
    surf.set_verts(verts)

    # Update psi_values
    psi_values[...] = psi_values_new

# Animate the plot
ani = FuncAnimation(fig, animate, fargs=(ax, surf, x_values, y_values, z_values, N, N_grid, L, threshold), interval=50)

# Show the animation
plt.show()
