import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

# Define simulation parameters
a = 1
N = int(input("Enter the number of particles (default 100): ") or "100")
N_grid = 100
L = 10
threshold = 0.1

# Initialize particle positions
x_values = np.zeros(N)
y_values = np.zeros(N)
z_values = np.zeros(N)


# Initialize wavefunction
x, y, z = np.meshgrid(np.linspace(-L, L, N_grid), np.linspace(-L, L, N_grid), np.linspace(-L, L, N_grid))
psi_values = np.exp(-(x**2 + y**2 + z**2)/(4*a**2))

def update(val):
    global surf, psi, X, Y, pot_surf, V
    # Get slider values
    sigma = ssigma.val
    k = sk.val
    L = sL.val
    x = np.linspace(-L/2, L/2, 100)
    y = np.linspace(-L/2, L/2, 100)
    X, Y = np.meshgrid(x, y)
    psi_new = np.exp(-((X**2 + Y**2)/(2*sigma**2))) * np.exp(1j*k*X)
    V = coulomb_potential(np.sqrt(X**2 + Y**2))  # update Coulomb potential
    psi_new *= np.exp(-1j*V*X)
    
    # Update surface plot
    surf.remove()
    surf = ax.plot_surface(X, Y, np.real(psi_new), cmap='jet', alpha=0.5)
    
    # Update wavefunction and potential
    psi = psi_new
    
    if pot_surf in ax.collections:
        ax.collections.remove(pot_surf)
    Z = np.tile(V.reshape((100, 100, 1)), (1, 1, X.shape[0])) # reshape V to match the shape of X and Y
    pot_surf = ax.plot_surface(X, Y, Z[:,:,0], cmap='coolwarm', alpha=0.5)
    
    # Redraw plot
    fig.canvas.draw_idle()


# Define the animation function
def animate(i, ax, x_values, y_values, z_values, N, N_grid, L, threshold):
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

    # Update the 3D plot
    ax.clear()
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-L, L)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=30)
    ax.voxels(psi_values_new > 10*threshold, edgecolor='k', alpha=0.7)

    # Update psi_values
    psi_values[...] = psi_values_new


# Coulomb potential function
def coulomb_potential(r):
    return -1 / np.sqrt(r**2 + a**2)

# Create figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.2)

# Initialize wavefunction
L = 1  # box length
x = np.linspace(-L/2, L/2, 100)
y = np.linspace(-L/2, L/2, 100)
X, Y = np.meshgrid(x, y)
sigma = L/10  # width of Gaussian wavepacket
k = 5*np.pi/L  # wavevector
a = 0.1  # Coulomb interaction strength
psi = np.exp(-((X**2 + Y**2)/(2*sigma**2))) * np.exp(1j*k*X)
V = coulomb_potential(np.sqrt(X**2 + Y**2))  # Coulomb potential

# Plot wavefunction and potential
surf = ax.plot_surface(X, Y, np.real(psi), cmap='jet', alpha=0.5)
pot_surf = ax.plot_surface(X, Y, V, cmap='coolwarm', alpha=0.5)

# Create sliders and connect to update function
axsigma = plt.axes([0.25, 0.1, 0.65, 0.03])
axk = plt.axes([0.25, 0.05, 0.65, 0.03])
axL = plt.axes([0.25, 0.15, 0.65, 0.03])

# Create sliders and connect to update function
ssigma = Slider(axsigma, 'Sigma', 0.01, 1.0, valinit=sigma)
sk = Slider(axk, 'K', 0.1, 10.0, valinit=k)
sL = Slider(axL, 'L', 0.1, 2.0, valinit=L)
ssigma.on_changed(update)
sk.on_changed(update)
sL.on_changed(update)

# Animate the plot
ani = FuncAnimation(fig, animate, fargs=(ax, x_values, y_values, z_values, N, N_grid, L, threshold), interval=50)


plt.show()


