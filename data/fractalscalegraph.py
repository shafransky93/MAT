import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set up the scaling factor and dimensionality ranges
S_range = np.arange(0.1, 100, 0.1)
D_range = np.arange(1, 4, 0.1)

# Calculate the number of quantum states N for each combination of S and D
N = np.zeros((len(S_range), len(D_range)))
for i in range(len(S_range)):
    for j in range(len(D_range)):
        N[i][j] = S_range[i] ** D_range[j]

# Create a 3D surface plot of N as a function of S and D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(S_range, D_range)
ax.plot_surface(X, Y, N.T, cmap='viridis') # Transpose N to match the shape of X and Y
ax.set_xlabel('Scaling Factor (S)')
ax.set_ylabel('Dimensionality (D)')
ax.set_zlabel('Number of Quantum States (N)')
plt.show()
