import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.widgets import Slider

# Create figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.1)

# Initialize wavefunction
amplitude = 1
frequency = 1
phase = 0
theta = np.linspace(0, 2*np.pi, 1000)
phi = np.linspace(0, np.pi, 1000)
x = amplitude * np.sin(frequency * theta + phase) * np.cos(phi)
y = amplitude * np.sin(frequency * theta + phase) * np.sin(phi)
z = amplitude * np.cos(frequency * theta + phase)

# Plot wavefunction
l, = ax.plot(x, y, z, lw=0.2, color='red')

# Create sliders
axamp = plt.axes([0.25, 0.1, 0.65, 0.03])
axfreq = plt.axes([0.25, 0.15, 0.65, 0.03])
axphase = plt.axes([0.25, 0.2, 0.65, 0.03])

samp = Slider(axamp, 'Amplitude', 0.1, 100.0, valinit=amplitude)
sfreq = Slider(axfreq, 'Frequency', 0.1, 100.0, valinit=frequency)
sphase = Slider(axphase, 'Phase', 0.0, 2*np.pi, valinit=phase)

# Update plot based on slider values
def update(val):
    amp = samp.val
    freq = sfreq.val
    phase = sphase.val
    l.set_xdata(amp*np.sin(freq*theta + phase)*np.cos(phi))
    l.set_ydata(amp*np.sin(freq*theta + phase)*np.sin(phi))
    l.set_3d_properties(amp*np.cos(freq*theta + phase))
    fig.canvas.draw_idle()

# Connect sliders to update function
samp.on_changed(update)
sfreq.on_changed(update)
sphase.on_changed(update)

plt.show()
