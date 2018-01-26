hlp = """
    Create a simple logo.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Parameters
b = np.pi / 2
c = 2 * b / 3
lim = 1.05 * b
xr = np.linspace(-b, b, 1000)
y = np.sin(xr)

# Colors
orange = "#b35900"
dgray = "#8c8c8c"
gray = "#cccccc"

# Fills
# wedge1 = mpl.patches.Wedge((0, 0), c, theta1=90, theta2=180, color=gray)
# wedge2 = mpl.patches.Wedge((0, 0), c, theta1=270, theta2=360, color=gray)
wedge1 = mpl.patches.Wedge((0, 0), c, theta1=0, theta2=90, color=gray)
wedge2 = mpl.patches.Wedge((0, 0), c, theta1=180, theta2=270, color=gray)
circle1 = plt.Circle((0, 0), c, color=dgray, fill=True)

# Plot
fig = plt.figure(figsize=(2.7, 2.7))
# plt.gca().add_artist(circle1)
plt.gca().add_artist(wedge1)
plt.gca().add_artist(wedge2)
for i in np.linspace(0, len(xr)-1, 17).astype(int):
    plt.plot([xr[i], xr[i]], [y[i], xr[i]], "--",
             linewidth=0.7, color=dgray)
plt.plot(xr, xr, color=dgray, linewidth=3)
plt.plot(xr, y, color=orange, linewidth=3)
plt.gca().set_xlim((-lim, lim))
plt.gca().set_ylim((-lim, lim))
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.savefig("img/logo.png", transparent=True)
fig.savefig("img/logo.svg", transparent=True)
plt.close("all")

