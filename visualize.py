import numpy as np
import torch
def sphere_points(n=128):
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    radius = np.sqrt(1 - z * z)

    points = np.zeros((n, 3))
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = z

    # xyz = points
    # x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return points
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')
points = sphere_points(128)
print(np.linalg.norm(points,axis=1))
np.savetxt('points.txt',points)
xdata = points[:,0]
ydata = points[:,1]
zdata = points[:,2]
ax.plot3D(xdata, ydata, zdata, 'gray')
plt.savefig('example.png')