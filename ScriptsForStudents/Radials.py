import numpy as np
import matplotlib.pyplot as plt
from Assistant_Functions.ImageImport import ImageStack
from Meshing.pvMesh import Mesh


path = r'F:\Project - MTF Validation\5 Identical Acquisitions\KULQCCT_LM_MTF_Reprod_Unknown_20240123_140423643'

stack = ImageStack(path)
print('Stack Complete')

mesh = Mesh(stack)
print('Mesh Complete')

# %%

# The mesh consists of a series of vertices (points) and faces (triangles) with the center of each triangle being
# called the centroid. Perpendicular to this triangle, we define the normal vector

faces = mesh.faces
vertices = mesh.vertices
centroids = mesh.centroids
normals = mesh.normals
radials = mesh.radial_distance

# For our MTF calculations, we are only interested in the triangles where the normals lie in the XY-plane
# (= perpendicular to z-axis). This because if we then draw a line along this normal, all pixels along that line, lie
# within the XY - plane aswell. In practice, we allow the normals to deviate max 10° from XY-plane.
# These are the selected triangles

# Radial distance = list of distances between isocenter of scan and each individual triangles
select_radials = mesh.select_radial
select_normals = mesh.select_normals
select_triangles = mesh.select_faces


# Values is the quantity we are going to make a histogram of
values = select_radials
# Bin width is width of the histogram bins
bin_width = 1

bins_x = np.arange(0, max(values) + bin_width, bin_width)
histogram, bins_x = np.histogram(values, bins=bins_x)

# Bin centers calculate the center of each bin = easier to plot
bin_centers = (bins_x[:-1] + bins_x[1:]) / 2

plt.bar(bin_centers, histogram, width=bin_width)
plt.show()


