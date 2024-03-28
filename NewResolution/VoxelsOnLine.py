from collections.abc import Iterable

import numpy as np


def bresenham_kwinten(point_a: Iterable[float], point_b: Iterable[float], vox_dim: Iterable[float]):
    """
    Find all voxels that lie on a line between point A and B. The size of the voxels depends on the voxel dimension

    Parameters
    ----------
    point_a: Iterable[float]
        Coordinate of point A in real space
    point_b: Iterable[float]
        Coordinate of point B in real space
    vox_dim: Iterable[float]
        Dimensions of voxel size

    Returns
    -------
    line_points: np.ndarray[float]
        Array of coordinates of voxels between A and B
    """
    z1, y1, x1 = point_a
    z2, y2, x2 = point_b
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    delta_z, delta_y, delta_x = vox_dim
    az, ay, ax = (z1 + delta_z / 2) // delta_z, (y1 + delta_y / 2) // delta_y, (x1 + delta_x / 2) // delta_x
    bz, by, bx = (z2 + delta_z / 2) // delta_z, (y2 + delta_y / 2) // delta_y, (x2 + delta_x / 2) // delta_x
    vox_dif_z, vox_dif_y, vox_dif_x = abs(bz - az), abs(by - ay), abs(bx - ax)
    max_dif = int(max(vox_dif_z, vox_dif_y, vox_dif_x))
    n = max_dif
    line_points = np.zeros([max_dif + 1, 3], dtype=int)
    z, y, x = z1 + delta_z / 2, y1 + delta_y / 2, x1 + delta_x / 2
    while n >= 0:
        pz, py, px = int(z // delta_z), int(y // delta_y), int(x // delta_x)
        line_points[n, :] = (pz, py, px)
        z, y, x = z + dz / max_dif, y + dy / max_dif, x + dx / max_dif

        n -= 1
    return line_points


def length_along_normal(point: Iterable[float], centroid: Iterable[float], normal: Iterable[float]):
    """
    Project vector between point and centroid onto the normal and calculate the length along normal

    Parameters
    ----------
    point: Iterable[float]
        Coordinate of point in real space
    centroid: Iterable[float]
        Coordinate of Centroid in real space
    normal: Iterable[float]
        Vector of the normal along the centroid

    Returns
    -------
    length_along_normal: float
        Length of vector between point and centroid, projected along the normal
    """
    az, ay, ax = point
    cz, cy, cx = centroid
    vz, vy, vx = az - cz, ay - cy, ax - cx
    nz, ny, nx = normal
    return (vz * nz + vy * ny + vx * nx) / np.sqrt(nz ** 2 + ny ** 2 + nx ** 2)


def esf_along_normal(image: np.ndarray[float], centroid: Iterable[float], normal: Iterable[float],
                     width: float, dimensions: Iterable[float]):
    """
    Draw ESF along the normal through the centroid.

    This function measures the HU values and positions along the normal for all voxels that are intersected by a line
    through the centroid, starting at a width along the normal before the centroid, ending at that same width after the
    centroid

    Parameters
    ----------
    image : np.ndarray[float, float, float]
        3D CT image stack
    centroid : Iterable[float]
        Coordinate of Centroid in real space
    normal : Iterable[float]
        Vector of the normal along the centroid
    width : float
        Distance along the normal where voxels are included
    dimensions : Iterable[float]
        Dimensions of voxel size

    Returns
    -------
    positions : np.ndarray[float]
        Distances along the normal of all voxels along the line
    hu_values : np.ndarray[float]
        HU values of all voxels along the line
    """
    cz, cy, cx = centroid
    nz, ny, nx = normal
    dz, dy, dx = dimensions
    begin_point = (cz - nz * width, cy - ny * width, cx - nx * width)
    end_point = (cz + nz * width, cy + ny * width, cx + nx * width)

    voxels = bresenham_kwinten(begin_point, end_point, dimensions)
    nb_voxels = len(voxels)

    positions = np.zeros(nb_voxels, dtype=float)
    hu_values = np.zeros(nb_voxels, dtype=float)

    for index, vox in enumerate(voxels):
        k, j, i = vox
        try:
            value = image[k, j, i]
        except IndexError:
            return None, None
        if np.isnan(value):
            return None, None
        else:
            coordinate = (k * dz, j * dy, i * dx)
            # We define positions outside the body as negative, but normals are outward!
            position_along_normal = -length_along_normal(coordinate, centroid, normal)
            hu_values[index], positions[index] = value, position_along_normal
    return positions, hu_values


def voxel_closest_to_point(point, dimensions, image):
    # Finds the voxel in lattice space that is closest to a point in linear space
    # Input:  point linear space = array 3 x 1
    #         dimensions of linear space = array 3 x 1
    #         image = array M x N x L (lattice space)
    # Output: voxel in lattice space  int array 3 x 1
    maximal_coordinates = np.add(image.shape, - 1)
    closest_voxel = np.round(np.divide(point, dimensions)).astype(int)
    closest_voxel = np.maximum(closest_voxel, np.array([0, 0, 0]))
    closest_voxel = np.minimum(closest_voxel, maximal_coordinates)
    return closest_voxel.astype(int)
