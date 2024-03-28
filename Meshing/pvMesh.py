import numpy as np
import pyacvd
import pymeshfix
import pyvista as pv
from PIL import ImageColor
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from pytictoc import TicToc
from skimage import measure

from NewResolution.VoxelsOnLine import voxel_closest_to_point


def faces_for_pyvista(faces: np.ndarray[int]):
    r"""Create a face array for PyVista

    Convert traditional face data (3 X N) into a single face array for PyVista

    Parameters
    ----------
    faces : np.ndarray[int]
        Array (3 x N) with vertices corresponding to each face

    Returns
    -------
    pv_faces : np.ndarray[int]
        Array (4N) with vertices corresponding to each face with an additional number indicating amount of vertices
        per face. E.g. [1, 0, 2] ---> [3, 1, 0, 2]
    """
    m, n = faces.shape
    pv_faces = np.zeros(m * (n + 1), dtype=int)
    for i, face in enumerate(faces):
        pv_faces[i * (n + 1)] = n
        pv_faces[i * (n + 1) + 1:(i + 1) * (n + 1)] = face
    return pv_faces


def create_mesh(mask: np.ndarray[int], voxel_dimension: np.ndarray[int], step_size: int, border=5):
    r"""Create a mesh from a binary image stack

    Convert a binary image stack (m x n x o) into a PyVista Polydata Mesh.

    Parameters
    ----------
    mask : np.ndarray[int]
        Binary array (m x n x o)

    voxel_dimension: np.ndarray[int]
        Dimensions of the voxels (z, y, x)

    step_size: int
        Coarseness of the marching cubes algorithm

    border: int
        Amount of empty space added to the mask to ensure the boundary is also meshed

    Returns
    -------
    mesh : PolyData
        PyVista PolyData object with all mesh info
    """
    t = TicToc()
    t.tic()
    z, y, x = mask.shape
    if border > 0:
        extended_mask = np.zeros([z + 2 * border, y + 2 * border, x + 2 * border])
        extended_mask[border:z + border, border:y + border, border:x + border] = mask
    else:
        extended_mask = mask
    t.toc('Create extended mask', restart=True)
    vertices, faces, _, _ = measure.marching_cubes(extended_mask, gradient_direction='ascent', level=0, step_size=step_size)
    vertices -= border
    t.toc('Marching Cubes', restart=True)
    for i, ds in enumerate(voxel_dimension):
        vertices[:, i] *= ds
    vertices, faces = pymeshfix.clean_from_arrays(vertices, faces)
    t.toc('Clean faces and vertices', restart=True)
    faces = faces_for_pyvista(faces)
    t.toc('Pyvista faces')
    return pv.PolyData(vertices, faces)


def magnitude(vector: np.ndarray[float]):
    r"""Calculate vector magnitude

    Calculate length of a 3D vector

    Parameters
    ----------
    vector : np.ndarray[float]
        3D vector

    Returns
    -------
    magnitude : float
        Length of the 3D vector
    """
    return np.linalg.norm(vector)


def distance_to_center(point, isocenter):
    r"""Calculate length between 3D point and 2D center
    Parameters
    ----------
    point : np.ndarray[float]
        3D coordinate
    isocenter: np.ndarray[float]
        2D center coordinate
    Returns
    -------
    length : float
        Length of the 3D vector
        """
    return magnitude(point[1:] - isocenter)


def angle_between(v1: np.ndarray[float], v2: np.ndarray[float]):
    r"""Calculate the angle between two 3D vectors
    Parameters
    ----------
    v1 : np.ndarray[float]
        3D vector
    v2 : np.ndarray[float]
        3D vector
    Returns
    -------
    angle : float
        Angle between vector v1 and v2 in degrees
    """
    return np.arccos(np.clip(np.dot(v1, v2) / (magnitude(v1) * magnitude(v2)), -1, 1)) / np.pi * 180


def centroid(face: np.ndarray[int], vertices: np.ndarray[float]):
    r"""Calculate the centroid of a 3D triangle
    Parameters
    ----------
    face : np.ndarray[int]
        Set of vertices that make up the triangle
    vertices : np.ndarray[float]
        Array with the physical coordinates of all vertices
    Returns
    -------
    centroid : np.ndarray[float]
        3D coordinate of the centroid
    """
    a, b, c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    return (a + b + c) / 3


def inside_fov(point: np.ndarray[float], dimensions: np.ndarray[float], image: np.ndarray[float]):
    r"""Check if a point in 3D lies within the image FOV
    Parameters
    ----------
    point : np.ndarray[float]
        Coordinate in 3D
    dimensions : np.ndarray[float]
        Dimensions of the image voxels
    image : np.ndarray[float]
        CT-image
    Returns
    -------
     boolean

    """
    voxel = tuple(voxel_closest_to_point(point, dimensions, image))
    return not np.isnan(image[voxel])


def select_in_plane_faces(faces: np.ndarray[int], angles: np.ndarray[float], normals: np.ndarray[float],
                          centroids: np.ndarray[float], voxel_dimension: np.ndarray[float], image: np.ndarray[float],
                          ref_angle=90, max_xy_deviation=10):
    r"""
    Find all faces that can be used for the MTF calculations.

    Compute all faces where the normal deviates less than a specific amount to the reference angle. Also, the centroid
    of said face has to lie within the FOV of the image
    Parameters
    ----------
    faces : np.ndarray[int]
        Set of vertices that make up the triangles
    angles : np.ndarray[float]
        Angle of the normal to the z-direction of each face
    normals : np.ndarray[float]
        Normal of each face
    centroids : np.ndarray[float]
        Centroid of each face
    voxel_dimension : np.ndarray[float]
        Dimensions of the image voxels
    image : np.ndarray[float]
        CT-image
    ref_angle: float
        Angle to which to compare the angles
    max_xy_deviation: float
        Max allowed between the angles and the reference angle
    Returns
    -------
    tuple(np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float])
    """
    centroids_in_fov = np.array(list(map(lambda c: inside_fov(c, voxel_dimension, image), centroids)))
    select = (angles <= ref_angle + max_xy_deviation) & (angles >= ref_angle - max_xy_deviation) & centroids_in_fov
    select_faces, select_angles, select_normals = faces[select], angles[select], normals[select]
    select_centroids = centroids[select]
    return select_faces, select_angles, select_normals, select_centroids, select


def plot_meshes(mesh, color='white', arrows=False, selected=False, scalars=None, show_scalar_bar=False, cmap=None,
                yes_color=None, no_color=None, arrow_color=None, condition_yes=None, radial_bin=10, save=None, zoom=1.4,
                **mesh_kwargs):
    m = mesh.mesh
    if save and type(save) == str and '.png' in save:
        off_screen = True
    else:
        off_screen = False
    pl = pv.Plotter(off_screen=off_screen)
    arrow_points = mesh.centroids
    arrow_normals = mesh.normals
    max_bin = 0

    if yes_color:
        yes = yes_color
    else:
        yes = '#ace6aa'
    if no_color:
        no = no_color
    else:
        no = 'white'

    if selected:
        # Define the colors we want to use
        arrow_points = mesh.select_centroids
        arrow_normals = mesh.select_normals

    if scalars == 'Radial':
        max_bin = int(np.max(np.ceil(mesh.radial_distance / radial_bin)))
        color_list = ['#fc03f8', '#8803fc', '#031cfc', '#03d7fc', '#03fc45', '#fcf003', '#fca103', '#fc1803']
        cmap = LinearSegmentedColormap.from_list('mappi', color_list, N=max_bin)
        show_scalar_bar = False
    if condition_yes and type(condition_yes) == str:
        mapping = np.linspace(m[scalars].min(), m[scalars].max(), 256)
        symbols = {'x': mapping}
        color_map = np.empty((256, 4))
        color_map[:] = np.append(np.array(ImageColor.getcolor(no, 'RGB')) / 256, 1)
        color_map[eval(condition_yes, symbols)] = np.append(np.array(ImageColor.getcolor(yes, 'RGB')) / 256, 1)
        cmap = ListedColormap(color_map)

    pl.add_mesh(m, scalars=scalars, color=color, cmap=cmap, show_scalar_bar=show_scalar_bar, **mesh_kwargs)
    if scalars == 'Radial':
        pl.add_scalar_bar(n_labels=max_bin + 1, label_font_size=10)
        pl.update_scalar_bar_range([0, max_bin * radial_bin])

    if arrows:
        if not arrow_color:
            arrow_color = 'k'
        pl.add_arrows(arrow_points, arrow_normals, mag=4, line_width=0.1, color=arrow_color)

    pl.view_vector([-1, -3, 1], [-1, 0, 0])
    pl.zoom_camera(value=zoom)
    if save and type(save) == str and '.png' in save:
        pl.screenshot(save)
    else:
        pl.show()


class Mesh:

    def __init__(self,
                 image_stack,
                 step_size=3,
                 pass_band=0.1,
                 max_xy_deviation=10,
                 max_xy_division_angle=10,
                 reference_vector=np.array([1, 0, 0]),
                 smooth=True,
                 equilateral=True,
                 import_mesh=None):
        self.stack = image_stack
        self.step_size = step_size
        self.pass_band = pass_band
        self.max_xy_deviation = max_xy_deviation
        self.max_xy_division_angle = max_xy_division_angle
        self.reference_vector = reference_vector
        self.valid = True
        self.smooth = smooth
        self.equilateral = equilateral
        self.import_mesh = import_mesh
        self._initialize_parameters()
        self._check_validness()
        self._initialise_mesh()
        self._smooth_mesh()
        self._make_equilateral_mesh()
        self._set_parameters()

    def _check_validness(self):
        self._check_is_array()
        self._check_is_binary()
        self._check_numbers()

    def _check_is_array(self):
        self.valid = self.valid and isinstance(self.mask, np.ndarray)

    def _check_is_binary(self):
        self.valid = self.valid and self.mask.all() in [0, 1]

    def _check_numbers(self):
        self.valid = self.valid and (isinstance(self.SliceThickness, float) or isinstance(self.SliceThickness, int))
        self.valid = self.valid and (isinstance(self.PixelSize, float) or isinstance(self.PixelSize, int))
        self.valid = self.valid and isinstance(self.step_size, int)
        self.valid = self.valid and (isinstance(self.pass_band, float) or isinstance(self.pass_band, int))
        self.valid = self.valid and (isinstance(self.max_xy_deviation, float) or isinstance(self.max_xy_deviation, int))

    def _initialize_parameters(self):
        try:
            self.mask = self.stack.mask_stack
            self.SliceThickness = self.stack.SliceThickness
            self.PixelSize = self.stack.PixelSize
            self.voxel_dimension = self.stack.dimensions
            self.hu = self.stack.raw_hu_stack
            self.z, self.y, self.x = np.multiply(self.mask.shape, self.voxel_dimension)
            self.center = np.array([np.floor(self.y / 2).astype(int), np.floor(self.x / 2).astype(int)])
            self.center = np.array([self.y / 2, self.x / 2])
            self.ReconstructionDiameter = self.stack.ReconstructionDiameter
        except AttributeError:
            self.mask = None
            self.SliceThickness = None
            self.PixelSize = None
            self.voxel_dimension = None
            self.ReconstructionDiameter = None
            self.valid = False

    def _initialise_mesh(self):
        if self.valid:
            if self.import_mesh and type(self.import_mesh) == str:
                try:
                    self.mesh = pv.read(self.import_mesh)
                except FileNotFoundError:
                    self.import_mesh = None
                    self._initialise_mesh()
            else:
                try:
                    self.mesh = create_mesh(self.mask, step_size=self.step_size, voxel_dimension=self.voxel_dimension)
                except AttributeError:
                    self.mesh = None
                    self.valid = False

    def _smooth_mesh(self):
        if self.valid and self.smooth:
            try:
                # First Taubin filtering will smooth the mesh while keeping the volume constant
                self.mesh = self.mesh.smooth_taubin(pass_band=self.pass_band)
            except AttributeError:
                self.mesh = None
                self.valid = False

    def _make_equilateral_mesh(self):
        if self.valid and self.equilateral:
            try:
                # Clustering of the mesh allows to re-mesh to more equilateral triangles
                cluster = pyacvd.Clustering(self.mesh)
                # Each triangle is divided in 4 with each iteration of subdivide
                cluster.subdivide(nsub=2)
                # n clus is the amount of vertices you want in the end. In order to have the same amount as before, you
                # need more vertices for the clustering re-mesh to work. Therefore, we subdivide the triangles in
                # two iterations. Gives in total 16 times the original faces
                cluster.cluster(nclus=self.mesh.n_points)
                self.mesh = cluster.create_mesh()
            except AttributeError:
                self.valid = False

    def _set_parameters(self):
        try:
            self.vertices = self.mesh.points
            self.faces = self.mesh.regular_faces
            self.normals = self.mesh.face_normals
            self.angles = np.array(list(map(lambda normal: angle_between(normal, self.reference_vector), self.normals)))
            self.mesh['Angles'] = self.angles
            self.centroids = np.array(list(map(lambda f: centroid(f, self.vertices), self.faces)))
            self.radial_distance = np.array(list(map(lambda c: distance_to_center(c, self.stack.ISOCENTER[1:]), self.centroids)))
            self.mesh['Radial'] = self.radial_distance
            self.reference_angle = 90
            self.select_faces, self.select_angles, self.select_normals, self.select_centroids, self.selected = \
                select_in_plane_faces(self.faces, self.angles, self.normals, self.centroids, self.voxel_dimension,
                                      self.hu, self.reference_angle, self.max_xy_deviation)
            self.mesh['Selected'] = self.selected * 1
            self.select_radial = self.radial_distance[self.selected]
            # self.faces_x, self.angles_x, self.normals_x, self.centers_x, self.radii_x, \
            #     self.faces_y, self.angles_y, self.normals_y, self.centers_y, self.radii_y = \
            #     split_faces_xy(self.select_faces, self.select_angles, self.select_normals,
            #                    self.select_circum_centers, self.select_radii,
            #                    max_angle=self.max_xy_division_angle)
        except AttributeError:
            self.mesh = None
            self.vertices = None
            self.faces = None
            self.normals = None
            self.angles = None
            self.centroids = None
            self.select_faces = None
            self.select_angles = None
            self.select_normals = None
            self.select_centroids = None
            self.selected = None
            self.valid = False

    def plot(self, color='white', show_normals=False, select_normals=False, scalars=None, show_scalar_bar=False,
             cmap=None, select_color=None, unselect_color=None, normal_color=None, condition_yes=None, radial_bin=10,
             save=None, zoom=1.4, **mesh_kwargs):
        r"""
        Plot the current Mesh object

        Parameters
        ----------
        save : str
            Filename how to save the view of this current mesh
        color: str
            Color of the mesh faces HEX or name
        show_normals : bool
            Plot the face normals as arrows
        select_normals : bool
            Plot only the normals of the faces that are useful for MTF measurements
        normal_color: str
            Color of the normals HEX or name
        select_color : str
            Color of selected faces HEX or name
        unselect_color : str
            Color of unselected faces HEX or name
        scalars : str
            Scalar value of the face you want plotted as color. Possible: Angles, Radial, Selected
        cmap: str
            Cmap used for scalar values
        condition_yes: str
            If scalars are indicated, the condition will only color the faces where the scalar value satisfies the
            condition. Conditions are defined as with variable x.  E.g. x==1, (x>=100) & (x< 140)...
        show_scalar_bar: bool
            Plot a scalar bar
        radial_bin: float
            If scalars == Radial, radial bin defines the width of the bin used for colormap
        zoom: float
            Zoom in amount
        Returns
        -------

        """
        mesh = self.mesh
        if save and type(save) == str and '.png' in save:
            off_screen = True
        else:
            off_screen = False
        pl = pv.Plotter(off_screen=off_screen)
        arrow_points = self.centroids
        arrow_normals = self.normals
        max_bin = 0

        if select_color:
            yes = select_color
        else:
            yes = '#ace6aa'
        if unselect_color:
            no = unselect_color
        else:
            no = 'white'

        if select_normals:
            # Define the colors we want to use
            arrow_points = self.select_centroids
            arrow_normals = self.select_normals

        if scalars == 'Radial':
            max_bin = int(np.max(np.ceil(self.radial_distance / radial_bin)))
            color_list = ['#fc03f8', '#8803fc', '#031cfc', '#03d7fc', '#03fc45', '#fcf003', '#fca103', '#fc1803']
            cmap = LinearSegmentedColormap.from_list('mappi', color_list, N=max_bin)
            show_scalar_bar = False
        if condition_yes and type(condition_yes) == str and scalars:
            mapping = np.linspace(mesh[scalars].min(), mesh[scalars].max(), 256)
            symbols = {'x': mapping}
            color_map = np.empty((256, 4))
            color_map[:] = np.append(np.array(ImageColor.getcolor(no, 'RGB')) / 256, 1)
            color_map[eval(condition_yes, symbols)] = np.append(np.array(ImageColor.getcolor(yes, 'RGB')) / 256, 1)
            cmap = ListedColormap(color_map)

        pl.add_mesh(mesh, scalars=scalars, color=color, cmap=cmap, show_scalar_bar=show_scalar_bar, **mesh_kwargs)
        if scalars == 'Radial':
            pl.add_scalar_bar(n_labels=max_bin + 1, label_font_size=10)
            pl.update_scalar_bar_range([0, max_bin * radial_bin])

        if show_normals:
            if not normal_color:
                normal_color = 'k'
            pl.add_arrows(arrow_points, arrow_normals, mag=4, line_width=0.1, color=normal_color)

        pl.view_vector([-1, -3, 1], [-1, 0, 0])
        pl.zoom_camera(value=zoom)
        if save and type(save) == str and '.png' in save:
            pl.screenshot(save)
        else:
            pl.show()

    def to_stl(self, location, file_format='stl'):
        file_location = location.split('.')[0] + '.%s' % file_format
        try:
            pv.save_meshio(file_location, self.mesh, file_format=file_format)
        except FileNotFoundError:
            pass

