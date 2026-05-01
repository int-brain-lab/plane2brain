"""a collection of linear algebra helpers, optimised with numba"""

import warnings
from typing import Tuple

import numpy as np
from numpy import linalg

import numba as nb


@nb.njit("Tuple((float64[:], float64[:]))(float64[:,:])", cache=True)
def plane_normal_form(face: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Form a plane from a face (3 points).

    Args:
        face: Array of shape (3, 3), rows are the three vertices in xyz.

    Returns:
        Tuple of (p0, n): p0 is a point on the plane, n is the unit normal.
    """

    p0, p1, p2 = face
    n = np.cross(p0 - p1, p0 - p2)
    n /= linalg.norm(n)
    return p0, n


# numba version fails with division by zero
def intersect_line_plane(
    l0: np.ndarray,
    l: np.ndarray,
    p0: np.ndarray,
    n: np.ndarray,
    warn: bool = True,
) -> np.ndarray:
    """Return the intersection point of a line and a plane in normal form.

    derivation: https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection

    point on line is p = l0 + d * l
    point on plane is (p - p0).n = 0
    substitute and solve for d
    ((l0 + d * l) - p0).n = 0

    Note:
    this function works in numpy so that a division-by-zero (parallel line/plane)
    raises a warning rather than crashing silently as in the numba version.

    Args:
        l0: Point on the line, shape (3,).
        l: Line direction vector, shape (3,).
        p0: Point on the plane, shape (3,).
        n: Plane normal vector, shape (3,).
        warn: If True, print a warning when division by zero occurs.

    Returns:
        The intersection point, shape (3,).
    """
    #

    if warn:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Catch all warnings
            d = np.dot(p0 - l0, n) / np.dot(l, n)
            if w:
                print(w, l0, l, p0, n)
            return l0 + d * l
    else:
        d = np.dot(p0 - l0, n) / np.dot(l, n)
        return l0 + d * l


@nb.njit("float64[:](float64[:],float64[:],float64[:],float64[:])", cache=True)
def intersect_line_plane_nb(
    l0: np.ndarray,
    l: np.ndarray,
    p0: np.ndarray,
    n: np.ndarray,
) -> np.ndarray:
    """Numba JIT version of intersect_line_plane().

    Unlike the NumPy version, a parallel line/plane raises ZeroDivisionError
    which cannot be caught inside Numba. Only call this when an intersection
    is guaranteed to exist.
    """
    l0 = np.ascontiguousarray(l0)
    l = np.ascontiguousarray(l)
    p0 = np.ascontiguousarray(p0)
    n = np.ascontiguousarray(n)

    # can only be called if we are sure such intersection point exists
    d = np.dot(p0 - l0, n) / np.dot(l, n)
    return l0 + d * l


@nb.njit("bool_(float64[:,:], float64[:])", cache=True)
def point_in_face(
    face: np.ndarray,
    point: np.ndarray,
) -> np.bool_:
    """Check if a point lies within a triangular face.

    3d form, barycentric coordinate based:
    https://math.stackexchange.com/questions/2582202/does-a-3d-point-lie-on-a-triangular-plane

    Args:
        face: Array of shape (3, 3), rows are the three vertices.
        point: Array of shape (3,).

    Returns:
        True if the point lies within the face.
    """

    ph = np.append(point, 1)
    A = np.ones((4, 3))
    A[:-1, :] = face.T  # numba can't deal well with concatenate
    w = linalg.pinv(A.T @ A) @ A.T @ ph
    return np.all(np.logical_and(w > 0, w < 1))


def point_in_face_np(
    face: np.ndarray,
    point: np.ndarray,
) -> np.bool_:
    """NumPy version of point_in_face()."""
    ph = np.concatenate([point, np.ones(1)])[:, np.newaxis]
    A = np.concatenate([face.T, np.ones(3)[np.newaxis, :]], axis=0)
    w = linalg.pinv(A.T @ A) @ A.T @ ph
    return np.all(np.logical_and(w > 0, w < 1))


def intersect_line_mesh_np(
    vertices: np.ndarray,
    edges: np.ndarray,
    line_point: np.ndarray,
    line_vector: np.ndarray,
    numba: bool = False,
    exclude: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the intersection of a line with a triangle mesh.

    Args:
        vertices: Mesh vertices, shape (V, 3).
        edges: Triangle index array, shape (F, 3), indexing into `vertices`.
        line_point: A point on the line, shape (3,).
        line_vector: Line direction vector, shape (3,).
        numba: If True, use the Numba-accelerated plane intersection.
        exclude: If True, skip faces with a horizontal normal (normal[2] == 0).

    Returns:
        Tuple of (faces, intersection_points, indices):
            faces: Intersected triangle vertices, shape (N, 3, 3).
            intersection_points: Intersection coordinates, shape (N, 3).
            indices: Indices into `edges` of the intersected faces, shape (N,).
    """
    # collect intersected faces, their intersection points, and their indices
    ix = []
    faces = []
    intersection_points = []
    for i in range(edges.shape[0]):
        face = vertices[edges[i]]
        plane_point, plane_normal = plane_normal_form(face)
        if exclude and plane_normal[2] == 0:
            # this excludes triangles from the mesh that can not be intersected
            continue
        if numba:
            func = intersect_line_plane_nb
        else:
            func = intersect_line_plane

        intersection_point = func(line_point, line_vector, plane_point, plane_normal)
        if point_in_face(face, intersection_point):
            intersection_points.append(intersection_point)
            faces.append(face)
            ix.append(i)
    return (
        np.array(faces).astype("float64"),
        np.array(intersection_points).astype("float64"),
        np.array(ix).astype("uint64"),
    )


@nb.njit("float64(float64[:],float64[:])", cache=True)
def get_angle(a, b):
    # the angle between two vectors a and b
    # to address the performance warnings: ensure contiguous arrays
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@nb.njit(
    "Tuple((float64[:,:,:], float64[:,:], int64[:]))(float64[:,:], int32[:,:], float64[:], float64[:])",
    parallel=True,
    cache=True,
)
def intersect_line_mesh_nb(
    vertices: np.ndarray,
    edges: np.ndarray,
    line_point: np.ndarray,
    line_vector: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """see intersect_line_mesh_np"""
    N = edges.shape[0]
    vertices_to_check = np.zeros(N, dtype="bool")

    for i in nb.prange(N):
        face = vertices[edges[i]]
        plane_point, plane_normal = plane_normal_form(face)
        # if plane normal and line vector are perpendicular, the line is parallel
        # to the plane and there is no intersection — skip those faces
        tol = 1e-5
        alpha = get_angle(plane_normal, line_vector)
        if np.abs((np.abs(alpha) - np.pi / 2)) > tol:
            face = vertices[edges[i]]
            plane_point, plane_normal = plane_normal_form(face)
            intersection_point = intersect_line_plane_nb(
                line_point, line_vector, plane_point, plane_normal
            )
            if point_in_face(face, intersection_point):
                vertices_to_check[i] = True

    ix = np.where(vertices_to_check)[0]
    intersection_points = np.zeros((ix.shape[0], 3), dtype="float64")
    faces = np.zeros((ix.shape[0], 3, 3), dtype="float64")
    for j in nb.prange(ix.shape[0]):
        i = ix[j]
        face = vertices[edges[i]]
        plane_point, plane_normal = plane_normal_form(face)
        intersection_points[j] = intersect_line_plane_nb(
            line_point, line_vector, plane_point, plane_normal
        )
        faces[j] = face

    return (
        faces,
        intersection_points,
        ix,
    )


# @nb.njit("Tuple((float64[:], int64))(float64[:,:,:],float64[:])")
def get_closest_face(
    faces: np.ndarray,
    point: np.ndarray,
) -> Tuple[np.ndarray, np.intp]:
    """Find the face closest to a point by comparing face centroid distances.

    Args:
        faces: Array of shape (N, 3, 3) for N triangular faces.
        point: Query point, shape (3,).

    Returns:
        Tuple of (closest face, its index in `faces`).
    """
    # calculate distance to all and get minimum
    dists = np.zeros(faces.shape[0], dtype="float64")
    for i in range(faces.shape[0]):
        dists[i] = linalg.norm(point - np.average(faces[i], axis=0))

    min_ix = np.argmin(dists)
    return faces[min_ix], min_ix


# numpy version
def find_closest_point_from_line_np(
    points: np.ndarray,
    l0: np.ndarray,
    l: np.ndarray,
) -> np.ndarray:
    """NumPy version of find_closest_point_from_line_nb()."""
    ds = linalg.norm((l0 - points) - np.dot(l0 - points, l)[:, np.newaxis] * l, axis=1)
    point = points[np.argmin(ds)]
    return point


# numba compatible version
@nb.njit("float64[:](float64[:,:], float64[:], float64[:])", cache=True)
def find_closest_point_from_line_nb(
    points: np.ndarray,
    l0: np.ndarray,
    l: np.ndarray,
) -> np.ndarray:
    """Return the point closest to a line from a set of candidate points.

    Args:
        points: Candidate points, shape (N, 3).
        l0: A point on the line, shape (3,).
        l: Line direction vector, shape (3,).

    Returns:
        The closest point, shape (3,).
    """
    l0 = np.ascontiguousarray(l0)
    l = np.ascontiguousarray(l)
    vs = (l0 - points) - np.dot(l0 - points, l)[:, np.newaxis] * l
    ds = np.array([linalg.norm(v) for v in vs])
    point = points[np.argmin(ds)]
    return point


@nb.njit(
    "float64[:,:](float64[:,:],float64[:,:],float64[:])", parallel=True, cache=True
)
def find_closest_points_on_surface(
    points_eval: np.ndarray,
    brain_surface_points: np.ndarray,
    n: np.ndarray,
) -> np.ndarray:
    # TODO this needs heavy refactoring
    # rename into: find_closest_point_from_lines
    # as this is essentially a parallelization wrapper find_closest_point_from_line_nb
    # change the call signature accordingly
    N = points_eval.shape[0]
    points_closest = np.zeros((N, 3))
    for i in nb.prange(N):
        points_closest[i, :] = find_closest_point_from_line_nb(
            brain_surface_points, points_eval[i], n
        )
    return points_closest


def get_rotation_between_vectors(
    a: np.ndarray,
    b: np.ndarray,
    as_affine: bool = True,
) -> np.ndarray:
    # returns the (3,3) transform or (4,4)
    # https://math.stackexchange.com/a/2470436

    # extend this to constrain one axis
    i = a
    ip = b
    j = np.cross(i, ip) / linalg.norm(np.cross(i, ip))
    jp = j
    k = np.cross(i, j)
    kp = np.cross(ip, jp)

    R = np.stack([ip, jp, kp], axis=1) @ np.stack([i, j, k], axis=1).T
    if as_affine:
        R_ = np.zeros((4, 4))
        R_[-1, -1] = 1
        R_[:3, :3] = R
        return R_
    else:
        return R


def get_vector_angles(
    v: np.ndarray,
    in_radians: bool = True,
) -> np.ndarray:
    # follows IBL conventions?
    # TODO verify the angles, again ...

    ml, ap, dv = v
    # theta is the angle for rotation around the ml axis = in plane in (ap, dv)
    # == pitch

    a = np.array([ap, dv])
    b = np.array([0, 1])
    theta = np.arccos(np.dot(a, b) / (linalg.norm(a) * linalg.norm(b)))

    # phi is the angle for rotation around the dv axis = in plane in (ml, ap)
    # == yaw
    a = np.array([ml, ap])
    b = np.array([1, 0])
    phi = np.arccos(np.dot(a, b) / (linalg.norm(a) * linalg.norm(b)))
    # beta is the angle for rotation in AP axis = in plane in (ml, dv)
    # == roll
    # not clearly defined in the IBL image (extent)

    a = np.array([ml, dv])
    b = np.array([0, 1])
    beta = np.arccos(np.dot(a, b) / (linalg.norm(a) * linalg.norm(b)))

    angles = np.array([phi, theta, beta])
    if in_radians:
        return angles
    else:
        return angles * 360 / (2 * np.pi)
