"""a collection of linear algebra helpers, optimised with numba"""

import warnings

import numba as nb
import numpy as np
from numpy import linalg


@nb.njit("Tuple((float64[:], float64[:]))(float64[:,:])", cache=True)
def plane_normal_form(face: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


# relative degeneracy threshold: det / (d11 * d22) is sin² of the angle between
# the two edge vectors, so this rejects slivers scale-independently
DEGENERATE_FACE_TOL = 1e-12


@nb.njit("bool_(float64[:,:], float64[:])", cache=True)
def point_in_face_barycentric(
    face: np.ndarray,
    point: np.ndarray,
) -> np.bool_:
    """Check if a point lies within a triangular face, without a matrix solve.

    Equivalent to point_in_face(), but for a point already known to lie in the
    plane of the face — as is the case for a line-plane intersection point.
    That makes the 4x3 system point_in_face() fits by pseudo-inverse consistent
    and exactly solvable, so it reduces to a 2x2 solve.

    With the first vertex as origin and the two edge vectors
    e1 = v1 - v0, e2 = v2 - v0, an in-plane point satisfies
    point - v0 = u * e1 + v * e2. Dotting with e1 and e2 gives the 2x2 system

        [e1.e1  e1.e2] [u]   [b.e1]
        [e1.e2  e2.e2] [v] = [b.e2]

    solved here by Cramer's rule. The barycentric weights are (1 - u - v, u, v),
    so the strict interior test all(0 < w < 1) is equivalent to
    u > 0, v > 0, u + v < 1.

    Args:
        face: Array of shape (3, 3), rows are the three vertices.
        point: Array of shape (3,), assumed to lie in the plane of the face.

    Returns:
        True if the point lies strictly within the face. Degenerate
        (zero-area) faces always return False.
    """
    # written out in scalars: the small array temporaries of the vectorised
    # form cost ~9x more than the arithmetic in this innermost kernel
    v0x, v0y, v0z = face[0, 0], face[0, 1], face[0, 2]
    e1x, e1y, e1z = face[1, 0] - v0x, face[1, 1] - v0y, face[1, 2] - v0z
    e2x, e2y, e2z = face[2, 0] - v0x, face[2, 1] - v0y, face[2, 2] - v0z
    bx, by, bz = point[0] - v0x, point[1] - v0y, point[2] - v0z

    # the Gram matrix of the two edge vectors
    d11 = e1x * e1x + e1y * e1y + e1z * e1z
    d12 = e1x * e2x + e1y * e2y + e1z * e2z
    d22 = e2x * e2x + e2y * e2y + e2z * e2z

    # by Lagrange's identity det == |e1 x e2|² == (2 * area)², so this is
    # positive for any non-degenerate triangle
    det = d11 * d22 - d12 * d12
    if det <= DEGENERATE_FACE_TOL * d11 * d22:
        return False

    b1 = bx * e1x + by * e1y + bz * e1z
    b2 = bx * e2x + by * e2y + bz * e2z
    u = (d22 * b1 - d12 * b2) / det
    v = (d11 * b2 - d12 * b1) / det
    return u > 0.0 and v > 0.0 and u + v < 1.0


def intersect_line_mesh_np(
    vertices: np.ndarray,
    edges: np.ndarray,
    line_point: np.ndarray,
    line_vector: np.ndarray,
    numba: bool = False,
    exclude: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        if np.abs(np.abs(alpha) - np.pi / 2) > tol:
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


@nb.njit(
    "Tuple((float64[:,:,:], float64[:,:], int64[:]))(float64[:,:], int32[:,:], float64[:,:], float64[:], float64[:])",
    parallel=True,
    cache=True,
)
def intersect_line_mesh_precomputed_nb(
    vertices: np.ndarray,
    edges: np.ndarray,
    normals: np.ndarray,
    line_point: np.ndarray,
    line_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Variant of intersect_line_mesh_nb() taking precomputed face normals.

    The fast path, equivalent to intersect_line_mesh_nb() but ~450x faster on a
    large mesh. Three things differ: the face normals are precomputed rather than
    rebuilt on every call, the interior test is the 2x2 solve derived in
    point_in_face_barycentric() rather than the pseudo-inverse of point_in_face(),
    and the candidate loop is written out in scalars.

    The scalar form is what most of the speedup comes from: materialising the face
    and the intersection point as arrays per candidate costs several times the
    arithmetic itself, and every face paid for it — even the ones rejected as
    parallel. Working in scalars also lets the intersection reuse the dot product
    computed for the parallel test. The helper functions remain the readable
    reference for the maths inlined here, and are what the tests check against.

    Args:
        vertices: Mesh vertices, shape (V, 3).
        edges: Triangle index array, shape (F, 3), indexing into `vertices`.
        normals: Unit face normals, shape (F, 3), as produced by
            ProjectionAtlas.precompute_normals().
        line_point: A point on the line, shape (3,).
        line_vector: Line direction vector, shape (3,).

    Returns:
        Tuple of (faces, intersection_points, indices), see intersect_line_mesh_np.
    """
    N = edges.shape[0]
    faces_to_check = np.zeros(N, dtype="bool")

    # loop invariants, hoisted out of the per-face test below
    ox, oy, oz = line_point[0], line_point[1], line_point[2]
    lx, ly, lz = line_vector[0], line_vector[1], line_vector[2]
    # the parallel-face test skips faces whose normal is within tol (in radians)
    # of perpendicular to the line. for the angle alpha between normal and line,
    # |alpha - pi/2| > tol is equivalent to |cos(alpha)| > sin(tol), hence to
    # |n.l| > sin(tol) * |n| * |l| — no arccos, and with unit normals the only
    # norm left is that of the line vector, which does not vary per face
    tol = 1e-5
    tol_dot = np.sin(tol) * np.sqrt(lx * lx + ly * ly + lz * lz)

    for i in nb.prange(N):
        nx, ny, nz = normals[i, 0], normals[i, 1], normals[i, 2]
        # if plane normal and line vector are perpendicular, the line is parallel
        # to the plane and there is no intersection — skip those faces
        denominator = nx * lx + ny * ly + nz * lz
        if np.abs(denominator) <= tol_dot:
            continue

        # plane_normal_form() uses the first vertex as the point on the plane
        i0, i1, i2 = edges[i, 0], edges[i, 1], edges[i, 2]
        v0x, v0y, v0z = vertices[i0, 0], vertices[i0, 1], vertices[i0, 2]

        # the line-plane intersection point, reusing l.n as the denominator
        d = ((v0x - ox) * nx + (v0y - oy) * ny + (v0z - oz) * nz) / denominator
        px, py, pz = ox + d * lx, oy + d * ly, oz + d * lz

        # the interior test of point_in_face_barycentric(), inlined: solve
        # p - v0 = u * e1 + v * e2 in the basis of the two edge vectors
        e1x, e1y, e1z = (
            vertices[i1, 0] - v0x,
            vertices[i1, 1] - v0y,
            vertices[i1, 2] - v0z,
        )
        e2x, e2y, e2z = (
            vertices[i2, 0] - v0x,
            vertices[i2, 1] - v0y,
            vertices[i2, 2] - v0z,
        )
        bx, by, bz = px - v0x, py - v0y, pz - v0z

        d11 = e1x * e1x + e1y * e1y + e1z * e1z
        d12 = e1x * e2x + e1y * e2y + e1z * e2z
        d22 = e2x * e2x + e2y * e2y + e2z * e2z
        det = d11 * d22 - d12 * d12
        if det <= DEGENERATE_FACE_TOL * d11 * d22:
            continue

        b1 = bx * e1x + by * e1y + bz * e1z
        b2 = bx * e2x + by * e2y + bz * e2z
        u = (d22 * b1 - d12 * b2) / det
        v = (d11 * b2 - d12 * b1) / det
        if u > 0.0 and v > 0.0 and u + v < 1.0:
            faces_to_check[i] = True

    ix = np.where(faces_to_check)[0]
    intersection_points = np.zeros((ix.shape[0], 3), dtype="float64")
    faces = np.zeros((ix.shape[0], 3, 3), dtype="float64")
    # only the handful of intersected faces are gathered here, so this pass stays
    # in the readable array form. recomputing the intersection point is cheaper
    # than carrying a (F, 3) buffer for it through the loop above
    for j in nb.prange(ix.shape[0]):
        i = ix[j]
        face = vertices[edges[i]]
        intersection_points[j] = intersect_line_plane_nb(
            line_point, line_vector, face[0], normals[i]
        )
        faces[j] = face

    return (
        faces,
        intersection_points,
        ix,
    )


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
