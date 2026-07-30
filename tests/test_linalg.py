import unittest

import numpy as np
import numpy.testing as nptest
from scipy.spatial import ConvexHull

from plane2brain.linalg import (
    find_closest_point_from_line_nb,
    find_closest_point_from_line_np,
    get_angle,
    get_rotation_between_vectors,
    intersect_line_mesh_nb,
    intersect_line_mesh_np,
    intersect_line_mesh_precomputed_nb,
    intersect_line_plane,
    intersect_line_plane_nb,
    plane_normal_form,
    point_in_face,
    point_in_face_barycentric,
)


def _unit_cube_mesh():
    # 8 vertices of the unit cube [-1, 1]^3, triangulated into 12 faces
    # (2 per cube face). edges is int32 because intersect_line_mesh_nb's numba
    # signature pins that dtype.
    vertices = np.array(
        [
            [-1.0, -1.0, -1.0],  # 0
            [1.0, -1.0, -1.0],  # 1
            [1.0, 1.0, -1.0],  # 2
            [-1.0, 1.0, -1.0],  # 3
            [-1.0, -1.0, 1.0],  # 4
            [1.0, -1.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [-1.0, 1.0, 1.0],  # 7
        ]
    )
    edges = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # bottom z = -1
            [4, 5, 6],
            [4, 6, 7],  # top    z = +1
            [0, 1, 5],
            [0, 5, 4],  # front  y = -1
            [2, 3, 7],
            [2, 7, 6],  # back   y = +1
            [0, 3, 7],
            [0, 7, 4],  # left   x = -1
            [1, 2, 6],
            [1, 6, 5],  # right  x = +1
        ],
        dtype=np.int32,
    )
    return vertices, edges


def _face_normals(vertices, edges):
    # mirrors ProjectionAtlas.precompute_normals()
    faces = vertices[edges]
    normals = np.cross(faces[:, 0] - faces[:, 1], faces[:, 0] - faces[:, 2])
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


class TestPlaneNormalForm(unittest.TestCase):
    def test_known_triangle_returns_first_vertex_and_unit_normal(self):
        # xy-plane triangle: normal should be along ±z with unit length
        face = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        p0, n = plane_normal_form(face)
        nptest.assert_array_almost_equal(p0, face[0])
        self.assertAlmostEqual(abs(n[2]), 1.0)
        self.assertAlmostEqual(np.linalg.norm(n), 1.0)


class TestIntersectLinePlane(unittest.TestCase):
    def test_golden_case(self):
        # z-axis line piercing the z = 5 plane -> intersection at (0, 0, 5)
        l0 = np.array([0.0, 0.0, 0.0])
        l = np.array([0.0, 0.0, 1.0])
        p0 = np.array([0.0, 0.0, 5.0])
        n = np.array([0.0, 0.0, 1.0])
        nptest.assert_array_almost_equal(
            intersect_line_plane(l0, l, p0, n, warn=False), [0.0, 0.0, 5.0]
        )

    def test_np_and_nb_agree(self):
        l0 = np.array([1.0, 2.0, 0.0])
        l = np.array([0.0, 0.0, 1.0])
        p0 = np.array([0.0, 0.0, 5.0])
        n = np.array([0.0, 0.0, 1.0])
        nptest.assert_array_almost_equal(
            intersect_line_plane(l0, l, p0, n, warn=False),
            intersect_line_plane_nb(l0, l, p0, n),
        )


class TestPointInFace(unittest.TestCase):
    def test_centroid_is_inside(self):
        face = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        centroid = face.mean(axis=0)
        self.assertTrue(bool(point_in_face(face, centroid)))

    def test_vertex_is_not_inside(self):
        # point_in_face uses strict 0 < w < 1; a vertex has barycentric weight 1
        # on itself and 0 on the others, so it returns False — boundary points
        # are not "inside"
        face = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertFalse(bool(point_in_face(face, face[0])))


class TestPointInFaceBarycentric(unittest.TestCase):
    def test_centroid_and_vertex_match_reference_semantics(self):
        face = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertTrue(bool(point_in_face_barycentric(face, face.mean(axis=0))))
        self.assertFalse(bool(point_in_face_barycentric(face, face[0])))

    def test_agrees_with_pinv_version_on_random_in_plane_points(self):
        # the barycentric version is only equivalent for points in the plane of
        # the face, so points are built from random barycentric weights.
        # the weight range straddles 0 and 1 so roughly half the points fall
        # outside the triangle
        rng = np.random.default_rng(0)
        n_cases = 2000
        faces = rng.uniform(-5000, 5000, (n_cases, 3, 3))
        weights = rng.uniform(-0.2, 0.9, (n_cases, 3))
        weights /= weights.sum(axis=1, keepdims=True)
        points = np.einsum("nij,ni->nj", faces, weights)

        n_inside = 0
        for face, point in zip(faces, points):
            expected = bool(point_in_face(face, point))
            self.assertEqual(expected, bool(point_in_face_barycentric(face, point)))
            n_inside += expected
        # guard against the assertions passing trivially on all-outside points
        self.assertGreater(n_inside, n_cases // 10)

    def test_degenerate_face_is_rejected(self):
        # collinear vertices have zero area; the pinv version returns some
        # minimum-norm solution there, this one rejects the face outright
        face = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        self.assertFalse(bool(point_in_face_barycentric(face, np.zeros(3))))
        # duplicated vertices, likewise zero area
        face = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        self.assertFalse(bool(point_in_face_barycentric(face, np.zeros(3))))


class TestIntersectLineMesh(unittest.TestCase):
    def test_vertical_ray_through_cube_returns_two_intersections(self):
        # ray slightly off-axis to avoid the diagonal where the two bottom
        # triangles meet (point_in_face is strict-inequality so points on the
        # shared edge are rejected from both triangles)
        vertices, edges = _unit_cube_mesh()
        line_point = np.array([0.1, 0.3, -10.0])
        line_vector = np.array([0.0, 0.0, 1.0])
        _, intersection_points, _ = intersect_line_mesh_nb(
            vertices, edges, line_point, line_vector
        )
        self.assertEqual(intersection_points.shape[0], 2)
        z_values = sorted(intersection_points[:, 2].tolist())
        nptest.assert_array_almost_equal(z_values, [-1.0, 1.0])

    def test_ray_missing_mesh_returns_empty_arrays(self):
        # this is the load-bearing case for the `except ValueError` branch in
        # projections.project_coords_onto_atlas_surface — empty `faces` then
        # makes downstream np.argmin raise ValueError
        vertices, edges = _unit_cube_mesh()
        line_point = np.array([10.0, 10.0, -10.0])
        line_vector = np.array([0.0, 0.0, 1.0])
        faces, intersection_points, ix = intersect_line_mesh_nb(
            vertices, edges, line_point, line_vector
        )
        self.assertEqual(intersection_points.shape[0], 0)
        self.assertEqual(faces.shape[0], 0)
        self.assertEqual(ix.shape[0], 0)

    def test_np_and_nb_agree_on_cube(self):
        vertices, edges = _unit_cube_mesh()
        line_point = np.array([0.1, 0.3, -10.0])
        line_vector = np.array([0.0, 0.0, 1.0])
        _, np_ips, _ = intersect_line_mesh_np(vertices, edges, line_point, line_vector)
        _, nb_ips, _ = intersect_line_mesh_nb(vertices, edges, line_point, line_vector)
        nptest.assert_array_almost_equal(
            sorted(np_ips[:, 2].tolist()), sorted(nb_ips[:, 2].tolist())
        )

    def test_precomputed_agrees_with_nb_on_random_mesh_and_rays(self):
        # intersect_line_mesh_precomputed_nb inlines the maths of
        # point_in_face_barycentric() and intersect_line_plane_nb() by hand, so
        # it is checked against the reference implementation over a mesh with
        # varied face orientations and many ray directions
        rng = np.random.default_rng(1)
        points = rng.normal(size=(60, 3))
        points /= np.linalg.norm(points, axis=1, keepdims=True)  # on a sphere
        vertices = points * 100.0
        edges = ConvexHull(vertices).simplices.astype(np.int32)
        normals = _face_normals(vertices, edges)

        n_hits = 0
        for _ in range(50):
            # aim each ray from outside the hull at a random point well inside
            # it, so that it actually intersects rather than missing
            origin = rng.normal(size=3)
            line_point = origin / np.linalg.norm(origin) * 300.0
            target = rng.uniform(-40, 40, 3)
            line_vector = target - line_point
            expected = intersect_line_mesh_nb(vertices, edges, line_point, line_vector)
            result = intersect_line_mesh_precomputed_nb(
                vertices, edges, normals, line_point, line_vector
            )
            # the intersected face indices must match exactly, the coordinates
            # only to floating point tolerance
            nptest.assert_array_equal(expected[2], result[2])
            nptest.assert_allclose(expected[1], result[1], rtol=1e-9, atol=1e-9)
            nptest.assert_allclose(expected[0], result[0], rtol=1e-9, atol=1e-9)
            n_hits += expected[2].shape[0]
        # guard against the assertions passing trivially on rays that all miss
        self.assertGreater(n_hits, 50)

    def test_precomputed_normals_agree_with_nb(self):
        # the precomputed variant must be a drop-in for intersect_line_mesh_nb
        vertices, edges = _unit_cube_mesh()
        normals = _face_normals(vertices, edges)
        # a hitting and a missing ray, plus an oblique one
        for line_point, line_vector in [
            (np.array([0.1, 0.3, -10.0]), np.array([0.0, 0.0, 1.0])),
            (np.array([10.0, 10.0, -10.0]), np.array([0.0, 0.0, 1.0])),
            (np.array([0.1, 0.3, -10.0]), np.array([0.05, 0.02, 1.0])),
        ]:
            expected = intersect_line_mesh_nb(vertices, edges, line_point, line_vector)
            result = intersect_line_mesh_precomputed_nb(
                vertices, edges, normals, line_point, line_vector
            )
            for a, b in zip(expected, result):
                nptest.assert_array_almost_equal(a, b)


class TestGetAngle(unittest.TestCase):
    def test_parallel_vectors_give_zero(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([2.0, 0.0, 0.0])
        self.assertAlmostEqual(get_angle(a, b), 0.0)

    def test_anti_parallel_vectors_give_pi(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        self.assertAlmostEqual(get_angle(a, b), np.pi)

    def test_perpendicular_vectors_give_pi_over_two(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(get_angle(a, b), np.pi / 2)


class TestFindClosestPointFromLine(unittest.TestCase):
    def test_picks_nearest_perpendicular_point(self):
        # candidates at perpendicular distances 0.5, 1.0, 2.0 from the z-axis
        points = np.array(
            [
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        l0 = np.array([0.0, 0.0, 0.0])
        l = np.array([0.0, 0.0, 1.0])
        nptest.assert_array_almost_equal(
            find_closest_point_from_line_np(points, l0, l), [0.5, 0.0, 0.0]
        )

    def test_np_and_nb_agree(self):
        points = np.array(
            [
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        l0 = np.array([0.0, 0.0, 0.0])
        l = np.array([0.0, 0.0, 1.0])
        nptest.assert_array_almost_equal(
            find_closest_point_from_line_np(points, l0, l),
            find_closest_point_from_line_nb(points, l0, l),
        )


class TestGetRotationBetweenVectors(unittest.TestCase):
    def test_maps_a_to_b(self):
        # R @ a ≈ b when a, b are unit vectors that are not (anti-)parallel
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        R = get_rotation_between_vectors(a, b, as_affine=False)
        nptest.assert_array_almost_equal(R @ a, b)

    def test_as_affine_returns_4x4_with_corner_one(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 1.0])
        R = get_rotation_between_vectors(a, b, as_affine=True)
        self.assertEqual(R.shape, (4, 4))
        self.assertAlmostEqual(R[3, 3], 1.0)


if __name__ == "__main__":
    unittest.main()
