import unittest

import numpy as np
import numpy.testing as nptest

from plane2brain.affine import (
    apply_transform,
    rotation_matrix,
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    translation_matrix,
)


class TestApplyTransform(unittest.TestCase):
    def test_identity_leaves_points_unchanged(self):
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        nptest.assert_array_almost_equal(
            apply_transform(points, np.identity(4)), points
        )

    def test_translation_shifts_points(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        T = translation_matrix(2.0, 3.0, 4.0)
        expected = np.array([[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
        nptest.assert_array_almost_equal(apply_transform(points, T), expected)

    def test_composition_matches_matrix_product(self):
        # apply(apply(p, T1), T2) == apply(p, T2 @ T1)
        points = np.array([[1.0, 2.0, 3.0]])
        T1 = translation_matrix(1.0, 0.0, 0.0)
        T2 = rotation_matrix_z(np.pi / 4)
        stepwise = apply_transform(apply_transform(points, T1), T2)
        combined = apply_transform(points, T2 @ T1)
        nptest.assert_array_almost_equal(stepwise, combined)

    def test_round_trip_with_inverse_returns_point(self):
        # apply(apply(p, T), inv(T)) == p
        points = np.array([[5.0, -3.0, 7.0]])
        T = rotation_matrix_x(0.3) @ translation_matrix(2.0, 1.0, -1.0)
        round_trip = apply_transform(
            apply_transform(points, T), np.linalg.inv(T)
        )
        nptest.assert_array_almost_equal(round_trip, points)


class TestRotationMatrices(unittest.TestCase):
    def test_rotation_x_fixes_x_axis(self):
        x_axis = np.array([[1.0, 0.0, 0.0]])
        R = rotation_matrix_x(0.7)
        nptest.assert_array_almost_equal(apply_transform(x_axis, R), x_axis)

    def test_rotation_y_fixes_y_axis(self):
        y_axis = np.array([[0.0, 1.0, 0.0]])
        R = rotation_matrix_y(0.7)
        nptest.assert_array_almost_equal(apply_transform(y_axis, R), y_axis)

    def test_rotation_z_fixes_z_axis(self):
        z_axis = np.array([[0.0, 0.0, 1.0]])
        R = rotation_matrix_z(0.7)
        nptest.assert_array_almost_equal(apply_transform(z_axis, R), z_axis)

    def test_rotation_orthogonal_and_unit_determinant(self):
        # R @ R.T == I and det(R[:3,:3]) == 1
        R = (
            rotation_matrix_x(0.3)
            @ rotation_matrix_y(0.5)
            @ rotation_matrix_z(0.7)
        )
        nptest.assert_array_almost_equal(R @ R.T, np.identity(4))
        self.assertAlmostEqual(np.linalg.det(R[:3, :3]), 1.0)

    def test_rotation_x_pi_over_two_maps_y_to_z(self):
        # right-handed convention: R_x(π/2) sends +y -> +z
        y_axis = np.array([[0.0, 1.0, 0.0]])
        result = apply_transform(y_axis, rotation_matrix_x(np.pi / 2))
        nptest.assert_array_almost_equal(result, [[0.0, 0.0, 1.0]])

    def test_rotation_in_degrees_matches_radians(self):
        R_deg = rotation_matrix_z(90, in_degrees=True)
        R_rad = rotation_matrix_z(np.pi / 2, in_degrees=False)
        nptest.assert_array_almost_equal(R_deg, R_rad)

    def test_compound_rotation_equals_xyz_product(self):
        # rotation_matrix(α, β, γ) == R_x(α) @ R_y(β) @ R_z(γ)
        alpha, beta, gamma = 0.3, 0.5, 0.7
        compound = rotation_matrix(alpha, beta, gamma)
        product = (
            rotation_matrix_x(alpha)
            @ rotation_matrix_y(beta)
            @ rotation_matrix_z(gamma)
        )
        nptest.assert_array_almost_equal(compound, product)


if __name__ == "__main__":
    unittest.main()
