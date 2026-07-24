import unittest

import numpy as np
import numpy.testing as nptest

from plane2brain.coordinate_systems import (
    coordinate_system_from_normal,
    create_coordinate_system_for_image,
    setup_coordinate_systems_3d,
)


class TestCoordinateSystems(unittest.TestCase):
    def test_create_coordinate_system_for_image_verification(self):
        img_size_px = np.array([100, 200])
        um_per_px = np.array([0.5, 0.3])
        ref_per_px = np.array([0.2, 0.4])
        img_topleft_ref = np.array([10.0, 20.0])

        coordinate_systems = create_coordinate_system_for_image(
            img_size_px=img_size_px,
            um_per_px=um_per_px,
            ref_per_px=ref_per_px,
            img_topleft_ref=img_topleft_ref,
        )

        img_size_um = img_size_px * um_per_px
        img_size_ref = img_size_px * ref_per_px
        img_bottomright_ref = img_topleft_ref + img_size_ref

        nptest.assert_array_almost_equal(
            coordinate_systems.transform(img_topleft_ref, "ref", "pixel"),
            np.zeros(2),
        )
        nptest.assert_array_almost_equal(
            coordinate_systems.transform(np.zeros(2), "pixel", "ref"),
            img_topleft_ref,
        )
        nptest.assert_array_almost_equal(
            coordinate_systems.transform(img_bottomright_ref, "ref", "pixel"),
            img_size_px,
        )
        nptest.assert_array_almost_equal(
            coordinate_systems.transform(np.zeros(2), "pixel", "ref"),
            img_topleft_ref,
        )
        nptest.assert_array_almost_equal(
            coordinate_systems.transform(img_bottomright_ref, "ref", "um_global")
            - coordinate_systems.transform(img_topleft_ref, "ref", "um_global"),
            img_size_um,
        )
        nptest.assert_array_almost_equal(
            coordinate_systems.transform(np.zeros(2), "pixel", "image"), np.zeros(2)
        )
        nptest.assert_array_almost_equal(
            coordinate_systems.transform(img_size_px, "pixel", "image"), np.ones(2)
        )
        nptest.assert_array_almost_equal(
            coordinate_systems.transform(np.ones(2), "image", "pixel"),
            img_size_px,
        )
        nptest.assert_array_almost_equal(
            coordinate_systems.transform(np.ones(2), "image", "um_image"),
            img_size_um,
        )
        nptest.assert_array_almost_equal(
            coordinate_systems.transform((0, 1), "image", "um_image"),
            (0, img_size_um[1]),
        )


class TestCoordinateSystemFromNormal(unittest.TestCase):
    def test_aligned_normal_gives_identity_basis(self):
        # n = [0,0,1]: DV straight up → basis should be the 3×3 identity
        p = np.array([1.0, 2.0, 3.0])
        n = np.array([0.0, 0.0, 1.0])
        cs = coordinate_system_from_normal(p, n)

        nptest.assert_array_almost_equal(cs.origin, p)
        nptest.assert_array_almost_equal(cs.basis, np.identity(3))

    def test_dv_axis_equals_normal(self):
        # for a tilted normal the DV axis (basis[:,2]) must match the normal
        p = np.zeros(3)
        angle = np.pi / 6  # 30 degrees
        n = np.array([0.0, np.sin(angle), np.cos(angle)])
        cs = coordinate_system_from_normal(p, n)

        nptest.assert_array_almost_equal(cs.basis[:, 2], n)

    def test_ap_axis_has_zero_ml_component(self):
        # AP axis (basis[:,1]) must have no ML component (index 0 == 0)
        p = np.zeros(3)
        n = np.array([0.0, np.sin(np.pi / 6), np.cos(np.pi / 6)])
        cs = coordinate_system_from_normal(p, n)

        self.assertAlmostEqual(cs.basis[0, 1], 0.0)

    def test_basis_is_orthogonal(self):
        p = np.zeros(3)
        n = np.array([0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
        cs = coordinate_system_from_normal(p, n)

        nptest.assert_array_almost_equal(cs.basis.T @ cs.basis, np.identity(3))

    def test_invert_dims_negates_axis(self):
        p = np.zeros(3)
        n = np.array([0.0, 0.0, 1.0])
        cs_default = coordinate_system_from_normal(p, n)
        cs_inverted = coordinate_system_from_normal(
            p, n, invert_dims=[True, False, False]
        )

        # ML axis (col 0) is negated; AP and DV axes are unchanged
        nptest.assert_array_almost_equal(
            cs_inverted.basis[:, 0], -cs_default.basis[:, 0]
        )
        nptest.assert_array_almost_equal(
            cs_inverted.basis[:, 1], cs_default.basis[:, 1]
        )
        nptest.assert_array_almost_equal(
            cs_inverted.basis[:, 2], cs_default.basis[:, 2]
        )

    def test_rotate_by_rotates_ml_ap_columns(self):
        # for a flat normal, rotate_by=π/2 around DV should send ML→AP and AP→-ML,
        # leaving the DV column unchanged
        p = np.zeros(3)
        n = np.array([0.0, 0.0, 1.0])
        cs = coordinate_system_from_normal(p, n, rotate_by=np.pi / 2)

        nptest.assert_array_almost_equal(cs.basis[:, 0], [0.0, 1.0, 0.0])
        nptest.assert_array_almost_equal(cs.basis[:, 1], [-1.0, 0.0, 0.0])
        nptest.assert_array_almost_equal(cs.basis[:, 2], [0.0, 0.0, 1.0])

    def test_rotate_by_preserves_orthonormal_basis(self):
        # the previous bug used apply_transform on the column-major basis, which
        # produced a non-orthogonal basis for tilted normals; verify orthonormality
        p = np.zeros(3)
        n = np.array([0.0, np.sin(np.pi / 6), np.cos(np.pi / 6)])
        cs = coordinate_system_from_normal(p, n, rotate_by=np.pi / 4)

        nptest.assert_array_almost_equal(cs.basis.T @ cs.basis, np.identity(3))

    def test_rotate_by_dv_axis_unchanged_for_flat_normal(self):
        # rotate_by rotates around DV, so the DV column must be invariant
        p = np.zeros(3)
        n = np.array([0.0, 0.0, 1.0])
        cs_no_rotation = coordinate_system_from_normal(p, n)
        cs_rotated = coordinate_system_from_normal(p, n, rotate_by=0.7)

        nptest.assert_array_almost_equal(
            cs_rotated.basis[:, 2], cs_no_rotation.basis[:, 2]
        )


class TestSetupCoordinateSystems3D(unittest.TestCase):
    def setUp(self):
        self.center = np.array([100.0, 200.0, 50.0])
        self.normal = np.array([0.0, 0.0, 1.0])  # flat brain surface
        self.cs = setup_coordinate_systems_3d(self.center, self.normal)

    def test_mlapdv_is_world_frame(self):
        nptest.assert_array_almost_equal(self.cs.get("mlapdv").origin, np.zeros(3))
        nptest.assert_array_almost_equal(self.cs.get("mlapdv").basis, np.identity(3))

    def test_imaging_plane_origin_equals_center(self):
        nptest.assert_array_almost_equal(
            self.cs.get("imaging_plane").origin, self.center
        )

    def test_center_maps_to_origin_in_imaging_plane(self):
        # the reference point should be at the origin of the imaging_plane system
        coords_in_plane = self.cs.transform(self.center, "mlapdv", "imaging_plane")
        nptest.assert_array_almost_equal(coords_in_plane, np.zeros(3))

    def test_round_trip(self):
        point_in_plane = np.array([10.0, -5.0, 0.0])
        point_mlapdv = self.cs.transform(point_in_plane, "imaging_plane", "mlapdv")
        point_back = self.cs.transform(point_mlapdv, "mlapdv", "imaging_plane")
        nptest.assert_array_almost_equal(point_back, point_in_plane)


if __name__ == "__main__":
    unittest.main()


# # %%
# # another coordinate system, rotated
# # origin = np.ones(2) + 2
# # basis_r = np.array([[[1,-1],[]]])

# # fail on rotating non-uniform coordinate axes!


# # rotatian: translate to origin, apply rotation matrix, translate back
# def rotation_matrix_2d(theta):
#     rotation_matrix = np.array(
#         [
#             [np.cos(theta), -np.sin(theta)],
#             [
#                 np.sin(theta),
#                 np.cos(theta),
#             ],
#         ]
#     )
#     return rotation_matrix


# basis_r = (basis_t.T @ rotation_matrix_2d(np.pi / 4)).T  # transpose because we have column vectors

# cs3 = CoordinateSystem(basis=basis_r, origin=np.ones(2) + 3, name="rotated")

# cs = LinkedCoordinateSystems([cs1, cs2, cs3])
# axes = cs.plot(color_by="axis")

# cs.transform(np.array([0, 1]), "translated", "original")

# axes.set_xlabel("x")
# axes.set_ylabel("y")

# %%
