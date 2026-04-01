import unittest

import numpy as np
import numpy.testing as nptest

from plane2brain.coordinate_systems import create_coordinate_system_for_image


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
