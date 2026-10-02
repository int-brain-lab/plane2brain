"""Tests for plane2brain.core."""

import unittest

import numpy as np
import numpy.testing as nptest

from plane2brain.core import (
    Anchor,
    CoordinateSystem,
    Image,
    LinkedCoordinateSystems,
    Orientation,
    Plane,
    _in_plane_axes_from_normal,
    create_coordinate_system_for_image,
)


def make_image(
    size_px: tuple[int, int] = (10, 20),
    um_per_px: float = 2.0,
    topleft_um: tuple[float, float] = (-10.0, -20.0),
) -> Image:
    """Return an image whose pixel grid starts at `topleft_um` in "um_global"."""
    coordinate_systems = LinkedCoordinateSystems(
        {
            "um_global": CoordinateSystem(basis=np.identity(2), origin=np.zeros(2)),
            "pixel": CoordinateSystem(
                basis=np.identity(2) * um_per_px, origin=np.array(topleft_um)
            ),
        }
    )
    return Image(size_px, coordinate_systems)


class TestCoordinateSystems(unittest.TestCase):
    """Tests for the 2D coordinate systems of an image."""

    def test_create_coordinate_system_for_image_verification(self) -> None:
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


class TestInPlaneAxesFromNormal(unittest.TestCase):
    """Tests for the in-plane ML and AP axes derived from a normal."""

    def test_vertical_normal_gives_ml_and_ap(self) -> None:
        # n = [0, 0, 1]: DV straight up -> the in-plane axes are ML and AP
        axes = _in_plane_axes_from_normal(np.array([0.0, 0.0, 1.0]))
        nptest.assert_array_almost_equal(axes, [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    def test_ap_axis_has_zero_ml_component(self) -> None:
        normal = np.array([0.2, np.sin(np.pi / 6), np.cos(np.pi / 6)])
        axes = _in_plane_axes_from_normal(normal / np.linalg.norm(normal))
        self.assertAlmostEqual(axes[0, 1], 0.0)

    def test_axes_and_normal_are_orthonormal(self) -> None:
        # also for a downward normal, where the former coordinate_system_from_normal failed
        for normal in [
            (0.0, 0.5, 0.8),
            (0.2, 0.3, 0.93),
            (0.2, 0.3, -0.93),
            (0.5, 0.6, -0.2),
        ]:
            normal = np.array(normal) / np.linalg.norm(normal)
            basis = np.column_stack([_in_plane_axes_from_normal(normal), normal])
            nptest.assert_array_almost_equal(basis.T @ basis, np.identity(3))

    def test_normal_along_ml_raises(self) -> None:
        with self.assertRaises(ValueError):
            _in_plane_axes_from_normal(np.array([1.0, 0.0, 0.0]))


class TestOrientation(unittest.TestCase):
    """Tests for the rig orientation matrix."""

    def test_default_is_identity(self) -> None:
        nptest.assert_array_almost_equal(Orientation().matrix, np.identity(3))

    def test_invert_axis_negates_column(self) -> None:
        matrix = Orientation(invert_axis=(True, False, True)).matrix
        nptest.assert_array_almost_equal(matrix, np.diag([-1.0, 1.0, -1.0]))

    def test_rotation_about_optical_axis(self) -> None:
        # 90 degrees: x -> y, y -> -x, z unchanged
        matrix = Orientation(rotation_degrees=90.0).matrix
        nptest.assert_array_almost_equal(matrix[:, 0], [0.0, 1.0, 0.0])
        nptest.assert_array_almost_equal(matrix[:, 1], [-1.0, 0.0, 0.0])
        nptest.assert_array_almost_equal(matrix[:, 2], [0.0, 0.0, 1.0])

    def test_inverts_first_then_rotates(self) -> None:
        # x is inverted first, then rotated by 90 degrees: x -> -x -> -y
        matrix = Orientation(
            rotation_degrees=90.0, invert_axis=(True, False, False)
        ).matrix
        nptest.assert_array_almost_equal(matrix @ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0])

    def test_matrix_is_orthonormal(self) -> None:
        for rotation_degrees in [0.0, 30.0, -75.0]:
            for invert_axis in [
                (False, False, False),
                (True, False, False),
                (True, True, True),
            ]:
                matrix = Orientation(rotation_degrees, invert_axis).matrix
                nptest.assert_array_almost_equal(matrix.T @ matrix, np.identity(3))
                self.assertAlmostEqual(abs(np.linalg.det(matrix)), 1.0)


class TestImage(unittest.TestCase):
    """Tests for images: validation, pixel indices, depths and corners."""

    def test_requires_pixel_and_um_global(self) -> None:
        for names in [("um_global",), ("pixel",)]:
            coordinate_systems = LinkedCoordinateSystems(
                {name: CoordinateSystem(np.identity(2), np.zeros(2)) for name in names}
            )
            with self.assertRaises(ValueError):
                Image((10, 10), coordinate_systems)

    def test_rejects_3d_set(self) -> None:
        coordinate_systems = LinkedCoordinateSystems(
            {
                name: CoordinateSystem(np.identity(3), np.zeros(3))
                for name in ("pixel", "um_global")
            }
        )
        with self.assertRaises(ValueError):
            Image((10, 10), coordinate_systems)

    def test_pixel_indices_first_index_slowest(self) -> None:
        image = make_image(size_px=(2, 3))
        nptest.assert_array_equal(
            image.pixel_indices, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
        )
        self.assertTrue(np.issubdtype(image.pixel_indices.dtype, np.integer))

    def test_depth_below_surface(self) -> None:
        image = make_image(size_px=(2, 3))
        self.assertIsNone(image.depth_below_surface)

        # a single depth applies to every pixel
        image.depth_below_surface = 300.0
        nptest.assert_array_equal(image.depth_below_surface, np.full(6, 300.0))

        # one depth per pixel, stored as a copy that cannot be changed in place
        depths = np.arange(6, dtype=float)
        image.depth_below_surface = depths
        depths[0] = 100.0
        self.assertEqual(image.depth_below_surface[0], 0.0)
        with self.assertRaises(ValueError):
            image.depth_below_surface[0] = 1.0

        with self.assertRaises(ValueError):
            image.depth_below_surface = np.arange(5, dtype=float)

    def test_get_corners(self) -> None:
        image = make_image(size_px=(10, 20), um_per_px=2.0, topleft_um=(-10.0, -20.0))
        corners_px = image.get_corners(in_="pixel")
        nptest.assert_array_almost_equal(corners_px["topleft"], [0.0, 0.0])
        nptest.assert_array_almost_equal(corners_px["bottomright"], [10.0, 20.0])
        corners_um = image.get_corners()
        nptest.assert_array_almost_equal(corners_um["topleft"], [-10.0, -20.0])
        nptest.assert_array_almost_equal(corners_um["center"], [0.0, 0.0])


class TestPlane(unittest.TestCase):
    """Tests for planes: the 3D link through the anchor, orientation and safeguards."""

    def setUp(self) -> None:
        """Define a tilted plane through a point, and an anchor at that point."""
        self.point = np.array([1000.0, -2000.0, -300.0])
        self.normal = np.array([0.2, 0.3, 0.93])
        self.anchor = Anchor(self.point, "um_global", (0.0, 0.0))

    def linked_plane(self, orientation: Orientation | None = None) -> Plane:
        """Return a plane with two images, linked through the anchor."""
        plane = Plane(self.point, self.normal)
        images = [make_image(), make_image(topleft_um=(690.0, -20.0))]
        plane.add_images(images, orientation or Orientation(), self.anchor)
        return plane

    def test_new_plane(self) -> None:
        plane = Plane(self.point, self.normal)
        self.assertEqual(
            list(plane.coordinate_systems.coordinate_systems), ["um_global"]
        )
        self.assertIsNone(plane.coordinate_systems_3d)
        self.assertIsNone(plane.anchor)
        self.assertEqual(plane.images, [])
        self.assertAlmostEqual(np.linalg.norm(plane.normal), 1.0)

    def test_mlapdv_is_world_frame(self) -> None:
        coordinate_systems_3d = self.linked_plane().coordinate_systems_3d
        nptest.assert_array_almost_equal(
            coordinate_systems_3d.get("mlapdv").origin, np.zeros(3)
        )
        nptest.assert_array_almost_equal(
            coordinate_systems_3d.get("mlapdv").basis, np.identity(3)
        )

    def test_anchor_lands_on_its_atlas_position(self) -> None:
        plane = Plane(self.point, self.normal)
        plane.add_images(
            [], Orientation(), Anchor(self.point, "um_global", (10.0, 20.0))
        )
        nptest.assert_array_almost_equal(
            plane.coordinate_systems_3d.transform(
                [10.0, 20.0, 0.0], "um_global", "mlapdv"
            ),
            self.point,
        )

    def test_round_trip(self) -> None:
        coordinate_systems_3d = self.linked_plane().coordinate_systems_3d
        point_in_plane = np.array([10.0, -5.0, 0.0])
        point_mlapdv = coordinate_systems_3d.transform(
            point_in_plane, "um_global", "mlapdv"
        )
        nptest.assert_array_almost_equal(
            coordinate_systems_3d.transform(point_mlapdv, "mlapdv", "um_global"),
            point_in_plane,
        )

    def test_third_coordinate_is_distance_from_plane(self) -> None:
        plane = self.linked_plane()
        offset = self.point + 5.0 * plane.normal
        coordinates = plane.coordinate_systems_3d.transform(
            offset, "mlapdv", "um_global"
        )
        self.assertAlmostEqual(coordinates[2], 5.0)

    def test_images_keep_their_distance(self) -> None:
        # the two image centers are 700 µm apart in "um_global", and stay so on the plane
        plane = self.linked_plane()
        centers = [
            plane.coordinate_systems_3d.transform(
                np.append(image.get_corners()["center"], 0.0), "um_global", "mlapdv"
            )
            for image in plane.images
        ]
        self.assertAlmostEqual(np.linalg.norm(centers[1] - centers[0]), 700.0)

    def test_orientation_inverts_in_plane_axis(self) -> None:
        plane = Plane(self.point, (0.0, 0.0, 1.0))
        plane.add_images([], Orientation(invert_axis=(True, False, False)), self.anchor)
        nptest.assert_array_almost_equal(
            plane.coordinate_systems_3d.get("um_global").basis,
            np.diag([-1.0, 1.0, 1.0]),
        )

    def test_add_images_with_the_same_placement_appends(self) -> None:
        plane = self.linked_plane()
        plane.add_images([make_image()], Orientation(), self.anchor)
        self.assertEqual(len(plane.images), 3)

    def test_safeguards(self) -> None:
        plane = self.linked_plane()
        coordinate_systems_3d = plane.coordinate_systems_3d
        only_pixel = LinkedCoordinateSystems(
            {"pixel": CoordinateSystem(np.identity(2), np.zeros(2))}
        )
        offset = self.point + 5.0 * plane.normal
        checks = {
            "3D set": lambda: Plane(
                self.point, self.normal, coordinate_systems=coordinate_systems_3d
            ),
            "no um_global": lambda: Plane(
                self.point, self.normal, coordinate_systems=only_pixel
            ),
            "anchor off the plane": lambda: Plane(self.point, self.normal).add_images(
                [], Orientation(), Anchor(offset, "um_global", (0.0, 0.0))
            ),
            "unknown anchor name": lambda: Plane(self.point, self.normal).add_images(
                [], Orientation(), Anchor(self.point, "pixel", (0.0, 0.0))
            ),
            "different placement": lambda: plane.add_images(
                [make_image()],
                Orientation(invert_axis=(True, False, False)),
                self.anchor,
            ),
            "normal along ML": lambda: Plane(self.point, (1.0, 0.0, 0.0)).add_images(
                [], Orientation(), self.anchor
            ),
        }
        for label, check in checks.items():
            with self.subTest(label), self.assertRaises(ValueError):
                check()
        # the failed add_images left the plane unchanged
        self.assertEqual(len(plane.images), 2)


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
