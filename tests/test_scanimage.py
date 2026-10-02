"""Tests for plane2brain.scanimage, on the ScanImage fixtures."""

import json
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as nptest

from plane2brain import scanimage

FIXTURES = Path(__file__).parent / "fixtures"

# enabled Roi uuids in the order of the IBL FOV lists (alf/FOV_xx) of the fixture sessions;
# copied from the IBL metadata for verification only, the code never reads it
SP058_UUIDS = [
    "8EADE89D4D3FE35",
    "405CA2328B13C216",
    "903057221591EEB1",
    "B8631D3799FEB206",
    "AAB5EED928ED2A1D",
    "9E8118A6F7804F58",
    "8409B0A6BA0BBA99",
    "8470AA45FF5F8FBD",
    "D43327735C479B20",
]
SP044_UUIDS = [
    "4A63E30D13846D98",
    "4B155B9419E39C6E",
    "995D6226029A042E",
    "AC32D2F9222B6AC0",
    "77D7BF53D3AA3A7",
    "5229A397B04C40F2",
]
SP061_UUIDS = [
    "93A40570CE348E0F",
    "B8F1CD16D14AC427",
    "6AB35EB486E1559D",
    "8C4144736C3EF472",
    "5E71509430C03C0F",
    "D30A52B143E97320",
    "788E7DF89938D8DF",
    "174031E36C676916",
]
# a disabled Roi of SP058 with two scanfields
SP058_MULTI_SCANFIELD_UUID = "255385A5966011B9"


def load_fixture(name: str) -> dict:
    """Load the ScanImage metadata of a fixture session."""
    with open(FIXTURES / f"{name}_scanimage_meta.json") as file:
        return json.load(file)


class ScanImageTestCase(unittest.TestCase):
    """Loads the single-plane (SP058, SP044) and dual-plane (SP061) fixtures."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the fixtures once for all tests of the class."""
        cls.sp058 = load_fixture("SP058_2024-08-01_001_raw_imaging_data_02")
        cls.sp044 = load_fixture("SP044_2023-06-27_001_raw_imaging_data_00")
        cls.sp061 = load_fixture("SP061_2025-02-21_001_raw_imaging_data_01")


class TestSoftwareValues(ScanImageTestCase):
    """Tests for reading values from the `Software` tag."""

    def test_parse_matlab_numeric(self) -> None:
        nptest.assert_array_equal(scanimage._parse_matlab_numeric("-515"), [[-515.0]])
        nptest.assert_array_equal(
            scanimage._parse_matlab_numeric("[-450 -250]"), [[-450.0, -250.0]]
        )
        nptest.assert_array_equal(
            scanimage._parse_matlab_numeric("[-450 -450;-250 -250]"),
            [[-450.0, -450.0], [-250.0, -250.0]],
        )
        self.assertEqual(scanimage._parse_matlab_numeric("[]").shape, (1, 0))

    def test_software_values(self) -> None:
        software_values = scanimage._get_software_values(self.sp044)
        self.assertEqual(software_values["SI.hStackManager.zs"], "[-450 -250]")

    def test_objective_resolution(self) -> None:
        for meta in [self.sp058, self.sp044, self.sp061]:
            self.assertEqual(scanimage.get_objective_resolution(meta), 150.0)

    def test_objective_resolution_matches_tiff_resolution(self) -> None:
        # µm per pixel from the scanfield geometry and from the TIFF tags agree
        for meta, uuid in [(self.sp058, SP058_UUIDS[0]), (self.sp044, SP044_UUIDS[0])]:
            roi_meta = scanimage.get_roi_meta(meta, uuid)
            size_ref, _ = scanimage.get_scanfield_size_ref(roi_meta)
            um_per_px = (
                size_ref
                * scanimage.get_objective_resolution(meta)
                / scanimage.get_scanfield_size_px(roi_meta)
            )
            nptest.assert_allclose(
                um_per_px, scanimage.get_resolution_from_scanimage_meta(meta), rtol=1e-5
            )


class TestRois(ScanImageTestCase):
    """Tests for reading Rois and their scanfields."""

    def test_uuids(self) -> None:
        for meta, n_enabled, n_all in [
            (self.sp058, 9, 26),
            (self.sp044, 6, 45),
            (self.sp061, 8, 20),
        ]:
            self.assertEqual(len(scanimage._get_uuids(meta)), n_enabled)
            self.assertEqual(len(scanimage._get_uuids(meta, enabled_only=False)), n_all)
        self.assertEqual(scanimage._get_uuids(self.sp044), SP044_UUIDS)

    def test_get_roi_meta(self) -> None:
        roi_meta = scanimage.get_roi_meta(self.sp044, SP044_UUIDS[0])
        self.assertEqual(roi_meta["roiUuid"], SP044_UUIDS[0])
        with self.assertRaises(ValueError):
            scanimage.get_roi_meta(self.sp044, "not a uuid")

    def test_scanfield_size_dims(self) -> None:
        roi_meta = scanimage.get_roi_meta(self.sp044, SP044_UUIDS[1])
        size_xy, center_xy = scanimage.get_scanfield_size_ref(roi_meta)
        size_yx, center_yx = scanimage.get_scanfield_size_ref(roi_meta, dims=("Y", "X"))
        nptest.assert_array_equal(center_xy, roi_meta["scanfields"]["centerXY"])
        nptest.assert_array_equal(center_yx, center_xy[::-1])
        nptest.assert_array_equal(size_yx, size_xy[::-1])


class TestSliceDepths(ScanImageTestCase):
    """Tests for the raw z of the slices of each Roi."""

    def test_single_slice(self) -> None:
        slice_depths = scanimage.extract_slice_depths_from_scanimage_meta(self.sp058)
        self.assertEqual(list(slice_depths), SP058_UUIDS)
        for zs in slice_depths.values():
            nptest.assert_array_equal(zs, [-515.0])

    def test_two_slices(self) -> None:
        slice_depths = scanimage.extract_slice_depths_from_scanimage_meta(self.sp044)
        for zs in slice_depths.values():
            nptest.assert_array_equal(zs, [-450.0, -250.0])

    def test_dual_plane_raises(self) -> None:
        with self.assertRaises(NotImplementedError):
            scanimage.extract_slice_depths_from_scanimage_meta(self.sp061)


class TestImageMap(ScanImageTestCase):
    """Tests for the image map, i.e. the order of the images."""

    def test_matches_ibl_fov_order(self) -> None:
        # ascending slice z, then the Roi order: the IBL FOV numbering (alf/FOV_xx)
        cases = [
            (self.sp058, [(uuid, -515.0) for uuid in SP058_UUIDS]),
            (
                self.sp044,
                [(uuid, -450.0) for uuid in SP044_UUIDS]
                + [(uuid, -250.0) for uuid in SP044_UUIDS],
            ),
        ]
        for meta, expected in cases:
            image_map = scanimage.get_image_map(meta)
            self.assertEqual(list(image_map.columns), ["index", "z", "uuid"])
            self.assertEqual(list(image_map["index"]), list(range(len(expected))))
            self.assertEqual(list(zip(image_map["uuid"], image_map["z"])), expected)

    def test_all_rois(self) -> None:
        self.assertEqual(
            len(scanimage.get_image_map(self.sp058, enabled_only=False)), 26
        )
        self.assertEqual(
            len(scanimage.get_image_map(self.sp044, enabled_only=False)), 90
        )

    def test_dual_plane_raises(self) -> None:
        with self.assertRaises(NotImplementedError):
            scanimage.get_image_map(self.sp061)


class TestCreateImages(ScanImageTestCase):
    """Tests for creating the images of single-plane acquisitions."""

    def test_images_follow_the_image_map(self) -> None:
        for meta in [self.sp058, self.sp044]:
            image_map = scanimage.get_image_map(meta)
            images = scanimage.create_images_from_scanimage_meta(meta, dims=("Y", "X"))
            self.assertEqual(len(images), len(image_map))
            for image, uuid in zip(images, image_map["uuid"]):
                # the image center is the Roi's scanfield center, in µm and in dims order
                center_xy = np.array(
                    scanimage.get_roi_meta(meta, uuid)["scanfields"]["centerXY"]
                )
                nptest.assert_array_almost_equal(
                    image.get_corners()["center"],
                    center_xy[::-1] * scanimage.get_objective_resolution(meta),
                )

    def test_image_geometry(self) -> None:
        image = scanimage.create_image_from_scanimage_meta(
            self.sp044, SP044_UUIDS[0], -450.0
        )
        nptest.assert_array_equal(image.size_px, [512, 512])
        self.assertIsNone(image.depth_below_surface)
        # one pixel: sizeXY (optical degrees) * objective resolution (µm per degree) / pixels
        one_pixel = image.coordinate_systems.transform(
            np.array([[0.0, 0.0], [1.0, 0.0]]), "pixel", "um_image"
        )
        self.assertAlmostEqual(one_pixel[1, 0] - one_pixel[0, 0], 3.9051 * 150 / 512)
        for name in ["ref", "um_global", "pixel", "um_image", "image"]:
            self.assertIn(name, image.coordinate_systems.coordinate_systems)

    def test_wrong_z_raises(self) -> None:
        with self.assertRaises(ValueError):
            scanimage.create_image_from_scanimage_meta(
                self.sp044, SP044_UUIDS[0], 123.0
            )

    def test_several_scanfields_raise(self) -> None:
        with self.assertRaises(NotImplementedError):
            scanimage.create_image_from_scanimage_meta(
                self.sp058, SP058_MULTI_SCANFIELD_UUID, -515.0
            )
        with self.assertRaises(NotImplementedError):
            scanimage.create_images_from_scanimage_meta(self.sp058, enabled_only=False)


class TestDualPlane(ScanImageTestCase):
    """Tests for the draft dual-plane loader on SP061."""

    def test_roi_order_matches_ibl_fovs(self) -> None:
        # with order "roi", images 2k and 2k + 1 are the two depths of IBL FOV_k (Zs [240, 200])
        images, image_map = scanimage.create_images_from_scanimage_dual_plane(
            self.sp061, actuator_channels=(1, 2), dims=("Y", "X")
        )
        self.assertEqual(len(images), 16)
        self.assertEqual(
            list(image_map.columns),
            ["index", "z", "uuid", "slice", "actuator", "channel"],
        )
        self.assertEqual(list(image_map["index"]), list(range(16)))
        self.assertEqual(
            list(image_map["uuid"]), [uuid for uuid in SP061_UUIDS for _ in range(2)]
        )
        self.assertEqual(list(image_map["z"]), [240.0, 200.0] * 8)
        self.assertEqual(list(image_map["actuator"]), [0, 1] * 8)
        self.assertEqual(list(image_map["channel"]), [1, 2] * 8)
        self.assertEqual(set(image_map["slice"]), {0})

    def test_both_depths_share_the_geometry(self) -> None:
        images, _ = scanimage.create_images_from_scanimage_dual_plane(
            self.sp061, (1, 2)
        )
        for k in range(8):
            nptest.assert_array_almost_equal(
                images[2 * k].get_corners()["center"],
                images[2 * k + 1].get_corners()["center"],
            )

    def test_other_orders(self) -> None:
        _, by_actuator = scanimage.create_images_from_scanimage_dual_plane(
            self.sp061, (1, 2), order="actuator"
        )
        self.assertEqual(list(by_actuator["z"]), [240.0] * 8 + [200.0] * 8)
        self.assertEqual(list(by_actuator["uuid"]), SP061_UUIDS * 2)
        _, by_z = scanimage.create_images_from_scanimage_dual_plane(
            self.sp061, (1, 2), order="z"
        )
        self.assertEqual(list(by_z["z"]), [200.0] * 8 + [240.0] * 8)

    def test_channel_mapping_is_an_argument(self) -> None:
        _, image_map = scanimage.create_images_from_scanimage_dual_plane(
            self.sp061, (2, 1)
        )
        self.assertEqual(list(image_map["channel"]), [2, 1] * 8)

    def test_invalid_arguments_raise(self) -> None:
        with self.assertRaises(ValueError):
            scanimage.create_images_from_scanimage_dual_plane(self.sp061, (1,))
        with self.assertRaises(ValueError):
            scanimage.create_images_from_scanimage_dual_plane(
                self.sp061, (1, 2), order="fov"
            )


if __name__ == "__main__":
    unittest.main()
