from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.testing as nptest

from plane2brain.coordinate_systems import (
    LinkedCoordinateSystems,
    create_coordinate_system_for_image,
)


def _get_fov_uuids(scanimage_meta: dict[str, Any]) -> List[str]:
    """Return all ScanImage field-of-view UUIDs from metadata."""
    scanimage_fov_metas = scanimage_meta["Artist"]["RoiGroups"]["imagingRoiGroup"][
        "rois"
    ]
    return [meta["roiUuid"] for meta in scanimage_fov_metas]


def get_scanfield_size_ref(
    scanimage_fov_meta: dict[str, Any],
    dims: Sequence[str] = ("X", "Y"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Read scanfield size and center from ScanImage FOV metadata.

    Args:
        scanimage_fov_meta: A single FOV metadata block from ScanImage.
        dims: The axis order for the returned arrays. Use
            `(X, Y)` by default, or `(Y, X)` to swap axes.
    """
    fov_size_ref = np.array(scanimage_fov_meta["scanfields"]["sizeXY"])
    fov_center_ref = np.array(scanimage_fov_meta["scanfields"]["centerXY"])
    if dims == ("Y", "X"):
        fov_size_ref = fov_size_ref[::-1]
        fov_center_ref = fov_center_ref[::-1]

    return fov_size_ref, fov_center_ref


def get_scanfield_size_px(
    scanimage_fov_meta: dict[str, Any],
    dims: Sequence[str] = ("X", "Y"),
) -> np.ndarray:
    """Read the scanfield pixel dimensions from ScanImage FOV metadata."""
    fov_size_px = np.array(scanimage_fov_meta["scanfields"]["pixelResolutionXY"])
    if dims == ("Y", "X"):
        fov_size_px = fov_size_px[::-1]

    return fov_size_px


def get_resolution_from_scanimage_meta(
    scanimage_meta: dict[str, Any],
    dims: Sequence[str] = ("X", "Y"),
) -> np.ndarray:
    """Convert ScanImage resolution metadata to micrometers per pixel.

    The returned array is ordered according to `dims`.
    """
    px_per_um = np.zeros(2)
    for i, d in enumerate(dims):
        res = scanimage_meta[f"{d}Resolution"]
        match scanimage_meta["ResolutionUnit"].casefold():
            case "centimeter":
                px_per_um[i] = res * 1e-4
            case _:
                raise ValueError(
                    "Reference image resolution unit must be in centimeters"
                )
    return 1 / px_per_um


def get_fov_meta(
    scanimage_meta: dict[str, Any],
    fov_uuid: str,
) -> Dict[str, Any]:
    """Return the metadata block corresponding to the specified FOV UUID."""
    (scanimage_fov_meta,) = [
        meta
        for meta in scanimage_meta["Artist"]["RoiGroups"]["imagingRoiGroup"]["rois"]
        if meta["roiUuid"] == fov_uuid
    ]
    return scanimage_fov_meta


def create_coordinate_systems_from_scanimage_meta(
    scanimage_meta: dict[str, Any],
    fov_uuids: Optional[List[str]] = None,
    dims: Sequence[str] = ("X", "Y"),
) -> Dict[str, LinkedCoordinateSystems]:
    """Build `LinkedCoordinateSystems` objects for all ScanImage scanfields.

    Args:
        scanimage_meta: Full ScanImage metadata dictionary.
        fov_uuids: Optional list of FOV UUIDs to process. If omitted, all FOVs
            in `scanimage_meta` are processed.
        dims: The axis order used for X/Y metadata values.

    Returns:
        A mapping from FOV UUIDs to `LinkedCoordinateSystems`.
    """
    if fov_uuids is None:
        fov_uuids = _get_fov_uuids(scanimage_meta)

    # pixel resolution from metadata
    um_per_px = get_resolution_from_scanimage_meta(scanimage_meta, dims=dims)
    coordinate_systems: Dict[str, LinkedCoordinateSystems] = {}

    for fov_uuid in fov_uuids:
        scanimage_fov_meta = get_fov_meta(scanimage_meta, fov_uuid)
        # misleading variable naming by ScanImage but here too X is the line = resonant scanner, and Y is the line number
        # this, combined with the fact that on the reference image, the strips are extended vertically
        # means: XY is AP, ML
        fov_size_px = get_scanfield_size_px(scanimage_fov_meta, dims=dims)
        fov_size_um = fov_size_px * um_per_px

        # the size of the scanfield is stored in the metadata in
        # "sizeXY: [width, height] size of the scanfield in optical degrees in the coordinate system in which it is defined"
        # (taken from the doc)
        # unclear if this is correct (reference space is not equal to optical degrees)
        # it rather seems it's in reference space
        # see below for proof at (*)

        # the center and size are expressed in the scanfield coordinate system
        fov_size_ref, fov_center_ref = get_scanfield_size_ref(
            scanimage_fov_meta, dims=dims
        )

        # transform to reference coordinate frame
        fov_topleft_ref = fov_center_ref - fov_size_ref / 2

        # (*) to show that sizeXY is in ref space
        # according to the docs: the affine transform to convert pixel coordinates to reference space
        T_p = np.array(scanimage_fov_meta["scanfields"]["pixelToRefTransform"])

        # this will actually fail due to imprecision!
        # np.testing.assert_allclose(
        #     (T_p @ np.append(np.zeros(2), 1))[:-1], fov_topleft_ref, rtol=1e-3
        # )

        # (T_p @ np.append(fov_size_px, 1))[:-1] - (T_p @ np.append(np.zeros(2), 1))[:-1]
        # T_a = np.array(scanimage_fov_meta["scanfields"]["affine"])

        fov_bottomright_ref = fov_topleft_ref + fov_size_ref
        nptest.assert_array_almost_equal(
            fov_size_ref, fov_bottomright_ref - fov_topleft_ref
        )
        ref_per_px = fov_size_ref / fov_size_px
        px_per_ref = 1 / ref_per_px

        # next wee need to know what is the size of a pixel in reference space?
        um_per_ref = um_per_px * px_per_ref
        ref_per_um = 1 / um_per_ref
        fov_topleft_um = fov_topleft_ref * um_per_ref

        coordinate_system = create_coordinate_system_for_image(
            fov_size_px,
            um_per_px,
            ref_per_px,
            fov_topleft_ref,
        )

        # the image size assertion TODO put this into a test
        nptest.assert_array_almost_equal(
            coordinate_system.transform(fov_topleft_ref, "ref", "pixel"),
            np.zeros(2),
        )
        nptest.assert_array_almost_equal(
            coordinate_system.transform(fov_bottomright_ref, "ref", "pixel"),
            fov_size_px,
        )
        nptest.assert_array_almost_equal(
            coordinate_system.transform(np.zeros(2), "pixel", "ref"),
            fov_topleft_ref,
        )
        nptest.assert_array_almost_equal(
            coordinate_system.transform(fov_bottomright_ref, "ref", "um_global")
            - coordinate_system.transform(fov_topleft_ref, "ref", "um_global"),
            fov_size_um,
        )
        coordinate_systems[fov_uuid] = coordinate_system

    return coordinate_systems


def extract_fov_depths_from_scanimage_meta(
    scanimage_meta: dict[str, Any],
    scanimage_params: dict[str, Any],
    fov_uuids: Optional[Sequence[str]] = None,
) -> Dict[str, np.float64]:
    """Extract the depth of each ScanImage FOV from metadata.

    Args:
        scanimage_meta: Full ScanImage metadata dictionary.
        scanimage_params: ScanImage acquisition parameters.
        fov_uuids: Optional list of FOV UUIDs to extract. If omitted, all FOVs
            in the metadata are extracted.

    Returns:
        Mapping from FOV UUID to imaged depth in micrometers.
    """
    if fov_uuids is None:
        fov_uuids = _get_fov_uuids(scanimage_meta)

    fov_depths = {}
    for fov_uuid in fov_uuids:
        # fov_meta = get_fov_meta(scanimage_meta, fov_uuid)
        # fov_depths[fov_uuid] = -1 * (fastz_pos + fov_meta["zs"])
        fov_depths[fov_uuid] = -1 * scanimage_params["hStackManager"]["zs"]
    return fov_depths
