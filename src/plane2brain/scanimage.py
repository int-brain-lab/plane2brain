"""scanimage specific code"""

from typing import Optional, Dict, List
import numpy as np
from plane2brain.coordinate_systems import (
    LinkedCoordinateSystems,
    create_coordinate_system_for_image,
)
import numpy.testing as nptest


def _get_fov_uuids(scanimage_meta: dict) -> list:
    scanimage_fov_metas = scanimage_meta["Artist"]["RoiGroups"]["imagingRoiGroup"][
        "rois"
    ]
    return [meta["roiUuid"] for meta in scanimage_fov_metas]


def get_resolution_from_scanimage_meta(scanimage_meta: dict) -> np.ndarray:
    # X is the line scan (resonant)
    # are the individual lines, e.g. 512 pixels per line and 512 lines
    px_per_um = np.zeros(2)
    for i, d in enumerate(["X", "Y"]):
        res = scanimage_meta[f"{d}Resolution"]
        match scanimage_meta["ResolutionUnit"].casefold():
            case "centimeter":
                px_per_um[i] = res * 1e-4
            case _:
                "Reference image resolution unit must be in centimeters"
    um_per_px = 1 / px_per_um
    return um_per_px


def create_coordinate_systems_from_scanimage_meta(
    scanimage_meta: dict,
    fov_uuids: Optional[List[str]] = None,
) -> Dict[str, LinkedCoordinateSystems]:
    # all FOVs or subselection
    if fov_uuids is None:
        fov_uuids = _get_fov_uuids(scanimage_meta)

    # get scanimage metadata for the selected FOVs
    scanimage_fov_metas = [
        [
            meta
            for meta in scanimage_meta["Artist"]["RoiGroups"]["imagingRoiGroup"]["rois"]
            if meta["roiUuid"] == uuid
        ][0]
        for uuid in fov_uuids
    ]

    # pixel resolution from metadata
    um_per_px = get_resolution_from_scanimage_meta(scanimage_meta)
    coordinate_systems = {}

    for scanimage_fov_meta in scanimage_fov_metas:
        # misleading variable naming by scanimage but here too X is the line and Y is the line number
        fov_size_px = np.array(scanimage_fov_meta["scanfields"]["pixelResolutionXY"])
        fov_size_um = fov_size_px * um_per_px

        # the size of the scanfield is stored in the metadata in
        # "sizeXY: [width, height] size of the scanfield in optical degrees in the coordinate system in which it is defined"
        # (taken from the doc)
        # seems like this is ALREADY stored in the "units" of the reference space
        # for proof:
        # (T_p @ np.append(fov_size_px, 1))[:-1] - (T_p @ np.append(np.zeros(2), 1))[:-1]

        fov_size_ref = np.array(scanimage_fov_meta["scanfields"]["sizeXY"])
        # the center and size are expressed in the scanfield coordinate system
        fov_center_ref = np.array(scanimage_fov_meta["scanfields"]["centerXY"])

        # the affine transformation to convert scanfield coordinates to reference space
        # T_a = np.array(scanimage_fov_meta["scanfields"]["affine"])

        # the affine transform to convert pixel coordinates to reference space
        # T_p = np.array(scanimage_fov_meta["scanfields"]["pixelToRefTransform"])

        # transform to reference coordinate frame
        fov_topleft_ref = fov_center_ref - fov_size_ref / 2
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

        cs2d = create_coordinate_system_for_image(
            fov_size_px,
            um_per_px,
            ref_per_px,
            fov_topleft_ref,
        )

        # the image size assertion
        nptest.assert_array_almost_equal(
            cs2d.transform(fov_topleft_ref, "ref", "pixel"), np.zeros(2)
        )
        nptest.assert_array_almost_equal(
            cs2d.transform(fov_bottomright_ref, "ref", "pixel"), fov_size_px
        )
        nptest.assert_array_almost_equal(
            cs2d.transform(np.zeros(2), "pixel", "ref"), fov_topleft_ref
        )
        nptest.assert_array_almost_equal(
            cs2d.transform(fov_bottomright_ref, "ref", "um_global")
            - cs2d.transform(fov_topleft_ref, "ref", "um_global"),
            fov_size_um,
        )
        coordinate_systems[scanimage_fov_meta["roiUuid"]] = cs2d

    return coordinate_systems


def extract_fov_depths_from_scanimage_meta(
    scanimage_meta: dict,
    scanimage_params: dict,
    fov_uuids: Optional[List[str]] = None,
) -> Dict[str, np.float64]:
    """from scanimage metadata, extract the imaged depths of all field of views,
    return as a dict with fov name (uuids) and corresponding depth"""
    if fov_uuids is None:
        fov_uuids = _get_fov_uuids(scanimage_meta)

    # get the metadata for the fovs by given uuids
    scanimage_fov_metas = scanimage_meta["Artist"]["RoiGroups"]["imagingRoiGroup"][
        "rois"
    ]
    # extract the depth - a combination of the voicecoil (fast-z)
    # and gantry position
    fastz_pos = scanimage_params["hFastZ"]["position"]
    fov_depths = {}
    for fov_uuid in fov_uuids:
        (fov_meta,) = [
            meta for meta in scanimage_fov_metas if meta["roiUuid"] == fov_uuid
        ]
        fov_depths[fov_uuid] = -1 * (fov_meta["zs"] + fastz_pos)
    return fov_depths
