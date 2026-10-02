"""Imaging geometry from ScanImage metadata.

The metadata is ScanImage's TIFF header (`rawScanImageMeta`): the TIFF tags, the
Roi groups in `Artist` and the `SI.*` properties in `Software`.
"""

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import numpy.testing as nptest
import pandas as pd

from plane2brain.core import (
    Image,
    LinkedCoordinateSystems,
    create_coordinate_system_for_image,
)


def _get_uuids(
    scanimage_meta: dict[str, Any],
    enabled_only: bool = True,
) -> list[str]:
    """Return the ScanImage Roi uuids from metadata.

    Args:
        scanimage_meta: Full ScanImage metadata dictionary.
        enabled_only: If True, only the Rois that were enabled (imaged) are returned.

    Returns:
        The Roi uuids, in the order of the Rois in the metadata.
    """
    scanimage_roi_metas = scanimage_meta["Artist"]["RoiGroups"]["imagingRoiGroup"][
        "rois"
    ]
    return [
        meta["roiUuid"]
        for meta in scanimage_roi_metas
        if meta["enable"] or not enabled_only
    ]


def _get_software_values(scanimage_meta: dict[str, Any]) -> dict[str, str]:
    """Return the raw values of the ScanImage properties in the `Software` tag.

    Keys are the property names (e.g. "SI.hStackManager.zs"), values the
    unparsed MATLAB literals (e.g. "[-450 -250]").
    """
    lines = scanimage_meta["Software"].splitlines()
    return dict(line.split(" = ", 1) for line in lines if " = " in line)


def _parse_matlab_numeric(value: str) -> np.ndarray:
    """Parse a numeric MATLAB literal (scalar, vector or matrix) into a 2D array."""
    rows = value.strip().strip("[]").split(";")
    return np.array(
        [[float(element) for element in row.replace(",", " ").split()] for row in rows]
    )


def get_scanfield_size_ref(
    scanimage_roi_meta: dict[str, Any],
    dims: tuple[str, str] = ("X", "Y"),
) -> tuple[np.ndarray, np.ndarray]:
    """Read the scanfield size and center of a ScanImage Roi.

    Args:
        scanimage_roi_meta: The metadata of a single-scanfield Roi.
        dims: The axis order for the returned arrays. Use
            `("X", "Y")` by default, or `("Y", "X")` to swap axes.

    Returns:
        Tuple of (scanfield_size_ref, scanfield_center_ref), each a shape (2,)
        array in reference space units (optical degrees), ordered according to
        `dims`.
    """
    scanfield_size_ref = np.array(scanimage_roi_meta["scanfields"]["sizeXY"])
    scanfield_center_ref = np.array(scanimage_roi_meta["scanfields"]["centerXY"])
    if dims == ("Y", "X"):
        scanfield_size_ref = scanfield_size_ref[::-1]
        scanfield_center_ref = scanfield_center_ref[::-1]

    return scanfield_size_ref, scanfield_center_ref


def get_scanfield_size_px(
    scanimage_roi_meta: dict[str, Any],
    dims: tuple[str, str] = ("X", "Y"),
) -> np.ndarray:
    """Read the scanfield size in pixels of a ScanImage Roi.

    Args:
        scanimage_roi_meta: The metadata of a single-scanfield Roi.
        dims: The axis order for the returned array, `("X", "Y")` or `("Y", "X")`.

    Returns:
        The size in pixels, shape (2,), ordered according to `dims`.
    """
    scanfield_size_px = np.array(scanimage_roi_meta["scanfields"]["pixelResolutionXY"])
    if dims == ("Y", "X"):
        scanfield_size_px = scanfield_size_px[::-1]

    return scanfield_size_px


def get_resolution_from_scanimage_meta(
    scanimage_meta: dict[str, Any],
    dims: tuple[str, str] = ("X", "Y"),
) -> np.ndarray:
    """Return the pixel size in micrometers from the TIFF resolution tags.

    Args:
        scanimage_meta: Full ScanImage metadata dictionary.
        dims: The axis order for the returned array, `("X", "Y")` or `("Y", "X")`.

    Returns:
        µm per pixel, shape (2,), ordered according to `dims`.

    Raises:
        ValueError: If the resolution unit is not centimeters.
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


def get_objective_resolution(scanimage_meta: dict[str, Any]) -> float:
    """Return the objective resolution (µm per optical degree) from ScanImage metadata."""
    return float(_get_software_values(scanimage_meta)["SI.objectiveResolution"])


def get_roi_meta(
    scanimage_meta: dict[str, Any],
    uuid: str,
) -> dict[str, Any]:
    """Return the metadata of the ScanImage Roi with the given uuid.

    Raises:
        ValueError: If not exactly one Roi has the uuid.
    """
    (roi_meta,) = [
        meta
        for meta in scanimage_meta["Artist"]["RoiGroups"]["imagingRoiGroup"]["rois"]
        if meta["roiUuid"] == uuid
    ]
    return roi_meta


def create_coordinate_systems_from_scanimage_meta(
    scanimage_meta: dict[str, Any],
    uuids: list[str] | None = None,
    dims: tuple[str, str] = ("X", "Y"),
) -> dict[str, LinkedCoordinateSystems]:
    """Build the 2D coordinate systems of ScanImage Rois (superseded).

    Superseded by `create_images_from_scanimage_meta`; keyed by Roi uuid, so it
    cannot tell the slices of a Roi apart. µm per pixel come from the TIFF
    resolution tags.

    Args:
        scanimage_meta: Full ScanImage metadata dictionary.
        uuids: Optional list of Roi uuids to process. If omitted, all enabled
            Rois in `scanimage_meta` are processed.
        dims: The axis order used for X/Y metadata values.

    Returns:
        A mapping from Roi uuid to `LinkedCoordinateSystems`.
    """
    if uuids is None:
        uuids = _get_uuids(scanimage_meta)

    # pixel resolution from metadata
    um_per_px = get_resolution_from_scanimage_meta(scanimage_meta, dims=dims)
    coordinate_systems: dict[str, LinkedCoordinateSystems] = {}

    for uuid in uuids:
        scanimage_roi_meta = get_roi_meta(scanimage_meta, uuid)
        # misleading variable naming by ScanImage but here too X is the line = resonant scanner, and Y is the line number
        # this, combined with the fact that on the reference image, the strips are extended vertically
        # means: XY is AP, ML
        fov_size_px = get_scanfield_size_px(scanimage_roi_meta, dims=dims)
        fov_size_um = fov_size_px * um_per_px

        # the size of the scanfield is stored in the metadata in
        # "sizeXY: [width, height] size of the scanfield in optical degrees in the coordinate system in which it is defined"
        # (taken from the doc)
        # unclear if this is correct (reference space is not equal to optical degrees)
        # it rather seems it's in reference space
        # see below for proof at (*)

        # the center and size are expressed in the scanfield coordinate system
        fov_size_ref, fov_center_ref = get_scanfield_size_ref(
            scanimage_roi_meta, dims=dims
        )

        # transform to reference coordinate frame
        fov_topleft_ref = fov_center_ref - fov_size_ref / 2

        # (*) to show that sizeXY is in ref space
        # according to the docs: the affine transform to convert pixel coordinates to reference space
        np.array(scanimage_roi_meta["scanfields"]["pixelToRefTransform"])

        # this will actually fail due to imprecision!
        # np.testing.assert_allclose(
        #     (T_p @ np.append(np.zeros(2), 1))[:-1], fov_topleft_ref, rtol=1e-3
        # )

        # (T_p @ np.append(fov_size_px, 1))[:-1] - (T_p @ np.append(np.zeros(2), 1))[:-1]
        # T_a = np.array(scanimage_roi_meta["scanfields"]["affine"])

        fov_bottomright_ref = fov_topleft_ref + fov_size_ref
        nptest.assert_array_almost_equal(
            fov_size_ref, fov_bottomright_ref - fov_topleft_ref
        )
        ref_per_px = fov_size_ref / fov_size_px
        px_per_ref = 1 / ref_per_px

        # next we need to know what is the size of a pixel in reference space?
        um_per_ref = um_per_px * px_per_ref
        1 / um_per_ref
        fov_topleft_ref * um_per_ref

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
        coordinate_systems[uuid] = coordinate_system

    return coordinate_systems


def extract_slice_depths_from_scanimage_meta(
    scanimage_meta: dict[str, Any],
    uuids: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Extract the raw z of the slices at which each ScanImage Roi is imaged.

    The values are raw ScanImage z positions in µm. They can only be interpreted
    as depth below the brain surface together with a brain surface.

    Args:
        scanimage_meta: Full ScanImage metadata dictionary.
        uuids: Optional list of Roi uuids. If omitted, all enabled Rois in the
            metadata are used.

    Returns:
        Mapping from Roi uuid to the raw z of its slices in µm.

    Raises:
        NotImplementedError: For dual-plane acquisitions, where the fast z
            actuators are at different z within a slice.
    """
    if uuids is None:
        uuids = _get_uuids(scanimage_meta)

    software_values = _get_software_values(scanimage_meta)
    # z of every slice, as seen by the first fast z actuator
    zs = _parse_matlab_numeric(software_values["SI.hStackManager.zs"]).flatten()
    # dual-plane: the actuators are at different z (e.g. SP061: zs = 240, all = [240 200])
    zs_all_actuators = software_values.get("SI.hStackManager.zsAllActuators")
    if zs_all_actuators is not None:
        zs_all_actuators = _parse_matlab_numeric(zs_all_actuators)
        if not np.all(zs_all_actuators == zs_all_actuators[:, :1]):
            raise NotImplementedError(
                "dual-plane acquisition, use create_images_from_scanimage_dual_plane"
            )

    slice_depths = {}
    for uuid in uuids:
        # this is the old - probably now deprecated approach - to be removed
        # fov_meta = get_roi_meta(scanimage_meta, uuid)
        # fov_depths[uuid] = -1 * (fastz_pos + fov_meta["zs"])
        roi_meta = get_roi_meta(scanimage_meta, uuid)
        if roi_meta["discretePlaneMode"]:
            # a discrete plane Roi is only imaged at the slices matching its own zs
            slice_depths[uuid] = zs[np.isin(zs, roi_meta["zs"])]
        else:
            slice_depths[uuid] = zs
    return slice_depths


def get_image_map(
    scanimage_meta: dict[str, Any],
    enabled_only: bool = True,
) -> pd.DataFrame:
    """Map image indices to the combinations of Roi uuid and raw slice z.

    Images are ordered by slice z (ascending), then by the order of the Rois in
    the metadata. This matches the IBL FOV numbering (alf/FOV_xx).

    Args:
        scanimage_meta: Full ScanImage metadata dictionary.
        enabled_only: If True, only the Rois that were enabled (imaged) are used.

    Returns:
        One row per image with the columns "index" (image index), "z" (raw ScanImage
        z in µm) and "uuid" (Roi uuid).
    """
    uuids = _get_uuids(scanimage_meta, enabled_only=enabled_only)
    slice_depths = extract_slice_depths_from_scanimage_meta(scanimage_meta, uuids=uuids)

    # all slice z of the acquisition, ascending
    zs = np.unique(np.concatenate(list(slice_depths.values())))
    rows = []
    # TODO - this is where the order is encoded. First, all
    # scanfields are imaged at slight height a, then all at b, etc
    # not sure how this behaves with multiple slices at multiple
    # scanfields
    for z in zs:
        for uuid in uuids:
            if z in slice_depths[uuid]:
                rows.append((len(rows), float(z), uuid))
    return pd.DataFrame(rows, columns=["index", "z", "uuid"])


def _create_image_from_roi_meta(
    scanimage_meta: dict[str, Any],
    roi_meta: dict[str, Any],
    dims: tuple[str, str] = ("X", "Y"),
) -> Image:
    """Create the image of a ScanImage Roi from its scanfield geometry.

    See `create_image_from_scanimage_meta` for the coordinate systems.
    """
    if isinstance(roi_meta["scanfields"], list):
        raise NotImplementedError("Rois with several scanfields are not supported yet")

    # µm per optical degree
    um_per_ref = get_objective_resolution(scanimage_meta)
    size_px = get_scanfield_size_px(roi_meta, dims=dims)
    size_ref, center_ref = get_scanfield_size_ref(roi_meta, dims=dims)
    ref_per_px = size_ref / size_px

    coordinate_systems = create_coordinate_system_for_image(
        size_px,
        um_per_px=ref_per_px * um_per_ref,
        ref_per_px=ref_per_px,
        img_topleft_ref=center_ref - size_ref / 2,
    )
    return Image(size_px=size_px, coordinate_systems=coordinate_systems)


def create_image_from_scanimage_meta(
    scanimage_meta: dict[str, Any],
    uuid: str,
    z: float,
    dims: tuple[str, str] = ("X", "Y"),
) -> Image:
    """Create the image of a ScanImage Roi at one slice.

    The 2D coordinate systems have ScanImage's reference space ("ref", optical
    degrees) as their common world, so "um_global" means the same in all images
    of an acquisition. No microscope orientation is involved yet, it only matters
    once the images are placed onto a plane in 3D.

    Coordinate systems:
        "ref": ScanImage reference space in optical degrees.
        "um_global": µm, origin and axes of "ref".
        "pixel": pixel indices, (0, 0) at the top-left corner of the image.
        "um_image": µm inside the image, origin at the top-left corner.
        "image": image size normalized to 1, origin at the top-left corner.

    Args:
        scanimage_meta: Full ScanImage metadata dictionary.
        uuid: uuid of the Roi.
        z: Raw ScanImage z of the slice in µm, as in the image map.
        dims: Axis order of the metadata values, `("X", "Y")` or `("Y", "X")`.

    Returns:
        The image, with `depth_below_surface` not assigned yet.
    """
    slice_depths = extract_slice_depths_from_scanimage_meta(scanimage_meta, [uuid])
    if z not in slice_depths[uuid]:
        raise ValueError(f"Roi {uuid} is not imaged at z = {z}")
    return _create_image_from_roi_meta(
        scanimage_meta, get_roi_meta(scanimage_meta, uuid), dims=dims
    )


def create_images_from_scanimage_meta(
    scanimage_meta: dict[str, Any],
    dims: tuple[str, str] = ("X", "Y"),
    enabled_only: bool = True,
) -> list[Image]:
    """Create the images of all Roi and slice combinations of an acquisition.

    Args:
        scanimage_meta: Full ScanImage metadata dictionary.
        dims: Axis order of the metadata values, `("X", "Y")` or `("Y", "X")`.
        enabled_only: If True, only the Rois that were enabled (imaged) are used.

    Returns:
        The images in the order of the image map: the list index is the image index
        (= IBL alf/FOV_xx).
    """
    image_map = get_image_map(scanimage_meta, enabled_only=enabled_only)
    return [
        create_image_from_scanimage_meta(scanimage_meta, uuid, z, dims=dims)
        for uuid, z in zip(image_map["uuid"], image_map["z"])
    ]


def create_images_from_scanimage_dual_plane(
    scanimage_meta: dict[str, Any],
    actuator_channels: Sequence[int],
    order: Literal["roi", "actuator", "z"] = "roi",
    dims: tuple[str, str] = ("X", "Y"),
    enabled_only: bool = True,
) -> tuple[list[Image], pd.DataFrame]:
    """Create the images of a dual-plane acquisition (draft).

    Assumes a dual-plane acquisition without checking: each slice is imaged at
    the same time at one depth per fast z actuator (`SI.hStackManager.zsAllActuators`),
    each depth recorded on its own channel. Both depths of a Roi share its
    scanfield geometry.

    Args:
        scanimage_meta: Full ScanImage metadata dictionary.
        actuator_channels: The channel recording the depth of each fast z
            actuator, e.g. `(1, 2)`. Rig specific, not part of the metadata.
        order: Order of the images: "roi" (by slice, Roi, actuator: the depths of
            a Roi next to each other, like the IBL FOVs), "actuator" (by slice,
            actuator, Roi) or "z" (by z ascending, then Roi, like `get_image_map`).
        dims: Axis order of the metadata values, `("X", "Y")` or `("Y", "X")`.
        enabled_only: If True, only the Rois that were enabled (imaged) are used.

    Returns:
        The images, and the image map with one row per image and the columns
        "index", "z", "uuid", "slice" and "actuator" (both counted from 0) and
        "channel". The list index of the images is the "index".
    """
    sort_columns = {
        "roi": ["slice", "roi_position", "actuator"],
        "actuator": ["slice", "actuator", "roi_position"],
        "z": ["z", "roi_position"],
    }
    if order not in sort_columns:
        raise ValueError(f"order must be one of {list(sort_columns)}, got {order!r}")
    # raw z of each slice (rows) and fast z actuator (columns)
    zs_all_actuators = _parse_matlab_numeric(
        _get_software_values(scanimage_meta)["SI.hStackManager.zsAllActuators"]
    )
    if len(actuator_channels) != zs_all_actuators.shape[1]:
        raise ValueError(
            f"expected one channel per fast z actuator ({zs_all_actuators.shape[1]}), "
            f"got {len(actuator_channels)}"
        )

    rows = []
    uuids = _get_uuids(scanimage_meta, enabled_only=enabled_only)
    for roi_position, uuid in enumerate(uuids):
        if get_roi_meta(scanimage_meta, uuid)["discretePlaneMode"]:
            raise NotImplementedError("discrete plane Rois are not supported here yet")
        for slice_index, slice_zs in enumerate(zs_all_actuators):
            for actuator, (z, channel) in enumerate(zip(slice_zs, actuator_channels)):
                rows.append(
                    (float(z), uuid, slice_index, actuator, channel, roi_position)
                )
    columns = ["z", "uuid", "slice", "actuator", "channel", "roi_position"]
    image_map = (
        pd.DataFrame(rows, columns=columns)
        .sort_values(sort_columns[order], kind="stable")
        .drop(columns="roi_position")
        .reset_index(drop=True)
    )
    image_map.insert(0, "index", range(len(image_map)))

    images = [
        _create_image_from_roi_meta(
            scanimage_meta, get_roi_meta(scanimage_meta, uuid), dims
        )
        for uuid in image_map["uuid"]
    ]
    return images, image_map
