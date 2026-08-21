import numpy as np

from plane2brain import scanimage


def load_reference_points_from_meta(
    ref_img_meta: dict,
) -> dict:
    # in our case the known point is the center of the craniotomy
    ref_point = {
        "mlap": np.array(
            [ref_img_meta["centerMM"][key] * 1e3 for key in ["ML", "AP"]],
        ),
        "xy": np.array(
            [ref_img_meta["centerMM"][key] for key in ["x", "y"]],
        ),
        "deg": np.array(
            [ref_img_meta["centerDeg"][key] for key in ["x", "y"]],
        ),
    }
    return ref_point


def infer_ref_stack_virtual_corner(
    ref_img_scanimage_meta: dict,
    ref_img_size_px: np.ndarray,  # in 2d (X,Y)
    dims: tuple[str, str] = ("X", "Y"),
) -> tuple[np.ndarray, np.ndarray]:
    # get the corner of the reference stack in ref space
    # TODO refactor me
    stripes = ref_img_scanimage_meta["Artist"]["RoiGroups"]["imagingRoiGroup"]["rois"]

    topleft_corners = []
    bottomright_corners = []

    for scanimage_fov_meta in stripes:
        # get size and center for each fov
        fov_size_ref, fov_center_ref = scanimage.get_scanfield_size_ref(
            scanimage_fov_meta, dims=dims
        )
        # transform to reference coordinate frame
        fov_topleft_ref = fov_center_ref - fov_size_ref / 2
        topleft_corners.append(fov_topleft_ref)

        fov_bottomright_ref = fov_center_ref + fov_size_ref / 2
        bottomright_corners.append(fov_bottomright_ref)

    ref_img_topleft_ref = np.min(topleft_corners, axis=0)
    ref_img_bottomright_ref = np.max(bottomright_corners, axis=0)
    ref_per_px = (ref_img_bottomright_ref - ref_img_topleft_ref) / ref_img_size_px

    return ref_img_topleft_ref, ref_per_px
