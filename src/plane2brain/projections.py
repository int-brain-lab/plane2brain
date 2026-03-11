"""code for coordinate system projections"""

import numpy as np
from tqdm import tqdm

from plane2brain.coordinate_systems import (
    LinkedCoordinateSystems,
    setup_coordinate_systems_3d,
)

from plane2brain.linalg import (
    plane_normal_form,
    intersect_line_plane,
    intersect_line_mesh_nb,
    get_closest_face,
)

from plane2brain.scanimage import create_coordinate_systems_from_scanimage_meta

from plane2brain.atlas import ProjectionAtlas

from typing import Literal, Tuple, Dict, List

"""
 
 ########  ########   #######        ## ########  ######  ######## ####  #######  ##    ##  ######  
 ##     ## ##     ## ##     ##       ## ##       ##    ##    ##     ##  ##     ## ###   ## ##    ## 
 ##     ## ##     ## ##     ##       ## ##       ##          ##     ##  ##     ## ####  ## ##       
 ########  ########  ##     ##       ## ######   ##          ##     ##  ##     ## ## ## ##  ######  
 ##        ##   ##   ##     ## ##    ## ##       ##          ##     ##  ##     ## ##  ####       ## 
 ##        ##    ##  ##     ## ##    ## ##       ##    ##    ##     ##  ##     ## ##   ### ##    ## 
 ##        ##     ##  #######   ######  ########  ######     ##    ####  #######  ##    ##  ######  
 
"""


def project_coords_onto_atlas_surface(
    coords_um: np.ndarray,  # in um
    coordinate_systems_3d: LinkedCoordinateSystems,
    atlas: ProjectionAtlas,
    projection_vector: np.ndarray,  # project along this axis. Positive is defined to point away from the brain surface
) -> np.ndarray:
    # coordinate_systems_3d needs to contain imaging plane and mlapv
    assert [
        key in coordinate_systems_3d.coordinate_systems.keys()
        for key in ["imaging_plane", "mlapdv"]
    ]

    # in um in the imaging plane
    coords_um_ = np.concatenate([coords_um, np.zeros((coords_um.shape[0], 1))], axis=1)
    coords_on_imaging_plane = coordinate_systems_3d.transform(
        coords_um_, "imaging_plane", "mlapdv"
    )
    # project the rois onto the brain surface along the brain normal
    coords_on_surface = np.zeros_like(coords_on_imaging_plane)
    for i, _coords in enumerate(tqdm(coords_on_imaging_plane)):
        try:
            faces, intersection_points, ix = intersect_line_mesh_nb(
                atlas.mesh["vertices"],
                atlas.mesh["edges"],
                _coords,
                projection_vector * -1,
            )
            _, ix = get_closest_face(faces, _coords)
            coords_on_surface[i] = intersection_points[ix]
        except ValueError:
            # TODO logger warn
            coords_on_surface[i] = np.nan

    return coords_on_surface


def project_down_from_surface(
    coords_on_surface: np.ndarray,
    atlas: ProjectionAtlas,
    coords_depths: np.ndarray,
) -> np.ndarray:
    coords_mlapdv = np.zeros_like(coords_on_surface)
    for i, point in enumerate(tqdm(coords_on_surface)):
        p, n = atlas.get_plane_at_point_mlap(point[0], point[1], numba=True)
        coords_mlapdv[i] = (
            p + n * -1 * coords_depths[i]
        )  # either depth of the imaging plane

    return coords_mlapdv


# def setup_coordinate_systems_from_scanimage_meta(
#     scanimage_meta: dict,
# ) -> Tuple[LinkedCoordinateSystems, LinkedCoordinateSystems]: ...


# def project_scanimage_multifov_data(coords: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]: ...


# TODO this function should be project multi fov or similar
def project_from_scanimage_meta(
    coords_px: Dict[str, np.ndarray],  # keys = scanimage fov uuids
    scanimage_meta: dict,
    scanner_orientation: dict,
    common_point_mlap: np.ndarray,
    atlas: ProjectionAtlas,
    ds: int = 1,  # downsample factor for debugging
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, LinkedCoordinateSystems],
    LinkedCoordinateSystems,
]:
    # and creating the coordinate system
    # TODO integrate ref point 0,0 differnces
    # ref_point_mlap == craniotomy center
    ref_point_mlapdv, brain_normal_at_ref = atlas.get_plane_at_point_mlap(
        *common_point_mlap,
        numba=True,
    )
    coordinate_systems_3d = setup_coordinate_systems_3d(
        ref_point_mlapdv,
        brain_normal_at_ref,
        rotate_by=scanner_orientation["rotation"],
        invert_dims=scanner_orientation["invert_axis"],
    )

    # the 2d coordinate systems, by fov name
    fov_uuids = sorted(list(coords_px.keys()))
    coordinate_systems = create_coordinate_systems_from_scanimage_meta(
        scanimage_meta,
        fov_uuids=fov_uuids,
    )
    # coordinate_systems_fovs = dict(zip(fov_names, coordinate_systems))
    # axes = plotters.plot_brain_surface_points(brain_surface_points)

    coords_projected = {}
    for fov_uuid in fov_uuids:
        coords_projected[fov_uuid] = {}
        # get the pixel data
        _coords_px = coords_px[fov_uuid][::ds]  # downsample factor for debugging
        # project into global um space
        _coords_um = coordinate_systems[fov_uuid].transform(
            _coords_px,
            "pixel",
            "um_global",
        )
        coords_projected[fov_uuid]["pixel"] = _coords_px
        coords_projected[fov_uuid]["um"] = _coords_um

        # project onto brain atlas
        coords_projected[fov_uuid]["on_surface"] = project_coords_onto_atlas_surface(
            _coords_um,
            coordinate_systems_3d,
            atlas,
            brain_normal_at_ref,
        )

    return coords_projected, coordinate_systems, coordinate_systems_3d


"""
 
 ######## #### ##       ########       ###    ########        ## ##     ##  ######  ######## 
    ##     ##  ##          ##         ## ##   ##     ##       ## ##     ## ##    ##    ##    
    ##     ##  ##          ##        ##   ##  ##     ##       ## ##     ## ##          ##    
    ##     ##  ##          ##       ##     ## ##     ##       ## ##     ##  ######     ##    
    ##     ##  ##          ##       ######### ##     ## ##    ## ##     ##       ##    ##    
    ##     ##  ##          ##       ##     ## ##     ## ##    ## ##     ## ##    ##    ##    
    ##    #### ########    ##       ##     ## ########   ######   #######   ######     ##    
 
"""


def get_brain_surface_normal(
    reference_brain_surface_points: Dict,
    ref_img_meta: dict,
    coordinate_systems_ref: LinkedCoordinateSystems,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """from the reference points, calculate a plane that approximates the brain surface
    and it's normal. Additionally, returns the average depth of the three points that is
    later used to adjust the apparent depth of a cell."""
    # TODO decouple here:
    # IBL specific
    # and scanimage specific

    # DOCME user selected
    stack_ixs = [
        point["stack_idx"] for point in reference_brain_surface_points["points"]
    ]
    # the position of the voice coil (for z offset calculation)
    # fastz_pos = ref_img_meta["scanImageParams"]['hFastZ']['position']
    # inversion of the sign: positive is up
    stack_dv = (
        -1 * np.array(ref_img_meta["scanImageParams"]["hStackManager"]["zs"])[stack_ixs]
    )
    dv_avg = np.average(
        stack_dv
    )  # horizontally average plane between the selected surface points
    brain_surface_points_rel = np.array(
        [point["coords"] for point in reference_brain_surface_points["points"]]
    )
    brain_surface_points_rel_um = coordinate_systems_ref.transform(
        brain_surface_points_rel, "image", "um"
    )  # NOTE this is um_global
    # these are the 3 points on the brain surface, relative, in um
    brain_surface_points_rel_um_3d = np.concatenate(
        [brain_surface_points_rel_um, stack_dv[:, np.newaxis]], axis=1
    )
    p_surface, n_surface = plane_normal_form(brain_surface_points_rel_um_3d)
    # invert if pointing downards
    if n_surface[2] < 0:
        n_surface *= -1

    return p_surface, n_surface, dv_avg


def correct_coords_for_tilt_2d(
    coords: Dict[str, Dict[str, np.ndarray]],  # uuid -> pixel etc
    coordinate_systems: Dict[str, LinkedCoordinateSystems],
    fov_depths: Dict[str, np.float64],
    p_surface: np.ndarray,
    n_surface: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    """when the bain surface is tilted to the optical axis the coordinates of
    deeper planes shift relative to those at the surface. The extracted positions
    of cells are a) shifted in x and y, and b) the depth of the plane is not the
    true morphological depth of the cell beneath the surface

    to correct for this, we take the apparent location of the cell, and project
    that point onto the the brain surface (as determined by the reference points)
    along the brain normal (likewise form the ref points)

    this yields: the corrected ml and ap coordinates
    the distance to the plane is the true dv depth

    NOTE this adds to the multi fov coords dict
    """

    for uuid in list(coords.keys()):
        _coords = coords[uuid]["pixel"]
        coords_um = coordinate_systems[uuid].transform(_coords, "pixel", "um_global")

        # coords_um = ... # 2d array in um_global!
        # dv_below_surface = np.zeros(coords_um.shape[0])
        coords_surface = np.zeros((coords_um.shape[0], 3))

        # turning this into 3d coordinates using the fov depth
        coords_um_3d = np.concatenate(
            [coords_um, np.ones((coords_um.shape[0], 1)) * fov_depths[uuid]], axis=1
        )

        for i, _coords in enumerate(coords_um_3d):
            # depth below plane

            # the ml, ap of these are the corrected values
            coords_surface[i] = intersect_line_plane(
                _coords, n_surface, p_surface, n_surface
            )

        # true dv is the distance between the point and the plane
        dv_below_surface = np.sqrt(np.sum((coords_um_3d - coords_surface) ** 2, axis=1))

        coords[uuid]["um_corrected"] = coords_surface[:, :-1]
        coords[uuid]["dv_below_surface"] = dv_below_surface

    return coords


def reproject_coords(  # FIXME refactor
    coords: Dict[str, Dict[str, np.ndarray]],
    coordinate_systems_3d: LinkedCoordinateSystems,
    atlas: ProjectionAtlas,
    projection_vector: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    for uuid in list(coords.keys()):
        coords_on_surface = project_coords_onto_atlas_surface(
            coords[uuid]["um_corrected"],
            coordinate_systems_3d,
            atlas,
            projection_vector,
        )
        coords_reprojected = project_down_from_surface(
            coords_on_surface,
            atlas,
            coords_depths=coords[uuid]["dv_below_surface"],
        )
        coords[uuid]["reprojected"] = coords_reprojected
    return coords
