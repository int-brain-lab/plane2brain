"""Projections of image pixels onto the brain surface and into the brain."""

import numpy as np
from tqdm import tqdm

from plane2brain.atlas import ProjectionAtlas
from plane2brain.core import Plane
from plane2brain.linalg import (
    intersect_line_mesh_precomputed_nb,
    intersect_line_plane,
)

"""
 
 ########  ########   #######        ## ########  ######  ######## ####  #######  ##    ##  ######  
 ##     ## ##     ## ##     ##       ## ##       ##    ##    ##     ##  ##     ## ###   ## ##    ## 
 ##     ## ##     ## ##     ##       ## ##       ##          ##     ##  ##     ## ####  ## ##       
 ########  ########  ##     ##       ## ######   ##          ##     ##  ##     ## ## ## ##  ######  
 ##        ##   ##   ##     ## ##    ## ##       ##          ##     ##  ##     ## ##  ####       ## 
 ##        ##    ##  ##     ## ##    ## ##       ##    ##    ##     ##  ##     ## ##   ### ##    ## 
 ##        ##     ##  #######   ######  ########  ######     ##    ####  #######  ##    ##  ######  
 
"""


# TODO deprecated
# def project_coords_onto_atlas_surface(
#     coords_um: np.ndarray,  # in um
#     coordinate_systems_3d: LinkedCoordinateSystems,
#     atlas: ProjectionAtlas,
#     projection_vector: np.ndarray,  # project along this axis. Positive is defined to point away from the brain surface
# ) -> np.ndarray:
#     # coordinate_systems_3d needs to contain imaging_plane and mlapdv
#     assert all(
#         key in coordinate_systems_3d.coordinate_systems
#         for key in ["imaging_plane", "mlapdv"]
#     )

#     # in um in the imaging plane
#     coords_um_ = np.concatenate([coords_um, np.zeros((coords_um.shape[0], 1))], axis=1)
#     coords_on_imaging_plane = coordinate_systems_3d.transform(
#         coords_um_, "imaging_plane", "mlapdv"
#     )
#     # project the rois onto the brain surface along the projection vector
#     coords_on_surface = np.zeros_like(coords_on_imaging_plane)
#     for i, _coords in enumerate(
#         tqdm(coords_on_imaging_plane, desc="projecting on surface")
#     ):
#         try:
#             _, intersection_points, _ = intersect_line_mesh_precomputed_nb(
#                 atlas.mesh["vertices"],
#                 atlas.mesh["edges"],
#                 atlas.mesh["normals"],
#                 _coords,
#                 projection_vector * -1,
#             )
#             # pick the intersection point nearest the imaging-plane point;
#             # face centroid was previously used as a proxy, which can flip
#             # the choice when one face is much more elongated than the other
#             ix = np.argmin(np.linalg.norm(intersection_points - _coords, axis=1))
#             coords_on_surface[i] = intersection_points[ix]
#         except ValueError:
#             # TODO logger warn
#             coords_on_surface[i] = np.nan

#     return coords_on_surface


def project_onto_surface(
    coords: np.ndarray,
    atlas: ProjectionAtlas,
    projection_vector: np.ndarray,
) -> np.ndarray:
    """Project points along a direction onto the brain surface.

    For each point, the line along the projection vector is intersected with the
    surface mesh, and the intersection nearest to the point is taken.

    Args:
        coords: Points in atlas space in µm, shape `(N, 3)`.
        atlas: Atlas with the surface mesh.
        projection_vector: Direction of the projection, pointing away from the
            brain, shape `(3,)`.

    Returns:
        The points on the brain surface in atlas space in µm, shape `(N, 3)`.
        Points whose line misses the surface are NaN.
    """
    coords_on_surface = np.full_like(coords, np.nan)
    for i, point in enumerate(tqdm(coords, desc="projecting onto surface")):
        # intersect the line along the projection vector with the surface mesh
        try:
            _, intersection_points, _ = intersect_line_mesh_precomputed_nb(
                atlas.mesh["vertices"],
                atlas.mesh["edges"],
                atlas.mesh["normals"],
                point,
                projection_vector * -1,
            )
            # pick the intersection point nearest the point on the plane;
            # face centroid was previously used as a proxy, which can flip
            # the choice when one face is much more elongated than the other
            ix = np.argmin(np.linalg.norm(intersection_points - point, axis=1))
            coords_on_surface[i] = intersection_points[ix]
        except ValueError:
            # TODO logger warn - the line misses the mesh, the point stays NaN
            pass
    return coords_on_surface


def project_down_from_surface(
    coords_on_surface: np.ndarray,
    atlas: ProjectionAtlas,
    coords_depths: np.ndarray,
) -> np.ndarray:
    """Project points on the brain surface into the brain by their depth.

    For each point, the local surface plane is looked up at its ML/AP location,
    and the point steps inward from there along the inverted surface normal.

    Args:
        coords_on_surface: Points on the brain surface in atlas space in µm,
            shape `(N, 3)`. Must not contain NaN.
        atlas: Atlas with the surface mesh.
        coords_depths: Depth below the surface in µm for each point, shape `(N,)`.

    Returns:
        The points in the brain in atlas space in µm, shape `(N, 3)`.
    """
    coords_mlapdv = np.zeros_like(coords_on_surface)
    for i, point in enumerate(tqdm(coords_on_surface, desc="projecting into brain")):
        surface_plane = atlas.get_plane_at_point_mlap(point[0], point[1], numba=True)
        coords_mlapdv[i] = (
            surface_plane.point + surface_plane.normal * -1 * coords_depths[i]
        )  # step inward from the surface along the inverted normal by the cell depth

    return coords_mlapdv


# def reproject_coords(  # FIXME refactor
#     coords: dict[str, dict[str, np.ndarray]],
#     coordinate_systems_3d: LinkedCoordinateSystems,
#     atlas: ProjectionAtlas,
#     projection_vector: np.ndarray,
# ) -> dict[str, dict[str, np.ndarray]]:
#     for uuid in list(coords.keys()):
#         coords_on_surface = project_coords_onto_atlas_surface(
#             coords[uuid]["um_corrected"],
#             coordinate_systems_3d,
#             atlas,
#             projection_vector,
#         )
#         coords_reprojected = project_down_from_surface(
#             coords_on_surface,
#             atlas,
#             coords_depths=coords[uuid]["dv_below_surface"],
#         )
#         coords[uuid]["reprojected"] = coords_reprojected
#     return coords


class Projection:
    """A configured projection of the pixels of a plane's images into the atlas.

    The results are kept in `coordinates`, which maps each stage to a list with
    one `(N, 3)` array per image, in the order of `plane.images` (the list index
    is the image index): "on_plane" (computed on construction), "on_surface"
    (`project_images_onto_surface`) and "in_brain" (`project_images_into_brain`).
    """

    def __init__(
        self,
        atlas: ProjectionAtlas,
        plane: Plane,
        pixels: list[np.ndarray] | None = None,
        *,
        projection_vector: np.ndarray | None = None,
        downsample: int = 1,
    ) -> None:
        """Configure the projection and place the pixels on the plane.

        Args:
            atlas: Atlas with the surface mesh to project onto.
            plane: Plane with images, linked to atlas space.
            pixels: Pixel indices to project, one `(N, 2)` array per image. If
                omitted, the full pixel grid of each image is used.
            projection_vector: Direction of the projection onto the surface,
                pointing away from the brain. Defaults to the plane's normal.
            downsample: Keep only every n-th pixel, for debugging.
        """
        if plane.coordinate_systems_3d is None:
            raise ValueError("plane is not linked to atlas space, it has no anchor")
        if pixels is None:
            pixels = [image.pixel_indices for image in plane.images]
        else:
            pixels = [pixels] * len(plane.images)

        self.atlas = atlas
        self.plane = plane
        self.pixels = [image_pixels[::downsample] for image_pixels in pixels]
        self.projection_vector = (
            plane.normal
            if projection_vector is None
            else np.asarray(projection_vector, dtype=float)
        )
        # initializing with the defaults
        # these are dicts with indices as keys, to be robust against
        # writing the coordinates in a different order (outside of
        # this repository)
        self.coordinates: dict[str, dict[int, np.ndarray]] = {
            "on_plane": {},
            "on_surface": {},
            "in_brain": {},
        }
        for i, image in enumerate(plane.images):
            points_um = image.coordinate_systems.transform(
                self.pixels[i], "pixel", "um_global"
            )
            # lift onto the plane: the third coordinate is the distance from the plane
            points_um = np.column_stack([points_um, np.zeros(len(points_um))])
            self.coordinates["on_plane"][i] = plane.coordinate_systems_3d.transform(
                points_um, "um_global", "mlapdv"
            )

    def project_images_onto_surface(self) -> list[np.ndarray]:
        """Project the points of each image on the plane onto the brain surface.

        Uses `coordinates["on_plane"]` and stores the result as
        `coordinates["on_surface"]`.

        Returns:
            The points on the brain surface in atlas space in µm, one `(N, 3)`
            array per image. Points whose projection misses the surface are NaN.
        """
        for i, image in enumerate(self.plane.images):
            self.coordinates["on_surface"][i] = project_onto_surface(
                self.coordinates["on_plane"][i],
                self.atlas,
                self.projection_vector,
            )

        return self.coordinates["on_surface"]

    def project_images_into_brain(self) -> list[np.ndarray]:
        """Project the points of each image on the surface into the brain by their depth.

        Steps from each surface point inward along the local surface normal by
        its pixel's `depth_below_surface`. The depths are known per pixel of the
        full pixel grid, so each image needs exactly one point per pixel.

        Uses `coordinates["on_surface"]` (run `project_images_onto_surface` first)
        and stores the result as `coordinates["in_brain"]`.

        Returns:
            The points in the brain in atlas space in µm, one `(N, 3)` array per
            image. Points that missed the surface stay NaN.
        """

        for i, image in enumerate(self.plane.images):
            on_surface = self.coordinates["on_surface"][i]
            depths = image.depth_below_surface
            assert len(depths) == len(on_surface)

            in_brain = np.full_like(on_surface, np.nan)
            # only the points that hit the surface can be projected further
            hit = ~np.isnan(on_surface).any(axis=1)
            in_brain[hit] = project_down_from_surface(
                on_surface[hit], self.atlas, depths[hit]
            )
            self.coordinates["in_brain"][i] = in_brain
        return self.coordinates["in_brain"]


# TODO deprecated
# def setup_coordinate_systems_from_scanimage_meta(
#     scanimage_meta: dict,
#     common_point_mlap: np.ndarray,
#     atlas: ProjectionAtlas,
#     scanner_orientation: dict,
#     fov_uuids: list[str],
# ) -> tuple[LinkedCoordinateSystems, LinkedCoordinateSystems]:
#     # and creating the coordinate system
#     # TODO integrate ref point 0,0 differences
#     # ref_point_mlap == craniotomy center
#     surface_plane_at_ref = atlas.get_plane_at_point_mlap(
#         *common_point_mlap,
#         numba=True,
#     )
#     coordinate_systems_3d = setup_coordinate_systems_3d(
#         surface_plane_at_ref.point,
#         surface_plane_at_ref.normal,
#         rotate_by=scanner_orientation["rotation"],
#         invert_dims=scanner_orientation["invert_axis"],
#     )

#     # the 2d coordinate systems, by fov name
#     coordinate_systems_2d = create_coordinate_systems_from_scanimage_meta(
#         scanimage_meta,
#         uuids=fov_uuids,
#     )
#     return coordinate_systems_2d, coordinate_systems_3d


# TODO deprecated
# def project_scanimage_fovs(
#     coords_px: dict[str, np.ndarray],
#     coordinate_systems_2d: dict[str, LinkedCoordinateSystems],
#     coordinate_systems_3d: LinkedCoordinateSystems,
#     atlas: ProjectionAtlas,
#     projection_vector: np.ndarray,
#     ds: int = 1,
# ) -> dict[str, dict[str, np.ndarray]]:
#     coords_projected = {}
#     fov_uuids = sorted(coords_px.keys())
#     for fov_uuid in fov_uuids:
#         coords_projected[fov_uuid] = {}
#         # get the pixel data
#         _coords_px = coords_px[fov_uuid][::ds]  # downsample factor for debugging
#         # project into global um space
#         _coords_um = coordinate_systems_2d[fov_uuid].transform(
#             _coords_px,
#             "pixel",
#             "um_global",
#         )
#         coords_projected[fov_uuid]["pixel"] = _coords_px
#         coords_projected[fov_uuid]["um_global"] = _coords_um

#         # project onto brain atlas
#         coords_projected[fov_uuid]["on_surface"] = project_coords_onto_atlas_surface(
#             _coords_um,
#             coordinate_systems_3d,
#             atlas,
#             projection_vector,
#         )

#     return coords_projected


"""
 
 ######## #### ##       ########       ###    ########        ## ##     ##  ######  ######## 
    ##     ##  ##          ##         ## ##   ##     ##       ## ##     ## ##    ##    ##    
    ##     ##  ##          ##        ##   ##  ##     ##       ## ##     ## ##          ##    
    ##     ##  ##          ##       ##     ## ##     ##       ## ##     ##  ######     ##    
    ##     ##  ##          ##       ######### ##     ## ##    ## ##     ##       ##    ##    
    ##     ##  ##          ##       ##     ## ##     ## ##    ## ##     ## ##    ##    ##    
    ##    #### ########    ##       ##     ## ########   ######   #######   ######     ##    
 
"""


# TODO potentially move outside of this repository
def correct_coords_for_tilt_2d(
    coords_um: np.ndarray,
    depth: float,
    p_surface: np.ndarray,
    n_surface: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Correct cell coordinates for the tilt between the brain surface and optical axis.

    When the brain surface is tilted relative to the optical axis, cells in
    deeper imaging planes appear shifted in x/y, and the z-stack depth is not
    the true morphological depth below the surface. This function projects each
    cell's apparent 3D position onto the tilted brain surface plane along the
    surface normal, yielding corrected ML/AP coordinates and true DV depth.

    Args:
        coords_um: Coordinates in "um_global" of one image, shape (N, 2).
        depth: Imaging depth of the image in µm.
        p_surface: A point on the brain surface plane, shape (3,).
        n_surface: Unit normal of the brain surface plane, shape (3,).

    Returns:
        Tuple of (um_corrected, dv_below_surface_corrected):
            um_corrected: Corrected coordinates in "um_global", shape (N, 2).
            dv_below_surface_corrected: Corrected depth below the surface in µm,
                shape (N,).
    """
    # turning this into 3d coordinates using the depth
    # ml and ap are in um_global
    # dv is depth below surface.
    # note that this is not in mlapdv

    coords_um_3d = np.concatenate(
        [coords_um, np.ones((coords_um.shape[0], 1)) * depth], axis=1
    )
    # from a given point, go along the brain normal until it intersects
    # the brain surface as defined by n_surface, p_surface
    coords_surface = np.zeros((coords_um.shape[0], 3))  # alloc
    for i, _coords in enumerate(coords_um_3d):
        coords_surface[i] = intersect_line_plane(
            _coords, n_surface, p_surface, n_surface
        )

    # the refined dv estimate is the distance between the original point
    # and the intersection point with the plane
    dv_below_surface_corrected = np.sqrt(
        np.sum((coords_um_3d - coords_surface) ** 2, axis=1)
    )

    return coords_surface[:, :-1], dv_below_surface_corrected
