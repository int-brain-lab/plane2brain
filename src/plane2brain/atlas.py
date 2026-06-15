from typing import Optional, Tuple, Dict
import numpy as np
from scipy.spatial import ConvexHull
from plane2brain.linalg import (
    intersect_line_mesh_np,
    intersect_line_mesh_nb,
    plane_normal_form,
)
from iblatlas.atlas import AllenAtlas


class ProjectionAtlas(AllenAtlas):
    def __init__(self, *args, **kwargs):
        # FIXME TODO figure out how to properly subclass MRIToronto
        # Scaling factors to align the MRI Toronto atlas to Allen CCF space,
        # derived empirically from the MRI->CCF affine transform.
        ML_SCALE = 0.952
        DV_SCALE = 0.885
        AP_SCALE = 1.031
        kwargs["scaling"] = np.array([ML_SCALE, AP_SCALE, DV_SCALE])
        self.mesh: Optional[Dict[str, np.ndarray]] = None
        super().__init__(*args, **kwargs)
        self.compute_surface()
        self.calculate_surface_triangulation()

    def calculate_surface_triangulation(self) -> None:
        """Compute the convex hull of surface points and store as a triangle mesh in `self.mesh`."""
        points = self.get_surface_points(dropna=True)
        hull = ConvexHull(points)
        connectivity_list = hull.simplices
        self.mesh = dict(vertices=points, edges=connectivity_list)

    def get_surface_points(self, dropna: bool = True) -> np.ndarray:
        """Return all brain surface points in micrometers.

        Args:
            dropna: If True, drop rows where the DV surface value is NaN.

        Returns:
            Array of shape (N, 3) in (ml, ap, dv) order, in µm.
        """

        ap_grid, ml_grid = np.meshgrid(
            self.bc.yscale, self.bc.xscale
        )  # now this indexes into AP, ML
        points = (
            np.stack(
                [ml_grid.T.flatten(), ap_grid.T.flatten(), self.top.flatten()], axis=1
            )
            * 1e6  # <- converts the atlas into um
        )
        if dropna:
            points = points[~np.isnan(points[:, 2])]
        self.surface_points = points
        return points

    def get_plane_at_point_mlap(
        self,
        ml: float,
        ap: float,
        upwards: bool = True,
        numba: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the brain surface plane in normal form at a given ML/AP location.

        Casts a vertical ray downward from above the brain and finds its intersection
        with the surface mesh.

        Args:
            ml: Medial-lateral coordinate in µm.
            ap: Anterior-posterior coordinate in µm.
            upwards: If True, flip the normal so it points away from the brain (upwards).
            numba: Use the Numba-accelerated mesh intersection. Defaults to False.

        Returns:
            Tuple of (point on surface, surface normal), each of shape (3,) in µm.
        """
        # projects from a point above the brain downwards until it intersects
        # the mesh
        l0 = np.array([ml, ap, 1000.0])
        l = np.array([0.0, 0.0, -1.0])
        if numba:
            func = intersect_line_mesh_nb
        else:
            func = intersect_line_mesh_np
        faces, ips, _ = func(self.mesh["vertices"], self.mesh["edges"], l0, l)
        # pick the intersection point nearest the starting point of the ray
        ix = np.argmin(np.linalg.norm(ips - l0, axis=1))
        face = faces[ix]
        _, n = plane_normal_form(face)  # the brain normal
        p = ips[ix]  # the intersection point in the mesh triangle
        if upwards:
            if n[2] < 0:
                n *= -1
        return p, n

    def get_dv_for_mlap(
        self,
        coords_mlap: np.ndarray,
    ) -> np.ndarray:
        """Complete ML/AP coordinates to ML/AP/DV by projecting onto the surface mesh.

        Args:
            coords_mlap: Array of shape (N, 2) in µm.

        Returns:
            Array of shape (N, 3) in µm, with DV filled from the surface mesh.
        """
        coords_mlapdv = np.zeros((coords_mlap.shape[0], 3))
        for i, _coords in enumerate(coords_mlap):
            _coords = np.append(_coords, 0.0)
            try:
                _, intersection_points, _ = intersect_line_mesh_nb(
                    self.mesh["vertices"],
                    self.mesh["edges"],
                    _coords,
                    np.array([0.0, 0.0, -1.0]),
                )
                # pick the intersection point nearest the imaging-plane point
                ix = np.argmin(np.linalg.norm(intersection_points - _coords, axis=1))
                coords_mlapdv[i] = intersection_points[ix]
            except ValueError:
                # TODO logger warn
                coords_mlapdv[i, :] = np.nan

        return coords_mlapdv

    def get_labels_for_mlapdv(
        self,
        coords_mlapdv: np.ndarray,
    ) -> Tuple[np.ndarray, list, np.ndarray, np.ndarray]:
        """Look up Allen Atlas region labels for a set of ML/AP/DV coordinates.

        Args:
            coords_mlapdv: Array of shape (N, 3) in µm.

        Returns:
            Tuple of (region IDs, region indices, RGBA colours, acronyms).
            Currently uses the default Allen mapping; beryl/cosmos mappings not yet supported.
        """
        # TODO choose mapping allen, beryl, cosmos etc
        ids = np.array(
            [self.get_labels(mlapdv / 1e6, mode="clip") for mlapdv in coords_mlapdv],
        )
        ix = [ix[0] for ix in self.regions.id2index(ids)[1]]
        rgba = self.regions.rgba[ix]
        acronym = self.regions.id2acronym(ids)
        return ids, ix, rgba, acronym
