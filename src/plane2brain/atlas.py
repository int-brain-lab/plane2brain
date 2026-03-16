from typing import Tuple, Dict
import numpy as np
from scipy.spatial import ConvexHull
from plane2brain.linalg import (
    intersect_line_mesh_np,
    intersect_line_mesh_nb,
    get_closest_face,
    plane_normal_form,
)
from iblatlas.atlas import AllenAtlas


class ProjectionAtlas(AllenAtlas):
    def __init__(self, *args, **kwargs):
        # FIXME TODO figure out how to properly subclass MRIToronto
        # MRI toronto scaling
        ML_SCALE = 0.952
        DV_SCALE = 0.885  # multiplicative factor on DV dimension, determined from MRI->CCF transform
        AP_SCALE = 1.031  # multiplicative factor on AP dimension
        kwargs["scaling"] = np.array([ML_SCALE, AP_SCALE, DV_SCALE])
        self.mesh: Dict[str, np.ndarray] = None  # better put Optional here
        super().__init__(*args, **kwargs)
        self.compute_surface()
        self.calculate_surface_triangulation()

    def calculate_surface_triangulation(self) -> dict:
        points = self.get_surface_points(dropna=True)
        hull = ConvexHull(points)
        connectivity_list = hull.simplices
        self.mesh = dict(vertices=points, edges=connectivity_list)

    def get_surface_points(self, dropna=True) -> np.ndarray:
        """for a given atlas, return all points that are on the brain surface in um.

        Args:
            atlas (BrainAtlas): _description_
            dropna (bool, optional): _description_. Defaults to True.

        Returns:
            np.ndarray: the surface points with shape (N,3) in (ml,ap,dv)
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
        ml: np.float64,
        ap: np.float64,
        upwards=True,
        numba=False,  # eventually drop this
    ) -> Tuple[np.ndarray, np.ndarray]:
        """for a given ml,ap coordinates, returns the plane on the brain surface
        in normal form

        Args:
            ml (np.float64): _description_
            ap (np.float64): _description_
            vertices (np.ndarray): _description_
            connectivity_list (np.ndarray): _description_
            upwards (bool, optional): enforce the normal pointing upwards. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray]: plane as defined by point and normal
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
        face, ix = get_closest_face(faces, l0)
        _, n = plane_normal_form(face)  # the brain normal
        p = ips[ix]  # the intersection point in the mesh triangle
        if upwards:
            if n[2] < 0:
                n *= -1
        return p, n

    def get_labels_for_mlapdv(
        self,
        coords_mlapdv: np.ndarray,
    ):
        # TODO choose mapping allen, beryl, cosmos etc
        ids = np.array(
            [self.get_labels(mlapdv / 1e6, mode="clip") for mlapdv in coords_mlapdv],
        )
        ix = [ix[0] for ix in self.regions.id2index(ids)[1]]
        rgba = self.regions.rgba[ix]
        acronym = self.regions.id2acronym(ids)
        return ids, ix, rgba, acronym
