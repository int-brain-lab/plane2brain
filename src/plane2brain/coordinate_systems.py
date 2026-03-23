from typing import Optional, List, Dict

import numpy as np
from numpy import linalg
import numpy.testing as nptest
import matplotlib.pyplot as plt
import seaborn as sns

from plane2brain.affine import rotation_matrix_z, apply_transform
from plane2brain.plotters import plot_line


class CoordinateSystem:
    def __init__(
        self,
        basis: np.ndarray = None,
        origin: np.ndarray = None,
    ):
        """
        A 2d or 3d coordinate system, defined by a set of basis vectors and an origin point.
        the basis vectors spanning the space in the columns
        the origin on the coordinate system
        """
        # basis is shape (3, 3), columns are vectors
        self.basis = basis
        # origin is a single point of shape (3,)
        self.origin = origin
        self.dim = basis.shape[0]

    def normalize(self):
        self.basis /= linalg.norm(self.basis, axis=0)[np.newaxis, :]

    def inverse_transform(self, points: np.ndarray) -> np.ndarray:
        # from this coordinate system to world frame
        return points @ self.basis.T + self.origin

    def transform(self, points_w):
        # from world frame to this coordinate system
        return (points_w - self.origin) @ linalg.pinv(self.basis.T)

    def plot(self, axes=None, scale=1, **kwargs):
        if axes is None:
            if self.dim == 2:
                _, axes = plt.subplots()
            if self.dim == 3:
                axes = plt.figure().add_subplot(projection="3d")

        if "color" not in kwargs or kwargs["color"] is None:
            colors = ["r", "g", "b"]
        else:
            colors = [kwargs["color"]] * self.dim
            kwargs.pop("color")

        for i in range(self.dim):
            axes = plot_line(
                self.origin,
                self.basis[:, i] * scale,
                length=[-2, 2],
                axes=axes,
                color=colors[i],
            )

        axes.set_aspect("equal")
        return axes


class LinkedCoordinateSystems:
    def __init__(
        self,
        coordinate_systems: Dict[str, CoordinateSystem],
    ):
        self.coordinate_systems = coordinate_systems
        # all have the same dim
        assert len(set([cs.dim for cs in self.coordinate_systems.values()])) == 1
        # no duplicate names
        assert len(set([name for name in self.coordinate_systems.keys()])) == len(
            self.coordinate_systems
        )
        self.dim = next(iter(self.coordinate_systems.values())).dim

    def transform(
        self,
        points: np.ndarray,
        name_from: str,
        name_target: str,
    ) -> np.ndarray:
        for name in [name_from, name_target]:
            if name not in self.coordinate_systems.keys():
                raise ValueError(
                    f"coordinate system with {name} not found. Present coordinate systems are:",
                    list(self.coordinate_systems.keys()),
                )

        # inverse - represent points back in world frame
        points_w = self.coordinate_systems[name_from].inverse_transform(points)
        return self.coordinate_systems[name_target].transform(points_w)

    def plot(self, scale=1, axes=None, color_by="system"):
        if axes is None:
            if self.dim == 2:
                _, axes = plt.subplots()
            if self.dim == 3:
                axes = plt.figure().add_subplot(projection="3d")

        if color_by == "system":
            colors = dict(
                zip(
                    list(self.coordinate_systems.keys()),
                    sns.color_palette("husl", n_colors=len(self.coordinate_systems)),
                )
            )
        if color_by == "axis":
            colors = dict(
                zip(
                    list(self.coordinate_systems.keys()),
                    [None] * len(list(self.coordinate_systems.keys())),
                )
            )

        for name, system in self.coordinate_systems.items():
            axes = system.plot(axes=axes, scale=scale, color=colors[name], label=name)
        axes.legend()
        return axes

    def get(self, name: str):
        return self.coordinate_systems[name]


# TODO refactor
def cs3d_from_normal(
    p: np.ndarray,  # point
    n: np.ndarray,  # normal
    rotate_by: Optional[float] = None,  # around the axis of the normal
    invert_dims: List[bool] = [False, False, False],
) -> CoordinateSystem:
    """creates a coordinates system with origin at point p and the upward pointing (DV) axis defined by n.
    Constraines AP vector to have a 0 ML component, all other direction are inferred.

    This mimics a horizontally aligned mouse, and the objective is tilted in the coronal plane.

    Args:
        p (np.ndarray): point
        n (np.ndarray): vector

    Returns:
        CoordinateSystem: _description_
    """

    ap, dv = n[1], n[2]
    r = np.linalg.norm(np.array([ap, dv]))
    phi = np.arccos(ap / r)
    phi -= np.pi / 2
    ap_, dv_ = r * np.cos(phi), r * np.sin(phi)

    dv_v = n
    ap_v = np.array([0, ap_, dv_])  # this is the 0 ML constrain
    ml_v = np.cross(ap_v, dv_v)

    # normalize
    ap_v /= np.linalg.norm(ap_v)
    ml_v /= np.linalg.norm(ml_v)
    basis = np.stack([ml_v, ap_v, dv_v], axis=1)

    if rotate_by:
        R = rotation_matrix_z(rotate_by)
        basis = apply_transform(basis, R)

    for i, invert in enumerate(invert_dims):
        if invert:
            basis[:, i] *= -1

    return CoordinateSystem(basis=basis, origin=p)


# def cs3d_from_normal_rot(p, n, name="imaging_plane"):
#     yaw, pitch, roll = get_vector_angles(n)
#     r1 = affine.rotation_matrix_x(pitch)
#     r2 = affine.rotation_matrix_y(-roll)
#     # TODO for the future, if the scanfield is rotated, this could be integrated here
#     # r3 = affine.rotation_matrix_z(yaw)

#     # R = r1 @ r2
#     R = r2 @ r1  # ZYX order

#     # for the new coordinate system
#     basis = np.identity(3)
#     basis = affine.apply_transform(basis, R)

#     return (CoordinateSystem(basis=basis, origin=p, name=name),)


def setup_coordinate_systems_3d(
    center_mlapdv: np.ndarray,
    brain_normal: np.ndarray,
    rotate_by: Optional[float] = None,
    invert_dims: List[bool] = [False, False, False],
) -> LinkedCoordinateSystems:
    """creates a coordinate system with an imaging plane oriented to the brain normal

    Args:
        center_mlapdv (np.ndarray): _description_
        brain_normal (np.ndarray): _description_

    Returns:
        LinkedCoordinateSystems: _description_
    """
    cs3d = LinkedCoordinateSystems(
        dict(
            mlapdv=CoordinateSystem(basis=np.identity(3), origin=np.zeros(3)),
            imaging_plane=cs3d_from_normal(
                center_mlapdv,
                brain_normal,
                rotate_by=rotate_by,
                invert_dims=invert_dims,
            ),
        )
    )

    # TODO verify

    return cs3d


def create_coordinate_system_for_ref(
    img_size_px: np.ndarray,
    um_per_px: np.ndarray,  # pixel resolution
    img_topleft_um: np.ndarray,
) -> LinkedCoordinateSystems:
    # this function is necessary because
    # we don't have the coordinates of the reference image in the scanimage reference space
    coordinate_systems = LinkedCoordinateSystems(
        dict(
            um=CoordinateSystem(basis=np.identity(2), origin=np.zeros(2)),
            pixel=CoordinateSystem(basis=np.diag(um_per_px), origin=img_topleft_um),
            image=CoordinateSystem(
                basis=np.diag(img_size_px * um_per_px), origin=img_topleft_um
            ),
        )
    )
    # some verifications
    nptest.assert_almost_equal(
        coordinate_systems.transform(np.zeros(2), "image", "pixel"),
        np.zeros(2),
    )
    nptest.assert_almost_equal(
        coordinate_systems.transform(np.ones(2), "image", "pixel"),
        img_size_px,
    )
    nptest.assert_almost_equal(
        coordinate_systems.transform(img_size_px, "pixel", "image"),
        np.ones(2),
    )
    nptest.assert_almost_equal(
        coordinate_systems.transform(np.array([0, 0]), "pixel", "um"),
        img_topleft_um,
    )
    return coordinate_systems


def create_coordinate_system_for_image(
    img_size_px: np.ndarray,  # in pixel
    um_per_px: np.ndarray,  # pixel size in um
    ref_per_px: np.ndarray,  # pixel size in ref space
    img_topleft_ref: np.ndarray,  # the top left corner of the image in the reference frame
    # img_topleft_um: np.ndarray,  # the top left corner of the image in um
    verify: bool = True,
) -> CoordinateSystem:
    # px_per_um = 1 / um_per_px
    img_size_um = img_size_px * um_per_px
    # ref_per_px = img_size_ref / img_size_px
    px_per_ref = 1 / ref_per_px
    um_per_ref = um_per_px * px_per_ref
    ref_per_um = 1 / um_per_ref

    # explain what this is really doing
    # create a coordinate system where: 0,0 in pixel is the topleft corner
    # the image is embedded in a larger reference frame, and it's topleft
    # corner is at the location specified by img_topleft ref

    # would somehow be more intuitive if ref_per_px is passed instead of size
    # so the dimensions of a pixel are described in both spaces

    coordinate_systems = LinkedCoordinateSystems(
        dict(
            ref=CoordinateSystem(basis=np.identity(2), origin=np.zeros(2)),
            pixel=CoordinateSystem(basis=np.diag(ref_per_px), origin=img_topleft_ref),
            um_image=CoordinateSystem(
                basis=np.diag(ref_per_um), origin=img_topleft_ref
            ),
            um_global=CoordinateSystem(basis=np.diag(ref_per_um), origin=np.zeros(2)),
            # TODO docme here why the origin of um is also 0,0
            # because in our use case both ref and um share the same origin
            # doens't need to be the case though!
        )
    )
    if verify:
        img_size_ref = img_size_px * ref_per_px
        img_bottomright_ref = img_topleft_ref + img_size_ref
        # the image size assertion
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
            coordinate_systems.transform(np.zeros(2), "pixel", "ref"),
            img_topleft_ref,
        )
        nptest.assert_array_almost_equal(
            coordinate_systems.transform(img_bottomright_ref, "ref", "um_global")
            - coordinate_systems.transform(img_topleft_ref, "ref", "um_global"),
            img_size_um,
        )
    return coordinate_systems


def get_image_corners(img_size_px, coordinate_systems, to="um"):
    img_size_px = np.array(img_size_px)
    corners = dict(
        topleft=[0, 0],
        topright=[0, img_size_px[1]],
        bottomleft=[img_size_px[0], 0],
        bottomright=[img_size_px[0], img_size_px[1]],
        center=img_size_px / 2,
    )
    # this is correct by visual inspection

    return {
        name: coordinate_systems.transform(np.array(corner), "pixel", to)
        for name, corner in corners.items()
    }
