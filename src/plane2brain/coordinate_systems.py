from typing import Any, Dict, List, Optional

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes

from plane2brain.affine import rotation_matrix_z, apply_transform
from plane2brain.plotters import plot_line

"""
 
  ######  ##          ###     ######   ######  ########  ######  
 ##    ## ##         ## ##   ##    ## ##    ## ##       ##    ## 
 ##       ##        ##   ##  ##       ##       ##       ##       
 ##       ##       ##     ##  ######   ######  ######    ######  
 ##       ##       #########       ##       ## ##             ## 
 ##    ## ##       ##     ## ##    ## ##    ## ##       ##    ## 
  ######  ######## ##     ##  ######   ######  ########  ######  
 
"""


class CoordinateSystem:
    """A linear coordinate system defined by an origin and basis vectors.

    The basis vectors are stored as the columns of `basis` and define the
    local axes relative to the world frame. Points can be mapped between the
    local coordinate system and the world coordinate frame.
    """

    def __init__(
        self,
        basis: np.ndarray,
        origin: np.ndarray,
    ) -> None:
        """Initialize a coordinate system.

        Args:
            basis: Array of shape `(D, D)` where each column is a basis vector.
            origin: Array of shape `(D,)` representing the origin in world
                coordinates.
        """
        self.basis = basis
        self.origin = origin
        self.dim = basis.shape[0]

    def normalize(self) -> None:
        """Normalize each basis vector to unit length."""
        self.basis /= linalg.norm(self.basis, axis=0)[np.newaxis, :]

    def inverse_transform(self, points: np.ndarray) -> np.ndarray:
        # from this coordinate system to world frame
        return points @ self.basis.T + self.origin

    def transform(self, points_w: np.ndarray) -> np.ndarray:
        # from world frame to this coordinate system
        return (points_w - self.origin) @ linalg.pinv(self.basis.T)

    def plot(
        self, axes: Optional[Axes] = None, scale: float = 1.0, **kwargs: Any
    ) -> Axes:
        """Plot the coordinate axes for this system.

        Args:
            axes: Optional Matplotlib axes object. If not provided, a new axes
                object is created.
            scale: Scale factor for the plotted basis vectors.
            **kwargs: Passed through to `plot_line`. The `color` keyword is
                interpreted per-axis if provided.

        Returns:
            The Matplotlib axes instance containing the plot.
        """
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
    """A named collection of coordinate systems sharing the same dimensionality."""

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
        """Transform points from one named system to another.

        Args:
            points: Coordinates in the source system, shape `(..., D)`.
            name_from: Name of the source coordinate system.
            name_target: Name of the target coordinate system.

        Returns:
            Coordinates in the target coordinate system.
        """
        for name in [name_from, name_target]:
            if name not in self.coordinate_systems.keys():
                raise ValueError(
                    f"coordinate system with {name} not found. Present coordinate systems are:",
                    list(self.coordinate_systems.keys()),
                )

        # inverse - represent points back in world frame
        points_w = self.coordinate_systems[name_from].inverse_transform(points)
        return self.coordinate_systems[name_target].transform(points_w)

    def plot(
        self,
        scale: float = 1.0,
        axes: Optional[Axes] = None,
        color_by: str = "system",
    ) -> Axes:
        """Plot all linked coordinate systems on a shared axes.

        Args:
            scale: Scale factor for each coordinate system's axes.
            axes: Optional Matplotlib axes object. If not provided, a new axes
                is created.
            color_by: If "system", assign each system a distinct color. If
                "axis", plot using the default axis colors.

        Returns:
            The Matplotlib axes instance containing the plot.
        """
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

    def get(self, name: str) -> CoordinateSystem:
        """Return a named coordinate system.

        Args:
            name: The name of the coordinate system to retrieve.

        Returns:
            The requested CoordinateSystem instance.
        """
        return self.coordinate_systems[name]

    def __repr__(self):
        return f"{type(self)} with named coordinate systems: {list(self.coordinate_systems.keys())}"


"""
 
  #######  ########  
 ##     ## ##     ## 
        ## ##     ## 
  #######  ##     ## 
 ##        ##     ## 
 ##        ##     ## 
 ######### ########  
 
"""


# TODO
def create_coordinate_system_for_image(
    img_size_px: np.ndarray,  # in pixel
    um_per_px: np.ndarray,  # pixel size in um
    ref_per_px: np.ndarray,  # pixel size in ref space
    img_topleft_ref: np.ndarray,  # the top left corner of the image in the reference frame
) -> LinkedCoordinateSystems:
    """Generate linked coordinate systems for an image and its reference frame.

    Args:
        img_size_px: Image dimensions in pixels, shape `(2,)`.
        um_per_px: Pixel size in micrometers, shape `(2,)`.
        ref_per_px: Pixel size in reference units, shape `(2,)`.
        img_topleft_ref: Top-left image corner in the reference frame,
            shape `(2,)`.
        verify: If true, validate coordinate relationships with assertions.

    Returns:
        A `LinkedCoordinateSystems` object for the image, pixel, um_image,
        reference, and global um coordinate systems.
    """
    img_size_um = img_size_px * um_per_px
    px_per_ref = 1 / ref_per_px
    um_per_ref = um_per_px * px_per_ref
    ref_per_um = 1 / um_per_ref

    # creates a coordinate system where: 0,0 in pixel indices is the topleft corner
    # the image is embedded in a reference frame, and it's topleft
    # corner is at the location specified by img_topleft ref

    coordinate_systems = LinkedCoordinateSystems(
        dict(
            ref=CoordinateSystem(
                basis=np.identity(2),
                origin=np.zeros(2),
            ),
            pixel=CoordinateSystem(
                basis=np.diag(ref_per_px),
                origin=img_topleft_ref,
            ),
            um_image=CoordinateSystem(
                basis=np.diag(ref_per_um),
                origin=img_topleft_ref,
            ),
            image=CoordinateSystem(
                basis=np.diag(ref_per_um * img_size_um),
                origin=img_topleft_ref,
            ),
            um_global=CoordinateSystem(
                basis=np.diag(ref_per_um),
                origin=np.zeros(2),
            ),
        )
    )
    return coordinate_systems


def get_image_corners(
    img_size_px: np.ndarray,
    coordinate_systems: LinkedCoordinateSystems,
    to: str = "um_global",  # TODO refactor me to 'in'
) -> Dict[str, np.ndarray]:
    """Return the corner coordinates of an image in a target coordinate system.

    Args:
        img_size_px: Image dimensions in pixels, shape `(2,)`.
        coordinate_systems: A linked coordinate system containing `image`.
        to: Name of the target coordinate system.

    Returns:
        A mapping of corner names to coordinates in the target system.
    """
    img_size_px = np.array(img_size_px)  # cast just in case
    corners = dict(
        topleft=[0, 0],
        topright=[0, 1],
        bottomleft=[1, 0],
        bottomright=[1, 1],
        center=img_size_px / 2,
    )
    return {
        name: coordinate_systems.transform(np.array(corner), "image", to)
        for name, corner in corners.items()
    }


"""
 
  #######  ########  
 ##     ## ##     ## 
        ## ##     ## 
  #######  ##     ## 
        ## ##     ## 
 ##     ## ##     ## 
  #######  ########  
 
"""


def coordinate_system_from_normal(
    p: np.ndarray,  # point
    n: np.ndarray,  # normal
    rotate_by: Optional[float] = None,  # around the axis of the normal
    invert_dims: Optional[List[bool]] = None,
) -> CoordinateSystem:
    """Create a coordinate system whose DV axis aligns with a given normal.

    The generated coordinate system uses `p` as its origin. The DV axis is set
    to `n`, and the AP axis is constrained to have zero ML component. The ML
    axis is inferred by the cross product.

    Args:
        p: Origin point of shape `(3,)`.
        n: Normal vector of shape `(3,)`, representing the DV direction.
        rotate_by: Optional rotation around the Z axis after basis construction.
        invert_dims: Optional length-3 boolean list to flip basis axes.

    Returns:
        The imaging-plane coordinate system aligned with the given normal.
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

    if invert_dims is None:
        invert_dims = [False, False, False]

    for i, invert in enumerate(invert_dims):
        if invert:
            basis[:, i] *= -1

    return CoordinateSystem(basis=basis, origin=p)


def setup_coordinate_systems_3d(
    center_mlapdv: np.ndarray,
    brain_normal: np.ndarray,
    rotate_by: Optional[float] = None,
    invert_dims: Optional[List[bool]] = None,
) -> LinkedCoordinateSystems:
    """Create a linked set of 3D coordinate systems for imaging.

    Args:
        center_mlapdv: Center point in ML/AP/DV order, shape `(3,)`.
        brain_normal: Normal vector in ML/AP/DV order, shape `(3,)`.
        rotate_by: Optional angle in radians to rotate the imaging plane about Z.
        invert_dims: Optional length-3 boolean list to invert basis axes.

    Returns:
        A `LinkedCoordinateSystems` object containing `mlapdv` and
        `imaging_plane` coordinate systems.
    """
    cs3d = LinkedCoordinateSystems(
        dict(
            mlapdv=CoordinateSystem(basis=np.identity(3), origin=np.zeros(3)),
            imaging_plane=coordinate_system_from_normal(
                center_mlapdv,
                brain_normal,
                rotate_by=rotate_by,
                invert_dims=invert_dims,
            ),
        )
    )

    # TODO add tests for the 3d case

    return cs3d
