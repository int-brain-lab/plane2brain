"""Coordinate systems, planes and the images inside them."""

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from numpy import linalg

from plane2brain.affine import rotation_matrix_z
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
        """Map points from this coordinate system to the world frame."""
        return points @ self.basis.T + self.origin

    def transform(self, points_w: np.ndarray) -> np.ndarray:
        """Map points from the world frame into this coordinate system."""
        return (points_w - self.origin) @ linalg.pinv(self.basis.T)

    def plot(self, axes: Axes | None = None, scale: float = 1.0, **kwargs: Any) -> Axes:
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
        coordinate_systems: dict[str, CoordinateSystem],
    ) -> None:
        """Initialize the collection.

        Args:
            coordinate_systems: Coordinate systems by name, all of the same
                dimensionality and defined in the same world frame.
        """
        self.coordinate_systems = coordinate_systems
        # all have the same dim
        assert len({cs.dim for cs in self.coordinate_systems.values()}) == 1
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

        Raises:
            ValueError: If one of the names is not in the collection.
        """
        for name in [name_from, name_target]:
            if name not in self.coordinate_systems:
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
        axes: Axes | None = None,
        color_by: Literal["system", "axis"] = "system",
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

    def __repr__(self) -> str:
        """Return the class and the names of the coordinate systems."""
        return f"{type(self)} with named coordinate systems: {list(self.coordinate_systems.keys())}"


@dataclass(frozen=True)
class Orientation:
    """Rig-constant orientation of the sample under the microscope.

    Maps the microscope axes (x and y after the reader's axis swap, z along the
    optical axis) onto the axes of a plane (in-plane ML, in-plane AP, normal).
    The axes are inverted first and then rotated about the optical axis. This
    reproduces the former `coordinate_system_from_normal`, which rotated the basis
    and then inverted its columns.

    Attributes:
        rotation_degrees: Rotation about the optical axis in degrees.
        invert_axis: Whether to invert the x, y and z axis. Invert z if
            increasing microscope z points into the brain.
    """

    rotation_degrees: float = 0.0
    invert_axis: tuple[bool, bool, bool] = (False, False, False)

    @property
    def matrix(self) -> np.ndarray:
        """Return the (3, 3) matrix that maps microscope axes onto plane axes."""
        rotation = rotation_matrix_z(self.rotation_degrees, in_degrees=True)[:3, :3]
        inversion = np.diag(np.where(self.invert_axis, -1.0, 1.0))
        return rotation @ inversion


@dataclass(frozen=True, eq=False)
class Anchor:
    """A point with a known position in both the 2D and the 3D coordinate systems of a plane.

    Attributes:
        mlapdv: Position in atlas space in µm, shape `(3,)`.
        name: Name of a coordinate system in the plane's 2D set.
        value: The same point in that 2D coordinate system, shape `(2,)`.
    """

    mlapdv: np.ndarray
    name: str
    value: np.ndarray

    def __post_init__(self) -> None:
        """Cast the positions to float arrays."""
        # frozen dataclass: attributes can only be set via object.__setattr__
        object.__setattr__(self, "mlapdv", np.asarray(self.mlapdv, dtype=float))
        object.__setattr__(self, "value", np.asarray(self.value, dtype=float))


class Image:
    """A pixel grid inside a plane.

    The 2D set must contain a coordinate system "pixel", and the coordinate system
    "um_global" that is shared with the plane and all other images in it.

    Values per pixel, such as the depth below the surface, are flat arrays with
    one entry per pixel, in the order of `pixel_indices`.
    """

    def __init__(
        self,
        size_px: np.ndarray,
        coordinate_systems: LinkedCoordinateSystems,
        depth_below_surface: float | np.ndarray | None = None,
    ) -> None:
        """Initialize an image and validate its 2D set.

        Args:
            size_px: Image size in pixels, shape `(2,)`.
            coordinate_systems: 2D set of linked coordinate systems.
            depth_below_surface: Depth below the brain surface in µm, a single
                depth for all pixels or one per pixel. `None` means not assigned yet.
        """
        if coordinate_systems.dim != 2:
            raise ValueError("coordinate_systems must be a 2D set")
        for name in ("pixel", "um_global"):
            if name not in coordinate_systems.coordinate_systems:
                raise ValueError(
                    f"coordinate_systems must contain a coordinate system {name!r}"
                )
        self.size_px = np.asarray(size_px)
        self.coordinate_systems = coordinate_systems
        # set through the property, which needs size_px to broadcast against
        self.depth_below_surface = depth_below_surface
        # holds corrected versions of the image's coordinates, e.g. "um_corrected", with
        # one row per pixel in the order of pixel_indices; empty until a correction runs
        self.coordinates: dict[str, np.ndarray] = {}

    def __repr__(self) -> str:
        """Return the image size and the depth (a single value or the range)."""
        depth = self.depth_below_surface
        if depth is not None:
            # one value for a uniform depth, the range otherwise, rather than every pixel
            depth = (
                float(depth[0])
                if np.all(depth == depth[0])
                else f"{depth.min():g}..{depth.max():g}"
            )
        return (
            f"{type(self).__name__}(size_px={self.size_px.tolist()}, "
            f"depth_below_surface={depth})"
        )

    @cached_property
    def pixel_indices(self) -> np.ndarray:
        """Return all pixel indices, shape `(N, 2)`, the first index running slowest.

        This is the order of every per-pixel value of the image.
        """
        grid = np.meshgrid(*map(np.arange, self.size_px), indexing="ij")
        return np.stack(grid, axis=-1).reshape(-1, 2)

    @property
    def depth_below_surface(self) -> np.ndarray | None:
        """Return the depth of each pixel below the brain surface in µm, shape `(N,)`.

        Measured along the plane's normal, in the order of `pixel_indices`.
        `None` means not assigned yet.
        """
        return self._depth_below_surface

    @depth_below_surface.setter
    def depth_below_surface(self, value: float | np.ndarray | None) -> None:
        """Set a single depth for all pixels, or one depth per pixel.

        Raises:
            ValueError: If an array does not hold exactly one depth per pixel.
        """
        if value is None:
            self._depth_below_surface = None
            return
        n_pixels = int(np.prod(self.size_px))
        try:
            # a read-only view of a private copy: a single depth is stored once, and
            # later changes to the caller's array cannot reach the image
            self._depth_below_surface = np.broadcast_to(
                np.array(value, dtype=float), (n_pixels,)
            )
        except ValueError:
            raise ValueError(
                f"expected a single depth or one per pixel ({n_pixels}), "
                f"got shape {np.shape(value)}"
            ) from None

    def get_corners(self, in_: str = "um_global") -> dict[str, np.ndarray]:
        """Return the corners and the center of the image in a coordinate system.

        Args:
            in_: Name of the coordinate system to return the corners in (`in` is
                a Python keyword).

        Returns:
            A mapping of corner names to coordinates.
        """
        # corners as fractions of the image size
        corners = {
            "topleft": [0, 0],
            "topright": [0, 1],
            "bottomleft": [1, 0],
            "bottomright": [1, 1],
            "center": [0.5, 0.5],
        }
        return {
            name: self.coordinate_systems.transform(
                np.array(corner) * self.size_px, "pixel", in_
            )
            for name, corner in corners.items()
        }


class Plane:
    """A plane in space, defined by a point and a normal, containing images.

    Positions within the plane are described by a 2D set of linked coordinate
    systems, which must contain the coordinate system "um_global" in
    micrometers, shared with all images in the plane. Adding images with an
    anchor and an orientation creates a 3D set ("mlapdv", "um_global") that
    links the 2D set to atlas space.
    """

    def __init__(
        self,
        point: np.ndarray,
        normal: np.ndarray,
        *,
        coordinate_systems: LinkedCoordinateSystems | None = None,
    ) -> None:
        """Initialize a plane and validate its 2D set.

        Args:
            point: A point on the plane in atlas space in µm, shape `(3,)`.
            normal: Normal of the plane, pointing away from the brain, shape
                `(3,)`. Normalized on construction.
            coordinate_systems: 2D set of linked coordinate systems. Defaults to
                a single coordinate system "um_global".
        """
        if coordinate_systems is None:
            coordinate_systems = LinkedCoordinateSystems(
                {
                    "um_global": CoordinateSystem(
                        basis=np.identity(2), origin=np.zeros(2)
                    )
                }
            )
        if coordinate_systems.dim != 2:
            raise ValueError("coordinate_systems must be a 2D set")
        if "um_global" not in coordinate_systems.coordinate_systems:
            raise ValueError(
                'coordinate_systems must contain a coordinate system "um_global"'
            )

        normal = np.asarray(normal, dtype=float)
        self.point = np.asarray(point, dtype=float)
        self.normal = normal / linalg.norm(normal)
        self.coordinate_systems = coordinate_systems
        self.anchor: Anchor | None = None
        self.images: list[Image] = []
        self.coordinate_systems_3d: LinkedCoordinateSystems | None = None

    def add_images(
        self,
        images: list[Image],
        orientation: Orientation,
        anchor: Anchor,
    ) -> None:
        """Place images onto the plane and link the plane to atlas space.

        Creates the 3D set from the anchor and the orientation; the orientation
        is not stored. All images of a plane share its placement, so images
        added later must result in the same 3D set.

        Args:
            images: Images to add, in the order of the image map.
            orientation: Orientation of the sample under the microscope.
            anchor: A point on the plane with known positions in atlas space and
                in one of the plane's 2D coordinate systems.
        """
        coordinate_systems_3d = self.create_coordinate_systems_3d(anchor, orientation)
        if self.coordinate_systems_3d is not None:
            new = coordinate_systems_3d.get("um_global")
            existing = self.coordinate_systems_3d.get("um_global")
            # the same placement means the same "um_global" basis and origin in atlas space
            if not (
                np.allclose(new.basis, existing.basis)
                and np.allclose(new.origin, existing.origin)
            ):
                raise ValueError(
                    "images are placed differently than the images in the plane"
                )
        self.anchor = anchor
        self.coordinate_systems_3d = coordinate_systems_3d
        self.images.extend(images)

    def __repr__(self) -> str:
        """Return the point, the normal, the number of images and whether it is linked."""
        return (
            f"{type(self).__name__}(point={self.point.tolist()}, "
            f"normal={self.normal.round(3).tolist()}, images={len(self.images)}, "
            f"linked={self.coordinate_systems_3d is not None})"
        )

    def create_coordinate_systems_3d(
        self,
        anchor: Anchor,
        orientation: Orientation | None = None,
    ) -> LinkedCoordinateSystems:
        """Create the 3D set that links the plane's 2D set to atlas space.

        The 3D coordinate system "um_global" spans the plane with the in-plane
        axes derived from the normal, rotated and inverted by the orientation,
        and has the normal as its third axis. Its third coordinate is thus the
        signed distance from the plane in µm.

        Args:
            anchor: A point on the plane with known positions in atlas space and
                in one of the plane's 2D coordinate systems.
            orientation: Orientation of the sample under the microscope. Only its
                in-plane part (x, y) is used. Defaults to no rotation and no
                inversions.

        Returns:
            A `LinkedCoordinateSystems` with "mlapdv" (atlas) and "um_global" (plane).
        """
        if anchor.name not in self.coordinate_systems.coordinate_systems:
            raise ValueError(
                f"anchor refers to unknown coordinate system {anchor.name!r}"
            )
        distance_from_plane = np.dot(anchor.mlapdv - self.point, self.normal)
        if not np.isclose(distance_from_plane, 0.0, atol=1e-6):
            raise ValueError(f"anchor is {distance_from_plane} µm away from the plane")

        if orientation is None:
            orientation = Orientation()
        # microscope x, y -> in-plane axes of the plane (rotation and inversions)
        in_plane_axes = (
            _in_plane_axes_from_normal(self.normal) @ orientation.matrix[:2, :2]
        )
        anchor_um = self.coordinate_systems.transform(
            anchor.value, anchor.name, "um_global"
        )
        # place the origin of "um_global" such that the anchor lands on its atlas position
        origin = anchor.mlapdv - in_plane_axes @ anchor_um
        return LinkedCoordinateSystems(
            {
                "mlapdv": CoordinateSystem(basis=np.identity(3), origin=np.zeros(3)),
                "um_global": CoordinateSystem(
                    basis=np.column_stack([in_plane_axes, self.normal]),
                    origin=origin,
                ),
            }
        )


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
    """Generate the linked 2D coordinate systems of an image in a reference frame.

    The reference frame ("ref") is the imaging system's reference space, e.g.
    ScanImage's optical degrees; it is the common world frame of all images of
    an acquisition.

    Args:
        img_size_px: Image dimensions in pixels, shape `(2,)`.
        um_per_px: Pixel size in micrometers, shape `(2,)`.
        ref_per_px: Pixel size in reference units, shape `(2,)`.
        img_topleft_ref: Top-left image corner in the reference frame,
            shape `(2,)`.

    Returns:
        A `LinkedCoordinateSystems` with "ref" (reference frame), "pixel" (pixel
        indices, (0, 0) at the top-left corner), "um_image" (µm, origin at the
        top-left corner), "image" (image size normalized to 1) and "um_global" (µm,
        origin of the reference frame).
    """
    img_size_um = img_size_px * um_per_px
    px_per_ref = 1 / ref_per_px
    um_per_ref = um_per_px * px_per_ref
    ref_per_um = 1 / um_per_ref

    # creates a coordinate system where: 0,0 in pixel indices is the topleft corner
    # the image is embedded in a reference frame, and it's topleft
    # corner is at the location specified by img_topleft ref

    coordinate_systems = LinkedCoordinateSystems(
        {
            "ref": CoordinateSystem(
                basis=np.identity(2),
                origin=np.zeros(2),
            ),
            "pixel": CoordinateSystem(
                basis=np.diag(ref_per_px),
                origin=img_topleft_ref,
            ),
            "um_image": CoordinateSystem(
                basis=np.diag(ref_per_um),
                origin=img_topleft_ref,
            ),
            "image": CoordinateSystem(
                basis=np.diag(ref_per_um * img_size_um),
                origin=img_topleft_ref,
            ),
            "um_global": CoordinateSystem(
                basis=np.diag(ref_per_um),
                origin=np.zeros(2),
            ),
        }
    )
    return coordinate_systems


"""
 
  #######  ########  
 ##     ## ##     ## 
        ## ##     ## 
  #######  ##     ## 
        ## ##     ## 
 ##     ## ##     ## 
  #######  ########  
 
"""


def _in_plane_axes_from_normal(normal: np.ndarray) -> np.ndarray:
    """Return the in-plane ML and AP axes of a plane as columns of a (3, 2) array.

    Same convention as the former `coordinate_system_from_normal`: DV is the
    (unit) normal, AP has no ML component, and ML = AP × DV. Unlike the former
    function, this also holds for normals with a negative DV component.

    Args:
        normal: Unit normal of the plane, shape `(3,)`.

    Returns:
        The in-plane ML and AP axes as columns, shape `(3, 2)`.

    Raises:
        ValueError: If the normal is along ML, where the AP axis is undefined.
    """
    # AP axis: the AP/DV part of the normal, rotated by -90° within the AP-DV plane
    ap_axis = np.array([0.0, normal[2], -normal[1]])
    ap_norm = linalg.norm(ap_axis)
    if np.isclose(ap_norm, 0.0):
        raise ValueError("normal is along ML, the in-plane AP axis is undefined")
    ap_axis /= ap_norm
    ml_axis = np.cross(ap_axis, normal)
    return np.column_stack([ml_axis, ap_axis])


# soon to be obsolete: replaced by Plane.create_coordinate_systems_3d (with Orientation and
# _in_plane_axes_from_normal), to be removed with the rework of projections.py
# def coordinate_system_from_normal(
#     p: np.ndarray,  # point
#     n: np.ndarray,  # normal
#     rotate_by: float | None = None,  # around the axis of the normal
#     invert_dims: list[bool] | None = None,
# ) -> CoordinateSystem:
#     """Create a coordinate system whose DV axis aligns with a given normal.

#     The generated coordinate system uses `p` as its origin. The DV axis is set
#     to `n`, and the AP axis is constrained to have zero ML component. The ML
#     axis is inferred by the cross product.

#     Args:
#         p: Origin point of shape `(3,)`.
#         n: Normal vector of shape `(3,)`, representing the DV direction.
#         rotate_by: Optional rotation around the Z axis after basis construction.
#         invert_dims: Optional length-3 boolean list to flip basis axes.

#     Returns:
#         The imaging-plane coordinate system aligned with the given normal.
#     """

#     ap, dv = n[1], n[2]
#     r = np.linalg.norm(np.array([ap, dv]))
#     phi = np.arccos(ap / r)
#     phi -= np.pi / 2
#     ap_, dv_ = r * np.cos(phi), r * np.sin(phi)

#     dv_v = n
#     ap_v = np.array([0, ap_, dv_])  # this is the 0 ML constraint
#     ml_v = np.cross(ap_v, dv_v)

#     # normalize
#     ap_v /= np.linalg.norm(ap_v)
#     ml_v /= np.linalg.norm(ml_v)
#     basis = np.stack([ml_v, ap_v, dv_v], axis=1)

#     if rotate_by:
#         R = rotation_matrix_z(rotate_by)
#         # basis vectors are stored as COLUMNS in this module, but
#         # affine.apply_transform interprets its input as Nx3 rows-as-points;
#         # to rotate the column vectors we multiply by R[:3, :3] directly.
#         basis = R[:3, :3] @ basis

#     if invert_dims is None:
#         invert_dims = [False, False, False]

#     for i, invert in enumerate(invert_dims):
#         if invert:
#             basis[:, i] *= -1

#     return CoordinateSystem(basis=basis, origin=p)


# soon to be obsolete: replaced by Plane.create_coordinate_systems_3d, to be removed with
# the rework of projections.py
# def setup_coordinate_systems_3d(
#     center_mlapdv: np.ndarray,
#     brain_normal: np.ndarray,
#     rotate_by: float | None = None,
#     invert_dims: list[bool] | None = None,
# ) -> LinkedCoordinateSystems:
#     """Create a linked set of 3D coordinate systems for imaging.

#     Args:
#         center_mlapdv: Center point in ML/AP/DV order, shape `(3,)`.
#         brain_normal: Normal vector in ML/AP/DV order, shape `(3,)`.
#         rotate_by: Optional angle in radians to rotate the imaging plane about Z.
#         invert_dims: Optional length-3 boolean list to invert basis axes.

#     Returns:
#         A `LinkedCoordinateSystems` object containing `mlapdv` and
#         `imaging_plane` coordinate systems.
#     """
#     cs3d = LinkedCoordinateSystems(
#         {
#             "mlapdv": CoordinateSystem(basis=np.identity(3), origin=np.zeros(3)),
#             "imaging_plane": coordinate_system_from_normal(
#                 center_mlapdv,
#                 brain_normal,
#                 rotate_by=rotate_by,
#                 invert_dims=invert_dims,
#             ),
#         }
#     )

#     # TODO add tests for the 3d case

#     return cs3d
