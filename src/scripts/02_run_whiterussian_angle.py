# %%
# %matplotlib qt5
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 300

# %%
from pathlib import Path
import numpy as np

from plane2brain import plotters, projections
from plane2brain.atlas import ProjectionAtlas
from plane2brain import scanimage

from plane2brain.coordinate_systems import (
    setup_coordinate_systems_3d,
    create_coordinate_system_for_image,
    get_image_corners,
)
import plane2brain.ibl as ibl
from plane2brain.suite2p import suite2p_data_loader

from one.api import ONE
import matplotlib.pyplot as plt

# %% whiterussian / local server base folder
# BASE_FOLDER = Path("/mnt/s0/Data/Subjects")
LOCATION = "local"
SAVE_OUTPUT = False

# %%
"""
 
 ##        #######     ###    ########  #### ##    ##  ######   
 ##       ##     ##   ## ##   ##     ##  ##  ###   ## ##    ##  
 ##       ##     ##  ##   ##  ##     ##  ##  ####  ## ##        
 ##       ##     ## ##     ## ##     ##  ##  ## ## ## ##   #### 
 ##       ##     ## ######### ##     ##  ##  ##  #### ##    ##  
 ##       ##     ## ##     ## ##     ##  ##  ##   ### ##    ##  
 ########  #######  ##     ## ########  #### ##    ##  ######   
 
"""

"""
load the suite2p data
"""
one = ONE()
# eid = one.ref2eid(dict(subject="SP058", date="2024-07-25", sequence="001"))
eid = one.ref2eid(dict(subject="SP058", date="2024-08-01", sequence="001"))

# load the reference image metadata
ref_img_meta = ibl.ibl_load_reference_stack_metadata(eid, one, location=LOCATION)
ref_point_mlap, ref_point_ref = ibl.get_reference_points_from_meta(
    ref_img_meta, use_resolved=False
)  # the craniotomy center, both in ml,ap (histology resolved) and in
# the reference space of scanimage (galvos)

# load the suite2p data
raw_imaging_meta, stat_paths, fov_map = ibl.ibl_load_fov_data(
    eid, one, location=LOCATION
)
fov_names = sorted(list(fov_map.keys()))
coords_px = suite2p_data_loader(stat_paths, fov_map)  # rename coords_px

# this is unfortunately defined
# scanner_orientation = dict(rotation=3 / 2 * np.pi, invert_axis=[True, False, False])
scanner_orientation = dict(rotation=0.0, invert_axis=[True, True, False])

# this is the atlas to project onto
atlas = ProjectionAtlas(res_um=50)


# %%
"""
 
 ########  ######## ########    #### ##     ##  ######   
 ##     ## ##       ##           ##  ###   ### ##    ##  
 ##     ## ##       ##           ##  #### #### ##        
 ########  ######   ######       ##  ## ### ## ##   #### 
 ##   ##   ##       ##           ##  ##     ## ##    ##  
 ##    ##  ##       ##           ##  ##     ## ##    ##  
 ##     ## ######## ##          #### ##     ##  ######   
 
"""

# load the actual reference image stack
# dv,ml,ap
ref_img_stack = ibl.ibl_load_reference_stack(eid, one, location=LOCATION)
ref_img_meta = ibl.ibl_load_reference_stack_metadata(eid, one, location=LOCATION)

# this is in ml, ap
ref_img_size_px = np.array(ref_img_stack[0].shape)

# scanimage metadata is by default stored as XY
# X is the resonant dimension, which in our reference image is the second
dims = ["Y", "X"]

# image resolution of the reference stack
um_per_px = scanimage.get_resolution_from_scanimage_meta(
    ref_img_meta["rawScanImageMeta"],
    dims=dims,
)
ref_img_size_um = ref_img_size_px * um_per_px
ref_img_topleft_um = ref_point_mlap - ref_img_size_um / 2

# the virtual corner of the image
ref_img_topleft_ref, ref_img_ref_per_px = ibl.infer_ref_stack_virtual_corner(
    ref_img_meta["rawScanImageMeta"],
    ref_img_size_px,
    dims=dims,
)
# the coordinate system in 2d for the reference image
coordinate_systems_ref = create_coordinate_system_for_image(
    ref_img_size_px,
    um_per_px,
    ref_img_ref_per_px,
    ref_img_topleft_ref,
)


# %% verify
# TODO this function is to be moved
def get_image_corners(img_size_px, coordinate_systems, to="um_global"):
    # img_size_px is in XY (scanimageXY)
    # X = resonant = AP
    img_size_px = np.array(img_size_px)  # cast just in case

    # when image is plotted
    # vertical axis is ML
    # horizontal is AP

    # image is of shape i,j
    # in a matshow, i corresponds to vertical = ML
    # in a plot, the first argument corresponds to xaxis = horizontal

    corners = dict(
        topleft=[0, 0],
        topright=[0, 1],  #
        bottomleft=[1, 0],
        bottomright=[1, 1],
        center=img_size_px / 2,
    )
    return {
        name: coordinate_systems.transform(np.array(corner), "image", to)
        for name, corner in corners.items()
    }


corners = get_image_corners(ref_img_size_px, coordinate_systems_ref, to="um_global")
extent = plotters.extent_from_corners(corners)
fig, axes = plt.subplots()

image_kwargs = dict(
    extent=plotters.extent_from_corners(corners),
    cmap="viridis",
    vmin=np.percentile(ref_img_stack, 0.1),
    vmax=np.percentile(ref_img_stack, 99.5),
)
axes.matshow(ref_img_stack[5, :], **image_kwargs)

# plotting the image corners
#  --- THIS IS IMPORTANT ---
# we need to invert here because of matplotlibs plotting convention
# first argument is xaxis which is NOT image X axis when the image is an array XY
# we have defined our image to be XY with, increasing image indices i,j <> -ML,-AP
axes.plot(*corners["topleft"][::-1], ".", color="r", markersize=50)
axes.plot(*corners["topright"][::-1], ".", color="g", markersize=50)
axes.plot(*corners["bottomleft"][::-1], ".", color="b", markersize=50)

# plot the stripes, in 2d
cs_stripes = scanimage.create_coordinate_systems_from_scanimage_meta(
    ref_img_meta["rawScanImageMeta"], dims=dims
)
edges = ["topleft", "topright", "bottomright", "bottomleft", "topleft"]
for uuid in cs_stripes.keys():
    img_size_px = scanimage.get_fov_meta(ref_img_meta["rawScanImageMeta"], uuid)[
        "scanfields"
    ]["pixelResolutionXY"]
    corners = get_image_corners(img_size_px, cs_stripes[uuid], to="um_global")

    # colored corners for each fov stripe
    axes.plot(*corners["topleft"][::-1], ".", color="r", markersize=10)
    axes.plot(*corners["topright"][::-1], ".", color="g", markersize=10)
    axes.plot(*corners["bottomleft"][::-1], ".", color="b", markersize=10)
    # bounding box
    box = np.array([corners[edge] for edge in edges])
    axes.plot(box[:, 1], box[:, 0], color="k", lw=2)


# %% plot the ref image the stripes

# plot them in 3d
axes = plotters.plot_brain_surface_points(atlas.get_surface_points())

# the coordinate systems
ref_point_mlapdv, brain_normal_at_ref = atlas.get_plane_at_point_mlap(
    *ref_point_mlap,
    numba=True,
)
coordinate_systems_3d = setup_coordinate_systems_3d(
    ref_point_mlapdv,
    brain_normal_at_ref,
    rotate_by=scanner_orientation["rotation"],
    invert_dims=scanner_orientation["invert_axis"],
)

coordinate_systems_3d.plot(axes=axes, color_by="axis", scale=500)

# in the imaging plane, plotting the boundaries of the FOVs
edges = ["topleft", "topright", "bottomright", "bottomleft", "topleft"]

#  plot the pixels of the ref img
from itertools import product

pixels = np.array(list(product(range(ref_img_size_px[0]), range(ref_img_size_px[1]))))
coords_um = coordinate_systems_ref.transform(pixels, "pixel", "um_global")
coords_um_ = np.concatenate([coords_um, np.zeros((coords_um.shape[0], 1))], axis=1)
cs = coordinate_systems_3d.transform(coords_um_, "imaging_plane", "mlapdv")
ds = 200
axes.scatter(*cs[::ds, :].T, c=ref_img_stack[5].reshape((-1, 1))[::ds])
# those look good!

# the stripes
cs_stripes = scanimage.create_coordinate_systems_from_scanimage_meta(
    ref_img_meta["rawScanImageMeta"],
    dims=dims,
)
for uuid in cs_stripes.keys():
    scanimage_fov_meta = scanimage.get_fov_meta(ref_img_meta["rawScanImageMeta"], uuid)
    fov_stripe_size_px = scanimage.get_scanfield_size_px(scanimage_fov_meta, dims=dims)
    corners = get_image_corners(img_size_px, cs_stripes[uuid], to="um_global")
    # the corners are expressed in the um global space and need to be
    # transformed into the mlapdv space first
    _corners = np.array([np.append(corners[e], 0) for e in edges])
    _corners = coordinate_systems_3d.transform(_corners, "imaging_plane", "mlapdv")
    axes.plot(*_corners.T, lw=1, color="k", zorder=10)

axes.set_aspect("equal")

# %% back to 2d, now with the reference points integrated
fig, axes = plt.subplots()
corners = get_image_corners(ref_img_size_px, coordinate_systems_ref, to="um_global")

# transform the um_global corners with the 3d system
for k, v in corners.items():
    corners[k] = coordinate_systems_3d.transform(
        np.append(v, 0), "imaging_plane", "mlapdv"
    )[:-1]

image_kwargs = dict(
    extent=plotters.extent_from_corners(corners),
    cmap="gray",
    vmin=np.percentile(ref_img_stack, 5),
    vmax=np.percentile(ref_img_stack, 99.9),
)
axes.matshow(ref_img_stack[6, :], **image_kwargs)

axes.set_aspect("equal")
line_kwargs = dict(linestyle=":", lw=1, alpha=1, color="r")
axes.axhline(0, **line_kwargs)
axes.axvline(0, **line_kwargs)
circle_center = ref_point_mlap[::-1]
# circle_center = [0, 0]
circle = plt.Circle(circle_center, 2500, fill=False, color="r")
axes.add_patch(circle)
axes.set_xlabel("AP")
axes.set_ylabel("ML")

# plot the reference points
brain_surface_points = ibl.ibl_load_brain_surface_points(eid, one, location=LOCATION)
brain_surface_points_rel = np.array(
    [point["coords"][::-1] for point in brain_surface_points["points"]]
)
brain_surface_points_um = coordinate_systems_ref.transform(
    brain_surface_points_rel, "image", "um_global"
)
# normally, we would expect here the plotting axis swappingh again as
# the xaxis is the second image dimension
# BUT
# the reference points are not stored in i,j as in image dimensions
# they are stored in screen coordinates as seen here
# https://www3.ntu.edu.sg/home/ehchua/programming/opengl/images/Graphics3D_DisplayCoord.png
for p in brain_surface_points_um:
    p_ = coordinate_systems_3d.transform(np.append(p, 0), "imaging_plane", "mlapdv")[
        :-1
    ]
    axes.plot(*p_[::-1], "+", color="r")


# %% proof: why do we need to swap the axes of the surface points
# they should be stored in image dim1,2
# they are not
# here loading the "old points" (as they were stored in the metadata)
# to be in perfect accordance with the ground truth example by samuel
brain_surface_points = ibl.ibl_load_brain_surface_points(eid, one, location=LOCATION)
brain_surface_points = {"points": ref_img_meta["points"]}
brain_surface_points_rel = np.array(
    [point["coords"] for point in brain_surface_points["points"]]
)
brain_surface_points_px = coordinate_systems_ref.transform(
    brain_surface_points_rel, "image", "pixel"
)
fig, axes = plt.subplots()
axes.matshow(ref_img_stack[5])

for p in brain_surface_points_px:
    axes.plot(*p[::1], ".", color="w")

# %% back to 3d and verify with the functional imaging FOVs

# plot them in 3d
axes = plotters.plot_brain_surface_points(atlas.get_surface_points())
coordinate_systems_3d.plot(axes=axes, color_by="axis", scale=500)

uuids = sorted(list(fov_map.values()))
coordinate_systems_fovs = scanimage.create_coordinate_systems_from_scanimage_meta(
    raw_imaging_meta["rawScanImageMeta"],
    fov_uuids=uuids,
    dims=dims,
)

for uuid, coordinate_system in coordinate_systems_fovs.items():
    fov_meta = scanimage.get_fov_meta(raw_imaging_meta["rawScanImageMeta"], uuid)
    fov_size_px = scanimage.get_scanfield_size_px(fov_meta, dims=dims)

    corners = get_image_corners(fov_size_px, coordinate_system, to="um_global")
    # the corners are expressed in the um global space and need to be
    # transformed into the mlapdv space first
    _corners = np.array([np.append(corners[e], 0) for e in edges])
    _corners = coordinate_systems_3d.transform(_corners, "imaging_plane", "mlapdv")
    axes.plot(*_corners.T, lw=1, color="k", zorder=10)

# %%
"""
######## #### ##       ########       ###    ########        ## ##     ##  ######  ########
   ##     ##  ##          ##         ## ##   ##     ##       ## ##     ## ##    ##    ##
   ##     ##  ##          ##        ##   ##  ##     ##       ## ##     ## ##          ##
   ##     ##  ##          ##       ##     ## ##     ##       ## ##     ##  ######     ##
   ##     ##  ##          ##       ######### ##     ## ##    ## ##     ##       ##    ##
   ##     ##  ##          ##       ##     ## ##     ## ##    ## ##     ## ##    ##    ##
   ##    #### ########    ##       ##     ## ########   ######   #######   ######     ##
"""

# %% adjusting for the fact that this is not the case: getting the optical axis
# load the brain surface points and get the normal
brain_surface_points = ibl.ibl_load_brain_surface_points(eid, one, location=LOCATION)

# this normal is expressed in the coordinate system of the reference stack
p_surface, n_surface, dv_avg = projections.get_brain_surface_normal(
    brain_surface_points,
    ref_img_meta,
    coordinate_systems_ref,
)

# the vector n_surface is expressed in the coordinate system of the reference stack
# express n_surface in mlapdv atlas 3d space

# this gets the dv component for the ref point, as well as the brain normal at that
# location
ref_point_mlapdv, brain_normal_at_ref = atlas.get_plane_at_point_mlap(
    *ref_point_mlap,
    numba=True,
)

# this sets up the 3d coordinate systems with the imaging plane, assuming it is
# brain normal and optical axis are colinear
coordinate_systems_3d = setup_coordinate_systems_3d(
    ref_point_mlapdv,
    brain_normal_at_ref,  # this is to be replaced with the optical axis
    rotate_by=scanner_orientation["rotation"],
    invert_dims=scanner_orientation["invert_axis"],
)

# this requires a coordinate system for 3d
optical_axis = (
    coordinate_systems_3d.transform(n_surface, "imaging_plane", "mlapdv")
    - ref_point_mlapdv
)

# set up a new 3d coordinate system with the imaging plane, now adjusted by the difference
# between the optical axis and the brain normal
coordinate_systems_3d_adjusted = setup_coordinate_systems_3d(
    ref_point_mlapdv,
    optical_axis,  # now adjusted for the optical axis
    rotate_by=scanner_orientation["rotation"],
    invert_dims=scanner_orientation["invert_axis"],
)

# %%
"""
########  ########   #######        ## ########  ######  ######## ####  #######  ##    ##
##     ## ##     ## ##     ##       ## ##       ##    ##    ##     ##  ##     ## ###   ##
##     ## ##     ## ##     ##       ## ##       ##          ##     ##  ##     ## ####  ##
########  ########  ##     ##       ## ######   ##          ##     ##  ##     ## ## ## ##
##        ##   ##   ##     ## ##    ## ##       ##          ##     ##  ##     ## ##  ####
##        ##    ##  ##     ## ##    ## ##       ##    ##    ##     ##  ##     ## ##   ###
##        ##     ##  #######   ######  ########  ######     ##    ####  #######  ##    ##
"""

# %% setting up the coordinate systems for the imaged fovs
fov_uuids = sorted(list(fov_map.values()))
coordinate_systems_2d = scanimage.create_coordinate_systems_from_scanimage_meta(
    raw_imaging_meta["rawScanImageMeta"],
    fov_uuids=fov_uuids,
)

coords = projections.project_scanimage_fovs(
    coords_px,  # the pixel coordinates as loaded from suite2p
    coordinate_systems_2d,
    coordinate_systems_3d_adjusted,
    atlas=atlas,
    projection_vector=optical_axis,  # now project along the optical axis
    ds=20,
)

# extract depths
fov_uuids = sorted(list(fov_map.values()))
fov_depths = scanimage.extract_fov_depths_from_scanimage_meta(
    scanimage_meta=raw_imaging_meta["rawScanImageMeta"],
    scanimage_params=raw_imaging_meta["scanImageParams"],
    fov_uuids=fov_uuids,
)
# this creates: the keys 'um_corrected' and 'dv_below_surface'
coords = projections.correct_coords_for_tilt_2d(
    coords,
    coordinate_systems_2d,
    fov_depths,
    p_surface,
    n_surface,
)

# the use of um_corrected requires a new projection
for uuid in list(coords.keys()):
    coords_on_surface = projections.project_coords_onto_atlas_surface(
        coords_um=coords[uuid]["um_corrected"],
        coordinate_systems_3d=coordinate_systems_3d_adjusted,
        atlas=atlas,
        projection_vector=optical_axis,
    )
    coords_reprojected = projections.project_down_from_surface(
        coords_on_surface=coords_on_surface,
        atlas=atlas,
        coords_depths=coords[uuid]["dv_below_surface"],
    )
    coords[uuid]["reprojected"] = coords_reprojected  # this is mlapdv


# %% some quantification of differences
for name, uuid in fov_map.items():
    _coords = coords[uuid]["pixel"]
    coords_um = coordinate_systems_2d[uuid].transform(_coords, "pixel", "um_global")
    xy_min = np.min(coords_um - coords[uuid]["um_corrected"], axis=0)
    xy_max = np.max(coords_um - coords[uuid]["um_corrected"], axis=0)
    dv_min = np.min((dv_avg - fov_depths[uuid]) - coords[uuid]["dv_below_surface"])
    dv_max = np.max((dv_avg - fov_depths[uuid]) - coords[uuid]["dv_below_surface"])
    print(f"-- {name} --")
    print(f"x: min/max {xy_min[0]:.2f}/{xy_max[0]:.2f}")
    print(f"y: min/max {xy_min[1]:.2f}/{xy_max[1]:.2f}")
    print(f"dv: min/max {dv_min:.2f}/{dv_max:.2f}")
    print()

# %% map anything mlapdv to brain area
for name, uuid in fov_map.items():
    ids, ix, rgba, acronym = atlas.get_labels_for_mlapdv(coords[uuid]["reprojected"])
    coords[uuid]["atlas_rgba"] = rgba
    coords[uuid]["atlas_acronym"] = acronym
    coords[uuid]["atlas_id"] = ids

# %%
"""
##     ## ####  ######
##     ##  ##  ##    ##
##     ##  ##  ##
##     ##  ##   ######
 ##   ##   ##        ##
  ## ##    ##  ##    ##
   ###    ####  ######
"""


# %% some diagnostic plotting
from plane2brain.coordinate_systems import get_image_corners

fig, axes = plt.subplots()
fov_uuids = sorted(list(coords.keys()))
for name, uuid in fov_map.items():
    # stat = np.load(stat_paths[name], allow_pickle=True)
    # we get the pixel coordinates as from suite2p
    _coords = coords[uuid]["pixel"]

    # convert to the global um -> this is how the FOVs relate to each other in
    # scanimage ref space
    coords_um = coordinate_systems_2d[uuid].transform(_coords, "pixel", "um_global")

    # this can be straightforward plotted
    img_size_px = scanimage.get_fov_meta(raw_imaging_meta["rawScanImageMeta"], uuid)[
        "scanfields"
    ]["pixelResolutionXY"]
    axes.scatter(*coords_um.T, c=coords[uuid]["atlas_rgba"] / 255)

    # in the same space we can plot the boundaries of the individual FOVs
    corners = get_image_corners(
        img_size_px, coordinate_systems_2d[uuid], to="um_global"
    )
    plotters.plot_fov_outline_from_corners(corners, axes=axes)

axes.set_aspect("equal")
kwargs = dict(linestyle=":", lw=1, alpha=1, color="k")
axes.axhline(0, **kwargs)
axes.axvline(0, **kwargs)
circle = plt.Circle((0, 0), 3000, fill=False, color="k")
axes.add_patch(circle)
axes.set_xlabel("X")
axes.set_ylabel("Y")

# while this looks wrong - keep in mind that this is in the ref space transformed to global um
# so in order to make this align with what you would get by a .matshow, the scanner orientation
# has to be taken into account

# %% don't really know where I was going with this
fig, axes = plt.subplots()
fov_uuids = sorted(list(coords.keys()))
for name, uuid in fov_map.items():
    _coords = coords[uuid]["pixel"]
    coords_um = coordinate_systems_2d[uuid].transform(_coords, "pixel", "um_global")
    coords_um_ = np.concatenate([coords_um, np.zeros((coords_um.shape[0], 1))], axis=1)
    coords_mlap = coordinate_systems_3d.transform(
        coords_um_, "imaging_plane", "mlapdv"
    )[:, :-1]
    axes.scatter(*coords_mlap.T, c=coords[uuid]["atlas_rgba"] / 255)

    steven_mlap = ibl.ibl_load_roi_mlapdv(eid, one, name, location="local")[:, :-1]
    axes.plot(*steven_mlap.T, ".", color="k")

axes.set_aspect("equal")
kwargs = dict(linestyle=":", lw=1, alpha=1, color="k")
axes.axhline(0, **kwargs)
axes.axvline(0, **kwargs)
circle = plt.Circle(ref_point_mlap, 3000, fill=False, color="k")
axes.add_patch(circle)
axes.set_xlabel("ML")
axes.set_ylabel("AP")


# %% combining everything in 3d

# the brain surface
axes = plotters.plot_brain_surface_points(atlas.get_surface_points())

# the coordinate systems
coordinate_systems_3d.plot(axes=axes, color_by="axis", scale=500)

# the ROIs on the surface
for name, uuid in fov_map.items():
    plotters.plot_points(
        coords[uuid]["on_surface"],
        axes=axes,
        s=2,
        color=coords[uuid]["atlas_rgba"] / 255,
    )

# in the imaging plane, plotting the boundaries of the FOVs
edges = ["topleft", "topright", "bottomright", "bottomleft", "topleft"]
for name, uuid in fov_map.items():
    img_size_px = scanimage.get_fov_meta(raw_imaging_meta["rawScanImageMeta"], uuid)[
        "scanfields"
    ]["pixelResolutionXY"]
    corners = get_image_corners(
        img_size_px, coordinate_systems_2d[uuid], to="um_global"
    )
    # the corners are expressed in the um global space and need to be
    # transformed into the mlapdv space first
    _corners = np.array([np.append(corners[e], 0) for e in edges])
    _corners = coordinate_systems_3d.transform(_corners, "imaging_plane", "mlapdv")
    axes.plot(*_corners.T, lw=1, color="k", zorder=10)

# in this 3d space, plot the outline of the reference image
corners = get_image_corners(ref_img_size_px, coordinate_systems_ref, to="um_global")
# notes about the following line
# the "um" space of the coordinate_systems_ref is not that straightforward
# to understand. It does not map to atlas space, but has the correct image
# scale. The coordinate_systems_3d has the mouse / microscope orientation
# and axis inversion
# however, unlike for the individual FOVs, we can't straight plot the corners
# but we need to offset by the origin (ref_point_mlap)
_corners = np.array([np.append(corners[e], 0) for e in edges])
_corners = coordinate_systems_3d.transform(_corners, "imaging_plane", "mlapdv")
axes.plot(*_corners.T, lw=1, color="k", zorder=10)
axes.plot(*_corners[0], ".", lw=1, color="r")
axes.plot(*_corners[1], ".", lw=1, color="g")  # this is a problem!
axes.plot(*_corners[3], ".", lw=1, color="b")  # this is a problem!

axes.set_aspect("equal")

# %% now, verify the tilt
axes = plotters.plot_brain_surface_points(atlas.get_surface_points())
for name, uuid in fov_map.items():
    plotters.plot_points(
        coords[uuid]["on_surface"],
        axes=axes,
        s=2,
        color=coords[uuid]["atlas_rgba"] / 255,
    )

    plotters.plot_points(
        coords[uuid]["reprojected"],
        axes=axes,
        s=2,
        color="k",
    )
coordinate_systems_3d.plot(axes=axes, color_by="system", scale=500)
coordinate_systems_3d_adjusted.plot(axes=axes, color_by="axis", scale=500)

# ok this looks like:
# the adjusted system is tilted more towards the anterior
# this means of the ref image the two ap points should be lower
# this is the case

# %% a plotter with a coordinate system
from plane2brain.coordinate_systems import get_image_corners

image = ref_img_stack[5]

fig, axes = plt.subplots()
corners = get_image_corners(ref_img_size_px, coordinate_systems_ref, to="um_image")
# in order to turn the corners into the mlap equivalents, we could try to:
# convert to actual mlapdv, throw away dv component
# this should capture the rotation and axis inversion
corners_ = {}
for k, v in corners.items():
    corners_[k] = coordinate_systems_3d.transform(
        np.append(v - ref_point_mlap, 0), "imaging_plane", "mlapdv"
    )[:-1]
# this is now with the hacked coordinate system for the ref

axes.plot(*coordinate_systems_ref.transform([0, 0], "image", "pixel"), ".", color="r")
axes.plot(*coordinate_systems_ref.transform([0, 1], "image", "pixel"), ".", color="g")
axes.plot(*coordinate_systems_ref.transform([1, 1], "image", "pixel"), ".", color="b")

kwargs = dict(
    # extent=plotters.extent_from_corners(corners_),
    vmin=np.percentile(ref_img_stack, 1),
    vmax=np.percentile(ref_img_stack, 99.9),
)
axes.matshow(image, **kwargs)
axes.set_aspect("equal")
kwargs = dict(linestyle=":", lw=1, alpha=1, color="w")
# axes.axhline(0, **kwargs)
# axes.axvline(0, **kwargs)


# %%
_ref_point_mlap = (
    ref_point_mlap[1],
    ref_point_mlap[0],
)  # this inversion is because the ml, ap (and ml is Y axis = 1)

for d in [2300, 2500, 2700]:
    circle = plt.Circle(_ref_point_mlap, d, fill=False, color="w", alpha=0.5)
    axes.add_patch(circle)
axes.set_xlabel("X")
axes.set_ylabel("Y")

# get the brain surface points
brain_surface_points = ibl.ibl_load_brain_surface_points(eid, one, location=LOCATION)

# FIXME an alternative source, and they are not equal!
brain_surface_points = {"points": ref_img_meta["points"]}

# DOCME user selected

# stack_ixs = [point["stack_idx"] for point in brain_surface_points["points"]]
# the position of the voice coil (for z offset calculation)
# fastz_pos = ref_img_meta["scanImageParams"]['hFastZ']['position']
# inversion of the sign: positive is up
# stack_dv = (
#     -1 * np.array(ref_img_meta["scanImageParams"]["hStackManager"]["zs"])[stack_ixs]
# )
# dv_avg = np.average(
#     stack_dv
# )  # horizontally average plane between the selected surface points
brain_surface_points_rel = np.array(
    [point["coords"] for point in brain_surface_points["points"]]
)
brain_surface_points_rel_um = coordinate_systems_ref.transform(
    brain_surface_points_rel, "image", "um_image"
)  # first transform from the relative image coordinates to something um

# for p in brain_surface_points_rel_um:
#     p_ = coordinate_systems_3d_ref.transform(
#         np.append(p - ref_point_mlap, 0), "imaging_plane", "mlapdv"
#     )
#     axes.plot(p_[1], p_[0], ".", color="w")
# these are the 3 points on the brain surface, relative, in um
# brain_surface_points_rel_um_3d = np.concatenate(
#     [brain_surface_points_rel_um, stack_dv[:, np.newaxis]], axis=1
# )

# same inversion of dimensions as above
# for p in brain_surface_points_rel_um_3d:
#     axes.plot(p[1], p[0], ".", color="w")

brain_surface_points_rel_px = coordinate_systems_ref.transform(
    brain_surface_points_rel, "image", "pixel"
)  # first transform from the relative image coordinates to some

# in pixels
fig, axes = plt.subplots()
axes.matshow(ref_img_stack[5])
axes.plot(*brain_surface_points_rel_px.T, ".", c="w")

# in um
corners = get_image_corners(ref_img_size_px, coordinate_systems_ref, to="um_global")
# transform the um_global corners with the 3d system
for k, v in corners.items():
    corners[k] = coordinate_systems_3d.transform(
        np.append(v, 0), "imaging_plane", "mlapdv"
    )[:-1]

extent = plotters.extent_from_corners(corners)
#
fig, axes = plt.subplots()
axes.matshow(ref_img_stack[5], extent=extent)


# _corners = np.array([np.append(corners[e], 0) for e in edges])
# _corners = coordinate_systems_3d.transform(_corners, "imaging_plane", "mlapdv")


# %%
"""
 
  ######     ###    ##     ## ########     #######  ##     ## ######## ########  ##     ## ######## 
 ##    ##   ## ##   ##     ## ##          ##     ## ##     ##    ##    ##     ## ##     ##    ##    
 ##        ##   ##  ##     ## ##          ##     ## ##     ##    ##    ##     ## ##     ##    ##    
  ######  ##     ## ##     ## ######      ##     ## ##     ##    ##    ########  ##     ##    ##    
       ## #########  ##   ##  ##          ##     ## ##     ##    ##    ##        ##     ##    ##    
 ##    ## ##     ##   ## ##   ##          ##     ## ##     ##    ##    ##        ##     ##    ##    
  ######  ##     ##    ###    ########     #######   #######     ##    ##         #######     ##    
 
"""
if SAVE_OUTPUT:
    for name, uuid in fov_map.items():
        session_folder = ibl._eid2path(eid, one, location=LOCATION)
        coords_mlapdv = coords[uuid]["reprojected"]
        # saving the updated coordinates
        np.save(
            session_folder / "alf" / name / "mpciROIs.mlapdv_angle_projection.npy",
            coords_mlapdv,
        )
        # saving the atlas ids
        atlas_ids = atlas.get_labels_for_mlapdv(coords_mlapdv)[0]
        np.save(
            session_folder
            / "alf"
            / name
            / "mpciROIs.brainLocationIds_ccf_2017_angle_projection.npy",
            atlas_ids,
        )

# %%
