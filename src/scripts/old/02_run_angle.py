# %%
# %matplotlib qt5
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 300

# %%
import numpy as np
from plane2brain import plotters, projections, scanimage, suite2p, ibl
from plane2brain.coordinate_systems import (
    setup_coordinate_systems_3d,
    create_coordinate_system_for_image,
    get_image_corners,
)
from plane2brain.atlas import ProjectionAtlas
from one.api import ONE
import matplotlib.pyplot as plt

# %% whiterussian / local server base folder
# BASE_FOLDER = Path("/mnt/s0/Data/Subjects")
LOCATION = "server"
SAVE_OUTPUT = False
PLOT = True

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

# this is defined
scanner_orientation = dict(rotation=0.0, invert_axis=[True, True, False])
dims = ("Y", "X")

one = ONE()
# eid = one.ref2eid(dict(subject="SP058", date="2024-07-25", sequence="001"))
eid = one.ref2eid(dict(subject="SP058", date="2024-08-01", sequence="001"))

# load the reference image metadata
ref_img_meta = ibl.load_reference_stack_metadata(eid, one, location=LOCATION)
ref_point = ibl.load_reference_points_from_meta(
    ref_img_meta
)  # the craniotomy center, both in ml,ap (histology resolved) and in

# load the suite2p data
raw_imaging_meta, stat_paths, fov_map = ibl.load_fov_data(eid, one, location=LOCATION)
fov_names = sorted(list(fov_map.keys()))
coords_px = suite2p.data_loader(
    stat_paths, fov_map, dims=dims
)  # refactor: rename coords_px

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
# which is stored on disk in: dv,ml,ap (!)
ref_img_stack = ibl.load_reference_stack(eid, one, location=LOCATION)
ref_img_meta = ibl.load_reference_stack_metadata(eid, one, location=LOCATION)
ref_img_size_px = np.array(ref_img_stack[0].shape)  # ml,ap

# scanimage metadata is by default stored as XY
# with: X is the resonant dimension
# which in our reference image is the second dimension
dims = ("Y", "X")

# image resolution and dimensions of the reference stack
# in um
um_per_px = scanimage.get_resolution_from_scanimage_meta(
    ref_img_meta["rawScanImageMeta"],
    dims=dims,
)
ref_img_size_um = ref_img_size_px * um_per_px

# the center of the craniotomy is not always exactly at the center
# of the image, we adjust for this here
ref_point["mlap_adjusted"] = ref_point["mlap"] + ref_point["xy"] * um_per_px
ref_img_topleft_um = ref_point["mlap_adjusted"] - ref_img_size_um / 2

# inferring the the "virtual corner" of the reference stack image
# in ref space
ref_img_topleft_ref, ref_img_ref_per_px = ibl.infer_ref_stack_virtual_corner(
    ref_img_meta["rawScanImageMeta"],
    ref_img_size_px,
    dims=dims,
)
# the 2d coordinate system in of the reference image
coordinate_systems_ref = create_coordinate_system_for_image(
    ref_img_size_px,
    um_per_px,
    ref_img_ref_per_px,
    ref_img_topleft_ref,
)

# %% verification by diagnostic plotting
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

# the reference stack is imaged in "stripes": the resonant axis is the short one
# plotting those field of views onto the image, with their respective corners
# colored as above
cs_stripes = scanimage.create_coordinate_systems_from_scanimage_meta(
    ref_img_meta["rawScanImageMeta"], dims=dims
)
edges = ["topleft", "topright", "bottomright", "bottomleft", "topleft"]
for uuid in cs_stripes.keys():
    scanimage_fov_meta = scanimage.get_fov_meta(ref_img_meta["rawScanImageMeta"], uuid)
    fov_stripe_size_px = scanimage.get_scanfield_size_px(scanimage_fov_meta, dims=dims)
    corners = get_image_corners(fov_stripe_size_px, cs_stripes[uuid], to="um_global")

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
    *ref_point["mlap_adjusted"],
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
ds = 200  # a downsample factor just for plotting
axes.scatter(*cs[::ds, :].T, c=ref_img_stack[5].reshape((-1, 1))[::ds])

# plotting the FOV stripes
cs_stripes = scanimage.create_coordinate_systems_from_scanimage_meta(
    ref_img_meta["rawScanImageMeta"],
    dims=dims,
)
for uuid in cs_stripes.keys():
    scanimage_fov_meta = scanimage.get_fov_meta(ref_img_meta["rawScanImageMeta"], uuid)
    fov_stripe_size_px = scanimage.get_scanfield_size_px(scanimage_fov_meta, dims=dims)
    corners = get_image_corners(fov_stripe_size_px, cs_stripes[uuid], to="um_global")
    # the corners are expressed in the um global space and need to be
    # transformed into the mlapdv space first
    _corners = np.array([np.append(corners[e], 0) for e in edges])
    _corners = coordinate_systems_3d.transform(_corners, "imaging_plane", "mlapdv")
    axes.plot(*_corners.T, lw=1, color="k", zorder=10)

axes.set_aspect("equal")

# %% back to 2d, verify the location of the points on the brain surface ("reference points")
fig, axes = plt.subplots()
corners = get_image_corners(ref_img_size_px, coordinate_systems_ref, to="um_global")

# transform the um_global corners with the 3d system
# to get real ml, ap coordinates
for k, v in corners.items():
    corners[k] = coordinate_systems_3d.transform(
        np.append(v, 0), "imaging_plane", "mlapdv"
    )[:-1]  # drop DV

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
circle_center = ref_point["mlap_adjusted"][::-1]
# circle_center = [0, 0]
circle = plt.Circle(circle_center, 2500, fill=False, color="r")
axes.add_patch(circle)
axes.set_xlabel("AP")
axes.set_ylabel("ML")

# plot the reference points
brain_surface_points = ibl.load_brain_surface_points(eid, one, location=LOCATION)
# CAREFUL - we are here swapping the axis, see for explanation below
brain_surface_points_rel = np.array(
    [point["coords"][::-1] for point in brain_surface_points["points"]]
)
brain_surface_points_um = coordinate_systems_ref.transform(
    brain_surface_points_rel, "image", "um_global"
)
# the reference points are not stored in i,j as in image dimensions
# they are stored in screen coordinates as seen here
# https://www3.ntu.edu.sg/home/ehchua/programming/opengl/images/Graphics3D_DisplayCoord.png
# this means they should be swapped _before_ the transform

# then when plotting, they need to be swapped to follow matplotlibs conventions
# explained above
for p in brain_surface_points_um:
    p_ = coordinate_systems_3d.transform(np.append(p, 0), "imaging_plane", "mlapdv")[
        :-1
    ]
    axes.plot(*p_[::-1], "+", color="r")


# %% proof: brain surface points are stored in screen coordinates

# the new way of loading the points
brain_surface_points = ibl.load_brain_surface_points(eid, one, location=LOCATION)

# here loading the "old points" (as they were stored in the metadata)
# to be in perfect accordance with the ground truth example by samuel
# (make sure to load with 'resolved=False' above to perfectly
# match the ground truth image provided by samuel)
brain_surface_points = {"points": ref_img_meta["points"]}
brain_surface_points_rel = np.array(
    [
        point["coords"] for point in brain_surface_points["points"]
    ]  # loading without axis inversion
)
brain_surface_points_px = coordinate_systems_ref.transform(
    brain_surface_points_rel, "image", "pixel"
)
fig, axes = plt.subplots()
axes.matshow(ref_img_stack[5])

for p in brain_surface_points_px:
    axes.plot(*p[::+1], ".", color="w")  # no inversion here (means they are inverted)!

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

# %% back in 2d: verify coordinates of ROIs

fig, axes = plt.subplots()

image_kwargs = dict(
    # extent=plotters.extent_from_corners(corners),
    cmap="gray",
    vmin=np.percentile(ref_img_stack, 5),
    vmax=np.percentile(ref_img_stack, 99.9),
)
axes.matshow(ref_img_stack[6, :], **image_kwargs)

for uuid, coordinate_system in coordinate_systems_fovs.items():
    fov_meta = scanimage.get_fov_meta(raw_imaging_meta["rawScanImageMeta"], uuid)
    fov_size_px = scanimage.get_scanfield_size_px(fov_meta, dims=dims)

    corners = get_image_corners(fov_size_px, coordinate_system, to="um_global")
    # _corners = np.array([np.append(corners[e], 0) for e in edges])
    _corners = np.array([corners[e] for e in edges])
    _corners = coordinate_systems_ref.transform(_corners, "um_global", "pixel")
    axes.plot(
        *_corners.T[::-1], lw=1, color="r", zorder=10
    )  # again, inverting here to have ml, ap

for uuid, coordinate_system in coordinate_systems_fovs.items():
    coords_um_global = coordinate_system.transform(
        coords_px[uuid], "pixel", "um_global"
    )
    px_global = coordinate_systems_ref.transform(coords_um_global, "um_global", "pixel")

    axes.scatter(*px_global.T[::-1], color="g", s=0.5, alpha=0.5)


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
brain_surface_points = ibl.load_brain_surface_points(eid, one, location=LOCATION)

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
    *ref_point["mlap_adjusted"],
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
    dims=dims,
)

coords = projections.project_scanimage_fovs(
    coords_px,  # the pixel coordinates as loaded from suite2p
    coordinate_systems_2d,
    coordinate_systems_3d_adjusted,
    atlas=atlas,
    projection_vector=optical_axis,  # now project along the optical axis
    ds=5,
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
        coords_depths=coords[uuid]["dv_below_surface_corrected"],
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

# plot them in 3d
axes = plotters.plot_brain_surface_points(atlas.get_surface_points())
coordinate_systems_3d.plot(axes=axes, color_by="axis", scale=500)
coordinate_systems_3d_adjusted.plot(axes=axes, color_by="axis", scale=500)

uuids = sorted(list(fov_map.values()))
coordinate_systems_fovs = scanimage.create_coordinate_systems_from_scanimage_meta(
    raw_imaging_meta["rawScanImageMeta"],
    fov_uuids=uuids,
    dims=dims,
)
edges = ["topleft", "topright", "bottomright", "bottomleft", "topleft"]

for uuid, coordinate_system in coordinate_systems_fovs.items():
    fov_meta = scanimage.get_fov_meta(raw_imaging_meta["rawScanImageMeta"], uuid)
    fov_size_px = scanimage.get_scanfield_size_px(fov_meta, dims=dims)

    corners = get_image_corners(fov_size_px, coordinate_system, to="um_global")
    # the corners are expressed in the um global space and need to be
    # transformed into the mlapdv space first
    _corners = np.array([np.append(corners[e], 0) for e in edges])
    _corners = coordinate_systems_3d_adjusted.transform(
        _corners, "imaging_plane", "mlapdv"
    )
    axes.plot(*_corners.T, lw=1, color="k", zorder=100)

for name, uuid in fov_map.items():
    axes.scatter(
        *coords[uuid]["on_surface"].T,
        c=coords[uuid]["atlas_rgba"] / 255,
        s=5,
        zorder=20,
    )
    axes.scatter(*coords[uuid]["reprojected"].T, c="k", s=5)

# adding the ref img FOV
corners = get_image_corners(ref_img_size_px, coordinate_systems_ref, to="um_global")
_corners = np.array([np.append(corners[e], 0) for e in edges])
_corners = coordinate_systems_3d_adjusted.transform(_corners, "imaging_plane", "mlapdv")
axes.plot(*_corners.T, lw=1, color="k", zorder=100)


# adding the craniotomy in the ref img
def approx_circle(center, radius, n_segments=128):
    cx, cy = center
    angles = np.linspace(0.0, 2.0 * np.pi, n_segments + 1)
    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)
    return np.vstack((x, y)).T  # shape (n_segments+1, 2)


points = approx_circle((0, 0), 2500)
points = np.concatenate([points, np.zeros((points.shape[0], 1))], axis=1)
_points = coordinate_systems_3d_adjusted.transform(points, "imaging_plane", "mlapdv")
axes.plot(*_points.T, lw=1, color="k", zorder=100)

axes.set_xlabel("ML")
axes.set_ylabel("AP")
axes.set_zlabel("DV")

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
