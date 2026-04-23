# %%
from pathlib import Path

import numpy as np

from plane2brain import projections

from plane2brain.atlas import ProjectionAtlas

from plane2brain.scanimage import (
    get_resolution_from_scanimage_meta,
    extract_fov_depths_from_scanimage_meta,
)
from plane2brain.coordinate_systems import (
    setup_coordinate_systems_3d,
    create_coordinate_system_for_ref,
)
from one.api import ONE

import plane2brain.ibl as ibl
from plane2brain.suite2p import suite2p_data_loader


import skimage
from ibllib.mpci.registration import register_reference_stacks

# %% whiterussian / local server base folder
BASE_FOLDER = Path("/mnt/s0/Data/Subjects")
LOCATION = "server"


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
eid = one.ref2eid(dict(subject="SP058", date="2024-07-25", sequence="001"))
# eid = one.ref2eid(dict(subject="SP058", date="2024-08-01", sequence="001"))

# load the reference image metadata
ref_img_meta = ibl.ibl_load_reference_stack_metadata(eid, one, location=LOCATION)
ref_point_mlap, ref_point_ref = ibl.get_reference_points_from_meta(
    ref_img_meta, use_resolved=True
)  # the craniotomy center, both in ml,ap (histology resolved) and in
# the reference space of scanimage (galvos)

# load the suite2p data
raw_imaging_meta, stat_paths, fov_map = ibl.ibl_load_fov_data(
    eid, one, location=LOCATION
)
fov_names = sorted(list(fov_map.keys()))
coords_px = suite2p_data_loader(stat_paths, fov_map)  # rename coords_px

# this is unfortunately defined
scanner_orientation = dict(rotation=3 / 2 * np.pi, invert_axis=[True, False, False])

# this is the atlas to project onto
atlas = ProjectionAtlas(res_um=50)

# %% integrating histology information from a reference session

# reference session for SP058: "SP058/2024-08-14/001"
eid_ref = one.ref2eid(dict(subject="SP058", date="2024-08-14", sequence="001"))

# get the path to the reference stack
ref_stack_path = ibl.ibl_get_reference_stack_path(
    eid,
    one,
    location=LOCATION,
    raw_imaging_collection=ibl.infer_imaging_collection(eid, one, location=LOCATION),
)

# correspondingly, to the reference stack of the reference session
ref_sess_ref_stack_path = ibl.ibl_get_reference_stack_path(
    eid_ref,
    one,
    location=LOCATION,
    raw_imaging_collection=ibl.infer_imaging_collection(
        eid_ref, one, location=LOCATION
    ),
)

# the transform between them
_, transform_params = register_reference_stacks(ref_stack_path, ref_sess_ref_stack_path)

# the transform between the reference stack and the "reference reference" stack
# = the reference stack of the reference session
ref_transform = skimage.transform.EuclideanTransform(
    rotation=transform_params["rotation"],
) + skimage.transform.EuclideanTransform(
    translation=transform_params["translation"],
)

# several problems here:
# this is a transform between two images that is in pixel space
# currently unclear what is the origin of the rotation

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
ref_img_stack = ibl.ibl_load_reference_stack(eid, one, location=LOCATION)

# metadata was already loaded above
# ref_img_meta = ibl.ibl_load_reference_stack_metadata(eid, one, location=LOCATION)
ref_img_size_px = np.array(ref_img_stack[0].shape)

# image resolution of the reference stack
um_per_px = get_resolution_from_scanimage_meta(
    ref_img_meta["rawScanImageMeta"]
)  # in X,Y
ref_img_size_um = ref_img_size_px * um_per_px

# ref_point_mlap, ref_point_ref = ibl.get_reference_points_from_meta(
#     ref_img_meta, use_resolved=True
# )  # the craniotomy center, both in mlap and in "ref" space of scanimage

# from the transform we computed between the reference stacks (of the session at hand
# and the reference session)
# we take the translation component to integrate it here -
# FIXME somthing similar to this
# check the dimensions
ref_point_mlap = ref_point_mlap + transform_params["translation"] * um_per_px

# TODO - deal with the entire situation of the reference image and it's axis
# we do plus here instead of minus because of the image axis inversion
ref_img_topleft_um = ref_point_mlap + np.array([1, 1]) * ref_img_size_um / 2

# the coordinate system of the reference image/stack
coordinate_systems_ref = create_coordinate_system_for_ref(
    ref_img_size_px,  # (ml, ap)
    um_per_px * -1,  # :(
    ref_img_topleft_um,
)
# what is this coordinate system used for?

# ref_img_center_um = get_image_corners(ref_img_size_px, coordinate_systems_ref)["center"]

# %% setting up the coordinate systems for the imaged fovs
fov_uuids = sorted(list(fov_map.values()))

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

# %% adjusting for the fact that this is not the case: getting the optical axis
# load the brain surface points and get the normal
brain_surface_points = ibl.ibl_load_brain_surface_points(eid, one, location=LOCATION)
p_surface, n_surface, dv_avg = projections.get_brain_surface_normal(
    brain_surface_points,
    ref_img_meta,
    coordinate_systems_ref,
)
# express n_surface in mlapdv, shift by origin
optical_axis = (
    coordinate_systems_3d.transform(n_surface, "imaging_plane", "mlapdv")
    - ref_point_mlapdv
)

# set up a new 3d coordinate system with the imaging plane, now adjusted by the difference
# between the optical axis and the brain normal
coordinate_systems_3d = setup_coordinate_systems_3d(
    ref_point_mlapdv,
    optical_axis,  # this is to be replaced with the optical axis
    rotate_by=scanner_orientation["rotation"],
    invert_dims=scanner_orientation["invert_axis"],
)

# %% the 2d coordinate systems, by fov name
from plane2brain.scanimage import create_coordinate_systems_from_scanimage_meta

coordinate_systems_2d = create_coordinate_systems_from_scanimage_meta(
    raw_imaging_meta["rawScanImageMeta"],
    fov_uuids=fov_uuids,
)

coords = projections.project_scanimage_fovs(
    coords_px,
    coordinate_systems_2d,
    coordinate_systems_3d,
    atlas=atlas,
    ds=10,
)

# %%


# %% this seems to be for the reference image??
ref_point_mlapdv, brain_normal_at_ref = atlas.get_plane_at_point_mlap(
    *ref_point_mlap, numba=True
)
# FIXME and here we don't rotate ...
coordinate_systems_3d = setup_coordinate_systems_3d(
    ref_point_mlapdv, brain_normal_at_ref, rotate_by=None
)

# %%
brain_surface_points = ibl.ibl_load_brain_surface_points(eid, one, location=LOCATION)

p_surface, n_surface, dv_avg = projections.get_brain_surface_normal(
    brain_surface_points,
    ref_img_meta,
    coordinate_systems_ref,
)

fov_uuids = sorted(list(fov_map.values()))
fov_depths = extract_fov_depths_from_scanimage_meta(
    raw_imaging_meta["rawScanImageMeta"],
    raw_imaging_meta["scanImageParams"],
    fov_uuids,
)


# %%
# coords, coordinate_systems_2d, coordinate_systems_3d = (
#     projections.project_from_scanimage_meta(
#         coords_px,  # this is what is read from suite2p
#         scanimage_meta=raw_imaging_meta["rawScanImageMeta"],
#         scanner_orientation=scanner_orientation,
#         common_point_mlap=ref_point_mlap,
#         atlas=atlas,
#         ds=10,  # FIXME DEBUGING
#     )
# )

# %%
# session_path = ibl._eid2path(eid, one=one, location=LOCATION)
# # list((session_path / 'alf' / 'FOV_00').glob('*mlapdv*'))
# rois_mlapdv = np.load(session_path / "alf" / "FOV_00" / "mpciROIs.mlapdv.npy")
# rois_mlapdv_transformed = skimage.transform.warp(
#     rois_mlapdv,
#     ref_transform,
#     order=1,
#     mode="constant",
#     cval=0,
#     clip=True,
#     preserve_range=True,
# )

# %%
# """
# if .mlapdv present
# apply transform!
# take corrected pixel xy or not?
# nearest neighbour pixels
# """
# session_path = ibl._eid2path(eid, one=one, location=LOCATION)
# list(
#     (session_path / ibl.infer_imaging_collection(eid, one=one, location=LOCATION)).glob(
#         "*"
#     )
# )
# list((session_path / "alf" / "FOV_00").glob("*mlapdv*"))
# Upload the saved image to Alyx as a note
# %%

# %%
# coords = projections.correct_coords_for_tilt_2d(
#     coords,
#     coordinate_systems_2d,
#     fov_depths,
#     p_surface,
#     n_surface,
# )

# # %% this is the reprojection
# coords = projections.reproject_coords(
#     coords, coordinate_systems_3d, atlas, brain_normal_at_ref
# )

# # %% diagnostic plots
# axes = plotters.plot_brain_surface_points(brain_surface_points)
# for name, uuid in fov_map.items():
#     plotters.plot_points(
#         coords[uuid]["reprojected"],
#         axes=axes,
#         s=2,
#         color="k",
#         # color=coords[uuid]["atlas_rgba"] / 255,
#     )
# coordinate_systems_3d.plot(axes=axes, color_by="axis", scale=500)

# # %% some quantification of differences
# for name, uuid in fov_map.items():
#     _coords = coords[uuid]["pixel"]
#     coords_um = coordinate_systems_2d[uuid].transform(_coords, "pixel", "um_global")
#     print(name, np.average(coords_um - coords[uuid]["um_corrected"], axis=0))
#     print(
#         name, np.average((dv_avg - fov_depths[uuid]) - coords[uuid]["dv_below_surface"])
#     )

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

for name, uuid in fov_map.items():
    session_folder = BASE_FOLDER / one.eid2path(eid).session_path_short()
    coords_mlapdv = coords[uuid]["reprojected"]["mlapdv"]
    # saving the updated coordinates
    np.save(
        session_folder / "alf" / name / "mpciROIs.mlapdv_v2.npy",
        coords_mlapdv,
    )
    # saving the atlas ids
    atlas_ids = atlas.get_labels_for_mlapdv(coords_mlapdv)[0]
    np.save(
        session_folder / "alf" / name / "mpciROIs.brainLocationIds_ccf_2017_v2.npy",
        atlas_ids,
    )

# %% offset DEV
