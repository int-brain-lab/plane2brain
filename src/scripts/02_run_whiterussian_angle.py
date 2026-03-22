# %%
from pathlib import Path

import numpy as np

from plane2brain import plotters, projections

from plane2brain.atlas import ProjectionAtlas

from plane2brain.scanimage import (
    extract_fov_depths_from_scanimage_meta,
    create_coordinate_systems_from_scanimage_meta,
    get_resolution_from_scanimage_meta,
)
from plane2brain.coordinate_systems import (
    setup_coordinate_systems_3d,
    create_coordinate_system_for_ref,
)

from one.api import ONE

import plane2brain.ibl as ibl
from plane2brain.suite2p import suite2p_data_loader

import matplotlib.pyplot as plt


# %% whiterussian / local server base folder
BASE_FOLDER = Path("/mnt/s0/Data/Subjects")
LOCATION = "server"
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

# TODO - deal with the entire situation of the reference image and it's axis
# we do plus here instead of minus because of the image axis inversion
ref_img_topleft_um = ref_point_mlap + np.array([1, 1]) * ref_img_size_um / 2

# the coordinate system of the reference image/stack
coordinate_systems_ref = create_coordinate_system_for_ref(
    ref_img_size_px,  # (ml, ap)
    um_per_px * -1,  # :(
    ref_img_topleft_um,
)
# what is this coordinate system used for? Potentially we will need this only for the mpci.meanImage coordinates
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
coordinate_systems_3d = setup_coordinate_systems_3d(
    ref_point_mlapdv,
    optical_axis,  # now adjusted for the optical axis
    rotate_by=scanner_orientation["rotation"],
    invert_dims=scanner_orientation["invert_axis"],
)
# ref_img_center_um = get_image_corners(ref_img_size_px, coordinate_systems_ref)["center"]
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
coordinate_systems_2d = create_coordinate_systems_from_scanimage_meta(
    raw_imaging_meta["rawScanImageMeta"],
    fov_uuids=fov_uuids,
)

coords = projections.project_scanimage_fovs(
    coords_px,  # the pixel coordinates as loaded from suite2p
    coordinate_systems_2d,
    coordinate_systems_3d,
    atlas=atlas,
    projection_vector=optical_axis,  # now project along the optical axis
    ds=10,
)

# extract depths
fov_uuids = sorted(list(fov_map.values()))
fov_depths = extract_fov_depths_from_scanimage_meta(
    scanimage_meta=raw_imaging_meta["rawScanImageMeta"],
    scanimage_params=raw_imaging_meta["scanImageParams"],
    fov_uuids=fov_uuids,
)

# this creates: the keys 'um_corrected' and 'dv_below_surface'
# the use of um_corrected requires a new projection
for uuid in list(coords.keys()):
    coords_on_surface = projections.project_coords_onto_atlas_surface(
        coords_um=coords[uuid]["um_corrected"],
        coordinate_systems_3d=coordinate_systems_3d,
        atlas=atlas,
        projection_vector=optical_axis,
    )
    coords_reprojected = projections.project_down_from_surface(
        coords_on_surface=coords_on_surface,
        atlas=atlas,
        coords_depths=coords[uuid]["dv_below_surface"],
    )
    coords[uuid]["reprojected"] = coords_reprojected  # this is mlapdv

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
fig, axes = plt.subplots()
fov_uuids = sorted(list(coords.keys()))
for name, uuid in fov_map.items():
    stat = np.load(stat_paths[name], allow_pickle=True)
    # _coords = np.stack([(np.average(s["xpix"]), np.average(s["ypix"])) for s in stat])
    _coords = coords[uuid]["pixel"]
    coords_um = coordinate_systems_2d[uuid].transform(_coords, "pixel", "um_global")
    # axes.plot(*coords_um.T, ".")
    axes.scatter(*coords_um.T, c=coords[uuid]["atlas_rgba"] / 255)

axes.set_aspect("equal")
kwargs = dict(linestyle=":", lw=1, alpha=1, color="k")
axes.axhline(0, **kwargs)
axes.axvline(0, **kwargs)
circle = plt.Circle((0, 0), 3000, fill=False, color="k")
axes.add_patch(circle)
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.invert_yaxis()  # because scanimage scanner coordinates follow image coordinate convention

# %% some 3d stuff
axes = plotters.plot_brain_surface_points(atlas.get_surface_points())
for name, uuid in fov_map.items():
    plotters.plot_points(
        coords[uuid]["on_surface"],
        axes=axes,
        s=2,
        color=coords[uuid]["atlas_rgba"] / 255,
    )
coordinate_systems_3d.plot(axes=axes, color_by="axis", scale=500)

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
if not SAVE_OUTPUT:
    for name, uuid in fov_map.items():
        session_folder = BASE_FOLDER / one.eid2path(eid).session_path_short()
        coords_mlapdv = coords[uuid]["reprojected"]
        # saving the updated coordinates
        np.save(
            session_folder
            / "alf"
            / name
            / "mpciROIs.mlapdv_v3.npy",  # FIXME better naming
            coords_mlapdv,
        )
        # saving the atlas ids
        atlas_ids = atlas.get_labels_for_mlapdv(coords_mlapdv)[0]
        np.save(
            session_folder
            / "alf"
            / name
            / "mpciROIs.brainLocationIds_ccf_2017_v3.npy",  # FIXME better naming
            atlas_ids,
        )
