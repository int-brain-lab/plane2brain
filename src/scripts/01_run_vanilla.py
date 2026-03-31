# %%
import numpy as np
from plane2brain import plotters, projections, scanimage, suite2p, ibl
from plane2brain.coordinate_systems import (
    setup_coordinate_systems_3d,
    get_image_corners,
)
from plane2brain.atlas import ProjectionAtlas
from one.api import ONE
import matplotlib.pyplot as plt

# %% whiterussian / local server base folder
# BASE_FOLDER = Path("/mnt/s0/Data/Subjects")
LOCATION = "local"
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

one = ONE()
# eid = one.ref2eid(dict(subject="SP058", date="2024-07-25", sequence="001"))
eid = one.ref2eid(dict(subject="SP058", date="2024-08-01", sequence="001"))

# load the reference image metadata
ref_img_meta = ibl.load_reference_stack_metadata(eid, one, location=LOCATION)
ref_point_mlap, ref_point_ref = ibl.load_reference_points_from_meta(
    ref_img_meta, use_resolved=True
)  # the craniotomy center, both in ml,ap (histology resolved) and in
# the reference space of scanimage (galvos)

# load the suite2p data
raw_imaging_meta, stat_paths, fov_map = ibl.load_fov_data(eid, one, location=LOCATION)
fov_names = sorted(list(fov_map.keys()))
coords_px = suite2p.data_loader(stat_paths, fov_map)  # refactor: rename coords_px

# this is defined
scanner_orientation = dict(rotation=0.0, invert_axis=[True, True, False])
dims = ["Y", "X"]

# this is the atlas to project onto
atlas = ProjectionAtlas(res_um=50)


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

# the 2d coordinate systems, by fov name
coordinate_systems_2d = scanimage.create_coordinate_systems_from_scanimage_meta(
    raw_imaging_meta["rawScanImageMeta"],
    fov_uuids=fov_uuids,
    dims=dims,
)

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

coords = projections.project_scanimage_fovs(
    coords_px,  # the pixel coordinates as loaded from suite2p
    coordinate_systems_2d,
    coordinate_systems_3d,
    atlas=atlas,
    projection_vector=brain_normal_at_ref,
    ds=50,
)

# %% projecting down from surface

# extract depths
fov_uuids = sorted(list(fov_map.values()))
fov_depths = scanimage.extract_fov_depths_from_scanimage_meta(
    scanimage_meta=raw_imaging_meta["rawScanImageMeta"],
    scanimage_params=raw_imaging_meta["scanImageParams"],
    fov_uuids=fov_uuids,
)

# project down
for fov_uuid in fov_uuids:
    _depths = np.ones(coords[fov_uuid]["pixel"].shape[0]) * fov_depths[fov_uuid]
    coords[fov_uuid]["mlapdv"] = projections.project_down_from_surface(
        coords[fov_uuid]["on_surface"],
        atlas=atlas,
        coords_depths=_depths,
    )


# %% map anything mlapdv to brain area
for name, uuid in fov_map.items():
    ids, ix, rgba, acronym = atlas.get_labels_for_mlapdv(coords[uuid]["mlapdv"])
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
edges = ["topleft", "topright", "bottomright", "bottomleft", "topleft"]

for uuid, coordinate_system in coordinate_systems_fovs.items():
    fov_meta = scanimage.get_fov_meta(raw_imaging_meta["rawScanImageMeta"], uuid)
    fov_size_px = scanimage.get_scanfield_size_px(fov_meta, dims=dims)

    corners = get_image_corners(fov_size_px, coordinate_system, to="um_global")
    # the corners are expressed in the um global space and need to be
    # transformed into the mlapdv space first
    _corners = np.array([np.append(corners[e], 0) for e in edges])
    _corners = coordinate_systems_3d.transform(_corners, "imaging_plane", "mlapdv")
    axes.plot(*_corners.T, lw=1, color="k", zorder=100)

for name, uuid in fov_map.items():
    axes.scatter(
        *coords[uuid]["on_surface"].T,
        c=coords[uuid]["atlas_rgba"] / 255,
        s=5,
        zorder=20,
    )
    axes.scatter(*coords[uuid]["mlapdv"].T, c="k", s=5)

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
        coords_mlapdv = coords[uuid]["mlapdv"]
        # saving the updated coordinates
        np.save(
            session_folder / "alf" / name / "mpciROIs.mlapdv_vanilla_projection.npy",
            coords_mlapdv,
        )
        # saving the atlas ids
        atlas_ids = atlas.get_labels_for_mlapdv(coords_mlapdv)[0]
        np.save(
            session_folder
            / "alf"
            / name
            / "mpciROIs.brainLocationIds_ccf_2017_vanilla_projection.npy",
            atlas_ids,
        )

# %%
