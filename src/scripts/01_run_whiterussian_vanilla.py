# %%
from pathlib import Path
import numpy as np

from plane2brain import plotters, projections
from plane2brain.atlas import ProjectionAtlas
from plane2brain.scanimage import (
    extract_fov_depths_from_scanimage_meta,
    create_coordinate_systems_from_scanimage_meta,
    get_fov_meta,
)
from plane2brain.coordinate_systems import (
    setup_coordinate_systems_3d,
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
# scanner_orientation = dict(rotation=0., invert_axis=[True, True, False])

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
coordinate_systems_2d = create_coordinate_systems_from_scanimage_meta(
    raw_imaging_meta["rawScanImageMeta"],
    fov_uuids=fov_uuids,
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
    ds=10,
)

# %% projecting down from surface

# extract depths
fov_uuids = sorted(list(fov_map.values()))
fov_depths = extract_fov_depths_from_scanimage_meta(
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
    ids, ix, rgba, acronym = atlas.get_labels_for_mlapdv(coords[uuid]["on_surface"])
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
    stat = np.load(stat_paths[name], allow_pickle=True)
    # _coords = np.stack([(np.average(s["xpix"]), np.average(s["ypix"])) for s in stat])
    _coords = coords[uuid]["pixel"]
    coords_um = coordinate_systems_2d[uuid].transform(_coords, "pixel", "um_global")
    # axes.plot(*coords_um.T, ".")
    axes.scatter(*coords_um.T, c=coords[uuid]["atlas_rgba"] / 255)
    img_size_px = (512, 512)  # TODO infer me
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

# in this 3d space, plot the outlines of the individual FOVs
from plane2brain.coordinate_systems import get_image_corners

for fov, uuid in fov_map.items():
    img_size_px = get_fov_meta(raw_imaging_meta["rawScanImageMeta"], uuid)[
        "scanfields"
    ]["pixelResolutionXY"]
    corners = get_image_corners(img_size_px, coordinate_systems_2d[uuid], to="um_global")
    edges = ["topleft", "topright", "bottomright", "bottomleft", "topleft"]
    _corners = np.array([np.append(corners[e], 0) for e in edges])
    _corners = coordinate_systems_3d.transform(_corners, "imaging_plane", "mlapdv")
    axes.plot(*_corners.T, lw=1, color="k")

axes.set_aspect("equal")

# plot stevens result on top
if 0:
    for fov_name, uuid in fov_map.items():
        mlapdv = ibl.ibl_load_roi_mlapdv(eid, one, fov_name, location=LOCATION)
        plotters.plot_points(mlapdv, axes=axes,s=2,color='g')

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
