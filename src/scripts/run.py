# %%
from typing import List
from pathlib import Path
import pickle
from itertools import product

import numpy as np

from plane2brain import plotters, projections

from plane2brain.brain_meshes import (
    get_labels_for_mlapdv,
    get_surface_points,
    calculate_surface_triangulation,
    get_plane_at_point_mlap,
)
from plane2brain.scanimage import get_resolution_from_scanimage_meta, extract_fov_depths_from_scanimage_meta
from plane2brain.coordinate_systems import create_coordinate_system_for_ref, get_image_corners, setup_coordinate_systems_3d
from one.api import ONE

from iblatlas.atlas import MRITorontoAtlas

import plane2brain.ibl as p2bi
import matplotlib.pyplot as plt
from plane2brain.suite2p import suite2p_data_loader


BASE_FOLDER = Path("/mnt/s0/Data/Subjects")

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
eid = one.ref2eid(dict(subject="SP058", date="2024-08-01", sequence="001"))
raw_imaging_collection = p2bi.infer_imaging_collection(eid, one)
ref_img_meta = p2bi.ibl_load_reference_stack_metadata(eid, one)
# TODO refactor me as this could be confusing
# or later down reference_points == surface_points
ref_point_mlap, ref_point_ref = p2bi.get_reference_points_from_meta(ref_img_meta, use_resolved=False)  # the craniotomy center
raw_imaging_meta, stat_paths, fov_map = p2bi.ibl_load_fov_data(eid, raw_imaging_collection, one)
fov_names = sorted(list(fov_map.keys()))
coords_px = suite2p_data_loader(stat_paths, fov_map)

# rename coords_px
# this is unfortunately defined
scanner_orientation = dict(rotation=3 / 2 * np.pi, invert_axis=[True, False, False])


# %%
"""
 
 ########  ########   #######        ## ########  ######  ######## #### ##    ##  ######   
 ##     ## ##     ## ##     ##       ## ##       ##    ##    ##     ##  ###   ## ##    ##  
 ##     ## ##     ## ##     ##       ## ##       ##          ##     ##  ####  ## ##        
 ########  ########  ##     ##       ## ######   ##          ##     ##  ## ## ## ##   #### 
 ##        ##   ##   ##     ## ##    ## ##       ##          ##     ##  ##  #### ##    ##  
 ##        ##    ##  ##     ## ##    ## ##       ##    ##    ##     ##  ##   ### ##    ##  
 ##        ##     ##  #######   ######  ########  ######     ##    #### ##    ##  ######   
 
"""

# NOTE that coords[uuid] is holding arrays of Nx? (Nx2 for 2d coords, Nx3 for 3d coords, Nx1 or Nx4 for others)

coords, coordinate_systems, coordinate_systems_3d = projections.project_from_scanimage_meta(
    coords_px,
    scanimage_meta=raw_imaging_meta["rawScanImageMeta"],
    scanner_orientation=scanner_orientation,
    craniotomy_center_mlap=ref_point_mlap,
    ds=100,  # TODO remove this debug flag
)

# %% map anything mlapdv to brain area
atlas = MRITorontoAtlas(50)

for name, uuid in fov_map.items():
    ids, ix, rgba, acronym = get_labels_for_mlapdv(coords[uuid]["on_surface"], atlas)
    coords[uuid]["atlas_rgba"] = rgba
    coords[uuid]["atlas_acronym"] = acronym
    coords[uuid]["atlas_id"] = ids

# %% some diagnostic plotting
fig, axes = plt.subplots()
fov_uuids = sorted(list(coords.keys()))
for name, uuid in fov_map.items():
    stat = np.load(stat_paths[name], allow_pickle=True)
    # _coords = np.stack([(np.average(s["xpix"]), np.average(s["ypix"])) for s in stat])
    _coords = coords[uuid]["pixel"]
    coords_um = coordinate_systems[uuid].transform(_coords, "pixel", "um_global")
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
atlas.compute_surface()
brain_surface_points = get_surface_points(atlas)
axes = plotters.plot_brain_surface_points(brain_surface_points)
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
 
 ########  ######## ########    #### ##     ##  ######   
 ##     ## ##       ##           ##  ###   ### ##    ##  
 ##     ## ##       ##           ##  #### #### ##        
 ########  ######   ######       ##  ## ### ## ##   #### 
 ##   ##   ##       ##           ##  ##     ## ##    ##  
 ##    ##  ##       ##           ##  ##     ## ##    ##  
 ##     ## ######## ##          #### ##     ##  ######   
 
"""

ref_img_stack = p2bi.ibl_load_reference_stack(eid, one)
ref_img_meta = p2bi.ibl_load_reference_stack_metadata(eid, one)
ref_img_size_px = np.array(ref_img_stack[0].shape)

um_per_px = get_resolution_from_scanimage_meta(ref_img_meta["rawScanImageMeta"])  # in X,Y
ref_img_size_um = ref_img_size_px * um_per_px

ref_point_mlap, ref_point_ref = p2bi.get_reference_points_from_meta(ref_img_meta, use_resolved=False)  # the craniotomy center

# MAJOR TODO - deal with the entire situation of the reference image and it's axis
# we do plus here instead of minus because of the image axis inversion
ref_img_topleft_um = ref_point_mlap + np.array([1, 1]) * ref_img_size_um / 2

coordinate_systems_ref = create_coordinate_system_for_ref(
    ref_img_size_px,  # (ml, ap)
    um_per_px * -1,  # :(
    ref_img_topleft_um,
)
ref_img_center_um = get_image_corners(ref_img_size_px, coordinate_systems_ref)["center"]

# %% # and creating the coordinate system
atlas = MRITorontoAtlas(50)
atlas.compute_surface()
brain_surface_points = get_surface_points(atlas)
mesh = calculate_surface_triangulation(atlas)

ref_point_mlapdv, brain_normal_at_ref = get_plane_at_point_mlap(*ref_point_mlap, mesh, numba=True)
# FIXME and here we don't rotate ...
coordinate_systems_3d = setup_coordinate_systems_3d(ref_point_mlapdv, brain_normal_at_ref, rotate_by=None)


# %% the coordinates of the individual pixels
pixels = np.array(list(product(range(ref_img_size_px[0]), range(ref_img_size_px[1]))))
coords_um = coordinate_systems_ref.transform(pixels, "pixel", "um")

# %%
coords = projections.project_coords_onto_atlas_surface(
    coords_um - ref_img_center_um,  # here we apply the center offset
    coordinate_systems_3d,
    mesh,
    brain_normal_at_ref,
)


with open(Path(__file__).parent / "reference_image_coords.pkl", "wb") as fH:
    pickle.dump(coords, fH)

# %%

with open(Path(__file__).parent / "reference_image_coords.pkl", "rb") as fH:
    coords = pickle.load(fH)

# %% revert the product


def extent_from_corners(corners: dict) -> List:
    return [
        corners["topleft"][1],
        corners["topright"][1],
        corners["topleft"][0],
        corners["bottomleft"][0],
    ]


fig, axes = plt.subplots()
corners = get_image_corners(ref_img_size_px, coordinate_systems_ref)
extent = extent_from_corners(corners)
axes.matshow(ref_img_stack[5], extent=extent)
axes.set_aspect("equal")
axes.invert_yaxis

# %%
crs = coords["on_imaging_plane"].reshape((*ref_img_size_px, -1))


# %%
def atlas_surface_adjust(coords_on_surface: np.ndarray, atlas):
    # for rounding errors when trying to get the labels
    xi = atlas.bc.x2i(coords_on_surface[:, 0] / 1e6, mode="clip")
    yi = atlas.bc.y2i(coords_on_surface[:, 1] / 1e6, mode="clip")
    surface_dv = atlas.top[yi, xi] * 1e6
    return np.concatenate([coords_on_surface[:, :-1], surface_dv[:, np.newaxis]], 1)


# %% revert the product
# crs = coords["on_imaging_plane"].reshape((*ref_img_size_px, -1))
# _coords_adjusted = atlas_surface_adjust(coords["on_surface"], atlas)
# ids, ix, rgba, acronym = get_labels_for_mlapdv(_coords_adjusted, atlas)

# %% diagnostic plotting
# _coords_adjusted = atlas_surface_adjust(coords["on_surface"], atlas)
# ids, ix, rgba, acronym = get_labels_for_mlapdv(_coords_adjusted, atlas)
# axes = plotters.plot_brain_surface_points(brain_surface_points)
# rgba[:, 3] = 255
# plotters.plot_points(coords["on_surface"], axes=axes, color=rgba / 255, s=2)
# coordinate_systems_3d.plot(axes=axes, color_by="axis", scale=500)


# %%
reference_brain_surface_points = p2bi.ibl_load_brain_surface_points(eid, one)

p_surface, n_surface, dv_avg = projections.get_brain_surface_normal(
    reference_brain_surface_points,
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
coords, coordinate_systems, coordinate_systems_3d = projections.project_from_scanimage_meta(
    coords_px,  # this is what is read from suite2p
    scanimage_meta=raw_imaging_meta["rawScanImageMeta"],
    scanner_orientation=scanner_orientation,
    craniotomy_center_mlap=ref_point_mlap,
    ds=10,  # FIXME DEBUGING
)

# %%
coords = projections.correct_coords_for_tilt_2d(
    coords,
    coordinate_systems,
    fov_depths,
    p_surface,
    n_surface,
)

# %% this is the reprojection
coords = projections.reproject_coords(coords, coordinate_systems_3d, mesh, brain_normal_at_ref)

# %% diagnostic plots
axes = plotters.plot_brain_surface_points(brain_surface_points)
for name, uuid in fov_map.items():
    plotters.plot_points(
        coords[uuid]["reprojected"],
        axes=axes,
        s=2,
        color="k",
        # color=coords[uuid]["atlas_rgba"] / 255,
    )
coordinate_systems_3d.plot(axes=axes, color_by="axis", scale=500)

# %% some quantification of differences
for name, uuid in fov_map.items():
    _coords = coords[uuid]["pixel"]
    coords_um = coordinate_systems[uuid].transform(_coords, "pixel", "um_global")
    print(name, np.average(coords_um - coords[uuid]["um_corrected"], axis=0))
    print(name, np.average((dv_avg - fov_depths[uuid]) - coords[uuid]["dv_below_surface"]))

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
    atlas_ids = get_labels_for_mlapdv(coords_mlapdv, atlas)[0]
    np.save(
        session_folder / "alf" / name / "mpciROIs.brainLocationIds_ccf_2017_v2.npy",
        atlas_ids,
    )
