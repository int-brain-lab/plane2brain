# %%
import sys
import numpy as np
from plane2brain import plotters, projections, scanimage, suite2p, ibl
from plane2brain.coordinate_systems import (
    create_coordinate_system_for_image,
)

from plane2brain.atlas import ProjectionAtlas
from one.api import ONE
import matplotlib.pyplot as plt

from ibllib.mpci.registration import register_reference_stacks
from ibllib.mpci.tasks import MesoscopeFOVHistology
from iblatlas.atlas import MRITorontoAtlas

import skimage
from pathlib import Path

# %% whiterussian / local server base folder
BASE_FOLDER = Path("/mnt/s0/Data/Subjects")

LOCATION = "server"
SAVE_OUTPUT = True
PLOT = False

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
if len(sys.argv) == 1:
    # NOTE this currently fails in vscode interactive mode
    # eid = one.ref2eid(dict(subject="SP058", date="2024-07-25", sequence="001"))
    eid = one.ref2eid(dict(subject="SP058", date="2024-08-01", sequence="001"))
    session_path = ibl._eid2path(eid, one, location=LOCATION)
else:
    session_path = Path(sys.argv[1])
    eid = one.path2eid(session_path)

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
atlas = ProjectionAtlas(res_um=25)


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
# but: in this case we have histology, so we infer the offset from there

# %%
"""
 
  ######  ##     ## #### ######## ######## 
 ##    ## ##     ##  ##  ##          ##    
 ##       ##     ##  ##  ##          ##    
  ######  #########  ##  ######      ##    
       ## ##     ##  ##  ##          ##    
 ##    ## ##     ##  ##  ##          ##    
  ######  ##     ## #### ##          ##    
 
"""

# from the transform we computed between the reference stacks (of the session at hand
# and the reference session)
# we take the translation component to integrate it here by shifting the ref point

# reference session for SP058: "SP058/2024-08-14/001"
eid_ref = one.ref2eid(dict(subject="SP058", date="2024-08-14", sequence="001"))

# get the path to the reference stack
ref_stack_path = ibl.get_reference_stack_path(
    eid,
    one,
    location=LOCATION,
    raw_imaging_collection=ibl.infer_imaging_collection(eid, one, location=LOCATION),
)

# correspondingly, to the reference stack of the reference session
ref_sess_ref_stack_path = ibl.get_reference_stack_path(
    eid_ref,
    one,
    location=LOCATION,
    raw_imaging_collection=ibl.infer_imaging_collection(
        eid_ref, one, location=LOCATION
    ),
)

# %% only for whiterussian use

# session_path = BASE_FOLDER / one.eid2path(eid).session_path_short()
reference_session_path = BASE_FOLDER / one.eid2path(eid_ref).session_path_short()

meso_task = MesoscopeFOVHistology(
    session_path=session_path, reference_session=reference_session_path, one=one
)
meso_task.setUp()

# meso_task.load_reference_stack()
ccf_idx = np.load(meso_task._get_atlas_registered_reference_mlap())


ba = MRITorontoAtlas(res_um=25)
ccf_idx[:, :, 1] = np.abs(ccf_idx[:, :, 1].astype("int64") - ba.label.shape[0]).astype(
    ccf_idx.dtype
)
# to be very explicit about: this is for the ref_img of the session that is aligned to the histo
ref_img_histo_mlapdv = (
    ba.ccf2xyz(ccf_idx * ba.res_um, ccf_order="mlapdv") * 1e6
)  # m -> μm

# %% the transform between this session and the ref stack of the histo session
# empirically determined that this way is the correct direction:
_, transform_params = register_reference_stacks(
    ref_sess_ref_stack_path,
    ref_stack_path,
    display=True,
    save_path=session_path / "alf" / "_gr_reference_stack_registration.gif",
)

# the transform between the reference stack and the "reference reference" stack
# = the reference stack of the reference session
ref_transform = skimage.transform.EuclideanTransform(
    rotation=transform_params["rotation"],
) + skimage.transform.EuclideanTransform(
    translation=transform_params["translation"],
)

# warp_matrix = np.concatenate([transform_params["warp_matrix"], [[0, 0, 1]]], axis=0)
# ref_transform = skimage.transform.EuclideanTransform(matrix=warp_matrix)

# the translation part can be easily used to shift ROIs in um_global space
# this is never used in this pipeline as well
# session_shift_um = transform_params["translation"] * um_per_px

# %% setting up the coordinate systems for the imaged fovs
fov_uuids = sorted(list(fov_map.values()))
coordinate_systems_2d = scanimage.create_coordinate_systems_from_scanimage_meta(
    raw_imaging_meta["rawScanImageMeta"],
    fov_uuids=fov_uuids,
    dims=dims,
)

ref_img_topleft_ref, ref_img_ref_per_px = ibl.infer_ref_stack_virtual_corner(
    ref_img_meta["rawScanImageMeta"],
    ref_img_size_px,
    dims=dims,
)

# the uncorrected version: 2d coordinate system in of the reference image
coordinate_systems_ref = create_coordinate_system_for_image(
    ref_img_size_px,
    um_per_px,
    ref_img_ref_per_px,
    ref_img_topleft_ref,
)

#### ##    ## #### ########
##  ###   ##  ##     ##
##  ####  ##  ##     ##
##  ## ## ##  ##     ##
##  ##  ####  ##     ##
##  ##   ###  ##     ##
#### ##    ## ####    ##

# %% setting up coords dict
coords = {}
fov_uuids = sorted(list(coords_px.keys()))
for fov_uuid in fov_uuids:
    coords[fov_uuid] = {}
    # get the pixel data
    _coords_px = coords_px[fov_uuid]
    # project into global um space
    _coords_um = coordinate_systems_2d[fov_uuid].transform(
        _coords_px,
        "pixel",
        "um_global",
    )
    coords[fov_uuid]["pixel"] = _coords_px
    coords[fov_uuid]["um_global"] = _coords_um

# extract depths
fov_uuids = sorted(list(fov_map.values()))
fov_depths = scanimage.extract_fov_depths_from_scanimage_meta(
    scanimage_meta=raw_imaging_meta["rawScanImageMeta"],
    scanimage_params=raw_imaging_meta["scanImageParams"],
    fov_uuids=fov_uuids,
)

# get the depth below brain surface by averaging the dv
# reference points on the brain surface
brain_surface_points = ibl.load_brain_surface_points(eid, one, location=LOCATION)

# this normal is expressed in the coordinate system of the reference stack
p_surface, n_surface, dv_avg = projections.get_brain_surface_normal(
    brain_surface_points,
    ref_img_meta,
    coordinate_systems_ref,
)
# the untilted plane
for uuid in fov_uuids:
    n = coords[uuid]["pixel"].shape[0]
    coords[uuid]["dv_below_surface"] = np.ones(n) * np.absolute(
        fov_depths[uuid] - dv_avg
    )

# %%
##     ## ####  ######  ########  #######     ##        #######   #######  ##    ## ##     ## ########
##     ##  ##  ##    ##    ##    ##     ##    ##       ##     ## ##     ## ##   ##  ##     ## ##     ##
##     ##  ##  ##          ##    ##     ##    ##       ##     ## ##     ## ##  ##   ##     ## ##     ##
#########  ##   ######     ##    ##     ##    ##       ##     ## ##     ## #####    ##     ## ########
##     ##  ##        ##    ##    ##     ##    ##       ##     ## ##     ## ##  ##   ##     ## ##
##     ##  ##  ##    ##    ##    ##     ##    ##       ##     ## ##     ## ##   ##  ##     ## ##
##     ## ####  ######     ##     #######     ########  #######   #######  ##    ##  #######  ##

# helper function for linear interpolation
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

grid = ref_img_histo_mlapdv[:, :, :-1]

xs = np.arange(grid.shape[0])
ys = np.arange(grid.shape[1])

interp = RegularGridInterpolator(
    (xs, ys),
    grid,
    method="linear",
    bounds_error=False,
    fill_value=np.nan,
)

sigma = 1
smoothed_grid = gaussian_filter(grid.astype(float), sigma=(sigma, sigma, 0))

interp_smooth = RegularGridInterpolator(
    (xs, ys),
    smoothed_grid,
    method="linear",
    bounds_error=False,
    fill_value=np.nan,
)

# %% first: just indexing
for fov_name, uuid in fov_map.items():
    # global px
    px = coords_px[uuid]
    coords_um_global = coordinate_systems_2d[uuid].transform(px, "pixel", "um_global")
    px = coordinate_systems_ref.transform(coords_um_global, "um_global", "pixel")

    # straight indexing
    px = px.astype("int")
    px[:, 0] = np.clip(px[:, 0], 0, grid.shape[0] - 1)
    px[:, 1] = np.clip(px[:, 1], 0, grid.shape[1] - 1)
    coords[uuid]["indexing"] = ref_img_histo_mlapdv[px[:, 0], px[:, 1], :]


# %% second: using interpolation
for fov_name, uuid in fov_map.items():
    # global px
    px = coords_px[uuid]
    coords_um_global = coordinate_systems_2d[uuid].transform(px, "pixel", "um_global")
    px = coordinate_systems_ref.transform(coords_um_global, "um_global", "pixel")

    # interpolation
    coords_mlap = interp(px)

    # projecting inward
    coords[uuid]["on_surface_interp"] = atlas.get_dv_for_mlap(
        coords_mlap + 1e-6  # DOCME
    )
    coords[uuid]["interp"] = projections.project_down_from_surface(
        coords_on_surface=coords[uuid]["on_surface_interp"],
        atlas=atlas,
        coords_depths=coords[uuid]["dv_below_surface"],
    )

# %% third: with session to session shift, no smoothing
for fov_name, uuid in fov_map.items():
    # global pixel
    px = coords_px[uuid]
    coords_um_global = coordinate_systems_2d[uuid].transform(px, "pixel", "um_global")
    px = coordinate_systems_ref.transform(coords_um_global, "um_global", "pixel")

    # session 2 session correction
    px = ref_transform(px)

    # histology lookup
    coords_mlap = interp(px)

    # projecting inward
    coords[uuid]["on_surface_interp_s2s"] = atlas.get_dv_for_mlap(coords_mlap + 1e-6)
    coords[uuid]["interp_s2s"] = projections.project_down_from_surface(
        coords_on_surface=coords[uuid]["on_surface_interp_s2s"],
        atlas=atlas,
        coords_depths=coords[uuid]["dv_below_surface"],
    )

# %% next: same as before, but with smoothed grid for interpolation

for fov_name, uuid in fov_map.items():
    # global pixel
    px = coords_px[uuid]
    coords_um_global = coordinate_systems_2d[uuid].transform(px, "pixel", "um_global")
    px = coordinate_systems_ref.transform(coords_um_global, "um_global", "pixel")

    # session 2 session correction
    px = ref_transform(px)

    # histology lookup
    mlap_interp = interp_smooth(px)

    # projecting inward
    coords[uuid]["on_surface_interp_smooth_s2s"] = atlas.get_dv_for_mlap(mlap_interp)
    coords[uuid]["interp_smooth_s2s"] = projections.project_down_from_surface(
        coords_on_surface=coords[uuid]["on_surface_interp_smooth_s2s"],
        atlas=atlas,
        coords_depths=coords[uuid]["dv_below_surface"],
    )

# %% next: include apparent xy shift
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
# just getting the DV for projecting down

# %% adjusting for the fact that this is not the case: getting the optical axis
# load the brain surface points and get the normal
brain_surface_points = ibl.load_brain_surface_points(eid, one, location=LOCATION)

# this normal is expressed in the coordinate system of the reference stack
p_surface, n_surface, dv_avg = projections.get_brain_surface_normal(
    brain_surface_points,
    ref_img_meta,
    coordinate_systems_ref,
)

# extract depths
fov_uuids = sorted(list(fov_map.values()))
fov_depths = scanimage.extract_fov_depths_from_scanimage_meta(
    scanimage_meta=raw_imaging_meta["rawScanImageMeta"],
    scanimage_params=raw_imaging_meta["scanImageParams"],
    fov_uuids=fov_uuids,
)
# this creates: the keys
# 'um_corrected' - for apparent xy shift based on tilt
# 'dv_below_surface_corrected'  - for apparent z shift based on tilt
coords = projections.correct_coords_for_tilt_2d(
    coords,
    fov_depths,
    p_surface,
    n_surface,
)

for fov_name, uuid in fov_map.items():
    # use the um_corrected to transform back to px
    px = coordinate_systems_ref.transform(
        coords[uuid]["um_corrected"], "um_global", "pixel"
    )
    # apply session to session correction
    px = ref_transform(px)

    # histo lookup
    mlap_interp = interp_smooth(px)

    # find point on surface
    coords[uuid]["on_surface_interp_smooth_s2s_apxy"] = atlas.get_dv_for_mlap(
        mlap_interp
    )
    # project down uncorrected amount
    coords[uuid]["interp_smooth_s2s_apxy"] = projections.project_down_from_surface(
        coords_on_surface=coords[uuid]["on_surface_interp_smooth_s2s_apxy"],
        atlas=atlas,
        coords_depths=coords[uuid]["dv_below_surface"],
    )
    # project down CORRECTED amount
    coords[uuid]["interp_smooth_s2s_apxyz"] = projections.project_down_from_surface(
        coords_on_surface=coords[uuid]["on_surface_interp_smooth_s2s_apxy"],
        atlas=atlas,
        coords_depths=coords[uuid]["dv_below_surface_corrected"],
    )


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
save_keys = [
    "indexing",
    "on_surface_interp",
    "interp",
    "on_surface_interp_s2s",
    "interp_s2s",
    "on_surface_interp_smooth_s2s",
    "interp_smooth_s2s",
    "on_surface_interp_smooth_s2s_apxy",
    "interp_smooth_s2s_apxy",
    "interp_smooth_s2s_apxyz",
]


if SAVE_OUTPUT:
    for name, uuid in fov_map.items():
        session_folder = ibl._eid2path(eid, one, location=LOCATION)
        # coords_mlapdv = coords[uuid]["reprojected_histo"]
        for key in save_keys:
            np.save(
                session_folder / "alf" / name / f"mpciROIs.mlapdv_repro_{key}.npy",
                coords[uuid][key],
            )

# %%
