# %%
import argparse
from itertools import product
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from iblatlas.atlas import MRITorontoAtlas
from ibllib.mpci.tasks import MesoscopeFOVHistology
from one.api import ONE

from plane2brain import ibl, projections, scanimage, suite2p
from plane2brain.atlas import ProjectionAtlas
from plane2brain.coordinate_systems import (
    create_coordinate_system_for_image,
)

# %% whiterussian / local server base folder
# BASE_FOLDER: Path = Path("/mnt/s0/Data/Subjects")
COORDS: Literal["rois", "px"] = "px"
LOCATION: Literal["server", "local", "popeye"] = "popeye"
SAVE_OUTPUT: bool = True
PLOT: bool = False
DEBUG: bool = True
ATLAS_RES: int = 25

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

if LOCATION == "popeye":
    from deploy.iblsdsc import OneSdsc

    # requires the unmerged PR #121 https://github.com/int-brain-lab/iblscripts/pull/121
    one = OneSdsc(location="popeye")
else:
    one = ONE()


parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("--session_path", type=Path)
group.add_argument("--eid")
args, _ = parser.parse_known_args()

if args.session_path is None and args.eid is None:
    # Neither provided: use both defaults
    match LOCATION:
        case "server":
            session_path = Path("/mnt/s0/Data/Subjects/SP058/2024-08-01/001")
            eid = one.path2eid(session_path)
        case "popeye":
            session_path = Path(
                "/mnt/sdceph/users/ibl/data/cortexlab/Subjects/SP058/2024-07-24/001"  # the direct comparison one for the regression
                # "/mnt/home/graiser/data/cortexlab/Subjects/SP058/2024-08-01/001"  # the original test case
                # "/mnt/sdceph/users/ibl/data/cortexlab/Subjects/SP058/2024-07-04/001"  # no ref stack
                # "/mnt/sdceph/users/ibl/data/cortexlab/Subjects/SP058/2024-06-28/001"  # ref stack wrong size but correct animal
                # "mnt/sdceph/users/ibl/data/cortexlab/Subjects/SP072/2025-08-26/001"  # ref stack of wrong size
            )
            eid = one.path2eid(session_path)
        case "local":
            eid = one.path2eid("SP058/2024-08-01/001")
            session_path = one.eid2path(eid)

elif args.session_path is not None:
    # if session path is provided, infer eid
    session_path = args.session_path
    eid = one.path2eid(session_path)
else:
    # or if eid is provided, infer session path
    eid = args.eid
    session_path = ibl._eid2path(eid, one, location=LOCATION)

print(f"{eid}:{session_path}")
# this is defined
scanner_orientation = {"rotation": 0.0, "invert_axis": [True, True, False]}
dims = ("Y", "X")

# load the reference image metadata
ref_img_meta = ibl.load_reference_stack_metadata(eid, one, location=LOCATION)
ref_point = ibl.load_reference_points_from_meta(
    ref_img_meta
)  # the craniotomy center, both in ml,ap (histology resolved) and in

# load the suite2p data
# load the suite2p data
raw_imaging_meta, stat_paths, fov_map = ibl.load_fov_data(
    eid,
    one,
    location=LOCATION,
    scratch_dir=Path.home() / "ibl_scratch" / str(eid)
    if LOCATION == "popeye"
    else None,
)

fov_names = sorted(fov_map.keys())

if COORDS == "rois":
    coords_px = suite2p.data_loader(
        stat_paths, fov_map, dims=dims
    )  # refactor: rename coords_px
elif COORDS == "px":
    coords_px = {}
    # cannot assert Width == Height because of the format of the FOVs being stitched
    # together vertically (mesoscope specific)
    n = raw_imaging_meta["rawScanImageMeta"]["Width"]
    for name, uuid in fov_map.items():
        coords_px[uuid] = np.array(list(product(range(n), repeat=2)), dtype="float")
        if DEBUG:
            coords_px[uuid] = coords_px[uuid][::100]

# this is the atlas to project onto
atlas = ProjectionAtlas(res_um=ATLAS_RES)


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

# reference session for SP058: "SP058/2024-08-14/001"
eid_ref = one.ref2eid({"subject": "SP058", "date": "2024-08-14", "sequence": "001"})

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

########  ######## ########     ######  ########  ######   ######     ########  ######## ########     ######  ########    ###     ######  ##    ##
##     ## ##       ##          ##    ## ##       ##    ## ##    ##    ##     ## ##       ##          ##    ##    ##      ## ##   ##    ## ##   ##
##     ## ##       ##          ##       ##       ##       ##          ##     ## ##       ##          ##          ##     ##   ##  ##       ##  ##
########  ######   ######       ######  ######    ######   ######     ########  ######   ######       ######     ##    ##     ## ##       #####
##   ##   ##       ##                ## ##             ##       ##    ##   ##   ##       ##                ##    ##    ######### ##       ##  ##
##    ##  ##       ##          ##    ## ##       ##    ## ##    ##    ##    ##  ##       ##          ##    ##    ##    ##     ## ##    ## ##   ##
##     ## ######## ##           ######  ########  ######   ######     ##     ## ######## ##           ######     ##    ##     ##  ######  ##    ##

# session_path = BASE_FOLDER / one.eid2path(eid).session_path_short()
reference_session_path = ibl._eid2path(eid_ref, one, location=LOCATION)
ba = MRITorontoAtlas(res_um=25)

match LOCATION:
    case "server":
        meso_task = MesoscopeFOVHistology(
            session_path=session_path, reference_session=reference_session_path, one=one
        )
        meso_task.setUp()

        # meso_task.load_reference_stack()
        ccf_idx = np.load(meso_task._get_atlas_registered_reference_mlap())
    case "local":
        raise NotImplementedError
    case "popeye":
        # the original path!
        # has local access to the histo
        # histology_path = (
        #     Path("/mnt/sdceph/users/ibl/data/histology/")
        #     / "cortexlab"
        #     / one.eid2path(eid_ref).session_path_short()
        #     / "referenceImage.mlapdv.npy"
        # )

        # test variants
        histology_path = (
            Path.home()
            / "repro"
            / "stevens_mlapdv_variants"
            # / "referenceImage.mlapdv.old_whiterussian.npy"
            # / "referenceImage.mlapdv.old_flatiron.npy"
            # / "referenceImage.mlapdv.new_run_1.npy"
            / "referenceImage.mlapdv.new_run_2.npy"
        )
        ccf_idx = np.load(histology_path)

        ccf_idx[:, :, 1] = np.abs(
            ccf_idx[:, :, 1].astype("int64") - ba.label.shape[0]
        ).astype(ccf_idx.dtype)


# to be very explicit about: this is for the ref_img of the session that is aligned to the histo
ref_img_histo_mlapdv = (
    ba.ccf2xyz(ccf_idx * ba.res_um, ccf_order="mlapdv") * 1e6
)  # m -> μm


########  ########  ######
##     ## ##       ##    ##
##     ## ##       ##
########  ######   ##   ####
##   ##   ##       ##    ##
##    ##  ##       ##    ##
##     ## ########  ######


# %% reimplementation of stack image registration
import tifffile
from registration import (
    apply_transform,
    evaluate,
    inspect_registration_delta,
    plot_keypoints,
    register_stacks,
)

# load the reference stack data from session and reference session
img_data = {}
for key, path in zip(
    ["stack", "target_stack"],
    [ref_stack_path, ref_sess_ref_stack_path],
):
    # key here: flipping dimensions
    img_data[key] = np.swapaxes(tifffile.imread(path), 1, 2)
    # img_data[key] = preprocess_vasculature(img_data[key]).astype("int16")

# find and apply transform
ref_transform, reg_details = register_stacks(
    img_data["stack"],
    img_data["target_stack"],
    transform_type="euclidean",
    return_details=True,
)
# NOTE affine is overall actually worse, but better for single plane

img_data["aligned"] = apply_transform(img_data["stack"], ref_transform)

# evaluate transform
ncc_before = evaluate(img_data["stack"], img_data["target_stack"])
ncc_after = evaluate(img_data["aligned"], img_data["target_stack"])

params = {
    "translation": ref_transform.translation,
    "rotation": ref_transform.rotation,
    "quality_ncc": ncc_after.mean(),
    "warp_matrix": np.array(ref_transform),
    "method": "orb_robust",
}

# %% save to gif
if PLOT:
    if SAVE_OUTPUT:
        save_path = session_path / "alf" / "_gr_reference_stack_registration.gif"
    else:
        save_path = None

    z = 8
    anim = inspect_registration_delta(
        img_data["stack"],
        img_data["target_stack"],
        img_data["aligned"],
        z=z,
        save_path=save_path,
        frames_per_second=1,  # 1s per frame in the saved gif
    )

# %% plot keypoints vis
if PLOT:
    if SAVE_OUTPUT:
        save_path = session_path / "alf" / "_gr_registration_keypoints.png"
    else:
        save_path = None

    plot_keypoints(
        img_data,
        reg_details,
        z,
        save_path=save_path,
    )

# %% save transform to json
# if SAVE_OUTPUT:
#     params = params.copy()
#     save_path = session_path / "alf" / "_gr_registration_keypoints.json"
#     for k, v in params.items():
#         if isinstance(v, np.ndarray):
#             params[k] = v.tolist()
#         elif isinstance(v, (np.float32, np.float64)):
#             params[k] = float(v)
#         else:
#             params[k] = v

#     with open(save_path, "w") as fp:
#         json.dump(params, fp, indent=4)

# %% setting up the coordinate systems for the imaged fovs
fov_uuids = sorted(fov_map.values())
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
fov_uuids = sorted(coords_px.keys())
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
fov_uuids = sorted(fov_map.values())
if COORDS == "rois":
    fov_depths = scanimage.extract_fov_depths_from_scanimage_meta(
        scanimage_meta=raw_imaging_meta["rawScanImageMeta"],
        scanimage_params=raw_imaging_meta["scanImageParams"],
        fov_uuids=fov_uuids,
    )
if COORDS == "px":
    fov_depths = {uuid: 0 for uuid in fov_uuids}

# get the depth below brain surface by averaging the dv
# reference points on the brain surface
match LOCATION:
    case "popeye":
        # on popeye we can only use the ones stored in the metadata
        brain_surface_points = {"points": ref_img_meta["points"]}
    case _:
        brain_surface_points = ibl.load_brain_surface_points(
            eid, one, location=LOCATION
        )

# this normal is expressed in the coordinate system of the reference stack
p_surface, n_surface, dv_avg = projections.get_brain_surface_normal(
    brain_surface_points,
    ref_img_meta,
    coordinate_systems_ref,
)
# the untilted plane
if COORDS == "rois":
    for uuid in fov_uuids:
        n = coords[uuid]["pixel"].shape[0]
        coords[uuid]["dv_below_surface"] = np.ones(n) * np.absolute(
            fov_depths[uuid] - dv_avg
        )
if COORDS == "px":
    for uuid in fov_uuids:
        n = coords[uuid]["pixel"].shape[0]
        coords[uuid]["dv_below_surface"] = np.zeros(n)


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
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

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

sigma = 25
smoothed_grid = gaussian_filter(grid.astype(float), sigma=(sigma, sigma, 0))

interp_smooth = RegularGridInterpolator(
    (xs, ys),
    smoothed_grid,
    method="linear",
    bounds_error=False,
    fill_value=np.nan,
)

# %% plot interpolation
if PLOT:
    fig, axes = plt.subplots(ncols=2)
    kwargs = {
        "cmap": "viridis", "vmin": np.percentile(grid, 1), "vmax": np.percentile(grid, 99)
    }

    titles = ["ml", "ap"]
    for i in range(2):
        axes[i].matshow(grid[:, :, i], **kwargs)
        axes[i].set_title(titles[i])
        axes[i].axhline(120, color="k", lw=1, alpha=0.5)

    fig, axes = plt.subplots()
    axes.plot(grid[120, :, 0], label="original")
    axes.plot(smoothed_grid[120, :, 0], label="smoothed")
    axes.legend()
    import seaborn as sns

    sns.despine(fig)

# %%


# %% first: just indexing
for uuid in fov_map.values():
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
for uuid in fov_map.values():
    # global px
    px = coords_px[uuid]
    coords_um_global = coordinate_systems_2d[uuid].transform(px, "pixel", "um_global")
    px = coordinate_systems_ref.transform(coords_um_global, "um_global", "pixel")

    # interpolation
    coords_mlap = interp(px)

    if not DEBUG:
        coords[uuid]["on_surface_interp"] = atlas.get_dv_for_mlap(
            coords_mlap + 1e-6  # DOCME
        )
        # projecting inward
        coords[uuid]["interp"] = projections.project_down_from_surface(
            coords_on_surface=coords[uuid]["on_surface_interp"],
            atlas=atlas,
            coords_depths=coords[uuid]["dv_below_surface"],
        )
    else:
        # keep fake 3d for debugging
        coords[uuid]["on_surface_interp"] = np.concatenate(
            [coords_mlap, np.zeros((coords_mlap.shape[0], 1))], axis=1
        )

# %% next: same as before, but with smoothed grid for interpolation
for uuid in fov_map.values():
    # global pixel
    px = coords_px[uuid]
    coords_um_global = coordinate_systems_2d[uuid].transform(px, "pixel", "um_global")
    px = coordinate_systems_ref.transform(coords_um_global, "um_global", "pixel")

    # histology lookup
    mlap_interp = interp_smooth(px)

    if not DEBUG:
        coords[uuid]["on_surface_interp_smooth"] = atlas.get_dv_for_mlap(mlap_interp)

        # projecting inward
        coords[uuid]["interp_smooth"] = projections.project_down_from_surface(
            coords_on_surface=coords[uuid]["on_surface_interp_smooth"],
            atlas=atlas,
            coords_depths=coords[uuid]["dv_below_surface"],
        )
    else:
        coords[uuid]["on_surface_interp_smooth"] = np.concatenate(
            [mlap_interp, np.zeros((mlap_interp.shape[0], 1))], axis=1
        )


# %% third: with session to session shift, no smoothing
for uuid in fov_map.values():
    # global pixel
    px = coords_px[uuid]
    coords_um_global = coordinate_systems_2d[uuid].transform(px, "pixel", "um_global")
    px = coordinate_systems_ref.transform(coords_um_global, "um_global", "pixel")

    # session 2 session correction
    px = ref_transform(px)

    # histology lookup
    coords_mlap = interp_smooth(px)

    if not DEBUG:
        coords[uuid]["on_surface_interp_smooth_s2s"] = atlas.get_dv_for_mlap(
            coords_mlap + 1e-6,
        )
        # projecting inward
        coords[uuid]["interp_smooth_s2s"] = projections.project_down_from_surface(
            coords_on_surface=coords[uuid]["on_surface_interp_smooth_s2s"],
            atlas=atlas,
            coords_depths=coords[uuid]["dv_below_surface"],
        )
    else:
        # keep fake 3d for debugging
        coords[uuid]["on_surface_interp_smooth_s2s"] = np.concatenate(
            [coords_mlap, np.zeros((coords_mlap.shape[0], 1))], axis=1
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
match LOCATION:
    case "popeye":
        # on popeye we can only use the ones stored in the metadata
        brain_surface_points = {"points": ref_img_meta["points"]}
    case _:
        brain_surface_points = ibl.load_brain_surface_points(
            eid, one, location=LOCATION
        )

# this normal is expressed in the coordinate system of the reference stack
p_surface, n_surface, dv_avg = projections.get_brain_surface_normal(
    brain_surface_points,
    ref_img_meta,
    coordinate_systems_ref,
)

# extract depths
fov_uuids = sorted(fov_map.values())

if COORDS == "rois":
    fov_depths = scanimage.extract_fov_depths_from_scanimage_meta(
        scanimage_meta=raw_imaging_meta["rawScanImageMeta"],
        scanimage_params=raw_imaging_meta["scanImageParams"],
        fov_uuids=fov_uuids,
    )
if COORDS == "px":
    fov_depths = {uuid: 0 for uuid in fov_uuids}

# this creates: the keys
# 'um_corrected' - for apparent xy shift based on tilt
# 'dv_below_surface_corrected'  - for apparent z shift based on tilt
coords = projections.correct_coords_for_tilt_2d(
    coords,
    fov_depths,
    p_surface,
    n_surface,
)

if not DEBUG:
    for uuid in fov_map.values():
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
        if COORDS == "rois":
            # project down uncorrected amount
            coords[uuid]["interp_smooth_s2s_apxy"] = (
                projections.project_down_from_surface(
                    coords_on_surface=coords[uuid]["on_surface_interp_smooth_s2s_apxy"],
                    atlas=atlas,
                    coords_depths=coords[uuid]["dv_below_surface"],
                )
            )
            # project down CORRECTED amount
            coords[uuid]["interp_smooth_s2s_apxyz"] = (
                projections.project_down_from_surface(
                    coords_on_surface=coords[uuid]["on_surface_interp_smooth_s2s_apxy"],
                    atlas=atlas,
                    coords_depths=coords[uuid]["dv_below_surface_corrected"],
                )
            )

# %% plotting

# fov_uuids = sorted(list(fov_map.values()))
# coordinate_systems_2d = scanimage.create_coordinate_systems_from_scanimage_meta(
#     raw_imaging_meta["rawScanImageMeta"],
#     fov_uuids=fov_uuids,
#     dims=dims,
# )

# setting up coords dict
# coords = {}
# fov_uuids = sorted(list(coords_px.keys()))
# for fov_uuid in fov_uuids:
#     coords[fov_uuid] = {}
#     # get the pixel data
#     _coords_px = coords_px[fov_uuid]
#     # project into global um space
#     _coords_um = coordinate_systems_2d[fov_uuid].transform(
#         _coords_px,
#         "pixel",
#         "um_global",
#     )
#     coords[fov_uuid]["pixel"] = _coords_px
#     coords[fov_uuid]["um_global"] = _coords_um

print(coords[fov_uuid].keys())
key = "indexing"
fig, axes = plt.subplots()
for uuid in fov_map.values():
    axes.plot(coords[uuid][key][::10, 0], coords[uuid][key][::10, 1], ".")

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
verification_key = histology_path.name.split(".")[-2]
if COORDS == "rois":
    save_keys = [
        "indexing",
        "on_surface_interp",
        "interp",
        "on_surface_interp_smooth",
        "interp_smooth",
        "on_surface_interp_smooth_s2s",
        "interp_smooth_s2s",
        "on_surface_interp_smooth_s2s_apxy",
        "interp_smooth_s2s_apxy",
        "interp_smooth_s2s_apxyz",
    ]

    if DEBUG:
        save_keys = [
            # "indexing",
            "on_surface_interp",
            # "interp",
            "on_surface_interp_smooth",
            # "interp_smooth",
            "on_surface_interp_smooth_s2s",
            # "interp_smooth_s2s",
            # "on_surface_interp_smooth_s2s_apxy",
            # "interp_smooth_s2s_apxy",
            # "interp_smooth_s2s_apxyz",
        ]

    if SAVE_OUTPUT:
        for name, uuid in fov_map.items():
            match LOCATION:
                case "server" | "local":
                    session_folder = ibl._eid2path(eid, one, location=LOCATION)
                    for key in save_keys:
                        np.save(
                            session_folder
                            / "alf"
                            / name
                            / f"mpciROIs.mlapdv_repro_ransac_ro_{ATLAS_RES}_{key}_{verification_key}.npy",
                            coords[uuid][key],
                        )
                case "popeye":
                    output_folder = (
                        Path.home()
                        / "ibl_scratch"
                        / "repro"
                        / one.eid2path(eid).session_path_short()
                        / "alf"
                        / name
                    )
                    output_folder.mkdir(exist_ok=True, parents=True)
                    for key in save_keys:
                        np.save(
                            output_folder
                            / f"mpciROIs.mlapdv_repro_ransac_ro_{ATLAS_RES}_{key}_{verification_key}.npy",
                            coords[uuid][key],
                        )


if COORDS == "px":
    save_keys = [
        "indexing",
        "on_surface_interp",
        "on_surface_interp_smooth",
        "on_surface_interp_smooth_s2s",
        "on_surface_interp_smooth_s2s_apxy",
    ]

    if DEBUG:
        save_keys = [
            "indexing",
            "on_surface_interp",
            # "interp",
            "on_surface_interp_smooth",
            # "interp_smooth",
            "on_surface_interp_smooth_s2s",
            # "interp_smooth_s2s",
            # "on_surface_interp_smooth_s2s_apxy",
            # "interp_smooth_s2s_apxy",
            # "interp_smooth_s2s_apxyz",
        ]

    if SAVE_OUTPUT:
        for name, uuid in fov_map.items():
            match LOCATION:
                case "server" | "local":
                    session_folder = ibl._eid2path(eid, one, location=LOCATION)
                    for key in save_keys:
                        np.save(
                            session_folder
                            / "alf"
                            / name
                            / f"mpciMeanImage.mlapdv_repro_ransac_ro_{ATLAS_RES}_{key}_{verification_key}.npy",
                            coords[uuid][key],
                        )
                case "popeye":
                    output_folder = (
                        Path.home()
                        / "ibl_scratch"
                        / "repro"
                        / one.eid2path(eid).session_path_short()
                        / "alf"
                        / name
                    )
                    output_folder.mkdir(exist_ok=True, parents=True)
                    for key in save_keys:
                        np.save(
                            output_folder
                            / f"mpciMeanImage.mlapdv_repro_ransac_ro_{ATLAS_RES}_{key}_{verification_key}.npy",
                            coords[uuid][key],
                        )
