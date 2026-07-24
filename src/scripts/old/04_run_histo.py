# %%
import sys
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

from ibllib.mpci.registration import register_reference_stacks
import skimage

# %% whiterussian / local server base folder
# BASE_FOLDER = Path("/mnt/s0/Data/Subjects")

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
    session_path = sys.argv[1]
    eid = one.path2eid(session_path)

# load the reference image metadata
ref_img_meta = ibl.load_reference_stack_metadata(eid, one, location=LOCATION)
ref_point = ibl.load_reference_points_from_meta(
    ref_img_meta
)  # the craniotomy center, both in ml,ap (histology resolved) and in

# load the suite2p data
raw_imaging_meta, stat_paths, fov_map = ibl.load_fov_data(eid, one, location=LOCATION)
fov_names = sorted(list(fov_map.keys()))
coords_px = suite2p.data_loader(stat_paths, fov_map)  # refactor: rename coords_px

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

# %% doesn't work right now
# from ibllib.mpci.tasks import MesoscopeFOVHistology

# session_path = one.eid2path(eid)
# reference_session_path = one.eid2path(eid_ref)

# meso_task = MesoscopeFOVHistology(
#     session_path=session_path, reference_session=reference_session_path, one=ONE()
# )

# meso_task.load_reference_stack()

# %% load the data from a local source and use miles code to convert to mlap
if LOCATION == "local":
    ccf_idx = np.load("/home/georg/data_local/referenceImage.mlapdv.npy")
elif LOCATION == "server":
    ccf_idx = np.load(
        "/home/ibladmin/Documents/georg/data_local/referenceImage.mlapdv.npy"
    )

from iblatlas.atlas import MRITorontoAtlas

ba = MRITorontoAtlas(res_um=25)
ccf_idx[:, :, 1] = np.abs(ccf_idx[:, :, 1].astype("int64") - ba.label.shape[0]).astype(
    ccf_idx.dtype
)
# to be very explicit about: this is for the ref_img of the session that is aligned to the histo
ref_img_histo_mlapdv = (
    ba.ccf2xyz(ccf_idx * ba.res_um, ccf_order="mlapdv") * 1e6
)  # m -> μm

# %% the transform between this session and the ref stack of the histo session
_, transform_params = register_reference_stacks(ref_stack_path, ref_sess_ref_stack_path)

# the transform between the reference stack and the "reference reference" stack
# = the reference stack of the reference session
ref_transform = skimage.transform.EuclideanTransform(
    rotation=transform_params["rotation"],
) + skimage.transform.EuclideanTransform(
    translation=transform_params["translation"],
)
# the translation part can be easily used to shift ROIs in um_global space
# this is never used in this pipeline as well
session_shift_um = transform_params["translation"] * um_per_px

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
# correcting by: shifting the topleft corner by the session shift
session_shift_ref = transform_params["translation"] * ref_img_ref_per_px
# the inversion of dimensions is necessary
ref_img_topleft_ref_corr = ref_img_topleft_ref + session_shift_ref[::-1]

# the 2d coordinate system in of the reference image
coordinate_systems_ref_corr = create_coordinate_system_for_image(
    ref_img_size_px,
    um_per_px,
    ref_img_ref_per_px,
    ref_img_topleft_ref_corr,
)

# and rotating the basis
rotation = skimage.transform.EuclideanTransform(
    rotation=transform_params["rotation"],
)
rotation = np.array(rotation)
from plane2brain.affine import apply_transform

for name in coordinate_systems_ref_corr.coordinate_systems.keys():
    basis = coordinate_systems_ref_corr.coordinate_systems[name].basis
    coordinate_systems_ref_corr.coordinate_systems[name].basis = apply_transform(
        basis, rotation
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
    coordinate_systems_ref,  # WARNING to be recomputed later for using the corrected version
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
from scipy.ndimage import map_coordinates, gaussian_filter


def interp_xy(grid, ii, jj):
    ii = np.asarray(ii, dtype=float)
    jj = np.asarray(jj, dtype=float)
    coords = np.stack([ii.ravel(), jj.ravel()])  # shape (2, K)
    x = map_coordinates(grid[..., 0], coords, order=1, mode="nearest")
    y = map_coordinates(grid[..., 1], coords, order=1, mode="nearest")
    return np.stack([x, y], axis=-1).reshape(ii.shape + (2,))


# %% first: without any interpolating
grid = ref_img_histo_mlapdv[:, :, :-1]
for fov_name, uuid in fov_map.items():
    px = coords_px[uuid]
    coords_um_global = coordinate_systems_2d[uuid].transform(px, "pixel", "um_global")
    px = coordinate_systems_ref.transform(coords_um_global, "um_global", "pixel")
    coords[uuid]["mlap_histo"] = interp_xy(grid, *px.T)
    coords[uuid]["mlapdv_on_surface_histo"] = atlas.get_dv_for_mlap(
        coords[uuid]["mlap_histo"] + 1e-6
    )
    coords[uuid]["mlapdv_histo"] = projections.project_down_from_surface(
        coords_on_surface=coords[uuid]["mlapdv_on_surface_histo"],
        atlas=atlas,
        coords_depths=coords[uuid]["dv_below_surface"],
    )

# %% next: with session to session shift, no interpolation

# recompute dv below surf
p_surface, n_surface, dv_avg = projections.get_brain_surface_normal(
    brain_surface_points,
    ref_img_meta,
    coordinate_systems_ref_corr,
)
# the untilted plane
for uuid in fov_uuids:
    n = coords[uuid]["pixel"].shape[0]
    coords[uuid]["dv_below_surface"] = np.ones(n) * np.absolute(
        fov_depths[uuid] - dv_avg
    )
# does actually not change, unsurpringly

grid = ref_img_histo_mlapdv[:, :, :-1]
for fov_name, uuid in fov_map.items():
    px = coords_px[uuid]
    coords_um_global = coordinate_systems_2d[uuid].transform(px, "pixel", "um_global")
    px = coordinate_systems_ref_corr.transform(coords_um_global, "um_global", "pixel")
    coords[uuid]["mlap_histo_s2s_corr"] = interp_xy(grid, *px.T)
    coords[uuid]["mlapdv_on_surface_histo_s2s_corr"] = atlas.get_dv_for_mlap(
        coords[uuid]["mlap_histo_s2s_corr"] + 1e-6
    )
    coords[uuid]["mlapdv_histo_s2s_corr"] = projections.project_down_from_surface(
        coords_on_surface=coords[uuid]["mlapdv_on_surface_histo_s2s_corr"],
        atlas=atlas,
        coords_depths=coords[uuid]["dv_below_surface"],
    )

# %% next: same as before, but with interpolation 25
sigma = 25
smoothed = gaussian_filter(grid.astype(float), sigma=(sigma, sigma, 0))
for fov_name, uuid in fov_map.items():
    px = coords_px[uuid]
    coords_um_global = coordinate_systems_2d[uuid].transform(px, "pixel", "um_global")
    px = coordinate_systems_ref_corr.transform(coords_um_global, "um_global", "pixel")
    coords[uuid]["mlap_histo_s2s_i25_corr"] = interp_xy(smoothed, *px.T)
    coords[uuid]["mlapdv_on_surface_histo_s2s_i25_corr"] = atlas.get_dv_for_mlap(
        coords[uuid]["mlap_histo_s2s_i25_corr"]
    )
    coords[uuid]["mlapdv_histo_s2s_i25_corr"] = projections.project_down_from_surface(
        coords_on_surface=coords[uuid]["mlapdv_on_surface_histo_s2s_i25_corr"],
        atlas=atlas,
        coords_depths=coords[uuid]["dv_below_surface"],
    )

# %% next: same as before, but with interpolation 1
sigma = 1
smoothed = gaussian_filter(grid.astype(float), sigma=(sigma, sigma, 0))
for fov_name, uuid in fov_map.items():
    px = coords_px[uuid]
    coords_um_global = coordinate_systems_2d[uuid].transform(px, "pixel", "um_global")
    px = coordinate_systems_ref_corr.transform(coords_um_global, "um_global", "pixel")
    coords[uuid]["mlap_histo_s2s_i1_corr"] = interp_xy(smoothed, *px.T)
    coords[uuid]["mlapdv_on_surface_histo_s2s_i1_corr"] = atlas.get_dv_for_mlap(
        coords[uuid]["mlap_histo_s2s_i1_corr"]
    )
    coords[uuid]["mlapdv_histo_s2s_i1_corr"] = projections.project_down_from_surface(
        coords_on_surface=coords[uuid]["mlapdv_on_surface_histo_s2s_i1_corr"],
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
    coordinate_systems_ref_corr,
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

# TODO verify that sigma 25 is the better one
sigma = 25
smoothed = gaussian_filter(grid.astype(float), sigma=(sigma, sigma, 0))
for fov_name, uuid in fov_map.items():
    # use the um_corrected to transform back to pix
    # transform to pixel in reference stack
    px = coordinate_systems_ref_corr.transform(
        coords[uuid]["um_corrected"],
        "um_global",
        "pixel",
    )
    # get the corresponding pixel
    coords[uuid]["mlap_histo_s2s_i25_apxy_corr"] = interp_xy(smoothed, *px.T)
    # find point on surface
    coords[uuid]["mlapdv_on_surface_histo_s2s_i25_apxy_corr"] = atlas.get_dv_for_mlap(
        coords[uuid]["mlap_histo_s2s_i25_apxy_corr"]
    )
    # project down uncorrected amount
    coords[uuid]["mlapdv_histo_s2s_i25_apxy_corr"] = (
        projections.project_down_from_surface(
            coords_on_surface=coords[uuid]["mlapdv_on_surface_histo_s2s_i25_apxy_corr"],
            atlas=atlas,
            coords_depths=coords[uuid]["dv_below_surface"],
        )
    )
    # project down CORRECTED amount
    coords[uuid]["mlapdv_histo_s2s_i25_apxyz_corr"] = (
        projections.project_down_from_surface(
            coords_on_surface=coords[uuid]["mlapdv_on_surface_histo_s2s_i25_apxy_corr"],
            atlas=atlas,
            coords_depths=coords[uuid]["dv_below_surface_corrected"],
        )
    )


# %%


# %% deal with this in a moment
i, j = ref_img_size_px // 2
center_mlap = interp_xy(smoothed, i, j)
ref_point["mlap_adjusted"] = center_mlap

# additionally, if this is not the reference session:
# does not matter for this projection but possibly for the others
if eid != eid_ref:
    # TODO figure out if this is plus or minus
    ref_point["mlap_adjusted"] = ref_point["mlap_adjusted"] + session_shift_um


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

# %% project downwards
# for uuid in list(coords.keys()):
#     coords_reprojected = projections.project_down_from_surface(
#         coords_on_surface=coords[uuid]["on_surface_histo_corrected"],
#         atlas=atlas,
#         coords_depths=coords[uuid]["dv_below_surface"],
#     )
#     coords[uuid]["reprojected_histo_corrected"] = coords_reprojected  # this is mlapdv

# %% some quantification of differences
# for name, uuid in fov_map.items():
#     _coords = coords[uuid]["pixel"]
#     coords_um = coordinate_systems_2d[uuid].transform(_coords, "pixel", "um_global")
#     xy_min = np.min(coords_um - coords[uuid]["um_corrected"], axis=0)
#     xy_max = np.max(coords_um - coords[uuid]["um_corrected"], axis=0)
#     dv_min = np.min((dv_avg - fov_depths[uuid]) - coords[uuid]["dv_below_surface"])
#     dv_max = np.max((dv_avg - fov_depths[uuid]) - coords[uuid]["dv_below_surface"])
#     print(f"-- {name} --")
#     print(f"x: min/max {xy_min[0]:.2f}/{xy_max[0]:.2f}")
#     print(f"y: min/max {xy_min[1]:.2f}/{xy_max[1]:.2f}")
#     print(f"dv: min/max {dv_min:.2f}/{dv_max:.2f}")
#     print()

# %% map anything mlapdv to brain area
# for name, uuid in fov_map.items():
#     ids, ix, rgba, acronym = atlas.get_labels_for_mlapdv(
#         coords[uuid]["reprojected_histo"]
#     )
#     coords[uuid]["atlas_rgba"] = rgba
#     coords[uuid]["atlas_acronym"] = acronym
#     coords[uuid]["atlas_id"] = ids

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

# %% 2d plotting
fig, axes = plt.subplots()
for fov_name, uuid in fov_map.items():
    axes.scatter(*coords[uuid]["mlap_histo"].T)

# %% plot them in 3d
#
plot_keys = [
    "mlapdv_on_surface_histo",
    # "mlapdv_histo",
    # "mlapdv_on_surface_histo_s2s_corr",
    # "mlapdv_histo_s2s_corr",
    # "mlapdv_on_surface_histo_s2s_i25_corr",
    # "mlapdv_histo_s2s_i25_corr",
    # "mlapdv_on_surface_histo_s2s_i1_corr",
    # "mlapdv_histo_s2s_i1_corr",
    # "mlapdv_on_surface_histo_s2s_i25_apxy_corr",
    # "mlapdv_histo_s2s_i25_apxy_corr",
    # "mlapdv_histo_s2s_i25_apxyz_corr",
]

axes = plotters.plot_brain_surface_points(atlas.get_surface_points())
# axes.view_init(elev=70, azim=-70)
for fov_name, uuid in fov_map.items():
    for key in plot_keys:
        plotters.plot_points(coords[uuid][key], axes=axes)

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
    "mlapdv_on_surface_histo",
    "mlapdv_histo",
    "mlapdv_on_surface_histo_s2s_corr",
    "mlapdv_histo_s2s_corr",
    "mlapdv_on_surface_histo_s2s_i25_corr",
    "mlapdv_histo_s2s_i25_corr",
    "mlapdv_on_surface_histo_s2s_i1_corr",
    "mlapdv_histo_s2s_i1_corr",
    "mlapdv_on_surface_histo_s2s_i25_apxy_corr",
    "mlapdv_histo_s2s_i25_apxy_corr",
    "mlapdv_histo_s2s_i25_apxyz_corr",
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
        # np.save(
        #     session_folder
        #     / "alf"
        #     / name
        #     / "mpciROIs.mlapdv_histo_projection_surface_5.npy",
        #     coords[uuid]["on_surface_histo_corrected"],
        # )

        # coords_mlapdv = np.concatenate(
        #     [
        #         coords[uuid]["on_surface_histo"],
        #         np.zeros((coords[uuid]["mlap_histo"].shape[0], 1)),
        #     ],
        #     axis=1,
        # )

        # saving the updated coordinates

        # saving the atlas ids
        # atlas_ids = atlas.get_labels_for_mlapdv(coords_mlapdv)[0]
        # np.save(
        #     session_folder
        #     / "alf"
        #     / name
        #     / "mpciROIs.brainLocationIds_ccf_2017_histo_projection_4.npy",
        #     atlas_ids,
        # )

# %%
