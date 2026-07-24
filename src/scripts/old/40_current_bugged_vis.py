# %%
import sys
from pathlib import Path
import numpy as np
from plane2brain import plotters, projections, scanimage, suite2p, ibl
from plane2brain.atlas import ProjectionAtlas
from one.api import ONE
import matplotlib.pyplot as plt


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

# NOTE this currently fails in vscode interactive mode
# eid = one.ref2eid(dict(subject="SP058", date="2024-07-25", sequence="001"))
eid = one.ref2eid(dict(subject="SP058", date="2024-08-01", sequence="001"))
session_path = ibl._eid2path(eid, one, location=LOCATION)
# session_path = Path("/mnt/s0/Data/Subjects/SP058/2024-06-19/001")
# eid = one.path2eid(session_path)


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

########  ##        #######  ########
##     ## ##       ##     ##    ##
##     ## ##       ##     ##    ##
########  ##       ##     ##    ##
##        ##       ##     ##    ##
##        ##       ##     ##    ##
##        ########  #######     ##

# %%
FOVs = [p.parts[-1] for p in sorted(list((session_path / "alf").glob("*FOV*")))]
mlapdv = {}
for dataset in [
    "mpciROIs.mlapdv.npy",
    "mpciROIs.mlapdv_estimate.npy",
    # "mpciROIs.mlapdv_repro_mlapdv_histo_s2s_i25_apxy_corr.npy",
    "mpciROIs.mlapdv_repro_mlapdv_histo.npy",
]:
    mlapdv[dataset] = {}
    for fov in FOVs:
        path = session_path / "alf" / fov / dataset
        mlapdv[dataset][fov] = np.load(path)

# %%
# plot them in 3d
dataset = "mpciROIs.mlapdv_estimate.npy"
dataset = "mpciROIs.mlapdv.npy"
dataset = "mpciROIs.mlapdv_repro_mlapdv_histo_s2s_i25_apxy_corr.npy"
dataset = "mpciROIs.mlapdv_repro_mlapdv_histo.npy"

axes = plotters.plot_brain_surface_points(atlas.get_surface_points())
axes.view_init(elev=90, azim=0)
for fov in FOVs:
    plotters.plot_points(mlapdv[dataset][fov], axes=axes, s=0.1)

axes.set_xlim(0, 4000)
axes.set_ylim(-4000, 0)

# %% quantify differences
A = np.concatenate([mlapdv["mpciROIs.mlapdv.npy"][fov] for fov in FOVs], axis=0)
B = np.concatenate(
    [
        mlapdv["mpciROIs.mlapdv_repro_mlapdv_histo_s2s_i25_apxy_corr.npy"][fov]
        for fov in FOVs
    ],
    axis=0,
)

np.sqrt(np.sum((A - B) ** 2))
