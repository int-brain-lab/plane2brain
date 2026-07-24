# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage
from ibllib.mpci.registration import register_reference_stacks
from one.api import ONE

from plane2brain import ibl

# %% whiterussian / local server base folder
BASE_FOLDER = Path("/mnt/s0/Data/Subjects")
LOCATION = "server"

# %% load in: multiple session data, transform, see if xy spread is smaller


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
eid_ref = one.ref2eid({"subject": "SP058", "date": "2024-08-14", "sequence": "001"})
session_refs = [
    # "SP058/2024-06-18/002",
    "SP058/2024-06-19/001",
    "SP058/2024-06-20/001",
    "SP058/2024-06-21/001",
    "SP058/2024-06-25/001",
    "SP058/2024-06-26/001",
    # "SP058/2024-06-28/001",
    "SP058/2024-07-02/001",
    "SP058/2024-07-04/001",
    "SP058/2024-07-05/001",
    "SP058/2024-07-09/001",
    "SP058/2024-07-10/001",
    "SP058/2024-07-12/001",
    "SP058/2024-07-16/001",
    "SP058/2024-07-18/001",
    "SP058/2024-07-19/001",
    "SP058/2024-07-23/001",
    "SP058/2024-07-24/001",
    "SP058/2024-07-25/001",
]

eids = [one.path2eid(ref) for ref in session_refs]


# %% get the transform from reference to target
transforms = {}

for i, eid in enumerate(eids):
    print(i)
    ref_stack_path = ibl.ibl_get_reference_stack_path(
        eid,
        one,
        location=LOCATION,
        raw_imaging_collection=ibl.infer_imaging_collection(
            eid, one, location=LOCATION
        ),
    )

    ref_sess_ref_stack_path = ibl.ibl_get_reference_stack_path(
        eid_ref,
        one,
        location=LOCATION,
        raw_imaging_collection=ibl.infer_imaging_collection(
            eid_ref, one, location=LOCATION
        ),
    )
    _, transform_params = register_reference_stacks(
        ref_stack_path, ref_sess_ref_stack_path
    )

    ref_transform = skimage.transform.EuclideanTransform(
        rotation=transform_params["rotation"],
    ) + skimage.transform.EuclideanTransform(
        translation=transform_params["translation"],
    )
    transforms[eid] = ref_transform

# %%
coords = {}
for eid in eids:
    session_path = ibl._eid2path(eid, one=one, location=LOCATION)
    transform = transforms[eid]

    rois_mlapdv = np.load(session_path / "alf" / "FOV_00" / "mpciROIs.mlapdv.npy")
    # rois_mlapdv_transformed = transform(rois_mlapdv[:, :-1])
    rois_mlapdv_transformed = skimage.transform.warp(
        rois_mlapdv,
        transform,
        order=1,
        mode="constant",
        cval=0,
        clip=True,
        preserve_range=True,
    )
    coords[eid] = {}
    coords[eid]["mlapdv"] = rois_mlapdv
    coords[eid]["mlapdv_t"] = rois_mlapdv_transformed

# %%
np.std(np.array([coords[eid]["mlapdv_t"][0][1] for eid in eids]))

# %%
coords[eid]["mlapdv"][:, :-1] - coords[eid]["mlapdv_t"]


# %%

fig, axes = plt.subplots()
axes.plot(*coords[eid]["mlapdv"][:, :-1].T, ".")
axes.plot(*coords[eid]["mlapdv_t"][:, :-1].T, ".")
axes.set_aspect("equal")

# %%

""" 

"""
