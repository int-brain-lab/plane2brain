# %% load via ONE
from one.api import ONE

one = ONE()
eid = one.ref2eid(dict(subject="SP058", date="2024-08-01", sequence="001"))
raw_imaging_meta = one.load_dataset(
    eid, "raw_imaging_data_02/_ibl_rawImagingData.meta.json"
)
ref_img_meta = one.load_dataset(
    eid, "raw_imaging_data_02/reference/referenceImage.meta.json"
)
# the file raw_imaging_data_02/reference/referenceImage.points.json is not registered on alyx
# there is a brain surface registered here:
brain_surface_points = ref_img_meta["points"]
# but it is slightly different from the content of the referenceImage.points.json
# however this discrepancy has little effect on the part below

# %% load locally on whiterussian
from pathlib import Path
import json

session_path = Path("/mnt/s0/Data/Subjects/SP058/2024-08-01/001")
with open(
    session_path / "raw_imaging_data_02/_ibl_rawImagingData.meta.json", "r"
) as fH:
    raw_imaging_meta = json.load(fH)
with open(
    session_path / "raw_imaging_data_02/reference/referenceImage.meta.json"
) as fH:
    ref_img_meta = json.load(fH)
ref_img_meta["points"]

# these are different from the ones stored in here:
with open(
    session_path / "raw_imaging_data_02/reference/referenceImage.points.json"
) as fH:
    brain_surface_points = json.load(fH)["points"]

# %%

# for the FOVs (raw imaging data)
scanimage_meta = raw_imaging_meta["rawScanImageMeta"]
scanimage_params = raw_imaging_meta["scanImageParams"]

z = scanimage_meta["Artist"]["RoiGroups"]["imagingRoiGroup"]["rois"][0]["zs"]
print(f"during FOV acquisition: z: {z}")  # = -50

z_vc = scanimage_params["hFastZ"]["position"]
print(f"during FOV acquisition: z_voicecoil: {z_vc}")  # = -515

# if we assume both values are combined additively and inverted
fov_depth = -1 * (z - z_vc)
print(f"relative depth of the FOV: {fov_depth}")  # -> 565

# for reference stack
ix = brain_surface_points[0]["stack_idx"]
print(f"reference stack slice index of a brain surface point: {ix}")
# = 3

# a list starting at -815 and going positive in 25 micron steps
z_ref = ref_img_meta["scanImageParams"]["hStackManager"]["zs"][ix]
print(f"brain surface z during reference stack acquisition: {z_ref}")
# -740

# voice coil position during the z-stack
z_vc_ref = ref_img_meta["scanImageParams"]["hFastZ"]["position"]
print(f"voice coil during reference stack acquisition: {z_vc_ref}")
# voicecoil is at -815 during reference stack acquisition

brain_surface = -1 * (z_ref - z_vc_ref)
print(f"relative brain surface in reference stack: {brain_surface}")
# -> 1555

# relative -> distance
depth_below_surf = brain_surface - fov_depth
print(f"FOV depth below surface: {depth_below_surf}")

# %% just using the stackmanager

# for the FOVs (raw imaging data)
fov_depth = -1 * raw_imaging_meta["scanImageParams"]["hStackManager"]["zs"]
ix = brain_surface_points[0]["stack_idx"]
brain_surface = -1 * ref_img_meta["scanImageParams"]["hStackManager"]["zs"][ix]
depth_below_surf = brain_surface - fov_depth
print(f"FOV depth below surface: {depth_below_surf}")
