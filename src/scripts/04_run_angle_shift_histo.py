# %%
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

# the transform between them
_, transform_params = register_reference_stacks(ref_stack_path, ref_sess_ref_stack_path)

# the transform between the reference stack and the "reference reference" stack
# = the reference stack of the reference session
ref_transform = skimage.transform.EuclideanTransform(
    rotation=transform_params["rotation"],
) + skimage.transform.EuclideanTransform(
    translation=transform_params["translation"],
)
# %% verify that calling transfrom directly
# is the same as appying the affine transform
# by hand

A = np.random.rand(10, 2)
B = ref_transform(A)
A = np.concatenate([A, np.ones((10, 1))], axis=1)

Bp = (np.array(ref_transform) @ A.T).T[:, :-1]
from numpy import testing as nptest

nptest.assert_array_equal(B, Bp)
# %%
"""

 ##        #######     ###    ########     ##     ## ####  ######  ########  #######  
 ##       ##     ##   ## ##   ##     ##    ##     ##  ##  ##    ##    ##    ##     ## 
 ##       ##     ##  ##   ##  ##     ##    ##     ##  ##  ##          ##    ##     ## 
 ##       ##     ## ##     ## ##     ##    #########  ##   ######     ##    ##     ## 
 ##       ##     ## ######### ##     ##    ##     ##  ##        ##    ##    ##     ## 
 ##       ##     ## ##     ## ##     ##    ##     ##  ##  ##    ##    ##    ##     ## 
 ########  #######  ##     ## ########     ##     ## ####  ######     ##     #######  
 
"""

coords = {
    uuid: {
        "mlapdv_histo": ibl.load_roi_mlapdv(eid, one, location=LOCATION, fov=fov_name)
    }
    for fov_name, uuid in fov_map.items()
}

shift = transform_params["translation"] * um_per_px


# shift is in the imaging plane
# we need to convert this to 3d

# to do this:
# create the 3d coordinate system
# convert shift vector to 3d
# apply the 3d shift to all the coordinates
# find new surface points
# project down by depth


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

ref_point_mlap = ref_point_mlap + transform_params["translation"] * um_per_px
ref_img_topleft_um = ref_point_mlap - ref_img_size_um / 2

# inferring the the "virtual corner" of the reference stack image
# in ref
ref_img_topleft_ref, ref_img_ref_per_px = ibl.infer_ref_stack_virtual_corner(
    ref_img_meta["rawScanImageMeta"],
    ref_img_size_px,
    dims=dims,
)
# the 2d coordinate system in of the reference image
coordinate_systems_ref = create_coordinate_system_for_image(
    ref_img_size_px,
    um_per_px,
    ref_img_ref_per_px,
    ref_img_topleft_ref,
)

# the coordinate systems
ref_point_mlapdv, brain_normal_at_ref = atlas.get_plane_at_point_mlap(
    *ref_point_mlap,
    numba=True,
)
coordinate_systems_3d = setup_coordinate_systems_3d(
    ref_point_mlapdv,
    brain_normal_at_ref,
    rotate_by=scanner_orientation["rotation"],
    invert_dims=scanner_orientation["invert_axis"],
)

# load the brain surface points and get the normal
brain_surface_points = ibl.load_brain_surface_points(eid, one, location=LOCATION)

# this normal is expressed in the coordinate system of the reference stack
p_surface, n_surface, dv_avg = projections.get_brain_surface_normal(
    brain_surface_points,
    ref_img_meta,
    coordinate_systems_ref,
)

# this requires a coordinate system for 3d
optical_axis = (
    coordinate_systems_3d.transform(n_surface, "imaging_plane", "mlapdv")
    - ref_point_mlapdv
)

# set up a new 3d coordinate system with the imaging plane, now adjusted by the difference
# between the optical axis and the brain normal
coordinate_systems_3d_adjusted = setup_coordinate_systems_3d(
    ref_point_mlapdv,
    optical_axis,  # now adjusted for the optical axis
    rotate_by=scanner_orientation["rotation"],
    invert_dims=scanner_orientation["invert_axis"],
)

# %%
"""
 
 ##     ## ####  ######  ########  #######        ###    ########        ## ##     ##  ######  ######## 
 ##     ##  ##  ##    ##    ##    ##     ##      ## ##   ##     ##       ## ##     ## ##    ##    ##    
 ##     ##  ##  ##          ##    ##     ##     ##   ##  ##     ##       ## ##     ## ##          ##    
 #########  ##   ######     ##    ##     ##    ##     ## ##     ##       ## ##     ##  ######     ##    
 ##     ##  ##        ##    ##    ##     ##    ######### ##     ## ##    ## ##     ##       ##    ##    
 ##     ##  ##  ##    ##    ##    ##     ##    ##     ## ##     ## ##    ## ##     ## ##    ##    ##    
 ##     ## ####  ######     ##     #######     ##     ## ########   ######   #######   ######     ##    
 
"""

# %% transform shift vector to 3d
shift_3d = (
    coordinate_systems_3d_adjusted.transform(
        np.append(shift, 0), "imaging_plane", "mlapdv"
    )
    - ref_point_mlapdv
)

# %% plot to verify
axes = plotters.plot_brain_surface_points(atlas.get_surface_points())
coordinate_systems_3d.plot(axes=axes, color_by="axis", scale=500)
coordinate_systems_3d_adjusted.plot(axes=axes, color_by="axis", scale=500)
plotters.plot_line(
    ref_point_mlapdv, shift_3d, axes=axes, color="magenta", length=[0, 10]
)
axes.set_aspect("equal")

# %% apply the 3d shift to all the coordinates
for fov_name, uuid in fov_map.items():
    # TODO verify if add or subtract, depends on the direction of the transform
    coords[uuid]["mlapdv_shifted"] = coords[uuid]["mlapdv_histo"] + shift_3d

# %% for these coordinates, find the new surface points
for fov_name, uuid in fov_map.items():
    coords[uuid]["on_surface_shifted"] = np.zeros_like(coords[uuid]["mlapdv_shifted"])
    points = coords[uuid]["mlapdv_shifted"][:, :-1]
    dummy_3d_system = setup_coordinate_systems_3d(
        np.array([0.0, 0.0, 100.0]), np.array([0.0, 0.0, 1.0])
    )
    coords[uuid]["on_surface_shifted"] = projections.project_coords_onto_atlas_surface(
        coords_um=points,
        coordinate_systems_3d=dummy_3d_system,
        atlas=atlas,
        projection_vector=np.array([0.0, 0.0, -1.0]),
    )

# %%
# how do stevens values get their dv component?

# coords[uuid]['mlapdv_histo']
coords[uuid]["on_surface_shifted"]

# %% from these, project downwards

# get the suite2p data
coords_px = suite2p.data_loader(stat_paths, fov_map)  # refactor: rename coords_px
fov_uuids = sorted(list(fov_map.values()))
# setup the 2d coordinate systems
coordinate_systems_2d = scanimage.create_coordinate_systems_from_scanimage_meta(
    raw_imaging_meta["rawScanImageMeta"],
    fov_uuids=fov_uuids,
    dims=dims,
)
ds = 1
for fov_name, fov_uuid in fov_map.items():
    # get the pixel data
    _coords_px = coords_px[fov_uuid][::ds]  # downsample factor for debugging
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
# this creates: the keys 'um_corrected' and 'dv_below_surface'
coords = projections.correct_coords_for_tilt_2d(
    coords,
    fov_depths,
    p_surface,
    n_surface,
)
# project down
for fov_name, fov_uuid in fov_map.items():
    _depths = np.ones(coords[fov_uuid]["pixel"].shape[0]) * fov_depths[fov_uuid]
    coords[fov_uuid]["mlapdv"] = projections.project_down_from_surface(
        coords[fov_uuid]["on_surface_shifted"],
        atlas=atlas,
        coords_depths=_depths,
    )

# %%

# %% some quantification of differences
for name, uuid in fov_map.items():
    _coords = coords[uuid]["pixel"]
    coords_um = coordinate_systems_2d[uuid].transform(_coords, "pixel", "um_global")
    xy_min = np.min(coords_um - coords[uuid]["um_corrected"], axis=0)
    xy_max = np.max(coords_um - coords[uuid]["um_corrected"], axis=0)
    dv_min = np.min((dv_avg - fov_depths[uuid]) - coords[uuid]["dv_below_surface"])
    dv_max = np.max((dv_avg - fov_depths[uuid]) - coords[uuid]["dv_below_surface"])
    print(f"-- {name} --")
    print(f"x: min/max {xy_min[0]:.2f}/{xy_max[0]:.2f}")
    print(f"y: min/max {xy_min[1]:.2f}/{xy_max[1]:.2f}")
    print(f"dv: min/max {dv_min:.2f}/{dv_max:.2f}")
    print()

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
    # axes.scatter(
    #     *coords[uuid]["on_surface"].T,
    #     c=coords[uuid]["atlas_rgba"] / 255,
    #     s=5,
    #     zorder=20,
    # )
    axes.scatter(*coords[uuid]["mlapdv_histo"][::10].T, c="k", s=5)
    axes.scatter(*coords[uuid]["mlapdv"][::10].T, c="r", s=5)

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
        coords_mlapdv = coords[uuid]["reprojected"]
        # saving the updated coordinates
        np.save(
            session_folder / "alf" / name / "mpciROIs.mlapdv_angle_projection.npy",
            coords_mlapdv,
        )
        # saving the atlas ids
        atlas_ids = atlas.get_labels_for_mlapdv(coords_mlapdv)[0]
        np.save(
            session_folder
            / "alf"
            / name
            / "mpciROIs.brainLocationIds_ccf_2017_angle_projection.npy",
            atlas_ids,
        )

# %%
