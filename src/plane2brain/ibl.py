from typing import Optional, Tuple, Dict
import zipfile
import tifffile
import json
import numpy as np

from one.api import ONE


def ibl_load_fov_data(eid: str, raw_imaging_collection: str, one: ONE):
    # TODO this definitely needs some work
    # get data
    # fov_collections = [c for c in one.list_collections(eid) if "FOV" in c]
    raw_imaging_meta = one.load_dataset(eid, "_ibl_rawImagingData.meta.json", collection=raw_imaging_collection)

    # get FOV depths from scanimage meta
    # scanimage_meta = raw_imaging_meta["rawScanImageMeta"]
    # scanimage_fov_metas = scanimage_meta["Artist"]["RoiGroups"]["imagingRoiGroup"]["rois"]

    # our fov names in ascending order
    fov_names = [f"FOV_0{i}" for i in range(len(raw_imaging_meta["FOV"]))]
    # TODO glob here on the folder and compare
    fov_uuids = [meta["roiUUID"] for meta in raw_imaging_meta["FOV"]]
    # fov_metas = [[meta for meta in scanimage_fov_metas if meta["roiUuid"] == uuid][0] for uuid in fov_uuids]
    # fov_depths = np.array([meta["zs"] for meta in fov_metas])
    fov_map = dict(zip(fov_names, fov_uuids))

    # the paths of the suite2p output
    stat_paths = {}
    for fov in fov_names:
        # if location == 'LOCAL_SERVER':
        # session_folder = BASE_FOLDER / one.eid2path(eid).session_path_short()
        session_folder = one.eid2path(eid)
        zip_path = next((session_folder / "alf" / fov).glob("*ROIData.raw.zip"))
        stat_path = zip_path.parent / zip_path.stem / "stat.npy"
        if not stat_path.exists():
            # if stat path doesn't exist, extract it
            stat_path.parent.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(stat_path.parent)
        stat_paths[fov] = stat_path

    return raw_imaging_meta, stat_paths, fov_map


def ibl_load_reference_stack(  # refactor me: IBL specific function
    eid: str,
    one: ONE,
    raw_imaging_collection: Optional[str] = None,
) -> np.ndarray:
    collection = raw_imaging_collection if raw_imaging_collection is not None else infer_imaging_collection(eid, one)
    filepath = one.load_dataset(
        eid,
        "*referenceImage.stack",
        collection=collection,
    )
    return tifffile.imread(filepath)  # (dv, ml, ap)


def ibl_load_reference_stack_metadata(
    eid: str,
    one: ONE,
    raw_imaging_collection: Optional[str] = None,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    # get the coordinates of the reference point
    # TODO
    collection = raw_imaging_collection if raw_imaging_collection is not None else infer_imaging_collection(eid, one)
    ref_img_meta = one.load_dataset(
        eid,
        "*referenceImage.meta",
        collection=collection,
    )
    return ref_img_meta


def get_reference_points_from_meta(
    ref_img_meta: dict,
    use_resolved: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    # we need a point in the imaged plane to be known in the atlas coordinate system
    # here we assume:
    # a) the center of the craniotomy is at 0,0 in scanimage reference frame
    # b) the optical axis of the glass window and the brain normal are collinear

    # the center of the craniotomy,
    ref_point_mlap = []
    for key in ["ML", "AP"]:
        if key + "_resolved" in ref_img_meta["centerMM"] and use_resolved:
            key = key + "_resolved"
        ref_point_mlap.append(ref_img_meta["centerMM"][key] * 1e3)
    ref_point_mlap = np.array(ref_point_mlap)

    ref_point_ref = []
    for key in ["x", "y"]:  # this here is to be replaced with the actual mapping
        ref_point_ref.append(ref_img_meta["centerDeg"][key])
    ref_point_ref = np.array(ref_point_ref)
    # MAJOR TODO FIXME ref point ref is never used, however the origin of the
    # coordinate system should be adjusted for it!!

    return ref_point_mlap, ref_point_ref


def infer_imaging_collection(eid: str, one: ONE) -> str:
    # infer the imaging collection
    collections = one.list_collections(eid, collection="raw_imaging_data_*")
    collections = [c for c in collections if "reference" in c]
    assert len(collections) == 1, "multiple imaging collections with reference stack found"
    return collections[0].split("/")[0]


def ibl_load_brain_surface_points(eid: str, one: ONE) -> Dict:
    # the local file
    # # FIXME
    filepath = "/home/georg/data_local/mesoscope/reference/referenceImage.points.json"
    with open(filepath, "r") as fH:
        reference_brain_surface_points = json.load(fH)
    # this should work
    # ref_img_meta = ibl_load_reference_stack_metadata(eid, one)
    # reference_brain_surface_points = ref_img_meta[some_key]

    return reference_brain_surface_points
