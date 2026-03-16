from pathlib import Path
from typing import Optional, Tuple, Dict
import zipfile
import tifffile
import json
import numpy as np

from one.api import ONE

BASE_FOLDER = Path("/mnt/s0/Data/Subjects")


def _eid2path(eid: str, one: ONE, location="server"):
    if location == "server":
        session_path = BASE_FOLDER / one.eid2path(eid).session_path_short()
    else:
        session_path = one.eid2path(eid)
    return session_path


def ibl_load_fov_data(
    eid: str,
    one: ONE,
    raw_imaging_collection: Optional[str] = None,
    location="server",
):
    # get data
    if raw_imaging_collection is None:
        raw_imaging_collection = infer_imaging_collection(eid, one, location)

    session_path = _eid2path(eid=eid, one=one, location=location)
    # raw_imaging_meta = one.load_dataset(eid, "_ibl_rawImagingData.meta.json", collection=raw_imaging_collection)
    with open(
        session_path / raw_imaging_collection / "_ibl_rawImagingData.meta.json", "r"
    ) as fH:
        raw_imaging_meta = json.load(fH)

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
        session_folder = _eid2path(eid, one, location)
        zip_path = next((session_folder / "alf" / fov).glob("*ROIData.raw.zip"))
        stat_path = zip_path.parent / zip_path.stem / "stat.npy"
        if not stat_path.exists():
            # if stat path doesn't exist, extract it
            stat_path.parent.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(stat_path.parent)
        stat_paths[fov] = stat_path

    return raw_imaging_meta, stat_paths, fov_map


def ibl_get_reference_stack_path(
    eid: str,
    one: ONE,
    raw_imaging_collection: Optional[str] = None,
    location: str = "server",
) -> np.ndarray:
    if raw_imaging_collection is None:
        raw_imaging_collection = infer_imaging_collection(eid, one, location)
    session_path = _eid2path(eid, one, location)
    reference_collection = session_path / raw_imaging_collection / "reference"
    filepath = [
        p for p in reference_collection.glob("*") if "referenceImage.stack" in str(p)
    ]
    assert len(filepath) == 1
    return filepath[0]


def ibl_load_reference_stack(
    eid: str,
    one: ONE,
    raw_imaging_collection: Optional[str] = None,
    location: str = "server",
) -> np.ndarray:
    if raw_imaging_collection is None:
        raw_imaging_collection = infer_imaging_collection(eid, one, location)
    filepath = ibl_get_reference_stack_path(eid, one, location)
    return tifffile.imread(filepath)  # (dv, ml, ap)


def ibl_load_reference_stack_metadata(
    eid: str,
    one: ONE,
    raw_imaging_collection: Optional[str] = None,
    location: str = "server",
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    # get the coordinates of the reference point
    if raw_imaging_collection is None:
        raw_imaging_collection = infer_imaging_collection(eid, one, location)
    if location == "server":
        session_path = _eid2path(eid, one, location)
        reference_collection = session_path / raw_imaging_collection / "reference"
        filepath = [
            p for p in reference_collection.glob("*") if "referenceImage.meta" in str(p)
        ]
        assert len(filepath) == 1
        filepath = filepath[0]

        with open(filepath, "r") as fH:
            ref_img_meta = json.load(fH)
    else:
        ref_img_meta = one.load_dataset(
            eid,
            "*referenceImage.meta",
            collection=raw_imaging_collection,
        )
    return ref_img_meta


def get_reference_points_from_meta(
    ref_img_meta: dict,
    use_resolved: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    # in our case the known point is the center of the craniotomy

    # the contents of ref_img_meta['centerMM'] are:
    #   x/y pixel offset difference from nx/2, ny/2
    #   ML/AP are ml ap coordinates
    ref_point_mlap = []
    for key in ["ML", "AP"]:
        if key + "_resolved" in ref_img_meta["centerMM"] and use_resolved:
            key = key + "_resolved"
        ref_point_mlap.append(ref_img_meta["centerMM"][key] * 1e3)
    ref_point_mlap = np.array(ref_point_mlap)

    # ref is the coordinate system of scanimage (galvo angle)
    # TODO although ... I think this is wrong
    ref_point_ref = []
    for key in ["x", "y"]:
        ref_point_ref.append(ref_img_meta["centerDeg"][key])
    ref_point_ref = np.array(ref_point_ref)
    return ref_point_mlap, ref_point_ref


def infer_imaging_collection(eid: str, one: ONE, location="server") -> str:
    # infer the imaging collection
    # TODO add non server usage
    session_path = _eid2path(eid=eid, one=one, location=location)
    assert session_path.exists()
    raw_imaging_collections = [
        c for c in session_path.glob("*") if c.is_dir() and "raw_imaging_data" in str(c)
    ]
    collections = [c for c in raw_imaging_collections if (c / "reference").exists()]
    assert len(collections) == 1, (
        "multiple imaging collections with reference stack found"
    )
    return collections[0].parts[-1]


def ibl_load_brain_surface_points(
    eid: str,
    one: ONE,
    raw_imaging_collection: Optional[Path] = None,
    location: str = "server",
) -> Dict:
    session_path = _eid2path(eid, one, location)
    if raw_imaging_collection is None:
        raw_imaging_collection = infer_imaging_collection(eid, one, location)

    with open(
        session_path
        / raw_imaging_collection
        / "reference"
        / "referenceImage.points.json",
        "r",
    ) as fH:
        brain_surface_points = json.load(fH)

    return brain_surface_points
    # the surface points are written into the metadata
    # ref_img_meta = ibl_load_reference_stack_metadata(eid, one)
    # assert "points" in ref_img_meta
    # brain_surface_points = ref_img_meta["points"]
    # return brain_surface_points
