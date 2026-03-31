from pathlib import Path
from typing import Optional, Tuple, Dict, List
import zipfile
import tifffile
import json
import numpy as np
import subprocess

from plane2brain import scanimage
from one.api import ONE

BASE_FOLDER = Path("/mnt/s0/Data/Subjects")


def _eid2path(eid: str, one: ONE, location="server"):
    if location == "server":
        session_path = BASE_FOLDER / one.eid2path(eid).session_path_short()
    else:
        session_path = one.eid2path(eid)
    return session_path


def load_fov_data(
    eid: str,
    one: ONE,
    raw_imaging_collection: Optional[str] = None,
    location="server",
):
    # get data
    if raw_imaging_collection is None:
        raw_imaging_collection = infer_imaging_collection(eid, one, location=location)

    session_path = _eid2path(eid=eid, one=one, location=location)

    match location:
        case "server":
            with open(
                session_path / raw_imaging_collection / "_ibl_rawImagingData.meta.json",
                "r",
            ) as fH:
                raw_imaging_meta = json.load(fH)
        case "local":
            raw_imaging_meta = one.load_dataset(
                eid, "_ibl_rawImagingData.meta.json", collection=raw_imaging_collection
            )

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
        session_folder = _eid2path(eid, one, location=location)
        match location:
            case "server":
                zip_path = next((session_folder / "alf" / fov).glob("*ROIData.raw.zip"))
                stat_path = zip_path.parent / zip_path.stem / "stat.npy"
                if not stat_path.exists():
                    # if stat path doesn't exist, extract it
                    stat_path.parent.mkdir(exist_ok=True)
                    with zipfile.ZipFile(zip_path, "r") as z:
                        z.extractall(stat_path.parent)
                stat_paths[fov] = stat_path
            case "local":
                zip_path = one.load_dataset(
                    eid, "*ROIData.raw.zip", collection=f"alf/{fov}", download_only=True
                )
                stat_path = zip_path.parent / zip_path.stem / "stat.npy"
                if not stat_path.exists():
                    # if stat path doesn't exist, extract it
                    stat_path.parent.mkdir(exist_ok=True)
                    with zipfile.ZipFile(zip_path, "r") as z:
                        z.extractall(stat_path.parent)
                stat_paths[fov] = stat_path

    return raw_imaging_meta, stat_paths, fov_map


def get_reference_stack_path(
    eid: str,
    one: ONE,
    raw_imaging_collection: Optional[str] = None,
    location: str = "server",
) -> np.ndarray:
    if raw_imaging_collection is None:
        raw_imaging_collection = infer_imaging_collection(eid, one, location=location)
    session_path = _eid2path(eid, one, location=location)
    match location:
        case "server":
            reference_collection = session_path / raw_imaging_collection / "reference"
            filepath = [
                p
                for p in reference_collection.glob("*")
                if "referenceImage.stack" in str(p)
            ]
            assert len(filepath) == 1, "multiple reference stacks found"
            filepath = filepath[0]
        case "local":
            filepath = one.load_dataset(
                eid,
                "*referenceImage.stack",
                collection=f"{raw_imaging_collection}/reference",
            )
    return filepath


def load_reference_stack(
    eid: str,
    one: ONE,
    raw_imaging_collection: Optional[str] = None,
    location: str = "server",
) -> np.ndarray:
    if raw_imaging_collection is None:
        raw_imaging_collection = infer_imaging_collection(eid, one, location=location)
    filepath = get_reference_stack_path(eid, one, location=location)
    return tifffile.imread(filepath)  # (dv, ml, ap)


def load_reference_stack_metadata(
    eid: str,
    one: ONE,
    raw_imaging_collection: Optional[str] = None,
    location: str = "server",
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    # get the coordinates of the reference point
    if raw_imaging_collection is None:
        raw_imaging_collection = infer_imaging_collection(eid, one, location=location)
    match location:
        case "server":
            session_path = _eid2path(eid, one, location=location)
            reference_collection = session_path / raw_imaging_collection / "reference"
            filepath = [
                p
                for p in reference_collection.glob("*")
                if "referenceImage.meta" in str(p)
            ]
            assert len(filepath) == 1
            filepath = filepath[0]

            with open(filepath, "r") as fH:
                ref_img_meta = json.load(fH)
        case "local":
            ref_img_meta = one.load_dataset(
                eid,
                "*referenceImage.meta",
                collection=raw_imaging_collection,
            )
    return ref_img_meta


def load_reference_points_from_meta(
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
    session_path = _eid2path(eid=eid, one=one, location=location)
    match location:
        case "server":
            assert session_path.exists()
            raw_imaging_collections = [
                c
                for c in session_path.glob("*")
                if c.is_dir() and "raw_imaging_data" in str(c)
            ]
            collections = [
                c for c in raw_imaging_collections if (c / "reference").exists()
            ]
            assert len(collections) == 1, (
                "multiple imaging collections with reference stack found"
            )
            return collections[0].parts[-1]
        case "local":
            collections = [
                c
                for c in one.list_collections(eid)
                if "raw_imaging_data" in c and "reference" in c
            ]
            assert len(collections) == 1, (
                "multiple imaging collections with reference stack found"
            )
            return collections[0].split("/")[0]


def load_brain_surface_points(
    eid: str,
    one: ONE,
    raw_imaging_collection: Optional[Path] = None,
    location: str = "server",
) -> Dict:
    session_path = _eid2path(eid, one, location)
    if raw_imaging_collection is None:
        raw_imaging_collection = infer_imaging_collection(eid, one, location)

    match location:
        case "server":
            with open(
                session_path
                / raw_imaging_collection
                / "reference"
                / "referenceImage.points.json",
                "r",
            ) as fH:
                brain_surface_points = json.load(fH)
        case "local":
            datasets = one.list_datasets(
                eid, collection=f"{raw_imaging_collection}/reference"
            )
            dataset = [d for d in datasets if "referenceImage.points" in d]
            # if dataset is registered on alyx, use one.
            if len(dataset) == 1:
                brain_surface_points = one.load_dataset(
                    eid,
                    "*referenceImage.points.json",
                    collection=f"{raw_imaging_collection}/reference",
                )
            # if it isn't, try to load locally (or copy)
            elif len(dataset) == 0:
                print("reference points not registered on alyx")
                remote_path = (
                    _eid2path(eid, one, location="server")
                    / raw_imaging_collection
                    / "reference"
                    / "referenceImage.points.json"
                )
                local_path = (
                    one.eid2path(eid)
                    / raw_imaging_collection
                    / "reference"
                    / "referenceImage.points.json"
                )
                if not local_path.exists():  # attempt to copy
                    cmd = f"scp -v mbox-whiterussian:{remote_path} {local_path}"  # This only work with my own SSH setup and is throwaway code ...
                    res = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True
                    )
                    if res.returncode != 0:
                        raise subprocess.CalledProcessError(
                            returncode=res.returncode,
                            cmd=res.args,
                            output=res.stdout,
                            stderr=res.stderr,
                        )
                print("loading reference points from local path")
                with open(local_path, "r") as fH:
                    brain_surface_points = json.load(fH)

    return brain_surface_points
    # the surface points are written into the metadata
    # ref_img_meta = load_reference_stack_metadata(eid, one)
    # assert "points" in ref_img_meta
    # brain_surface_points = ref_img_meta["points"]
    # return brain_surface_points


def load_roi_mlapdv(
    eid: str,
    one: ONE,
    fov: str = "FOV_00",
    location: str = "server",
    provenance: str = "resolved",
) -> np.ndarray:
    session_path = _eid2path(eid, one, location)
    dataset = (
        "mpciROIs.mlapdv.npy"
        if provenance == "resolved"
        else "mpciROIs.mlapdv_estimate.npy"
    )
    match location:
        case "server":
            path = session_path / "alf" / fov / dataset
            assert path.exists()
            mlapdv = np.load(path)
        case "local":
            datasets = one.list_datasets(eid, collection=f"alf/{fov}")
            if f"alf/{fov}/{dataset}" in datasets:
                mlapdv = one.load_dataset(eid, dataset, collection=f"alf/{fov}")
            else:
                # dataset it not available via one
                # make a general copy dataset function from the sraps above
                raise NotImplementedError
    return mlapdv


def infer_ref_stack_virtual_corner(
    ref_img_scanimage_meta: dict,
    ref_img_size_px: np.ndarray,  # in 2d (X,Y)
    dims: List = ["X", "Y"],
):
    # get the corner of the reference stack in ref space
    # TODO refactor me
    stripes = ref_img_scanimage_meta["Artist"]["RoiGroups"]["imagingRoiGroup"]["rois"]

    topleft_corners = []
    bottomright_corners = []

    for scanimage_fov_meta in stripes:
        # get size and center for each fov
        fov_size_ref, fov_center_ref = scanimage.get_scanfield_size_ref(
            scanimage_fov_meta, dims=dims
        )
        # transform to reference coordinate frame
        fov_topleft_ref = fov_center_ref - fov_size_ref / 2
        topleft_corners.append(fov_topleft_ref)

        fov_bottomright_ref = fov_center_ref + fov_size_ref / 2
        bottomright_corners.append(fov_bottomright_ref)

    ref_img_topleft_ref = np.min(topleft_corners, axis=0)
    ref_img_bottomright_ref = np.max(bottomright_corners, axis=0)
    ref_per_px = (ref_img_bottomright_ref - ref_img_topleft_ref) / ref_img_size_px

    return ref_img_topleft_ref, ref_per_px
