import json
import subprocess
import zipfile
from pathlib import Path
from typing import Literal

import numpy as np
import tifffile
from one.api import ONE

from plane2brain import scanimage

# Server-side data root — machine-specific, only used when location="server"
BASE_FOLDER = Path("/mnt/s0/Data/Subjects")
BASE_FOLDER_POPEYE = Path("/mnt/sdceph/users/ibl/data/Subjects")


def _eid2path(eid: str, one: ONE, location: str = "server") -> Path:
    if location == "server":
        session_path = BASE_FOLDER / one.eid2path(eid).session_path_short()
    else:
        session_path = one.eid2path(eid)
    return session_path


def get_fov_map(
    raw_imaging_meta: dict,
    session_path: Path | None = None,  # optional, for verification
) -> dict:
    # our fov names in ascending order
    fov_names = [f"FOV_0{i}" for i in range(len(raw_imaging_meta["FOV"]))]
    # if session path is given, check if metadata and extracted data
    # contain the same FOVs
    if session_path:
        assert sorted(fov_names) == sorted(
            path.name for path in (session_path / "alf").glob("FOV_*")
        )

    fov_uuids = [meta["roiUUID"] for meta in raw_imaging_meta["FOV"]]
    # fov_metas = [[meta for meta in scanimage_fov_metas if meta["roiUuid"] == uuid][0] for uuid in fov_uuids]
    # fov_depths = np.array([meta["zs"] for meta in fov_metas])
    fov_map = dict(zip(fov_names, fov_uuids))
    return fov_map


def load_fov_data(
    eid: str,
    one: ONE,
    raw_imaging_collection: str | None = None,
    location: str = "server",
    scratch_dir: Path | None = None,
) -> tuple[dict, dict[str, Path], dict[str, str]]:
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
        case "local" | "popeye":
            raw_imaging_meta = one.load_dataset(
                eid, "_ibl_rawImagingData.meta.json", collection=raw_imaging_collection
            )
        case _:
            raise NotImplementedError

    fov_map = get_fov_map(raw_imaging_meta)

    # the paths of the suite2p output
    stat_paths = {}
    for fov_name in fov_map:
        session_folder = _eid2path(eid, one, location=location)
        match location:
            case "server":
                zip_path = next(
                    (session_folder / "alf" / fov_name).glob("*ROIData.raw.zip")
                )
            case "popeye":
                zip_path = next(
                    (session_folder / "alf" / fov_name).glob("*ROIData.raw.*.zip")
                )
            case "local":
                zip_path = one.load_dataset(
                    eid,
                    "*ROIData.raw.zip",
                    collection=f"alf/{fov_name}",
                    download_only=True,
                )

        if scratch_dir is None:
            stat_path = zip_path.parent / zip_path.stem / "stat.npy"
        else:
            stat_path = scratch_dir / zip_path.stem / "stat.npy"

        if not stat_path.exists():
            # if stat path doesn't exist, extract it
            stat_path.parent.mkdir(exist_ok=True, parents=True)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(stat_path.parent)
        stat_paths[fov_name] = stat_path

    return raw_imaging_meta, stat_paths, fov_map


def get_reference_stack_path(
    eid: str,
    one: ONE,
    raw_imaging_collection: str | None = None,
    location: str = "server",
) -> Path:
    if raw_imaging_collection is None:
        raw_imaging_collection = infer_imaging_collection(eid, one, location=location)
    session_path = _eid2path(eid, one, location=location)
    match location:
        case "server" | "popeye":
            reference_collection = session_path / raw_imaging_collection / "reference"
            filepath = [
                p
                for p in reference_collection.glob("*")
                if "referenceImage.stack" in str(p)
            ]

            assert len(filepath) == 1, (
                f"number of reference stacks is: {len(filepath)} - and has to be exactly 1"
            )
            filepath = filepath[0]
        case "local":
            filepath = one.load_dataset(
                eid,
                "*referenceImage.stack",
                collection=f"{raw_imaging_collection}/reference",
                download_only=True,
            )
    return filepath


def load_reference_stack(
    eid: str,
    one: ONE,
    raw_imaging_collection: str | None = None,
    location: str = "server",
) -> np.ndarray:
    """Load the reference image stack for a session.

    Returns:
        Array of shape (dv, ml, ap).
    """
    if raw_imaging_collection is None:
        raw_imaging_collection = infer_imaging_collection(eid, one, location=location)
    filepath = get_reference_stack_path(eid, one, location=location)
    return tifffile.imread(filepath)


def load_reference_stack_metadata(
    eid: str,
    one: ONE,
    raw_imaging_collection: str | None = None,
    location: str = "server",
) -> dict:
    # load the referenceImage.meta JSON file
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
            reference_collection = raw_imaging_collection + "/reference"
            ref_img_meta = one.load_dataset(
                eid,
                "*referenceImage.meta",
                collection=reference_collection,
            )
        case "popeye":
            session_path = _eid2path(eid, one, location=location)
            reference_collection = session_path / raw_imaging_collection / "reference"
            filepath = list(reference_collection.glob("*referenceImage.meta.*.json"))
            assert len(filepath) == 1
            filepath = filepath[0]
            with open(filepath, "r") as fH:
                ref_img_meta = json.load(fH)

    return ref_img_meta


def load_reference_points_from_meta(
    ref_img_meta: dict,
) -> dict:
    # in our case the known point is the center of the craniotomy
    ref_point = {
        "mlap": np.array(
            [ref_img_meta["centerMM"][key] * 1e3 for key in ["ML", "AP"]],
        ),
        "xy": np.array(
            [ref_img_meta["centerMM"][key] for key in ["x", "y"]],
        ),
        "deg": np.array(
            [ref_img_meta["centerDeg"][key] for key in ["x", "y"]],
        ),
    }
    return ref_point


def load_reference_stack_miles(
    eid: str,
    one: ONE,
    raw_imaging_collection: str | None = None,
    location: str = "server",
) -> np.ndarray:
    """Load reference stack using Miles' loader. Not yet implemented."""
    raise NotImplementedError


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
                f"number of collections with reference stacks is: {len(collections)} - and has to be exactly 1"
            )
            return collections[0].parts[-1]
        case "local" | "popeye":
            collections = [
                c
                for c in one.list_collections(eid)
                if "raw_imaging_data" in c and "reference" in c
            ]
            assert len(collections) == 1, (
                f"number of collections with reference stacks is: {len(collections)} - and has to be exactly 1"
            )
            return collections[0].split("/")[0]


def load_brain_surface_points(
    eid: str,
    one: ONE,
    raw_imaging_collection: Path | None = None,
    location: str = "server",
) -> dict:
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
        case "popeye":
            ref_points_path = list(
                (session_path / raw_imaging_collection / "reference").glob(
                    "referenceImage.points.*.json"
                )
            )
            assert len(ref_points_path) == 1
            with open(ref_points_path[0], "r") as fH:
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
                    cmd = f"scp -v mbox-whiterussian:{remote_path} {local_path}"  # This only works with my own SSH setup and is throwaway code ...
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
        case "local" | "popeye":
            datasets = one.list_datasets(eid, collection=f"alf/{fov}")
            if f"alf/{fov}/{dataset}" in datasets:
                mlapdv = one.load_dataset(eid, dataset, collection=f"alf/{fov}")
            else:
                # dataset is not available via one
                # make a general copy dataset function from the scraps above
                raise NotImplementedError
    return mlapdv


def infer_ref_stack_virtual_corner(
    ref_img_scanimage_meta: dict,
    ref_img_size_px: np.ndarray,  # in 2d (X,Y)
    dims: tuple[str, str] = ("X", "Y"),
) -> tuple[np.ndarray, np.ndarray]:
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


def infer_reference_session(subject: str) -> str:
    # this function should return the eid of the session
    # that was used for the alignment of the histology.
    # skip implementation for now
    ...


def _validate_eid_and_session_path(
    eid: str,
    session_folder: str | Path,
    one: ONE,
    location: Literal["server", "local", "popeye"] = "server",
) -> bool:
    match location:
        case "server":
            base_folder = BASE_FOLDER
        case "popeye":
            base_folder = BASE_FOLDER_POPEYE
        case "local":
            raise NotImplementedError

    session_folder_inferred = base_folder / one.eid2path(eid).session_path_short()
    if session_folder != session_folder_inferred:
        raise ValueError(
            f"eid and session folder mismatch: {eid}, {session_folder} != {session_folder_inferred}"
        )


def infer_possible_corrections(
    eid: str | None,
    session_path: str | Path | None,
    one: ONE,
    location: str = "server",
) -> dict[str, bool]:
    one = one if one is not None else ONE()  # not ideal


# files necessary for the reprojection
# and how to get them with the data loader
