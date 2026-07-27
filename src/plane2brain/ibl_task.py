from __future__ import annotations
import json
from pathlib import Path
from typing import Literal
from uuid import UUID
from one.api import ONE
from one.alf.path import ALFPath
import numpy as np
import tifffile
from iblatlas.atlas import MRITorontoAtlas
from ibllib.oneibl import data_handlers
from ibllib.oneibl.data_handlers import ExpectedDataset, ServerGlobusDataHandler
from mpci.alyx.tasks import MesoscopeTask
from plane2brain.registration import (
    apply_transform,
    evaluate,
    inspect_registration_delta,
    plot_keypoints,
    register_stacks,
)
from skimage.transform import ProjectiveTransform

IBL_MESOSCOPE_DEFINITIONS = {
    "scanner_orientation": {"rotation": 0.0, "invert_axis": [True, True, False]},
    "scanimage_dimensions": ("Y", "X"),
}
import logging

_logger = logging.getLogger(__name__)


class ReprojectionTask(MesoscopeTask):
    """development of the reprojection task"""

    priority = 100
    io_charge = 100
    cpu = -1
    job_size = "large"

    def __init__(
        self,
        *args,
        FOV: str | None = None,
        reference_session_path: str | Path,
        one: ONE | None = None,
        raw_imaging_collection: str | None = None,
        reference_session_raw_imaging_collection: str | None = None,
        **kwargs,
    ):
        """Initialize the task with FOV and reference session identifiers.

        Infers raw imaging collections for both the current and reference
        session if not given explicitly.
        """
        self.one = one or ONE()
        assert not self.one.offline
        session_path = args[0]
        self.eid = self.one.eid2path(session_path)

        self.FOV = FOV
        self.raw_imaging_collection = (
            raw_imaging_collection or self.infer_raw_imaging_collection(session_path)
        )
        self.reference_session_path = ALFPath(reference_session_path)
        self.reference_session_eid = one.path2eid(self.reference_session_path)
        self.reference_session_raw_imaging_collection = (
            reference_session_raw_imaging_collection
            or self.infer_raw_imaging_collection(self.reference_session_path)
        )

        super().__init__(*args, **kwargs)

    @property
    def signature(self):
        """Build the task's expected input and output dataset signature."""
        I = ExpectedDataset.input
        O = ExpectedDataset.output
        # how does this fan out accross FOVs?

        signature = {
            "input_files": [
                I(
                    "_ibl_rawImagingData.meta.json",
                    self.raw_imaging_collection,
                    True,
                    unique=False,
                ),
            ],
            "output_files": [
                O("referenceImage.mlapdv.npy", "alf/FOV*", True),
                O("referenceImage.mlapdv.npy", "alf/FOV*", True),
            ],  # TODO ask about ExpectedDataset.output (or rather the lack of them)
        }

        return signature

    def validate_reference_session(self, reference_session_eid: str | UUID) -> UUID:
        assert (
            self.one.eid2ref(reference_session_eid)["subject"]
            == self.one.eid2ref(self.eid)["subject"]
        ), "reference session does not match to this session: wrong subject"
        return reference_session_eid

    @staticmethod
    def infer_raw_imaging_collection(session_path: str | Path) -> str:
        """Find the raw imaging collection containing reference measurements.

        If multiple matching collections exist, the last one is chosen.
        """
        assert session_path.exists()
        collections = [
            c
            for c in session_path.glob("raw_imaging_data_*")
            if c.is_dir() and (c / "reference").exists()
        ]
        if len(collections) > 1:
            _logger.warning(
                f"number of collections with reference stacks is: {len(collections)} - taking the last one"
            )

        return collections[-1].parts[-1]

    def load_imaging_metadata(self) -> dict:
        """Load the IBL-specific raw imaging metadata JSON."""
        metadata_filepath = (
            self.session_path
            / self.raw_imaging_collection
            / "_ibl_rawImagingData.meta.json"
        )
        return json.loads(Path(metadata_filepath).read_text(encoding="utf-8"))

    def load_reference_stack_metadata(self) -> dict:
        """Load the IBL metadata JSON of the reference stack."""
        reference_collection = (
            self.session_path / self.raw_imaging_collection / "reference"
        )
        filepath = [
            p for p in reference_collection.glob("*") if "referenceImage.meta" in str(p)
        ]
        assert len(filepath) == 1
        return json.loads(Path(filepath[0]).read_text(encoding="utf-8"))

    def get_reference_stack_path(self) -> Path:
        """Return the path to the reference stack of the current session."""
        # check if this is necessary or if it can be folded with the function below
        return self._get_ref_stack_path(self.session_path, self.raw_imaging_collection)

    def get_reference_session_reference_stack_path(self) -> Path:
        """Return the path to the reference stack of the reference session."""
        # NOTE this should fail loudly when it fails. All the checks whether
        # the reference stack has the correct shape (or exists at all) should
        # not live here
        # All loaders behave as if "happy path"
        return self._get_ref_stack_path(
            self.reference_session_path,
            self.reference_session_raw_imaging_collection,
        )

    def _get_ref_stack_path(
        self, session_path: Path, raw_imaging_collection: str
    ) -> Path:
        """Find the reference stack file within a session's raw imaging collection."""
        path = session_path / raw_imaging_collection / "reference"
        filepath = [p for p in path.glob("*") if "referenceImage.stack" in str(p)]

        assert len(filepath) == 1, (
            f"number of reference stacks is: {len(filepath)} - and has to be exactly 1"
        )
        return filepath[0]

    def load_reference_stack(self) -> np.ndarray:
        # load the reference stack
        return tifffile.imread(self.get_reference_stack_path())

    def load_reference_session_reference_stack(self) -> np.ndarray:
        # load the reference stack of the reference session
        return tifffile.imread(self.get_reference_session_reference_stack_path())

    def load_histology(self) -> np.ndarray:
        # returns the mlapdv volume
        atlas = MRITorontoAtlas(res_um=25)
        local_histo_path = self._get_atlas_registered_reference_mlap()
        ccf_idx = np.load(local_histo_path)

        ccf_idx[:, :, 1] = np.abs(
            ccf_idx[:, :, 1].astype("int64") - atlas.label.shape[0]
        ).astype(ccf_idx.dtype)
        # to be very explicit about: this is for the ref_img of the session that is aligned to the histo
        ref_img_histo_mlapdv = (
            atlas.ccf2xyz(ccf_idx * atlas.res_um, ccf_order="mlapdv") * 1e6
        )  # m -> μm
        return ref_img_histo_mlapdv

    def _load_brain_surface_points_from_metadata(self) -> dict:
        # load the brain imaging points from the metadata file
        # this is the function that should fail if they don't exist
        ref_img_meta = self.load_reference_stack_metadata()
        return {"points": ref_img_meta["points"]}

    def _load_brain_surface_points_from_file(self) -> dict:
        ref_points_path = list(
            (self.session_path / self.raw_imaging_collection / "reference").glob(
                "referenceImage.points.*.json"
            )
        )
        if len(ref_points_path) == 0:
            raise FileNotFoundError
        if len(ref_points_path) > 1:
            raise ValueError("multiple reference point files found")
        return json.loads(Path(ref_points_path[0]).read_text(encoding="utf-8"))

    def load_brain_surface_points(
        self,
        prefer: Literal["metadata", "file"] = "metadata",
    ) -> dict:
        # the brain surface points are now stored in the metadata and not in the points.json

        # returns None if can't load any form of brain surface points!
        try:
            brain_surface_points_file = self._load_brain_surface_points_from_file()
        except FileNotFoundError:
            brain_surface_points_file = None

        # otherwise just return from the metadata
        try:
            brain_surface_points_meta = self._load_brain_surface_points_from_metadata()
        except KeyError:
            brain_surface_points_meta = None

        # if none exists
        if brain_surface_points_file is None and brain_surface_points_meta is None:
            raise ValueError("no brain surface points found")

        # if both exist
        if (
            brain_surface_points_file is not None
            and brain_surface_points_meta is not None
        ):
            # and they are the same, it doesn't matter
            if brain_surface_points_file == brain_surface_points_meta:
                return brain_surface_points_file
            # if they aren't, return the preferred
            match prefer:
                case "metadata":
                    return brain_surface_points_meta
                case "file":
                    return brain_surface_points_file
                case _:
                    raise ValueError(f"invalid preference: {prefer}")
        # if only one exists:
        if brain_surface_points_file is None and brain_surface_points_meta is not None:
            if prefer == "file":
                _logger.warning(
                    "using metadata as a non-prefered source of brain surface points"
                )
            return brain_surface_points_meta
        if brain_surface_points_file is not None and brain_surface_points_meta is None:
            if prefer == "metadata":
                _logger.warning(
                    "using points.json file as a non-prefered source of brain surface points"
                )
            return brain_surface_points_file

    def _get_atlas_registered_reference_mlap(self, clobber=False):
        """Download the aligned reference stack Allen atlas indices.

        This is the file created by the histology pipeline, one per subject.
        This file contains the Allen atlas image volume indices for each pixel of the reference stack.

        Parameters
        ----------
        clobber : bool
            If True, re-download the file even if it exists locally.

        Returns
        -------
        one.alf.path.ALFPath
            The local filepath of the aligned reference stack.
            A uint16 array with shape (h, w, 3), comprising Allen atlas image volume indices for
            dimensions representing (ml, ap, dv).  The first two dimensions (h, w) should equal
            those of the reference stack.
        """

        # Ensure reference session reference files present
        # this is order sensitive
        # signature = {
        #     "input_files": self.signature["input_files"][-3:],
        #     "output_files": [],
        # }
        # assert all(
        #     x.identifiers[-1].startswith("reference") for x in signature["input_files"]
        # )
        reference_collection = self.raw_imaging_collection + "/reference"
        signature = {
            "input_files": [
                ExpectedDataset.input(
                    "referenceImage.mlapdv.npy", reference_collection, True
                )
            ],
            "output_files": [],
        }
        if self.location == "server" and self.force:
            handler = data_handlers.ServerGlobusDataHandler(
                self.reference_session_path, signature, one=self.one
            )
        else:
            handler = self.data_handler.__class__(
                self.reference_session_path, signature, one=self.one
            )
        handler.setUp()

        _logger.info(
            "Looking for reference MLAPDV in %s",
            self.reference_session_path.joinpath(
                self.raw_imaging_collection, "reference"
            ),
        )
        # NB: The local reference folder is expected to exist after handler.setUp()
        local_file = (
            self.reference_session_path
            / self.raw_imaging_collection
            / "reference"
            / "referenceImage.mlapdv.npy"
        )

        if not local_file.exists():
            _logger.warning("getting histology via data handler failed!")

        if clobber or not local_file.exists():
            _logger.info("attempting to download histology file from flatiron")
            # Download remote file
            assert self.one, "ONE required"
            local_file.parent.mkdir(parents=True, exist_ok=True)
            lab = self.one.get_details(self.reference_session_path)["lab"]
            remote_file = f"{lab}/{self.reference_session_path.session_path_short()}/{local_file.name}"
            try:
                # assert isinstance(self.data_handler, dh.ServerGlobusDataHandler)  # If not, assume Globus not configured
                handler = ServerGlobusDataHandler(
                    self.reference_session_path,
                    {"input_files": [], "output_files": []},
                    one=self.one,
                )
                endpoint_id = next(
                    v["id"]
                    for k, v in handler.globus.endpoints.items()
                    if k.startswith("flatiron")
                )
                handler.globus.add_endpoint(
                    endpoint_id, label="flatiron_histology", root_path="/histology/"
                )
                handler.globus.mv(
                    "flatiron_histology",
                    "local",
                    [remote_file],
                    ["/".join(local_file.parts[-5:])],
                )
                assert local_file.exists(), (
                    f"failed to download {remote_file} to {local_file}"
                )
            except Exception as e:
                _logger.error(
                    f"Failed to download via Globus: {e}, attempting via HTTP"
                )
                remote_file = (
                    f"{self.one.alyx._par.HTTP_DATA_SERVER}/histology/" + remote_file
                )
                _logger.warning(f"Using HTTP download for {remote_file}")
                local_file = self.one.alyx.download_file(
                    remote_file, target_dir=local_file.parent
                )
                assert local_file.exists(), (
                    f"failed to download {remote_file} to {local_file}"
                )
        return local_file

    def register_reference_stacks(
        self,
        ref_stack_path: str | Path,
        ref_sess_ref_stack_path: str | Path,
        display: bool = False,  # TODO discuss what to do (where to store) the transform output
        save_plots: bool = False,
        save_transform: bool = False,
    ) -> ProjectiveTransform:
        # TODO refactor
        # naming is confusing: this is image registration and not dataset registration

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

        if display:
            # TODO find a different place and different namespace
            save_path = (
                self.session_path / "alf" / "_gr_reference_stack_registration.gif"
                if save_plots
                else None
            )

            z = 8  # FIXME this is almost certainly dataset specific and needs to be inferred
            # in some way, take peak brightness for example
            anim = inspect_registration_delta(
                img_data["stack"],
                img_data["target_stack"],
                img_data["aligned"],
                z=z,
                save_path=save_path,
                frames_per_second=1,  # 1s per frame in the saved gif
            )

        # plot keypoints vis
        if display:
            # TODO find a different place and different namespace
            save_path = (
                self.session_path / "alf" / "_gr_registration_keypoints.png"
                if save_plots
                else None
            )

            plot_keypoints(
                img_data,
                reg_details,
                z,
                save_path=save_path,
            )

        # save transform to json
        if save_transform:
            params = params.copy()
            # TODO find a better namespace, and place for this dataset
            save_path = self.session_path / "alf" / "_gr_registration_keypoints.json"
            for k, v in params.items():
                if isinstance(v, np.ndarray):
                    params[k] = v.tolist()
                elif isinstance(v, (np.float32, np.float64)):
                    params[k] = float(v)
                else:
                    params[k] = v

            with open(save_path, "w") as fp:
                json.dump(params, fp, indent=4)

        return ref_transform  # output signature might change

    def _run(self):
        return None

    def pipeline(self):
        # the first draft of the conditional pipeline

        # has reference image? yes / no
        self.load_reference_stack()

        # has brain surface? yes / no
        # self.load_brain_surface_points()

        # reference session has reference stack?
        self.load_reference_session_reference_stack()

        # reference session has histology?
        self.load_histology()
