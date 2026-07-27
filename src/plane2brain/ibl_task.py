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
    """Assign MLAPDV brain coordinates to the pixels of a mesoscope session's reference stack.

    The session's reference stack is registered to the reference stack of a *reference session*
    of the same subject. Only that reference session has been aligned to histology, so the
    resulting image transform carries the atlas coordinates over to the current session.

    Notes
    -----
    This task is under development: `_run` is still a no-op and the output signature is not final.
    """

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

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to `MesoscopeTask`; the first is the session path.
        FOV : str, optional
            Name of the field of view to process. If None, all FOVs are considered.
        reference_session_path : str or pathlib.Path
            Session path of the histology-aligned reference session of the same subject.
        one : one.api.ONE, optional
            An online ONE instance. A new one is created if not given.
        raw_imaging_collection : str, optional
            Raw imaging collection of this session. Inferred if not given.
        reference_session_raw_imaging_collection : str, optional
            Raw imaging collection of the reference session. Inferred if not given.
        **kwargs : dict
            Keyword arguments forwarded to `MesoscopeTask`.
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

        super().__init__(*args, one=self.one, **kwargs)

    @property
    def signature(self):
        """Build the task's expected input and output dataset signature."""
        I = ExpectedDataset.input
        O = ExpectedDataset.output
        # how does this fan out accross FOVs?

        signature = {
            "input_files": [
                I("_ibl_rawImagingData.meta.json", self.raw_imaging_collection, True),
                I(
                    "referenceImage.stack.tif",
                    f"{self.raw_imaging_collection}/reference",
                    True,
                ),
                I(
                    "referenceImage.meta.json",
                    f"{self.raw_imaging_collection}/reference",
                    True,
                ),
            ],
            "output_files": [],  # TODO ask about ExpectedDataset.output (or rather the lack of them)
        }

        return signature

    def validate_reference_session(self, reference_session_eid: str | UUID) -> UUID:
        """Check that the reference session belongs to the same subject as this session.

        Parameters
        ----------
        reference_session_eid : str or uuid.UUID
            Experiment ID of the candidate reference session.

        Returns
        -------
        uuid.UUID
            The validated experiment ID, unchanged.

        Raises
        ------
        AssertionError
            If the reference session was recorded from a different subject.
        """
        assert (
            self.one.eid2ref(reference_session_eid)["subject"]
            == self.one.eid2ref(self.eid)["subject"]
        ), "reference session does not match to this session: wrong subject"
        return reference_session_eid

    @staticmethod
    def infer_raw_imaging_collection(session_path: str | Path) -> str:
        """Find the raw imaging collection that contains a reference stack.

        Parameters
        ----------
        session_path : str or pathlib.Path
            Path of the session to search.

        Returns
        -------
        str
            Name of the raw imaging collection, e.g. 'raw_imaging_data_00'. If several
            collections hold a reference folder, the last one is returned.
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
        """Load the raw imaging metadata of this session.

        Returns
        -------
        dict
            Contents of `_ibl_rawImagingData.meta.json`.
        """
        metadata_filepath = (
            self.session_path
            / self.raw_imaging_collection
            / "_ibl_rawImagingData.meta.json"
        )
        return json.loads(Path(metadata_filepath).read_text(encoding="utf-8"))

    def load_reference_stack_metadata(self) -> dict:
        """Load the metadata of this session's reference stack.

        Returns
        -------
        dict
            Contents of `referenceImage.meta.json`.

        Raises
        ------
        AssertionError
            If not exactly one metadata file is found.
        """
        reference_collection = (
            self.session_path / self.raw_imaging_collection / "reference"
        )
        filepath = list(reference_collection.glob("*referenceImage.meta*"))

        assert len(filepath) == 1
        return json.loads(Path(filepath[0]).read_text(encoding="utf-8"))

    def get_reference_stack_path(self) -> Path:
        """Return the path to the reference stack of this session.

        Returns
        -------
        pathlib.Path
            Path of the `referenceImage.stack` tif.
        """
        # TODO check whether this can be folded into the method below
        return self._get_ref_stack_path(self.session_path, self.raw_imaging_collection)

    def get_reference_session_reference_stack_path(self) -> Path:
        """Return the path to the reference stack of the reference session.

        On popeye the stack is not directly readable and is symlinked into the task
        quarantine folder first.

        Returns
        -------
        pathlib.Path
            Path of the `referenceImage.stack` tif, or of its symlink when on popeye.
        """
        if self.location == "popeye":
            return self._symlink_reference_session_reference_stack()
        else:
            return self._get_ref_stack_path(
                self.reference_session_path,
                self.reference_session_raw_imaging_collection,
            )

    def _get_ref_stack_path(
        self, session_path: Path, raw_imaging_collection: str
    ) -> Path:
        """Find the reference stack file within a session's raw imaging collection.

        Parameters
        ----------
        session_path : pathlib.Path
            Path of the session to search.
        raw_imaging_collection : str
            Name of the raw imaging collection holding the reference folder.

        Returns
        -------
        pathlib.Path
            Path of the `referenceImage.stack` tif.

        Raises
        ------
        AssertionError
            If not exactly one reference stack is found.
        """
        path = session_path / raw_imaging_collection / "reference"
        filepath = list(path.glob("*referenceImage.stack*"))

        assert len(filepath) == 1, (
            f"number of reference stacks is: {len(filepath)} - and has to be exactly 1"
        )
        return filepath[0]

    def load_reference_stack(self) -> np.ndarray:
        """Load the reference stack of this session.

        Returns
        -------
        numpy.ndarray
            Image stack with shape (Z, Y, X).
        """
        return tifffile.imread(self.get_reference_stack_path())

    def load_reference_session_reference_stack(self) -> np.ndarray:
        """Load the reference stack of the reference session.

        Returns
        -------
        numpy.ndarray
            Image stack with shape (Z, Y, X).
        """
        return tifffile.imread(self.get_reference_session_reference_stack_path())

    def _symlink_reference_session_reference_stack(self) -> None:
        """Symlink the reference session's reference stack into the popeye quarantine folder.

        Returns
        -------
        pathlib.Path
            Path of the created symlink. An existing symlink is replaced.

        Raises
        ------
        AssertionError
            If not exactly one reference stack is found in the source folder.
        """
        base_folder = Path("/mnt/sdceph/users/ibl/data/quarantine/tasks")
        path_short = self.one.eid2path(self.reference_session_eid).session_path_short()
        lab = self.one.get_details(self.reference_session_path)["lab"]
        symlinked_reference_stack = (
            base_folder
            / type(self).__name__
            / lab
            / "Subjects"
            / path_short
            / self.reference_session_raw_imaging_collection
            / "reference"
            / "referenceImage.stack.tif"
        )

        _session_folder = (
            Path("/mnt/sdceph/users/ibl/data")
            / lab
            / "Subjects"
            / path_short
            / self.reference_session_raw_imaging_collection
            / "reference"
        )

        reference_stack_path = list(_session_folder.glob("*referenceImage.stack.*.tif"))
        assert len(reference_stack_path) == 1, (
            "none or multiple referenceImage stacks found during symlinking"
        )

        if symlinked_reference_stack.exists():
            symlinked_reference_stack.unlink()
        symlinked_reference_stack.parent.mkdir(parents=True, exist_ok=True)
        symlinked_reference_stack.symlink_to(reference_stack_path[0])
        return symlinked_reference_stack

    def load_histology(self) -> np.ndarray:
        """Load the MLAPDV coordinates of the reference session's reference image.

        Returns
        -------
        numpy.ndarray
            Array with shape (h, w, 3) holding the (ml, ap, dv) coordinates in μm of each
            pixel of the reference session's reference image.
        """
        atlas = MRITorontoAtlas(res_um=25)
        local_histo_path = self._get_atlas_registered_reference_mlap()
        ccf_idx = np.load(local_histo_path)

        # flip the ap axis to match the atlas volume orientation
        ccf_idx[:, :, 1] = np.abs(
            ccf_idx[:, :, 1].astype("int64") - atlas.label.shape[0]
        ).astype(ccf_idx.dtype)
        # NB: these coordinates belong to the reference session, i.e. the one aligned to histology
        ref_img_histo_mlapdv = (
            atlas.ccf2xyz(ccf_idx * atlas.res_um, ccf_order="mlapdv") * 1e6
        )  # m -> μm
        return ref_img_histo_mlapdv

    def _load_brain_surface_points_from_metadata(self) -> dict:
        """Read the brain surface points from the reference stack metadata.

        Returns
        -------
        dict
            Mapping with a 'points' key holding the brain surface points.

        Raises
        ------
        KeyError
            If the metadata does not contain any points.
        """
        ref_img_meta = self.load_reference_stack_metadata()
        return {"points": ref_img_meta["points"]}

    def _load_brain_surface_points_from_file(self) -> dict:
        """Read the brain surface points from the dedicated points file.

        Returns
        -------
        dict
            Contents of `referenceImage.points.json`.

        Raises
        ------
        FileNotFoundError
            If no points file exists.
        ValueError
            If more than one points file exists.
        """
        ref_points_path = list(
            (self.session_path / self.raw_imaging_collection / "reference").glob(
                "referenceImage.points.json"
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
        """Load the brain surface points, from either the points file or the stack metadata.

        Both sources are tried. If they exist and disagree, `prefer` decides which one wins;
        if only the non-preferred source exists, it is used and a warning is logged.

        Parameters
        ----------
        prefer : {'metadata', 'file'}
            Source to use when both exist and their contents differ.

        Returns
        -------
        dict
            The brain surface points.

        Raises
        ------
        ValueError
            If neither source provides points, or if `prefer` is not a valid source.
        """
        # from the points file
        try:
            brain_surface_points_file = self._load_brain_surface_points_from_file()
        except FileNotFoundError:
            brain_surface_points_file = None

        # from the reference stack metadata
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
                    "using metadata as a non-preferred source of brain surface points"
                )
            return brain_surface_points_meta
        if brain_surface_points_file is not None and brain_surface_points_meta is None:
            if prefer == "metadata":
                _logger.warning(
                    "using points.json file as a non-preferred source of brain surface points"
                )
            return brain_surface_points_file

    def _get_atlas_registered_reference_mlap(self, clobber=False):
        """Download the aligned reference stack Allen atlas indices.

        This is the file created by the histology pipeline, one per subject.
        This file contains the Allen atlas image volume indices for each pixel of the reference stack.

        On popeye the file is read in place from the histology folder. Elsewhere it is fetched
        with a data handler, falling back to a direct Globus transfer and then to HTTP.

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

        Raises
        ------
        AssertionError
            If the file could neither be transferred via Globus nor downloaded via HTTP.
        """
        reference_collection = (
            self.reference_session_raw_imaging_collection + "/reference"
        )

        if self.location == "popeye":
            lab = self.one.get_details(self.reference_session_path)["lab"]
            base_folder = Path(f"/mnt/sdceph/users/ibl/data/histology/{lab}")
            local_file = (
                base_folder
                / self.reference_session_path.session_path_short()
                / "referenceImage.mlapdv.npy"
            )
            return local_file

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
                self.reference_session_raw_imaging_collection, "reference"
            ),
        )
        # NB: The local reference folder is expected to exist after handler.setUp()
        local_file = (
            self.reference_session_path
            / self.reference_session_raw_imaging_collection
            / "reference"
            / "referenceImage.mlapdv.npy"
        )

        if not local_file.exists():
            _logger.warning("getting histology via data handler failed!")

        if clobber or not local_file.exists():
            _logger.info("attempting to download histology file from flatiron")
            assert self.one, "ONE required"
            local_file.parent.mkdir(parents=True, exist_ok=True)
            lab = self.one.get_details(self.reference_session_path)["lab"]
            remote_file = f"{lab}/{self.reference_session_path.session_path_short()}/{local_file.name}"
            try:
                # the histology folder is not part of the standard endpoints, so mount it as its own
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
        display: bool = False,
        save_plots: bool = False,
        save_transform: bool = False,
    ) -> ProjectiveTransform:
        """Find the image transform mapping this session's reference stack onto the reference session's.

        Note that this is *image* registration, not dataset registration to Alyx.

        Parameters
        ----------
        ref_stack_path : str or pathlib.Path
            Path of this session's reference stack, the stack that is being moved.
        ref_sess_ref_stack_path : str or pathlib.Path
            Path of the reference session's reference stack, the target of the registration.
        display : bool
            If True, build the registration delta animation and the keypoint plot.
        save_plots : bool
            If True, write those plots to the session's alf folder. Requires `display`.
        save_transform : bool
            If True, write the transform parameters and their quality metric to a json file
            in the session's alf folder.

        Returns
        -------
        skimage.transform.ProjectiveTransform
            Transform mapping coordinates of this session's stack onto the reference session's.
        """
        # TODO refactor, and settle on where the transform output should be stored

        # load the stacks of this session and of the reference session
        img_data = {}
        for key, path in zip(
            ["stack", "target_stack"],
            [ref_stack_path, ref_sess_ref_stack_path],
        ):
            # swap Y and X to bring both stacks into the same convention
            img_data[key] = np.swapaxes(tifffile.imread(path), 1, 2)
            # img_data[key] = preprocess_vasculature(img_data[key]).astype("int16")

        # find and apply transform
        ref_transform, reg_details = register_stacks(
            img_data["stack"],
            img_data["target_stack"],
            transform_type="euclidean",
            return_details=True,
        )
        # NB: 'affine' is worse overall, but better when registering a single plane

        img_data["aligned"] = apply_transform(img_data["stack"], ref_transform)

        # score the transform by normalized cross-correlation, before and after
        ncc_before = evaluate(img_data["stack"], img_data["target_stack"])
        ncc_after = evaluate(img_data["aligned"], img_data["target_stack"])

        params = {
            "translation": ref_transform.translation,
            "rotation": ref_transform.rotation,
            "quality_ncc": ncc_after.mean(),
            "warp_matrix": np.array(ref_transform),
            "method": "orb_robust",
        }

        # plot the before/after delta of the registration
        if display:
            # TODO find a different place and a different namespace for this plot
            save_path = (
                self.session_path / "alf" / "_gr_reference_stack_registration.gif"
                if save_plots
                else None
            )

            # FIXME this plane is almost certainly dataset specific and should be inferred,
            # for example from the plane of peak brightness
            z = 8
            anim = inspect_registration_delta(
                img_data["stack"],
                img_data["target_stack"],
                img_data["aligned"],
                z=z,
                save_path=save_path,
                frames_per_second=1,  # 1s per frame in the saved gif
            )

        # plot the keypoint matches that the transform was fit on
        if display:
            # TODO find a different place and a different namespace for this plot
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
            # TODO find a better namespace and place for this dataset
            save_path = self.session_path / "alf" / "_gr_registration_keypoints.json"
            # cast numpy types to their python equivalents for json serialization
            for k, v in params.items():
                if isinstance(v, np.ndarray):
                    params[k] = v.tolist()
                elif isinstance(v, (np.float32, np.float64)):
                    params[k] = float(v)
                else:
                    params[k] = v

            with open(save_path, "w") as fp:
                json.dump(params, fp, indent=4)

        return ref_transform  # the output signature might still change

    def _run(self):
        """Run the task and return the paths of the registered output datasets.

        Not implemented yet.
        """
        return None

    def verify_data_presence(self):
        """Check that all inputs the task needs can be loaded.

        Each loader raises if its input is missing, so a silent return means the session is
        ready to be processed.
        """
        # this session has a reference stack
        self.load_reference_stack()

        # this session has brain surface points
        self.load_brain_surface_points()

        # the reference session has a reference stack
        self.load_reference_session_reference_stack()

        # the reference session has histology
        self.load_histology()
