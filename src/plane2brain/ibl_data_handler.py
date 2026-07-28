# %%
from pathlib import Path
from typing import Literal
from uuid import UUID

from ibllib.oneibl.data_handlers import (
    DataHandler,
    ExpectedDataset,
    LocalDataHandler,
    PopeyeDataHandler,
    ServerGlobusDataHandler,
)
from one.alf.path import ALFPath
from one.api import ONE

BASE_FOLDER = Path("/mnt/s0/Data/Subjects")
BASE_FOLDER_POPEYE = Path("/mnt/sdceph/users/ibl/data/Subjects")


def init_data_handler(
    eid: str | UUID,
    datasets: list[ALFPath],
    location: Literal["server", "local", "sdsc"],
    one: ONE,
) -> DataHandler:
    # based on location, return the corresponding data handler

    # build signature from list of files
    signature = {
        "input_files": [
            ExpectedDataset.input(dataset.parts[-1], "/".join(dataset.parts[:-1]), True)
            for dataset in datasets
        ],
        "output_files": [],
    }
    match location:
        case "local":
            session_path = one.eid2path(eid)
            dh = LocalDataHandler(session_path, signature, one)
        case "server":
            session_path = BASE_FOLDER / one.eid2path(eid).session_path_short()
            dh = ServerGlobusDataHandler(session_path, signature, one)
        case "popeye":
            session_path = BASE_FOLDER_POPEYE / one.eid2path(eid).session_path_short()
            dh = PopeyeDataHandler(session_path, signature, one)
    dh.setUp()  # download missing data
    dh.assert_expected_inputs(raise_error=True)  # Ensure everything present
    file_paths = [session_path / dataset for dataset in datasets]
    if not all(file_path.exists() for file_path in file_paths):
        for file in file_paths:
            if not file.exists():
                print(f"missing file: {file}")
        raise FileNotFoundError
    return dh, file_paths


# raw imaging metadata for the and the FOVs
# %%
one = ONE()
eid = one.ref2eid({"subject": "SP058", "date": "2024-08-14", "sequence": "001"})
dataset = ALFPath("raw_imaging_data_01/_ibl_rawImagingData.meta.json")
location = "local"
init_data_handler(eid, [dataset], "local", one=one)


# %%
# reference stack for the session

# reference stack of the reference session


# histology files

# brain surface


# # do thing...

# files_to_register = []
# dh.uploadData(files_to_register)
# dh.cleanUp()  # remove downloaded data
