# %%
from one.api import ONE
from plane2brain.ibl_task import ReprojectionTask
from pathlib import Path

one = ONE()
session_path = Path("/mnt/s0/Data/Subjects/SP058/2024-08-01/001")
reference_session_path = Path("/mnt/s0/Data/Subjects/SP058/2024-08-14/001")
repro_task = ReprojectionTask(
    session_path,
    reference_session_path=reference_session_path,
    FOV="FOV_00",
    one=one,
)

repro_task.setUp()
repro_task.pipeline()
# %%
