# %%
from one.api import ONE
from plane2brain.ibl_task import ReprojectionTask
from pathlib import Path

LOCATION = "server"
BASE_FOLDER_LOCAL_SERVER = Path("/mnt/s0/Data/Subjects")

session_path = "SP058/2024-07-24/001"
reference_session_path = "SP058/2024-08-14/001"

# %%
match LOCATION:
    case "popeye":
        from deploy.iblsdsc import OneSdsc

        # requires the unmerged PR #121 https://github.com/int-brain-lab/iblscripts/pull/121
        one = OneSdsc(location="popeye")
        session_path = one.eid2path(one.path2eid(session_path))
        reference_session_path = one.eid2path(one.path2eid(reference_session_path))
    case "server":
        one = ONE()
        session_path = BASE_FOLDER_LOCAL_SERVER / session_path
        reference_session_path = BASE_FOLDER_LOCAL_SERVER / reference_session_path


repro_task = ReprojectionTask(
    session_path,
    reference_session_path=reference_session_path,
    one=one,
    location=LOCATION,
)

repro_task.setUp()
repro_task.verify_data_presence()
repro_task.pipeline()
# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots()
ds = 1
for uuid, _coords in repro_task.coords.items():
    points = _coords["mlapdv"]
    axes.plot(points[::ds, 0], points[::ds, 1], ".")
