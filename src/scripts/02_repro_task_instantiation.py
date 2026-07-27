# %%
from one.api import ONE
from plane2brain.ibl_task import ReprojectionTask

one = ONE()
eid = one.path2eid("SP058/2024-08-01/001")
eid_ref = one.path2eid("SP058/2024-08-14/001")
session_path = one.eid2path(eid)
repro_task = ReprojectionTask(
    session_path, reference_session_eid=eid_ref, one=one, FOV="FOV_00"
)

# %%
