from pathlib import Path
import subprocess
import sys

BASE_FOLDER = Path("/mnt/s0/Data/Subjects")
session_paths = [
    # "SP058/2024-06-18/002", # multiple ref stacks
    "SP058/2024-06-19/001",
    "SP058/2024-06-20/001",
    "SP058/2024-06-21/001",
    "SP058/2024-06-25/001",
    "SP058/2024-06-26/001",
    # "SP058/2024-06-28/001", # fails bc of different z stack
    "SP058/2024-07-02/001",
    "SP058/2024-07-04/001",
    "SP058/2024-07-05/001",
    "SP058/2024-07-09/001",
    "SP058/2024-07-10/001",
    "SP058/2024-07-12/001",
    "SP058/2024-07-16/001",
    "SP058/2024-07-18/001",
    "SP058/2024-07-19/001",
    "SP058/2024-07-23/001",
    "SP058/2024-07-24/001",
    "SP058/2024-07-25/001",
]

# NOTE early sessions are much "worse" than the later ones
# session_paths = [
#     "SP058/2024-06-19/001",
#     "SP058/2024-06-20/001",
#     "SP058/2024-06-21/001",
#     "SP058/2024-06-25/001",
#     "SP058/2024-06-26/001",
# ]

# LATE
# session_paths = [
#     "SP058/2024-07-18/001",
#     "SP058/2024-07-19/001",
#     "SP058/2024-07-23/001",
#     "SP058/2024-07-24/001",
#     "SP058/2024-07-25/001",
# ]

for i, session_path in enumerate(session_paths):
    print(f" --- processing {i + 1}/{len(session_paths)} : {session_path} --- ")
    script = Path(__file__).parent / "09_run_histo_ransac_reorder.py"
    result = subprocess.run(
        [sys.executable, script, "--session_path", BASE_FOLDER / session_path]
    )
    if result.returncode != 0:
        print(f"failed with code {result.returncode}")
    else:
        print("done.")
