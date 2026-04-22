from pathlib import Path
import subprocess 
import sys

BASE_FOLDER = Path('/mnt/s0/Data/Subjects')\

session_paths = [
    'SP058/2024-06-18/002',
    'SP058/2024-06-19/001',
    'SP058/2024-06-20/001',
    'SP058/2024-06-21/001',
    'SP058/2024-06-25/001',
    'SP058/2024-06-26/001',
    'SP058/2024-06-28/001',
    'SP058/2024-07-02/001',
    'SP058/2024-07-04/001',
    'SP058/2024-07-05/001',
    'SP058/2024-07-09/001',
    'SP058/2024-07-10/001',
    'SP058/2024-07-12/001',
    'SP058/2024-07-16/001',
    'SP058/2024-07-18/001',
    'SP058/2024-07-19/001',
    'SP058/2024-07-23/001',
    'SP058/2024-07-24/001',
    'SP058/2024-07-25/001',
]

for session_path in session_paths:
    print(f" --- processing {session_path} --- ")
    # for script, args in [("first.py", ["arg1"]), ("second.py", ["arg2"])]:
    # left in here as a template to compare runs
    script = "04_run_histo.py"
    result = subprocess.run([sys.executable, script, BASE_FOLDER / session_path])
    if result.returncode != 0:
        print(f"failed with code {result.returncode}")
    else:
        print(f"done.")
        