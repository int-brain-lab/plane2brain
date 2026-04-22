from pathlib import Path
import subprocess 
import sys
session_paths = []

for session_path in session_paths:
    # for script, args in [("first.py", ["arg1"]), ("second.py", ["arg2"])]:
    # left in here as a template to compare runs
    script = "04_run_histo.py"

    result = subprocess.run([sys.executable, script, session_path])
    if result.returncode != 0:
        print(f"failed with code {result.returncode}")