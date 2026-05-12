"""suite2p specific code"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np


# this is "kind of" ibl specific
def data_loader(
    stat_paths: Dict[str, Path],
    fov_map: Dict[str, str],
    dims: Tuple[str, str] = ("X", "Y"),
) -> Dict[str, np.ndarray]:
    coords = {}
    for fov_name, path in stat_paths.items():
        stat = np.load(path, allow_pickle=True)
        fov_uuid = fov_map[fov_name]
        coords[fov_uuid] = np.stack(
            [(np.average(s["xpix"]), np.average(s["ypix"])) for s in stat],
        )
        if dims == ("Y", "X"):
            coords[fov_uuid] = coords[fov_uuid][:, ::-1]
    return coords
