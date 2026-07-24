"""suite2p specific code"""

from pathlib import Path

import numpy as np


# this is "kind of" ibl specific
def data_loader(
    stat_paths: dict[str, Path],
    fov_map: dict[str, str],
    dims: tuple[str, str] = ("X", "Y"),
) -> dict[str, np.ndarray]:
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
