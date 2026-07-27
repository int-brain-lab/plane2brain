# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

session_folder = Path("/mnt/s0/Data/Subjects/SP058/2024-07-24/001")
fov_folders = sorted(list((session_folder / "alf").glob("FOV_*")))

# %%
fov_data = {}
for fov_folder in fov_folders:
    fov = fov_folder.parts[-1]
    fov_data[fov] = np.load(
        fov_folder
        # / "mpciMeanImage.mlapdv_repro_ransac_ro_100_on_surface_interp_smooth_s2s_apxy.npy"
        # / "mpciROIs.mlapdv_repro_ransac_ro_100_on_surface_interp_smooth_s2s_apxy.npy"
        # / "mpciROIs.mlapdv_repro_ransac_ro_25_on_surface_interp_smooth_s2s_verify.npy"
        # / 'mpciMeanImage.mlapdv_repro_ransac_ro_25_on_surface_interp_smooth_s2s_verify.npy'
        # / 'mpciMeanImage.mlapdv_repro_ransac_ro_25_on_surface_interp_smooth_s2s_with_popeye_iblpy.npy'
        # / 'mpciROIs.mlapdv_repro_ransac_ro_25_on_surface_interp_smooth_s2s_with_popeye_iblpy.npy'
        / 'mpciMeanImage.mlapdv_repro_ransac_ro_25_on_surface_interp_smooth_s2s_with_popeye_iblpy.npy'
    )

ds = 1
fig, axes = plt.subplots()
for fov, px in fov_data.items():
    axes.plot(px[::ds, 0], px[::ds, 1], ".")

# %%
