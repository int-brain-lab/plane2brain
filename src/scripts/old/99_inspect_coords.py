# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

session_folder = Path.home() / "ibl_scratch/repro/SP058/2024-07-24/001"
fov_folders = sorted((session_folder / "alf").glob("FOV_*"))

# %%
fov_data = {}
for fov_folder in fov_folders:
    fov = fov_folder.parts[-1]
    fov_data[fov] = np.load(
        fov_folder
        # / "mpciMeanImage.mlapdv_repro_ransac_ro_100_on_surface_interp_smooth_s2s_apxy.npy"
        # / "mpciROIs.mlapdv_repro_ransac_ro_25_on_surface_interp_smooth_s2s.npy"
        # / "mpciROIs.mlapdv_repro_ransac_ro_25_on_surface_interp_smooth_s2s_popeye_variant.npy"
        # / "mpciMeanImage.mlapdv_repro_ransac_ro_25_on_surface_interp_stevens_variant_1.npy"
        # / "mpciMeanImage.mlapdv_repro_ransac_ro_25_on_surface_interp_stevens_variant_2.npy"
        # / "mpciMeanImage.mlapdv_repro_ransac_ro_25_on_surface_interp_old.npy"
        / "mpciMeanImage.mlapdv_repro_ransac_ro_25_on_surface_interp_old_whiterussian.npy"
    )

ds = 1
fig, axes = plt.subplots()
for fov, px in fov_data.items():
    axes.plot(px[::ds, 0], px[::ds, 1], ".")

# %% with variants
fig, axes = plt.subplots()

variants = ["old_whiterussian", "old_flatiron", "new_run_1", "new_run_2"]
import seaborn as sns

variant_colors = dict(zip(variants, sns.color_palette("tab10", n_colors=len(variants))))

fov_data = {}
for variant in variants:
    fov_data[variant] = {}
    for fov_folder in fov_folders:
        fov = fov_folder.parts[-1]
        fov_data[variant][fov] = np.load(
            fov_folder
            / f"mpciMeanImage.mlapdv_repro_ransac_ro_25_indexing_{variant}.npy"
        )

    ds = 1
    for i, (fov, px) in enumerate(fov_data[variant].items()):
        label = variant if i == 0 else None
        axes.plot(
            px[::ds, 0],
            px[::ds, 1],
            ".",
            color=variant_colors[variant],
            alpha=0.2,
            label=label,
        )
axes.legend()
# fig.suptitle(variant)

# %%
