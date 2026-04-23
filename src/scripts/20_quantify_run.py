# %%
import sys

sys.path.append("/home/ibladmin/Documents/georg/code/mesoscope/")
import chronic_data_loader
from pathlib import Path
from one.api import ONE
import pandas as pd


with open(Path(__file__).parent / "projected_sessions.txt", "r") as fH:
    session_paths = [line.strip() for line in fH.readlines()]

one = ONE()
eids = [one.path2eid(path) for path in session_paths]

# %%
chronic_data_ = chronic_data_loader.load_chronic_imaging(
    eids,
    one=one,
    location="server",
    mlapdv_file="mpciROIs.mlapdv.npy",
    metadata_only=True,
)

# %%
chronic_data_est = chronic_data_loader.load_chronic_imaging(
    eids,
    one=one,
    location="server",
    mlapdv_file="mpciROIs.mlapdv_estimate.npy",
    metadata_only=True,
)

# %%
chronic_data_repro = chronic_data_loader.load_chronic_imaging(
    eids,
    one=one,
    location="server",
    mlapdv_file="mpciROIs.mlapdv_histo_projection.npy",
    metadata_only=True,
)

# %%
chronic_data_repro_2 = chronic_data_loader.load_chronic_imaging(
    eids,
    one=one,
    location="server",
    mlapdv_file="mpciROIs.mlapdv_histo_projection_2.npy",
    metadata_only=True,
)


# %%
# use a function to get these
fovs = ["FOV_00", "FOV_01", "FOV_02", "FOV_03", "FOV_04", "FOV_05", "FOV_06", "FOV_07"]

# %%

# select_chronic_data_by_roicat_UCIDs
# %%
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

chronic_data = chronic_data_repro_2

common_UCIDs = {}
for fov in fovs:
    sets = [set(chronic_data[eid][fov]["roicat_UCID"].dropna().values) for eid in eids]
    common_UCIDs[fov] = list(set.intersection(*sets))

mlapdv_chronic = {}
for eid in eids:
    mlapdv_session = []
    for fov in fovs:
        meta = chronic_data[eid][fov].set_index("roicat_UCID")
        mlapdv_fov = meta.loc[common_UCIDs[fov]][["ml", "ap", "dv"]]
        mlapdv_session.append(mlapdv_fov)
    mlapdv_session = pd.concat(mlapdv_session, axis=0)
    mlapdv_chronic[eid] = mlapdv_session

# %% the pairwise
df = pd.DataFrame(index=eids, columns=eids)
for eid_a, eid_b in combinations(eids, 2):
    a = mlapdv_chronic[eid_a].values
    b = mlapdv_chronic[eid_b].values
    df.loc[eid_a, eid_b] = np.median(np.sqrt(np.sum((a - b) ** 2, 1)))


sns.heatmap(df.astype("float"), vmin=0, vmax=50)

# %% the 3d scatterplot
from plane2brain import plotters, atlas

atlas = atlas.ProjectionAtlas()
brain_surface_points = atlas.get_surface_points()
axes = plotters.plot_brain_surface_points(brain_surface_points, ds=50)
# axes = plt.figure().add_subplot(projection="3d")
for eid in eids:
    d = mlapdv_chronic[eid][["ml", "ap", "dv"]].values
    axes.scatter3D(*d.T, s=1)

# %% the 2d scatterplot
fig, axes = plt.subplots()
for eid in eids:
    d = mlapdv_chronic[eid][["ml", "ap"]].values
    axes.scatter(*d.T, s=1)

# %%
