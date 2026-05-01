# %%
import sys

sys.path.append("/home/ibladmin/Documents/georg/code/mesoscope/")
import chronic_data_loader
from pathlib import Path
from one.api import ONE
import pandas as pd


with open(Path(__file__).parent / "projected_sessions.txt", "r") as fH:
    session_paths = [line.strip() for line in fH.readlines()]

# session_paths = session_paths[:3]

one = ONE()
eids = [one.path2eid(path) for path in session_paths]


# %% this is the faster iteration tmp
chronic_data_repro_surface = chronic_data_loader.load_chronic_imaging(
    eids,
    one=one,
    location="server",
    mlapdv_file="mpciROIs.mlapdv_histo_projection_surface_2.npy",
    metadata_only=True,
)

# %%
chronic_data_repro_mlapdv = chronic_data_loader.load_chronic_imaging(
    eids,
    one=one,
    location="server",
    mlapdv_file="mpciROIs.mlapdv_histo_projection_mlapdv_2.npy",
    metadata_only=True,
)

# %%

# select_chronic_data_by_roicat_UCIDs
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# chronic_data = chronic_data_est
chronic_data = chronic_data_repro_mlapdv
fovs = ["FOV_00", "FOV_01", "FOV_02", "FOV_03", "FOV_04", "FOV_05", "FOV_06", "FOV_07"]
fovs = ["FOV_05"]

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

# the 2d scatterplot
# color code by day
# sort eids by day
start_times = [one.alyx.rest("sessions", "read", eid)["start_time"] for eid in eids]
# np.argsort(start_times)

colors = dict(zip(eids, sns.color_palette("viridis", n_colors=len(eids))))

fig, axes = plt.subplots(ncols=2)
for eid in eids:
    d = mlapdv_chronic[eid][["ml", "ap"]].values
    axes[0].scatter(*d.T, s=1, c=colors[eid])
axes[0].set_xlabel("ml")
axes[0].set_ylabel("ap")
for eid in eids:
    d = mlapdv_chronic[eid][["ml", "dv"]].values
    axes[1].scatter(*d.T, s=1, c=colors[eid])
axes[1].set_xlabel("ml")
axes[1].set_ylabel("dv")
for ax in axes:
    ax.set_aspect("equal")
fig.suptitle("all sessions all tracked ROIs")
sns.despine(fig)

# %%
fig, axes = plt.subplots()
dv_values = [np.average(mlapdv_chronic[eid]["dv"]) for eid in eids]
from datetime import datetime

start_times_ = [datetime.fromisoformat(d) for d in start_times]
axes.scatter(start_times_, dv_values, c=colors.values())
import matplotlib.dates as mdates

ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))

# 3) Rotate labels and align
plt.setp(axes.get_xticklabels(), rotation=45, ha="right")  # or ha='center' / 'left'
sns.despine(fig)
axes.set_ylabel("average dv")

# %% the pairwise
df = pd.DataFrame(index=eids, columns=eids)
for eid_a, eid_b in combinations(eids, 2):
    a = mlapdv_chronic[eid_a].values
    b = mlapdv_chronic[eid_b].values
    df.loc[eid_a, eid_b] = np.mean(np.sqrt(np.sum((a - b) ** 2, 1)))

sns.heatmap(df.astype("float"), vmin=0, vmax=70)
fig = plt.gcf()
fig.suptitle("euclidean distance, averaged over ROIs, all sessions pairwise")

np.nanmean(df.values.flatten())
# %% the 3d scatterplot
from plane2brain import plotters, atlas

atlas = atlas.ProjectionAtlas()
brain_surface_points = atlas.get_surface_points()
axes = plotters.plot_brain_surface_points(brain_surface_points, ds=50)
# axes = plt.figure().add_subplot(projection="3d")
for eid in eids:
    d = mlapdv_chronic[eid][["ml", "ap", "dv"]].values
    axes.scatter3D(*d.T, s=1)

# %%
