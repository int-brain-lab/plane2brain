# %%
import sys

sys.path.append("/home/ibladmin/Documents/georg/code/mesoscope/")
import sys
from itertools import combinations
from pathlib import Path

import chronic_data_loader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from one.api import ONE

SAVE = False
# WHEN = "early"
# WHEN = "late"
WHEN = "all"
# WHICH = "surface"
WHICH = "deep"
# WHICH = "all"

# %%
with open(Path(__file__).parent / "projected_sessions.txt", "r") as fH:
    session_paths = [line.strip() for line in fH]
match WHEN:
    case "early":
        session_paths = session_paths[:5]
    case "late":
        session_paths = session_paths[-5:]
    case "all":
        ...

one = ONE()
eids = [one.path2eid(path) for path in session_paths]

# %%
match WHICH:
    case "deep":
        save_keys = [
            "indexing",
            # "on_surface_interp",
            "interp",
            # "on_surface_interp_s2s",
            "interp_s2s",
            # "on_surface_interp_smooth_s2s",
            "interp_smooth_s2s",
            # "on_surface_interp_smooth_s2s_apxy",
            "interp_smooth_s2s_apxy",
            "interp_smooth_s2s_apxyz",
        ]
    case "surface":
        save_keys = [
            # "indexing",
            "on_surface_interp",
            # "interp",
            "on_surface_interp_s2s",
            # "interp_s2s",
            "on_surface_interp_smooth_s2s",
            # "interp_smooth_s2s",
            "on_surface_interp_smooth_s2s_apxy",
            # "interp_smooth_s2s_apxy",
            # "interp_smooth_s2s_apxyz",
        ]
    case "all":
        save_keys = [
            "indexing",
            "on_surface_interp",
            "interp",
            "on_surface_interp_s2s",
            "interp_s2s",
            "on_surface_interp_smooth_s2s",
            "interp_smooth_s2s",
            "on_surface_interp_smooth_s2s_apxy",
            "interp_smooth_s2s_apxy",
            "interp_smooth_s2s_apxyz",
        ]

# %%
save_keys = [
    # "indexing",
    "on_surface_interp",
    # "interp",
    "on_surface_interp_smooth",
    # "interp_smooth",
    "on_surface_interp_smooth_s2s",
    # "interp_smooth_s2s",
    "on_surface_interp_smooth_s2s_apxy",
    # "interp_smooth_s2s_apxy",
    # "interp_smooth_s2s_apxyz",
]

# %%
save_keys = [
    # "indexing",
    # "on_surface_interp",
    "interp",
    # "on_surface_interp_smooth",
    "interp_smooth",
    # "on_surface_interp_smooth_s2s",
    "interp_smooth_s2s",
    # "on_surface_interp_smooth_s2s_apxy",
    "interp_smooth_s2s_apxy",
    "interp_smooth_s2s_apxyz",
]

# %% load
chronic_data = {}
# fovs = ["FOV_04"]
fovs = ["FOV_00", "FOV_01", "FOV_02", "FOV_03", "FOV_04", "FOV_05", "FOV_06", "FOV_07"]

for key in save_keys:
    chronic_data[key] = chronic_data_loader.load_chronic_imaging(
        eids,
        FOVs=fovs,
        one=one,
        location="server",
        mlapdv_file=f"mpciROIs.mlapdv_repro_ransac_ro_100_{key}.npy",
        metadata_only=True,
    )

# %%
res = []
mlapdv_chronic = {}

for key in save_keys:
    common_UCIDs = {}
    mlapdv_chronic[key] = {}
    for fov in fovs:
        sets = [
            set(chronic_data[key][eid][fov]["roicat_UCID"].dropna().values)
            for eid in eids
        ]
        common_UCIDs[fov] = list(set.intersection(*sets))

    for eid in eids:
        mlapdv_session = []
        for fov in fovs:
            meta = chronic_data[key][eid][fov].set_index("roicat_UCID")
            mlapdv_fov = meta.loc[common_UCIDs[fov]][["ml", "ap", "dv"]]
            mlapdv_session.append(mlapdv_fov)
        mlapdv_session = pd.concat(mlapdv_session, axis=0)
        n_cells = mlapdv_session.shape[0]
        mlapdv_chronic[key][eid] = mlapdv_session

    # the pairwise
    for eid_a, eid_b in combinations(eids, 2):
        a = mlapdv_chronic[key][eid_a].values
        b = mlapdv_chronic[key][eid_b].values
        vals = {
            "key": key,
            "eid_a": eid_a,
            "eid_b": eid_b,
            "mean_dist": np.mean(np.sqrt(np.sum((a - b) ** 2, 1))),
        }
        res.append(vals)

df = pd.DataFrame(res, columns=["key", "eid_a", "eid_b", "mean_dist"])
if SAVE:
    df.to_csv(Path(__file__).parent / "quantify_all_runs.csv", index=None)
    sys.exit()

# pd.read_csv(Path(__file__).parent / 'quantify_all_runs.csv')

# %% barplor quant
# order = save_keys[1:]
order = save_keys
# colors = dict(zip(order, sns.color_palette('tab10',n_colors=len(save_keys))))
# colors = list(np.array(sns.color_palette('viridis',n_colors=len(save_keys))))
colors = sns.color_palette("tab10", n_colors=(len(save_keys) + 1))[1:]
fig, axes = plt.subplots(figsize=[3, 6])
sns.barplot(df, y="mean_dist", hue="key", hue_order=order, ax=axes, palette=colors)
sns.despine(fig)
# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1)
plt.tight_layout()
plt.show()

# %%
# the 2d scatterplot - only top view
# color code by day
# colors = dict(zip(eids, sns.color_palette("viridis", n_colors=len(eids))))
# color by cells
colors = {eid: sns.color_palette("tab10", n_colors=n_cells) for eid in eids}
for key in save_keys:
    fig, axes = plt.subplots(figsize=[6, 4])
    for eid in eids:
        d = mlapdv_chronic[key][eid][["ml", "ap"]].values
        axes.scatter(*d.T, s=4, c=colors[eid], alpha=1.0)
    for eid in eids:
        d = mlapdv_chronic[key][eid][["ml", "ap"]].values
        axes.scatter(*d[0, :], s=45, c="magenta", alpha=1.0)
        axes.scatter(
            *d[0, :],
            s=45,
            alpha=1.0,
            edgecolor="black",
            facecolors="none",
            linewidths=1.5,
        )

    axes.set_xlabel("ml")
    axes.set_ylabel("ap")
    axes.set_aspect("equal")
    v = df.groupby("key").get_group(key)["mean_dist"].mean()
    fig.suptitle(f"{key}, {v:.2f}")
    sns.despine(fig)
    fig.savefig(f"/mnt/s0/Data/georg_tmp/repro_{key}.png", dpi=300)


# %%
# the 2d scatterplot
# color code by day
# colors = dict(zip(eids, sns.color_palette("viridis", n_colors=len(eids))))
# color by cells
colors = {eid: sns.color_palette("tab10", n_colors=n_cells) for eid in eids}

for key in save_keys:
    fig, axes = plt.subplots(ncols=2)
    for eid in eids:
        d = mlapdv_chronic[key][eid][["ml", "ap"]].values
        axes[0].scatter(*d.T, s=1, c=colors[eid])
    axes[0].set_xlabel("ml")
    axes[0].set_ylabel("ap")
    for eid in eids:
        d = mlapdv_chronic[key][eid][["ml", "dv"]].values
        axes[1].scatter(*d.T, s=1, c=colors[eid])
    axes[1].set_xlabel("ml")
    axes[1].set_ylabel("dv")
    for ax in axes:
        ax.set_aspect("equal")
    v = df.groupby("key").get_group(key)["mean_dist"].mean()
    fig.suptitle(f"all sessions all tracked ROIs, {key}, {v:.2f}")
    sns.despine(fig)

# %% debugging: why are deep coordinates systematically
# further from each other?
fig, axes = plt.subplots()
df.groupby(["key", "eid_a", "eid_b"]).get_group(("interp", eid_a, eid_b))
# mlapdv_chronic['interp'][eid_b]

a = mlapdv_chronic[key][eid_a].values
b = mlapdv_chronic[key][eid_b].values

dists = np.sqrt(np.sum((a - b) ** 2, 1))
# %%

f"mpciROIs.mlapdv_repro_ransac_ro_100_{key}.npy"
# with keys being any of the following:
keys = [
    "indexing",
    "on_surface_interp",
    "interp",
    "on_surface_interp_smooth",
    "interp_smooth",
    "on_surface_interp_smooth_s2s",
    "interp_smooth_s2s",
    "on_surface_interp_smooth_s2s_apxy",
    "interp_smooth_s2s_apxy",
    "interp_smooth_s2s_apxyz",
]
# 'on_surface' denotes the surface version


# %% inspecting the distribution of distances on a cell by cell level
dfs = []
key = "interp_smooth_s2s_apxy"
for eid_a, eid_b in combinations(eids, 2):
    df = pd.DataFrame(columns=["ucid", "eid_a", "eid_b", "dist"])
    a = mlapdv_chronic[key][eid_a].values
    b = mlapdv_chronic[key][eid_b].values
    df["dist"] = np.sqrt(np.sum((a - b) ** 2, 1))
    df["ucid"] = mlapdv_chronic[key][eid_a].index
    df["eid_a"] = eid_a
    df["eid_b"] = eid_b
    dfs.append(df)
    # vals = dict(
    #     key=key,
    #     eid_a=eid_a,
    #     eid_b=eid_b,
    # )
    # res.append(vals)
df = pd.concat(dfs)

# %% plot the matrix
ucids = mlapdv_chronic[key][eid_a].index
eid_pairs = list(combinations(eids, 2))

dists = np.zeros((len(ucids), len(eid_pairs)))
for i, (eid_a, eid_b) in enumerate(eid_pairs):
    dists[:, i] = (
        df.groupby(["eid_a", "eid_b"]).get_group((eid_a, eid_b))["dist"].values
    )

# %%
fig, axes = plt.subplots()
mat = axes.matshow(dists)
axes.set_aspect("auto")
plt.colorbar(mat, label="distance (µm)")
axes.set_ylabel("ROIs")
axes.set_xlabel("eid combo")

# %%
mu = np.average(dists, axis=1)
sds = np.std(dists, axis=1)
fig, axes = plt.subplots(figsize=[14, 4])
for i in range(len(mu)):
    axes.plot([i, i], [mu[i] - sds[i], mu[i] + sds[i]], lw=0.1, color="k")
    axes.plot(i, mu[i], ".", color="k", alpha=0.5)

sns.despine(fig)
axes.set_xlabel("ROI")
axes.set_ylabel("mean dist")
axes.set_title("per ROI distance, all session pairwise")

# %%
bins = np.linspace(0, 100, 20)
dists_hist = np.zeros((len(ucids), len(bins) - 1))
for i in range(len(dists)):
    dists_hist[i, :] = np.histogram(dists[i, :], bins=bins)[0]

fig, axes = plt.subplots()
ax = axes.matshow(dists_hist, extent=(bins[0], bins[-1], 0, len(ucids)))
axes.set_aspect("auto")
plt.colorbar(ax, label="count")
axes.set_ylabel("ROI")
axes.set_xlabel("bin")


# %%
# ddf = pd.DataFrame(dists, columns=eid_pairs, index=ucids)
# sns.barplot(ddf.melt(),hue='variable',y='value',legend=None)
df["eid_combo"] = [
    ":".join(e)
    for e in zip(df["eid_a"].astype("str").values, df["eid_b"].astype("str").values)
]
sns.barplot(df, hue="ucid", y="dist", legend=None)
ax = plt.gca()
ax.set_ylabel("distance (µm)")
ax.set_xlabel("eid combination")
ax.set_title("per ROI session to session distance")
sns.despine(fig)

# %%

# %%

fig, axes = plt.subplots()
for i in range(200):
    vals = df.groupby("ucid").get_group(ucids[i])["dist"].values
    axes.hist(vals, bins=np.linspace(0, 100, 20), alpha=0.5)
