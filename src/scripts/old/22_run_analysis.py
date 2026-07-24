# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv(Path(__file__).parent / "quantify_all_runs.csv")

# %%
order = [
    # "mlapdv_on_surface_histo",
    "mlapdv_histo",
    # "mlapdv_on_surface_histo_s2s_corr",
    "mlapdv_histo_s2s_corr",
    # "mlapdv_on_surface_histo_s2s_i25_corr",
    "mlapdv_histo_s2s_i25_corr",
    # "mlapdv_on_surface_histo_s2s_i1_corr",
    "mlapdv_histo_s2s_i1_corr",
    # "mlapdv_on_surface_histo_s2s_i25_apxy_corr",
    "mlapdv_histo_s2s_i25_apxy_corr",
    "mlapdv_histo_s2s_i25_apxyz_corr",
]
sns.barplot(df, y="mean_dist", hue="key", hue_order=order)
plt.gca().set_ylim(30, 42)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
# %%
