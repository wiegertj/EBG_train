import pandas as pd
import matplotlib.pyplot as plt
import os

df_ebg = pd.read_csv(os.pardir + "/data/comparison/simulations/ebg_simulation.csv")
df_alrt = pd.read_csv(os.pardir + "/data/comparison/simulations/alrt_simulation.csv")
df_ufb = pd.read_csv(os.pardir + "/data/comparison/simulations/ufb_simulation.csv")
df_sbs = pd.read_csv(os.pardir + "/data/comparison/simulations/sbs_simulation.csv")
df_rb = pd.read_csv(os.pardir + "/data/comparison/simulations/rb_simulation.csv")

proportion_truth_is_1 = df_ebg.groupby('EBG_Support')['in_true'].mean()
alrt_proportion_truth_is_1 = df_alrt.groupby('alrt_Support')['inTrue'].mean()
ufb_proportion_truth_is_1 = df_ufb.groupby('ufb_Support')['inTrue'].mean()
sbs_proportion_truth_is_1 = df_sbs.groupby('sbs_Support')['inTrue'].mean()
rb_proportion_truth_is_1 = df_rb.groupby('rb_Support')['inTrue'].mean()

moving_average_alrt = alrt_proportion_truth_is_1.rolling(window=5, min_periods=1).mean()
moving_average_ebg = proportion_truth_is_1.rolling(window=5, min_periods=1).mean()
moving_average_ufb = ufb_proportion_truth_is_1.rolling(window=5, min_periods=1).mean()
moving_average_sbs = sbs_proportion_truth_is_1.rolling(window=5, min_periods=1).mean()
moving_average_rb = rb_proportion_truth_is_1.rolling(window=5, min_periods=1).mean()

plt.plot(alrt_proportion_truth_is_1.index, moving_average_alrt, label='SH-aLRT', color="black", linewidth=1.5)
plt.plot(proportion_truth_is_1.index, moving_average_ebg, label='EBG', color="lightgrey",  linewidth=1.5)
plt.plot(ufb_proportion_truth_is_1.index, moving_average_ufb, label='UFBoot2', linestyle="dashdot",  linewidth=1.5)
plt.plot(moving_average_sbs.index, moving_average_sbs, label='SBS', linestyle="dotted",  linewidth=2.0)
plt.plot(moving_average_rb.index, moving_average_rb, label='RB', color="red",  linewidth=1.5)

plt.plot(proportion_truth_is_1.index, proportion_truth_is_1.index/100, linestyle='--', color='red')

plt.xlabel('Branch support', fontsize=14)
plt.ylabel('Fraction in true tree', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.grid(True)
plt.savefig("paper_figures/simulated_data_smoothed_new_high.png", dpi=900)

