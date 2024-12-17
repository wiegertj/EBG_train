import pandas as pd
import matplotlib.pyplot as plt
import os

df_ebg = pd.read_csv(os.pardir + "/data/comparison/simulations_pandit/ebg_simulation_pandit.csv")
df_alrt = pd.read_csv(os.pardir + "/data/comparison/simulations_pandit/alrt_simulation_pandit.csv")
df_ufb = pd.read_csv(os.pardir + "/data/comparison/simulations_pandit/ufb_simulation_pandit.csv")
df_rb = pd.read_csv(os.pardir + "/data/comparison/simulations_pandit/rb_simulation_pandit.csv")
df_sbs = pd.read_csv(os.pardir + "/data/comparison/simulations_pandit/sbs_simulation_pandit.csv")

duplicates = df_ebg[df_ebg.duplicated(subset=["dataset", "branchID_True"], keep=False)]

# If there are duplicates, save them to a new CSV file
if not duplicates.empty:
    duplicates.to_csv("duplicates.csv", index=False)
    print("Duplicates found and saved to 'duplicates.csv'.")
else:
    print("No duplicates found.")
print(df_ebg.groupby('EBG_Support').size())
proportion_truth_is_1 = df_ebg.groupby('EBG_Support')['inTrue'].sum() / df_ebg.groupby('EBG_Support').size()
alrt_proportion_truth_is_1 = df_alrt.groupby('alrt_Support')['inTrue'].sum() / df_alrt.groupby('alrt_Support').size()
ufb_proportion_truth_is_1 = df_ufb.groupby('ufb_Support')['inTrue'].sum() / df_ufb.groupby('ufb_Support').size()
sbs_proportion_truth_is_1 = df_sbs.groupby('sbs_Support')['inTrue'].sum() / df_sbs.groupby('sbs_Support').size()
rb_proportion_truth_is_1 = df_rb.groupby('rb_Support')['inTrue'].sum() / df_rb.groupby('rb_Support').size()

moving_average_alrt = alrt_proportion_truth_is_1.rolling(window=5, min_periods=1).mean()
moving_average_ebg = proportion_truth_is_1.rolling(window=5, min_periods=1).mean()
moving_average_ufb = ufb_proportion_truth_is_1.rolling(window=5, min_periods=1).mean()
moving_average_rb = rb_proportion_truth_is_1.rolling(window=5, min_periods=1).mean()
moving_average_sbs = sbs_proportion_truth_is_1.rolling(window=5, min_periods=1).mean()

#plt.plot(moving_average_alrt.index, moving_average_alrt, label='SH-aLRT')
#plt.plot(moving_average_ebg.index, moving_average_ebg, label='EBG')
#plt.plot(moving_average_ufb.index, moving_average_ufb, label='UFBoot2')
#plt.plot(moving_average_sbs.index, moving_average_sbs, label='SBS')
#plt.plot(moving_average_rb.index, moving_average_rb, label='RB')

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

# Set the limits of both axes
plt.xlim(0, 100)
plt.ylim(0, 1)
plt.savefig("paper_figures/simulated_data_pandit_high.png", dpi=900)

