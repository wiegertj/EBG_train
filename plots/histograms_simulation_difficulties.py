import pandas as pd
import matplotlib.pyplot as plt
import os

df_alisim = pd.read_csv(os.pardir + "/data/comparison/difficulties/alisim_dna_simulations_difficulties.csv")
df_pandit = pd.read_csv(os.pardir + "/data/comparison/difficulties/pandit_dna_simulations_difficulties.csv").sample(982)

diff_mean = df_pandit['diff'].mean()
diff_std = df_pandit['diff'].std()

print("Mean of difficulty PANDIT:", round(diff_mean, 2))
print("Standard deviation of difficulty PANDIT", round(diff_std, 2))

diff_mean = df_alisim['diff'].mean()
diff_std = df_alisim['diff'].std()

print("Mean of difficulty AliSim:", round(diff_mean, 2))
print("Standard deviation of difficulty AliSim", round(diff_std, 2))

plt.figure(figsize=(8, 6))
plt.hist(df_alisim['diff'], bins=10, alpha=0.5, color='darkgray', label='AliSim simulations', edgecolor='black')
plt.hist(df_pandit['diff'], bins=10, alpha=0.5, color='lightgray', label='PANDIT simulations', edgecolor='black', hatch="/")

plt.xlabel('Pythia difficulty', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("paper_figures/simulations_difficulties_histogram.png", dpi=900)



