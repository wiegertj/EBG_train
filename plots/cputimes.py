import pandas as pd
import matplotlib.pyplot as plt
import os

cpu_times = pd.read_csv(os.path.join(os.pardir, "data/comparison/cpu_times/test_times.csv"))
pd.set_option('display.max_columns', None)

print(cpu_times[cpu_times['ratio'] == cpu_times['ratio'].max()])
sum_ebg = cpu_times["sum_ebg_inf"].sum()
sum_iq = cpu_times["total_cpu_iqtree"].sum()
sum_inf =  cpu_times["total_cpu_inf"].sum()
sum_only = cpu_times["total_cpu_ebg"].sum()

print(f"CPU time EBG + inference {sum_ebg}")
print(f"Percentage inference of (EBG + inference): {sum_only / sum_ebg}" )
print(f"CPU time inference {sum_inf}")
print(f"CPU time UFBoot {sum_iq}")
print(f"Ratio CPU time (EBG + inference) / UFBoot : {sum_ebg / sum_iq}")

df_sorted = cpu_times.sort_values(by='difficulty')

plt.figure(figsize=(10, 6))
plt.plot(df_sorted['difficulty'], df_sorted['ratio'], linestyle='-', color='blue', label='Running Average')
plt.xlabel('Difficulty', fontsize=14)
plt.ylabel('CPU time EBG + Inf / CPU time UFBoot2', fontsize=14)
plt.grid(True)
plt.xticks(rotation=45, ha='right', fontsize=12)

plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("paper_figures/cpu_times.png")