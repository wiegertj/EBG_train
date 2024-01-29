import pandas as pd
import matplotlib.pyplot as plt
import os

runtimes = pd.read_csv(os.path.join(os.pardir, "data/comparison/wallclock_times/wallclock_times.csv"))

count_smaller = len(runtimes[runtimes['elapsed_time_ebg_inf'] < runtimes['elapsed_time_iqtree']])
percentage_smaller = (count_smaller / len(runtimes)) * 100
print(f"The percentage of rows where EBG + Inference is smaller than UFBoot: {percentage_smaller}%")
print("#" * 100)

runtimes["diff_iqebg"] = (runtimes['elapsed_time_iqtree'] - runtimes['elapsed_time_ebg_inf'])
runtimes["perc_diff"] = runtimes["diff_iqebg"] / runtimes['elapsed_time_iqtree']
runtimes["speedup_to_iq"] = (runtimes['elapsed_time_iqtree'] / runtimes['elapsed_time_ebg_inf'])
avg_speedup = runtimes["speedup_to_iq"].mean()
std_speedup = runtimes["speedup_to_iq"].std()
print(f"Average speedup of EBG + Inference to UFBoot: {avg_speedup}, Std.: {std_speedup}")
print("#" * 100)

runtimes_large_msas = runtimes[runtimes["msa_size"] >= 250000]
avg_speedup_large_msas = runtimes_large_msas["speedup_to_iq"].mean()
print(f"Average speedup of EBG + Inference to UFBoot: {avg_speedup_large_msas}")
print("#" * 100)

sorted_time_merged = runtimes.sort_values(by='msa_size')

window_size = 20  # Adjust this value as needed
sorted_time_merged['moving_avg_iqtree_1'] = sorted_time_merged[sorted_time_merged['isDNA'] == 1][
    'elapsed_time_iqtree'].rolling(window=window_size).median()
sorted_time_merged['moving_avg_ebg'] = sorted_time_merged['elapsed_time_ebg'].rolling(window=window_size).median()
sorted_time_merged['moving_avg_sbs'] = sorted_time_merged['elapsed_time_sbs'].rolling(window=window_size).median()
sorted_time_merged['moving_avg_ebg_inf_1'] = sorted_time_merged[sorted_time_merged['isDNA'] == 1][
    'elapsed_time_ebg_inf'].rolling(window=window_size).median()

sorted_time_merged['moving_avg_iqtree_0'] = sorted_time_merged[sorted_time_merged['isDNA'] == 0][
    'elapsed_time_iqtree'].rolling(window=window_size).median()
sorted_time_merged['moving_avg_ebg_0'] = sorted_time_merged[sorted_time_merged['isDNA'] == 0][
    'elapsed_time_ebg'].rolling(window=window_size).median()
sorted_time_merged['moving_avg_ebg_inf_0'] = sorted_time_merged[sorted_time_merged['isDNA'] == 0][
    'elapsed_time_ebg_inf'].rolling(window=window_size).median()

sorted_time_merged['moving_avg_alrt_1'] = sorted_time_merged[sorted_time_merged['isDNA'] == 1][
    'elapsed_time_alrt'].rolling(window=window_size).median()
sorted_time_merged['moving_avg_alrt_0'] = sorted_time_merged[sorted_time_merged['isDNA'] == 0][
    'elapsed_time_alrt'].rolling(window=window_size).median()

plt.plot(sorted_time_merged[sorted_time_merged['isDNA'] == 1]['msa_size'],
         sorted_time_merged[sorted_time_merged['isDNA'] == 1]['moving_avg_iqtree_1'], label='UFBoot2 (DNA)',
         color='blue')
plt.plot(sorted_time_merged[sorted_time_merged['isDNA'] == 0]['msa_size'],
         sorted_time_merged[sorted_time_merged['isDNA'] == 0]['moving_avg_iqtree_0'], label='UFBoot2 (AA)',
         color='blue', linestyle="dashed")

plt.plot(sorted_time_merged[sorted_time_merged['isDNA'] == 1]['msa_size'],
         sorted_time_merged[sorted_time_merged['isDNA'] == 1]['moving_avg_alrt_1'], label='SH-aLRT (DNA)',
         color='black')
plt.plot(sorted_time_merged[sorted_time_merged['isDNA'] == 0]['msa_size'],
         sorted_time_merged[sorted_time_merged['isDNA'] == 0]['moving_avg_alrt_0'], label='SH-aLRT (AA)', color='black',
         linestyle="dashed")

plt.plot(sorted_time_merged[sorted_time_merged['isDNA'] == 1]['msa_size'],
         sorted_time_merged[sorted_time_merged['isDNA'] == 1]['moving_avg_ebg_inf_1'], label='EBG + inference (DNA)',
         color='green')
plt.plot(sorted_time_merged[sorted_time_merged['isDNA'] == 0]['msa_size'],
         sorted_time_merged[sorted_time_merged['isDNA'] == 0]['moving_avg_ebg_inf_0'], label='EBG + inference (AA)',
         color='green', linestyle="dashed")
plt.plot(sorted_time_merged["msa_size"], sorted_time_merged["moving_avg_ebg"], label='EBG', color='green',
         linestyle="dotted")

plt.plot(sorted_time_merged["msa_size"], sorted_time_merged["moving_avg_sbs"], label='SBS', color='red')

plt.xlabel('MSA size (#sequences × sequence length)')
plt.yscale('log', base=10)
plt.xscale('log', base=10)
plt.ylabel('Moving median elapsed time (window size = 20)')
plt.legend()
plt.tight_layout()
plt.savefig("runtimes.png")
plt.show()
