import pandas as pd
import matplotlib.pyplot as plt
import os

runtimes_ = pd.read_csv(os.path.join(os.pardir, "data/comparison/wallclock_times/wallclock_times.csv"))
no_pats = pd.read_csv(os.path.join(os.pardir, "data/processed/features/dataset_pattern_counts.csv"))
print(runtimes_.shape)
runtimes = runtimes_.merge(no_pats, on=["dataset"], how="inner")
print(runtimes.shape)
runtimes["msa_size_patterns"] = runtimes["num_seq"] * runtimes["no_pat"]

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

runtimes_large_msas = runtimes[runtimes["msa_size_patterns"] >= 200000]
avg_speedup_large_msas = runtimes_large_msas["speedup_to_iq"].mean()
print(f"Average speedup of EBG + Inference to UFBoot (filtered by size >= 200000): {avg_speedup_large_msas}")
print("#" * 100)

sorted_time_merged = runtimes.sort_values(by='msa_size_patterns')

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

plt.plot(sorted_time_merged[sorted_time_merged['isDNA'] == 1]['msa_size_patterns'],
         sorted_time_merged[sorted_time_merged['isDNA'] == 1]['moving_avg_iqtree_1'], label='IQTREE 2 Ultrafast Bootstrap (DNA)',
         color='blue')
plt.plot(sorted_time_merged[sorted_time_merged['isDNA'] == 0]['msa_size_patterns'],
         sorted_time_merged[sorted_time_merged['isDNA'] == 0]['moving_avg_iqtree_0'], label='IQTREE 2 Ultrafast Bootstrap (AA)',
         color='blue', linestyle="dashed")

#plt.plot(sorted_time_merged[sorted_time_merged['isDNA'] == 1]['msa_size_patterns'],
#         sorted_time_merged[sorted_time_merged['isDNA'] == 1]['moving_avg_alrt_1'], label='SH-aLRT (DNA)',
 #        color='black')
#plt.plot(sorted_time_merged[sorted_time_merged['isDNA'] == 0]['msa_size_patterns'],
 #        sorted_time_merged[sorted_time_merged['isDNA'] == 0]['moving_avg_alrt_0'], label='SH-aLRT (AA)', color='black',
  #       linestyle="dashed")

plt.plot(sorted_time_merged[sorted_time_merged['isDNA'] == 1]['msa_size_patterns'],
         sorted_time_merged[sorted_time_merged['isDNA'] == 1]['moving_avg_ebg_inf_1'], label='EBG (DNA)',
         color='green')
plt.plot(sorted_time_merged[sorted_time_merged['isDNA'] == 0]['msa_size_patterns'],
         sorted_time_merged[sorted_time_merged['isDNA'] == 0]['moving_avg_ebg_inf_0'], label='EBG (AA)',
         color='green', linestyle="dashed")
#plt.plot(sorted_time_merged["msa_size_patterns"], sorted_time_merged["moving_avg_ebg"], label='EBG', color='green',
 #        linestyle="dotted")

plt.plot(sorted_time_merged["msa_size_patterns"], sorted_time_merged["moving_avg_sbs"], label='SBS', color='red')

plt.xlabel('MSA size (#sequences × #site patterns)')
plt.yscale('log', base=10)
plt.xscale('log', base=10)
plt.ylabel('Moving median time-to-completion (window size = 20)')
plt.legend()
plt.tight_layout()
plt.savefig("paper_figures/runtimes.png")
