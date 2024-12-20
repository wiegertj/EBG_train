import pandas as pd
import os
import matplotlib.pyplot as plt

# Define the file path pattern
base_dir = "/hits/fast/cme/wiegerjs/EBG_train/EBG_train/data/processed/final"
file_pattern = "test_regressor_out_{group_id}_fold_{fold_id}.csv" #test_regressor_out_0_fold_3.csv

# Dictionary to store MAE per group
mae_results = {group_id: [] for group_id in range(6)}

# Loop through all group and fold combinations
for group_id in range(6):
    for fold_id in range(4):
        try:
            file_name = file_pattern.format(group_id=group_id, fold_id=fold_id)
            file_path = os.path.join(base_dir, file_name)

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Compute Mean Absolute Error
            mae = (df['prediction_median'] - df['support']).abs().mean()

            # Store the result
            mae_results[group_id].append(mae)
        except:
            pass

# Convert results to DataFrame for plotting
mae_df = pd.DataFrame.from_dict(mae_results, orient='index').transpose()
mae_df.columns = [f'Group {i}' for i in range(6)]
# Plot the results
plt.figure(figsize=(10, 6))
mae_df.boxplot()

# Plot the means as text labels
group_means = mae_df.mean()
for i, mean_value in enumerate(group_means, start=1):
    plt.text(i, mean_value, f'{mean_value:.4f}', ha='center', va='bottom', color='red')

plt.title("MAE per Group Over Folds")
plt.ylabel("Mean Absolute Error")
plt.xlabel("Group (Dropped Features)")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
output_file = "/hits/fast/cme/wiegerjs/feature_sel.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()
