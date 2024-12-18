import pandas as pd
import os

# Define file path pattern
file_path_pattern = "/Users/juliuswiegert/Downloads/test_regressor_out_{}.csv"

# Loop through file indices from 0 to 3
for i in range(4):
    file_path = file_path_pattern.format(i)

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Compute mean absolute error
    mean_abs_error = (df['prediction_median'] - df['support']).abs().mean()

    # Compute median absolute error
    median_abs_error = (df['prediction_median'] - df['support']).abs().median()

    # Print results
    print(f"File: test_regressor_out_{i}.csv")
    print(f"Mean Absolute Error: {mean_abs_error:.4f}")
    print(f"Median Absolute Error: {median_abs_error:.4f}\n")
