import os
import pandas as pd

features = pd.read_csv( os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "features", "features.csv"))
targets = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "target", "branch_supports.csv"))

merged_df = features.merge(targets, how="inner", on=["dataset", "branchId"])
print(merged_df.shape)
merged_df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "final", "final.csv"))

