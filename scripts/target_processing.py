import warnings
from ete3 import Tree
import pandas as pd
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def calculate_support_statistics(support_file_path, dataset_name) -> list:
    """
    Function for extracting the SBS values from the datasets at data/raw.
    The branches are numbered by ete3 default traversing strategy 'level-order'.
    Support values are stored as fractions.
    Stores the final target file at data/processed/target/branch_supports.csv.

            Parameters:
                    :param support_file_path: path to the support file to process
                    :param dataset_name: name of the dataset to store in the result file

            Returns:
                    :return list of (str, str, float): list of tuples: (dataset_name, branchId, sbs fraction)

    """
    results = []
    with open(support_file_path, "r") as support_file:
        tree_str = support_file.read()
        phylo_tree = Tree(tree_str)
    branch_id_counter = 0
    for node in phylo_tree.traverse():
        branch_id_counter += 1
        if node.support is not None and not node.is_leaf():
            node.__setattr__("name", branch_id_counter)
            results.append((dataset_name, node.name, node.support / 100))
    return results


if __name__ == '__main__':

    raw_path = os.path.join(os.path.pardir, "data", "raw")
    folder_names = [folder for folder in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, folder))]
    counter = 0
    results_final = []
    for file in folder_names:
        support_path = os.path.join(raw_path, file, file + "_1000.raxml.support")

        counter += 1
        if counter % 100 == 0:
            print(f"{counter} / {len(folder_names)}")
        if os.path.exists(support_path):
            results_tmp = calculate_support_statistics(support_path, file.replace(".newick", ""))
            results_final.extend(results_tmp)

    df_final = pd.DataFrame(results_final, columns=["dataset", "branchId", "support"])
    df_final.to_csv(os.path.join(os.pardir, "data/processed/target/branch_supports.csv"), index=False)