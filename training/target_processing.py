import warnings
from ete3 import Tree
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import os


def calculate_support_statistics(support_file_path, dataset_name):
    results = []
    with open(support_file_path, "r") as support_file:
        tree_str = support_file.read()
        phylo_tree = Tree(tree_str)
    branch_id_counter = 0
    for node in phylo_tree.traverse():
        branch_id_counter += 1
        if node.support is not None and not node.is_leaf():
            length = node.dist
            node.__setattr__("name", branch_id_counter)

            number_nodes = sum([1 for node in phylo_tree.traverse()])
            number_children = sum([1 for node in phylo_tree.traverse() if node.is_leaf()])

            farthest_topo = phylo_tree.get_farthest_leaf(topology_only=True)[1]
            farthest_branch = phylo_tree.get_farthest_leaf(topology_only=False)[1]
            length_relative = length / farthest_branch
            depth = node.get_distance(topology_only=True, target=phylo_tree.get_tree_root())
            num_children = sum([1 for child in node.traverse()])
            num_children_inner = sum([1 for child in node.traverse() if not child.is_leaf()])
            num_children_leaf = sum([1 for child in node.traverse() if child.is_leaf()])

            results.append((dataset_name, node.name, node.support / 100, length, length_relative,depth, depth / farthest_topo, num_children,
                            num_children / number_nodes, num_children_inner ,num_children_inner / num_children, num_children_leaf, num_children_leaf / number_children,
                            num_children_inner / num_children_leaf))

    return results


loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
loo_selection["dataset"] = loo_selection["verbose_name"].str.replace(".phy", "")

already_dedup_bs = pd.read_csv(os.path.join(os.pardir, "data/bs_support_pred_selection.csv"))
already_dedup_bs["dataset"] = already_dedup_bs["dataset"].str.replace(".newick", "")

loo_selection = loo_selection.merge(already_dedup_bs, on="dataset", how="inner")
print(loo_selection.shape)
filenames = loo_selection['verbose_name'].str.replace(".phy", "").tolist()
results_final = []
counter = 0

for file in filenames:
    print(len(filenames))
    support_path = os.path.join(os.pardir, "scripts/") + file + "_1000.raxml.support"
    print(support_path)
    counter +=1
    print(counter)

    if os.path.exists(support_path):
        print("Found")
        results_tmp = calculate_support_statistics(support_path, file.replace(".newick", ""))
        results_final.extend(results_tmp)

df_final = pd.DataFrame(results_final, columns=["dataset", "branchId", "support", "length", "length_relative", "depth", "depth_relative","num_children",
                                                "rel_num_children", "num_children_inner", "rel_num_children_inner", "num_children_leaf", "rel_num_children_leaf",
                                                "child_inner_leaf_ratio"])

df_final.to_csv(os.path.join(os.pardir, "data/processed/target/branch_supports.csv"))
