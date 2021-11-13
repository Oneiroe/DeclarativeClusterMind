import sys
from DeclarativeClusterMind.declare_trees.declare_trees import build_declare_tree
from DeclarativeClusterMind.declare_trees.declare_trees import build_declare_tree_dynamic
from DeclarativeClusterMind.declare_trees.declare_trees import build_declare_tree_variance

if __name__ == '__main__':
    print(sys.argv)
    clusters_file = sys.argv[1]
    constraints_threshold = float(sys.argv[2])
    output_file = sys.argv[3]
    branching_policy = sys.argv[4]  # [frequency, dynamic, variance]
    minimize_flag = sys.argv[5] == "True"  # single choice splits are discarded, keep only the separating constraints
    reverse_flag = sys.argv[6] == "True"  # True descending order, False: ascending order

    # frequency: total frequency among all the clusters (each level in each branch splits with the same constraint)
    # dynamic: each split is given by the most frequent constraint among the clusters in the branch
    if branching_policy == "frequency":
        build_declare_tree(clusters_file, constraints_threshold, output_file, minimize_flag, reverse_flag)
    elif branching_policy == "dynamic":
        build_declare_tree_dynamic(clusters_file, constraints_threshold, output_file,minimize_flag, reverse_flag)
    elif branching_policy == "variance":
        build_declare_tree_variance(clusters_file, constraints_threshold, output_file, minimize_flag, reverse_flag)
    else:
        print("Branching policy not recognized. Supported policies: [frequency, dynamic, variance] ")
