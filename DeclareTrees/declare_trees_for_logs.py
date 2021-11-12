import sys
import ClusterMind.IO.SJ2T_import as cmio
from DeclareTrees.declare_trees import build_declare_tree
from DeclareTrees.declare_trees import build_declare_tree_dynamic
from DeclareTrees.declare_trees import build_declare_tree_variance

if __name__ == '__main__':
    print(sys.argv)
    sj2t_file = sys.argv[1]
    constraints_threshold = float(sys.argv[2])
    preprocessed_file = sys.argv[3]
    output_file = sys.argv[4]
    branching_policy = sys.argv[5]  # [frequency, dynamic, variance]
    minimization_flag = sys.argv[6] == "True"
    reverse_flag = sys.argv[7] == "True"  # True descending order, Flase: ascending order

    # Pre-prcessing
    cmio.extract_detailed_perspective(sj2t_file, preprocessed_file, measure="Confidence")

    # Declare tree
    if branching_policy == "frequency":
        build_declare_tree(preprocessed_file, constraints_threshold, output_file, minimization_flag, reverse_flag)
    elif branching_policy == "dynamic":
        build_declare_tree_dynamic(preprocessed_file, constraints_threshold, output_file, minimization_flag,
                                   reverse_flag)
    elif branching_policy == "variance":
        build_declare_tree_variance(preprocessed_file, constraints_threshold, output_file, minimization_flag,
                                   reverse_flag)
    else:
        print("Branching policy not recognized. Supported policies: [frequency, dynamic, variance] ")
