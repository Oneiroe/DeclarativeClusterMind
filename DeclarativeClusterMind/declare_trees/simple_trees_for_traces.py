"""
OUT-DATED LAUNCHER to split the traces of the log according to their trace measures in a hierarchical fashion.
"""
import os.path
import sys
import DeclarativeClusterMind.io.Janus3_import as j3io
from DeclarativeClusterMind.declare_trees.simple_trees import build_declare_tree_static, \
    build_clusters_from_traces_simple_tree
from DeclarativeClusterMind.declare_trees.simple_trees import build_declare_tree_dynamic

if __name__ == '__main__':
    print(sys.argv)
    janus_results_csv_file = sys.argv[1]
    constraints_threshold = float(sys.argv[2])
    temp_transposed_trace_measures_file = sys.argv[3]
    output_file = sys.argv[4]
    branching_policy = sys.argv[5]  # [frequency, dynamic, variance]
    minimization_flag = sys.argv[6] == "True"
    reverse_flag = sys.argv[7] == "True"  # True descending order, Flase: ascending order
    original_log = sys.argv[8]

    # Pre-prcessing
    if not os.path.exists(temp_transposed_trace_measures_file):
        print("Transposing data...")
        j3io.extract_detailed_trace_perspective_csv(janus_results_csv_file, temp_transposed_trace_measures_file,
                                                    measure="Confidence")
    else:
        print("Transposed data already exists")

    # Declare tree
    if branching_policy == "static-frequency":
        result_tree = build_declare_tree_static(temp_transposed_trace_measures_file, constraints_threshold, output_file,
                                                minimization_flag,
                                                reverse_flag)
    elif branching_policy == "dynamic-frequency" or branching_policy == "dynamic-variance":
        result_tree = build_declare_tree_dynamic(temp_transposed_trace_measures_file, constraints_threshold,
                                                 branching_policy,
                                                 output_file,
                                                 minimization_flag,
                                                 False,
                                                 min_leaf_size=50)
    else:
        print(
            "Branching policy not recognized. Supported policies: [static-frequency, dynamic-frequency, dynamic-variance] ")

    os.makedirs(os.path.join(os.path.dirname(output_file), "tree-clusters"), exist_ok=True)
    build_clusters_from_traces_simple_tree(result_tree,
                                           original_log,
                                           os.path.join(os.path.dirname(output_file), "tree-clusters"))
