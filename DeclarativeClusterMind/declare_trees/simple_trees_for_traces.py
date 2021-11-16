"""
OUT-DATED LAUNCHER to split the traces of the log according to their trace measures in a hierarchical fashion.
"""
import os.path
import sys
import DeclarativeClusterMind.io.Janus3_import as j3io
from DeclarativeClusterMind.declare_trees.simple_trees import build_declare_tree_static
from DeclarativeClusterMind.declare_trees.simple_trees import build_declare_tree_dynamic

if __name__ == '__main__':
    print(sys.argv)
    janus_file = sys.argv[1]
    constraints_threshold = float(sys.argv[2])
    temp_transposed_trace_measures_file = sys.argv[3]
    output_file = sys.argv[4]
    branching_policy = sys.argv[5]  # [frequency, dynamic, variance]
    minimization_flag = sys.argv[6] == "True"
    reverse_flag = sys.argv[7] == "True"  # True descending order, Flase: ascending order

    # Pre-prcessing
    if not os.path.exists(temp_transposed_trace_measures_file):
        print("Transposing data...")
        j3io.extract_detailed_trace_perspective_csv(janus_file, temp_transposed_trace_measures_file, measure="Confidence")
    else:
        print("Transposed data already exists")

# Declare tree
    if branching_policy == "static-frequency":
        build_declare_tree_static(temp_transposed_trace_measures_file, constraints_threshold, output_file,
                                  minimization_flag,
                                  reverse_flag)
    elif branching_policy == "dynamic-frequency" or branching_policy == "dynamic-variance":
        build_declare_tree_dynamic(temp_transposed_trace_measures_file, constraints_threshold, branching_policy,
                                   output_file,
                                   minimization_flag,
                                   reverse_flag)
    else:
        print(
            "Branching policy not recognized. Supported policies: [static-frequency, dynamic-frequency, dynamic-variance] ")
