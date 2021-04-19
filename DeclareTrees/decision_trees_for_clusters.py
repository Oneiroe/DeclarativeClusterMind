import sys
from DeclareTrees.declare_trees import retrieve_decision_tree_for_clusters

if __name__ == '__main__':
    print(sys.argv)
    labels_file = sys.argv[1]
    # sj2t_trace_result_csv = sys.argv[2]
    j3tree_trace_measures_csv = sys.argv[2]
    output_file = sys.argv[3]
    label_feature_index = int(sys.argv[4])

    retrieve_decision_tree_for_clusters(labels_file, j3tree_trace_measures_csv, output_file, label_feature_index)
