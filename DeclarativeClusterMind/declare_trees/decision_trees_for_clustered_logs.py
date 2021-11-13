import sys
from DeclarativeClusterMind.declare_trees.declare_trees import retrieve_decision_tree_rules_for_clustered_logs

if __name__ == '__main__':
    print(sys.argv)
    labels_csv_file = sys.argv[1]
    output_file = sys.argv[2]
    label_feature_index = int(sys.argv[3])
    # split_policy = sys.argv[5]
    # 'rules'
    # 'attributes'
    # 'specific-attribute'
    # 'mixed'

    # focussed_csv = os.path.join(os.path.dirname(output_file), "focus.csv")

    # if split_policy == 'mixed':
    #     # MIXED
    #     print("multi-perspective decision tree")
    #     retrieve_decision_tree_multi_perspective_for_clusters(labels_csv_file,
    #                                                           j3tree_trace_measures_csv,
    #                                                           output_file,
    #                                                           focussed_csv,
    #                                                           label_feature_index)
    # elif split_policy == 'attributes':
    #     # ATTRIBUTES
    #     print("attributes-only decision tree")
    #     retrieve_decision_tree_attributes_for_clusters(labels_csv_file,
    #                                                    j3tree_trace_measures_csv,
    #                                                    output_file,
    #                                                    focussed_csv,
    #                                                    label_feature_index)
    # elif split_policy == 'specific-attribute':
    #     # SPECIFIC ATTRIBUTE
    #     print("split on single specific attribute not yet implemented")
    #     pass
    # elif split_policy == 'performances':
    #     # PERFORMANCES
    #     print("performances-only decision tree")
    #     retrieve_decision_tree_performances_for_clusters(labels_csv_file,
    #                                                      j3tree_trace_measures_csv,
    #                                                      output_file,
    #                                                      focussed_csv,
    #                                                      label_feature_index)
    # elif split_policy == 'rules':
    #     # RULES
    print("rules-only decision tree")
    retrieve_decision_tree_rules_for_clustered_logs(labels_csv_file,
                                                    output_file,
                                                    label_feature_index)
    # else:
    #     print("ERROR: Decision tree split policy not recognized")
