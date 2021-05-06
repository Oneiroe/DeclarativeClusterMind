import os
import sys
from DeclareTrees.declare_trees import retrieve_decision_tree_rules_for_clusters
from DeclareTrees.declare_trees import retrieve_decision_tree_multi_perspective_for_clusters
from DeclareTrees.declare_trees import retrieve_decision_tree_attributes_for_clusters

if __name__ == '__main__':
    print(sys.argv)
    labels_file = sys.argv[1]
    # sj2t_trace_result_csv = sys.argv[2]
    j3tree_trace_measures_csv = sys.argv[2]
    output_file = sys.argv[3]
    label_feature_index = int(sys.argv[4])
    split_policy = sys.argv[5]
    # 'rules'
    # 'attributes'
    # 'specific-attribute'
    # 'mixed'

    focussed_csv = os.path.join(os.path.dirname(output_file), "focus.csv")

    if split_policy == 'mixed':
        # MIXED
        print("multi-perspective decision tree")
        retrieve_decision_tree_multi_perspective_for_clusters(labels_file,
                                                              j3tree_trace_measures_csv,
                                                              output_file,
                                                              focussed_csv,
                                                              label_feature_index)
    elif split_policy == 'attributes':
        # ATTRIBUTES
        print("attributes-only decision tree")
        retrieve_decision_tree_attributes_for_clusters(labels_file,
                                                       j3tree_trace_measures_csv,
                                                       output_file,
                                                       focussed_csv,
                                                       label_feature_index)
    elif split_policy == 'specific-attribute':
        # SPECIFIC ATTRIBUTE
        print("split on single specific attribute not yet implemented")
        pass
    elif split_policy == 'rules':
        # RULES
        print("rules-only decision tree")
        retrieve_decision_tree_rules_for_clusters(labels_file,
                                                  j3tree_trace_measures_csv,
                                                  output_file,
                                                  focussed_csv,
                                                  label_feature_index)
    else:
        print("ERROR: Decision tree split policy not recognized")
