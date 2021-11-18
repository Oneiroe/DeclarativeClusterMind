""" Construct a decision tree from declarative rules plus additional perspectives

The decision tree is build using the Scikit CART algorithm.
For more info: https://scikit-learn.org/stable/modules/tree.html (last access on November 2021)

The decision tree is constructed from a feature vector for either the logs of the singles traces using different perspectives.

The perspectives supported are:
    - rules:    declarative rules (trace/log measured through Janus measurement framework)
    - attributes:   numerical log/trace attributes from the Event Log (categorical attributed are not supported by SciKit decision tree implementation)
    - performances: case performances of the log
"""

import csv
import sys

import DeclarativeClusterMind.io.Janus3_import as j3tio

import numpy as np
import pandas as pd
from sklearn import tree


def import_labels(labels_csv_file, label_feature_index):
    """
Util function to import the labels and selected feature name form featured data

    :param labels_csv_file:
    :param label_feature_index:
    :return:
    """
    labels = []
    selected_feature_name = ""

    with open(labels_csv_file, 'r') as input_file:
        csv_reader = csv.reader(input_file, delimiter=';')
        header = True
        for line in csv_reader:
            if header:
                header = False
                print("Label feature: " + line[label_feature_index])
                selected_feature_name = line[label_feature_index]
                continue
            labels += [line[label_feature_index]]

    return labels, selected_feature_name


def import_trace_labels_rules(labels_csv_file, janus_trace_measures_csv, focussed_csv, label_feature_index=1):
    """
Imports the featured data for each trace of the event log and import it for decision tree building.
Only declarative rules are considered as features.

The labels file header has the following structure:
TRACE-INDEX | CLUSTER | ATTRIBUTE_1 | ... | ATTRIBUTE_n | case-duration | case-length | case-unique-tasks

the janus measures file header has the following structure:
Trace |	Constraint name | measure-1-name |  measure-2-name
For every column there is only one log measure for each constraint.
if more than one measure is present, by default it is take "Confidence"

    :param labels_csv_file:
    :param janus_trace_measures_csv:
    :param focussed_csv:
    :param label_feature_index:
    :return:
    """
    # Import labels
    labels, selected_feature_name = import_labels(labels_csv_file, label_feature_index)
    # import data
    data, constraints_names = j3tio.extract_detailed_trace_rules_perspective_csv(janus_trace_measures_csv, focussed_csv)
    return data, labels, constraints_names, selected_feature_name


def import_trace_labels_multi_perspective(labels_csv_file, j3tree_trace_measures_csv, focussed_csv,
                                          label_feature_index=1, performances_index=-3):
    """
Imports the featured data for each trace of the event log and import it for decision tree building.
All the features except the performances are considered.
Performances are excluded because they obscure the other features given their high variance

The labels file header has the following structure:
TRACE-INDEX | CLUSTER | ATTRIBUTE_1 | ... | ATTRIBUTE_n | case-duration | case-length | case-unique-tasks

the janus measures file header has the following structure:
Trace |	Constraint name | measure-1-name |  measure-2-name
For every column there is only one log measure for each constraint.
if more than one measure is present, by default it is take "Confidence"

    :param labels_csv_file:
    :param j3tree_trace_measures_csv:
    :param focussed_csv:
    :param label_feature_index:
    :return:
    """
    # Import labels
    labels, selected_feature_name = import_labels(labels_csv_file, label_feature_index)
    # import data
    data, features_names = j3tio.extract_detailed_trace_multi_perspective_csv(j3tree_trace_measures_csv,
                                                                              labels_csv_file,
                                                                              focussed_csv,
                                                                              label_feature_index,
                                                                              performances_index)
    return data, labels, features_names, selected_feature_name


def import_labels_attributes(labels_csv_file, label_feature_index=1, performances_index=-3):
    """
Imports the featured data for each trace of the event log and import it for decision tree building.
Only the attributes are considered as features.
Only numerical attributes are supported by SciKit, categorical data are discarded.

The labels file header has the following structure for TRACES:
TRACE-INDEX | CLUSTER | ATTRIBUTE_1 | ... | ATTRIBUTE_n | case-duration | case-length | case-unique-tasks
label_feature_index=1, performances_index=-3

The labels file header has the following structure for CLUSTERS:
CLUSTER_NUM | TRACES | TRACE-LEN-AVG	TRACE-LEN-MIN	TRACE-LEN-MAX	DURATION-MEDIAN	DURATION-MIN	DURATION-MAX	CASE-ARRIVAL-AVG	VARIANTS-NUM	TASKS-NUM	TASKS
 | ATTRIBUTE_1 | ... | ATTRIBUTE_n
label_feature_index=0, performances_index=12

the janus measures file header has the following structure:
Trace |	Constraint name | measure-1-name |  measure-2-name
For every column there is only one log measure for each constraint.
if more than one measure is present, by default it is take "Confidence"

    :param labels_csv_file:
    :param label_feature_index:
    :param performances_index: index to skip the performances attributes
                            (if negative cut the tail, if positive cut the head)
    :return:
    """
    # Import labels
    labels, selected_feature_name = import_labels(labels_csv_file, label_feature_index)
    # import data
    data, features_names = j3tio.extract_detailed_attributes_csv(labels_csv_file,
                                                                 label_feature_index,
                                                                 performances_index)
    return data, labels, features_names, selected_feature_name


def import_labels_performances(labels_csv_file, label_feature_index=1, performances_index=-3):
    """
Imports the featured data for each trace of the event log and import it for decision tree building.
Only the performances are considered as features: case-duration, case-length, case-unique-tasks.
Usually case-duration is the only one which matter.

The labels file header has the following structure:
TRACE-INDEX | CLUSTER | ATTRIBUTE_1 | ... | ATTRIBUTE_n | case-duration | case-length | case-unique-tasks
label_feature_index=1, performances_index=-3

The labels file header has the following structure for CLUSTERS:
CLUSTER_NUM | TRACES | TRACE-LEN-AVG	TRACE-LEN-MIN	TRACE-LEN-MAX	DURATION-MEDIAN	DURATION-MIN	DURATION-MAX	CASE-ARRIVAL-AVG	VARIANTS-NUM	TASKS-NUM	TASKS
 | ATTRIBUTE_1 | ... | ATTRIBUTE_n
label_feature_index=0, performances_index=12

the janus measures file header has the following structure:
Trace |	Constraint name | measure-1-name |  measure-2-name
For every column there is only one log measure for each constraint.
if more than one measure is present, by default it is take "Confidence"

    :param labels_csv_file:
    :param label_feature_index:
    :param performances_index:
    :return:
    """
    # Import labels
    labels, selected_feature_name = import_labels(labels_csv_file, label_feature_index)
    # import data
    data, features_names = j3tio.extract_detailed_performances_csv(labels_csv_file,
                                                                   label_feature_index,
                                                                   performances_index)
    return data, labels, features_names, selected_feature_name


def import_log_labels_rules(labels_csv_file, label_feature_index=0):
    """
Imports the featured data aggregate at the level of the event log and import it for decision tree building.

The file header has the following structure: CLUSTER | CONSTRAINT_1 | ... | CONSTRAINT_n
for every column there is only one log measure for each constraint
    :param labels_csv_file:
    :param label_feature_index:
    :return:
    """
    # Import labels
    labels = []
    feature_name = ""
    data = []
    constraints_names = []

    with open(labels_csv_file, 'r') as input_file:
        csv_reader = csv.reader(input_file, delimiter=';')
        header = True
        for line in csv_reader:
            if header:
                header = False
                print("Label feature: " + line[label_feature_index])
                feature_name = line[label_feature_index]
                constraints_names += [constraint for constraint in
                                      line[:label_feature_index] + line[label_feature_index + 1:]]
                continue
            labels += [line[label_feature_index]]
            data += [[float(constraint) for constraint in line[:label_feature_index] + line[label_feature_index + 1:]]]

    return data, labels, constraints_names, feature_name


def import_log_labels_multi_perspective(measures_csv_file,
                                        attributes_performances_csv_file,
                                        label_feature_index=0,
                                        performances_index=12):
    """
Imports the featured data aggregate at the level of the event log and import it for decision tree building.

The file header has the following structure: CLUSTER | CONSTRAINT_1 | ... | CONSTRAINT_n
for every column there is only one log measure for each constraint
    :param measures_csv_file:
    :param label_feature_index:
    :return:
    """
    # Import labels rulse
    data_rules, labels_rules, features_names_rules, feature_name_rules = import_log_labels_rules(measures_csv_file,
                                                                                                 label_feature_index)
    # Import labels attributes
    data_attributes, labels_attributes, features_names_attributes, feature_name_attributes = import_labels_attributes(
        attributes_performances_csv_file, label_feature_index, performances_index)
    # Import labels performances
    data_performances, labels_performances, features_names_performances, feature_name_performances = import_labels_performances(
        attributes_performances_csv_file, label_feature_index, performances_index)

    if feature_name_rules != feature_name_performances != feature_name_attributes:
        print("ERROR labels between rules and attributes do not match!")
        sys.exit(1)
    feature_name = feature_name_rules

    data = np.concatenate([pd.DataFrame(data_rules, columns=features_names_rules),
                           data_attributes
                              , data_performances
                           ], axis=1)

    if len(labels_rules) != len(labels_attributes) != len(labels_performances):
        print("ERROR different number of labels between rules and attributes!")
        sys.exit(1)
    labels = set(labels_rules)
    labels.update(set(labels_attributes))
    labels.update(set(labels_performances))
    labels = sorted(list(labels))

    features_names = np.concatenate([features_names_rules,
                                     features_names_attributes
                                        , features_names_performances
                                     ])

    return data, labels, features_names, feature_name


def retrieve_decision_tree(featured_data, labels, output_file, features_names, selected_feature_name,
                           alpha_generic=0.0, alpha_specific=0.0, infinite_cap=1.7976931348623157e+100):
    """
Builds a decision tree given the input feature data.
A series of specific decision trees is also produced to classify a label against all the others.

This part is common to any perspective chosen at previous step.

# X: [n_samples, n_features] --> featured data: for each trace put the constraint feature vector
# Y: [n_samples] --> target: for each trace put the clusters label

    :param featured_data: table containing in each row a label and in each column its value for a certain feature
    :param labels: list of labels to which the tree tries to classify the entries. aka values of
    :param output_file: path to the output DOT/SVG tree file
    :param features_names:  ordered list of the features of the featured_data
    :param selected_feature_name: name of the feature used for the classification labels

    :param alpha_generic: ccp_alpha parameter to prune the general tree (0 by default, leading to un-pruned over-fitting trees)
    :param alpha_specific: ccp_alpha parameter to prune each specific tree (0 by default, leading to un-pruned over-fitting trees)
    :param infinite_cap: finite number to which map +/-infinite values
    """
    print("Building decision Tree...")
    featured_data = np.nan_to_num(np.array(featured_data), posinf=infinite_cap, neginf=-infinite_cap)
    labels = np.nan_to_num(np.array(labels), posinf=infinite_cap, neginf=-infinite_cap)
    print(f"number of labels to classify: {len(set(labels))}")

    clf = tree.DecisionTreeClassifier(
        ccp_alpha=alpha_generic
    )
    clf = clf.fit(featured_data, labels)
    print("Exporting decision Tree...")
    tree.plot_tree(clf)
    tree.export_graphviz(clf,
                         out_file=output_file,
                         feature_names=features_names,
                         class_names=[selected_feature_name + "_" + str(i) for i in clf.classes_],
                         filled=True,
                         rounded=True,
                         # special_characters = True
                         )

    left = 0
    clusters_labels = sorted(set(labels))
    for cluster in clusters_labels:
        print(f"Decision tree of cluster {left}/{len(clusters_labels)}", end="\r")
        left += 1
        current_labels = np.where(labels != cluster, 'others', labels)
        clf = tree.DecisionTreeClassifier(
            ccp_alpha=alpha_specific
        )
        clf = clf.fit(featured_data, current_labels)
        tree.plot_tree(clf)
        tree.export_graphviz(clf,
                             # out_file=output_file + "_" + selected_feature_name + "_" + cluster + ".dot",
                             out_file=output_file + "_" + cluster + ".dot",
                             feature_names=features_names,
                             class_names=[selected_feature_name + "_" + str(i) for i in clf.classes_],
                             filled=True,
                             rounded=True,
                             # special_characters = True
                             )

# if __name__ == '__main__':
#     log_attributes_csv_file = "/home/alessio/Data/Phd/my_code/ClusterMind/experiments/REAL-LIFE-EXPLANATION/SEPSIS_age/3-results/clusters-stats.csv"
#     trace_measures_csv_file = "/home/alessio/Data/Phd/my_code/ClusterMind/experiments/REAL-LIFE-EXPLANATION/SEPSIS_age/2-merged-log/SEPSIS_age-output[tracesMeasures].csv"
#     log_measures_csv_file = "/home/alessio/Data/Phd/my_code/ClusterMind/experiments/REAL-LIFE-EXPLANATION/SEPSIS_age/3-results/clusters-labels.csv"
#     trace_labels_file = "/home/alessio/Data/Phd/my_code/ClusterMind/experiments/REAL-LIFE-EXPLANATION/SEPSIS_age/3-results/traces-labels.csv"
#     output_file = "/home/alessio/Data/Phd/my_code/ClusterMind/experiments/REAL-LIFE-EXPLANATION/SEPSIS_age/3-results/TEST.dot"
#
#     focus = "/home/alessio/Data/Phd/my_code/ClusterMind/experiments/REAL-LIFE-EXPLANATION/SEPSIS_age/3-results/focus.csv"
#     trace_stats="/home/alessio/Data/Phd/my_code/ClusterMind/experiments/REAL-LIFE-EXPLORATION/SEPSIS/clusters_rules-treeSplit_rules/3-results/traces-labels.csv"
#
#     # data, labels, features_names, selected_feature_name = import_log_labels_rules(log_measures_csv_file, 0)
#     # data, labels, features_names, selected_feature_name = import_labels_attributes(log_attributes_csv_file, 0, 12)
#     # data, labels, features_names, selected_feature_name = import_labels_performances(log_attributes_csv_file, 0, 12)
#     # data, labels, features_names, selected_feature_name = import_log_labels_multi_perspective(log_measures_csv_file,log_attributes_csv_file,0, 12)
#
#     # data, labels, features_names, selected_feature_name = import_trace_labels_rules(trace_labels_file,
#     #                                                                                 trace_measures_csv_file,
#     #                                                                                 focus,
#     #                                                                                 1)
#     # data, labels, features_names, selected_feature_name = import_labels_attributes(trace_stats, 1, -3)
#     # data, labels, features_names, selected_feature_name = import_labels_performances(trace_stats, 1, -3)
#     # data, labels, features_names, selected_feature_name = import_trace_labels_multi_perspective(trace_stats,trace_measures_csv_file,focus,1, -3)
#
#     retrieve_decision_tree(data, labels, output_file, features_names, selected_feature_name)
