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

import DeclarativeClusterMind.io.Janus3_import as j3tio

import numpy as np
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
    data, constraints_names = j3tio.extract_detailed_trace_perspective_csv(janus_trace_measures_csv, focussed_csv)
    return data, labels, constraints_names, selected_feature_name


def import_trace_labels_multi_perspective(labels_csv_file, j3tree_trace_measures_csv, focussed_csv, label_feature_index=1):
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
                                                                              label_feature_index)
    return data, labels, features_names, selected_feature_name


def import_trace_labels_attributes(labels_csv_file, focussed_csv, label_feature_index=1):
    """
Imports the featured data for each trace of the event log and import it for decision tree building.
Only the attributes are considered as features.
Only numerical attributes are supported by SciKit, categorical data are discarded.
# todo handle categorical attributes (at least the true/false ones in 1/0)

The labels file header has the following structure:
TRACE-INDEX | CLUSTER | ATTRIBUTE_1 | ... | ATTRIBUTE_n | case-duration | case-length | case-unique-tasks

the janus measures file header has the following structure:
Trace |	Constraint name | measure-1-name |  measure-2-name
For every column there is only one log measure for each constraint.
if more than one measure is present, by default it is take "Confidence"

    :param labels_csv_file:
    :param focussed_csv:
    :param label_feature_index:
    :return:
    """
    # Import labels
    labels, selected_feature_name = import_labels(labels_csv_file, label_feature_index)
    # import data
    data, features_names = j3tio.extract_detailed_trace_attributes_csv(labels_csv_file,
                                                                       focussed_csv,
                                                                       label_feature_index)
    return data, labels, features_names, selected_feature_name


def import_trace_labels_performances(labels_csv_file, focussed_csv, label_feature_index=1):
    """
Imports the featured data for each trace of the event log and import it for decision tree building.
Only the performances are considered as features: case-duration, case-length, case-unique-tasks.
Usually case-duration is the only one which matter.

The labels file header has the following structure:
TRACE-INDEX | CLUSTER | ATTRIBUTE_1 | ... | ATTRIBUTE_n | case-duration | case-length | case-unique-tasks

the janus measures file header has the following structure:
Trace |	Constraint name | measure-1-name |  measure-2-name
For every column there is only one log measure for each constraint.
if more than one measure is present, by default it is take "Confidence"

    :param labels_csv_file:
    :param focussed_csv:
    :param label_feature_index:
    :return:
    """
    # Import labels
    labels, selected_feature_name = import_labels(labels_csv_file, label_feature_index)
    # import data
    data, features_names = j3tio.extract_detailed_trace_performances_csv(labels_csv_file,
                                                                         focussed_csv,
                                                                         label_feature_index)
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
                             out_file=output_file + "_" + selected_feature_name + "_" + cluster + ".dot",
                             feature_names=features_names,
                             class_names=[selected_feature_name + "_" + str(i) for i in clf.classes_],
                             filled=True,
                             rounded=True,
                             # special_characters = True
                             )
