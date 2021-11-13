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


def import_labels(labels_file, j3tree_trace_measures_csv, focussed_csv, label_feature_index=1):
    # Import labels
    labels = []
    feature_name = ""

    with open(labels_file, 'r') as input_file:
        csv_reader = csv.reader(input_file, delimiter=';')
        header = True
        for line in csv_reader:
            if header:
                header = False
                print("Label feature: " + line[label_feature_index])
                feature_name = line[label_feature_index]
                continue
            labels += [line[label_feature_index]]

    # import data
    featured_data = [[] for i in range(len(labels))]

    # data, constraints_names = cmio.extract_detailed_perspective(j3tree_trace_measures_csv, focussed_csv)
    data, constraints_names = j3tio.extract_detailed_trace_perspective_csv(j3tree_trace_measures_csv, focussed_csv)
    # transpose_sj2t(data)
    # for constraint in data:
    #     for trace in data[constraint]:
    #         if trace == 'Constraint':
    #             continue
    #         featured_data[int(trace.strip("T"))] += [float(data[constraint][trace])]

    return data, labels, constraints_names, feature_name


def import_labels_multi_perspective(labels_file, j3tree_trace_measures_csv, focussed_csv, label_feature_index=1):
    # Import labels
    labels = []
    selected_feature_name = ""

    with open(labels_file, 'r') as input_file:
        csv_reader = csv.reader(input_file, delimiter=';')
        header = True
        for line in csv_reader:
            if header:
                header = False
                print("Label feature: " + line[label_feature_index])
                selected_feature_name = line[label_feature_index]
                continue
            labels += [line[label_feature_index]]

    # import data

    # data, constraints_names = cmio.extract_detailed_perspective(j3tree_trace_measures_csv, focussed_csv)
    data, features_names = j3tio.extract_detailed_trace_multi_perspective_csv(j3tree_trace_measures_csv,
                                                                              labels_file,
                                                                              focussed_csv,
                                                                              label_feature_index)
    # transpose_sj2t(data)
    # featured_data = [[] for i in range(len(labels))]
    # for constraint in data:
    #     for trace in data[constraint]:
    #         if trace == 'Constraint':
    #             continue
    #         featured_data[int(trace.strip("T"))] += [float(data[constraint][trace])]

    return data, labels, features_names, selected_feature_name


def import_labels_attributes(labels_file, focussed_csv, label_feature_index=1):
    # Import labels
    labels = []
    selected_feature_name = ""

    with open(labels_file, 'r') as input_file:
        csv_reader = csv.reader(input_file, delimiter=';')
        header = True
        for line in csv_reader:
            if header:
                header = False
                print("Label feature: " + line[label_feature_index])
                selected_feature_name = line[label_feature_index]
                continue
            labels += [line[label_feature_index]]

    # import data

    # data, constraints_names = cmio.extract_detailed_perspective(j3tree_trace_measures_csv, focussed_csv)
    # data, features_names = j3tio.extract_detailed_trace_multi_perspective_csv(j3tree_trace_measures_csv,
    #                                                                           labels_file,
    #                                                                           focussed_csv,
    #                                                                           label_feature_index)
    data, features_names = j3tio.extract_detailed_trace_attributes_csv(labels_file,
                                                                       focussed_csv,
                                                                       label_feature_index)
    # transpose_sj2t(data)
    # featured_data = [[] for i in range(len(labels))]
    # for constraint in data:
    #     for trace in data[constraint]:
    #         if trace == 'Constraint':
    #             continue
    #         featured_data[int(trace.strip("T"))] += [float(data[constraint][trace])]

    return data, labels, features_names, selected_feature_name


def import_labels_performances(labels_file, focussed_csv, label_feature_index=1):
    # Import labels
    labels = []
    selected_feature_name = ""

    with open(labels_file, 'r') as input_file:
        csv_reader = csv.reader(input_file, delimiter=';')
        header = True
        for line in csv_reader:
            if header:
                header = False
                print("Label feature: " + line[label_feature_index])
                selected_feature_name = line[label_feature_index]
                continue
            labels += [line[label_feature_index]]

    data, features_names = j3tio.extract_detailed_trace_performances_csv(labels_file,
                                                                         focussed_csv,
                                                                         label_feature_index)
    return data, labels, features_names, selected_feature_name


def import_clusters_labels(labels_csv_file, label_feature_index):
    """
The file header has the following structure: CLUSTER | CONSTRAINT_1 | ... | CONSTRAINT_n
for every column there is only one measure for each constraint
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


def retrieve_decision_tree_rules_for_logs_to_clusters(labels_csv_file,
                                                      output_file,
                                                      label_feature_index):
    """
Given the labels of clusters and their log measures
    :param labels_csv_file:
    :param output_file:
    :param label_feature_index:
    :return:
    """
    print("Importing data...")
    featured_data, labels, constraints_names, feature_name = import_clusters_labels(labels_csv_file,
                                                                                    label_feature_index)
    # X: [n_samples, n_features] --> featured data: for each trace put the constraint feature vector
    # Y: [n_samples] --> target: for each trace put the clusters label
    print("Building decision Tree...")
    featured_data = np.nan_to_num(np.array(featured_data), posinf=1.7976931348623157e+32,
                                  neginf=-1.7976931348623157e+32)
    labels = np.nan_to_num(np.array(labels), posinf=1.7976931348623157e+32, neginf=-1.7976931348623157e+32)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(featured_data, labels)
    print("Exporting decision Tree...")
    tree.plot_tree(clf)
    # dot_data = tree.export_graphviz(clf, out_file=sj2t_trace_output_file)
    dot_data = tree.export_graphviz(clf,
                                    out_file=output_file,
                                    feature_names=constraints_names,
                                    class_names=[feature_name + "_" + str(i) for i in clf.classes_],
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
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(featured_data, current_labels)
        tree.plot_tree(clf)
        # dot_data = tree.export_graphviz(clf, out_file=sj2t_trace_output_file)
        dot_data = tree.export_graphviz(clf,
                                        out_file=output_file + "_" + feature_name + "_" + cluster + ".dot",
                                        feature_names=constraints_names,
                                        class_names=[feature_name + "_" + str(i) for i in clf.classes_],
                                        filled=True,
                                        rounded=True,
                                        # special_characters = True
                                        )


def retrieve_decision_tree_rules_for_traces_to_clusters(labels_file,
                                                        j3tree_trace_measures_csv,
                                                        sj2t_trace_output_file,
                                                        focussed_csv,
                                                        label_feature_index=1):
    """
Use existing decision tree building techniques to retrieve a decision tree for your clusters
where the splits are declare rules

    label_index 1 is always the clusters label
    :param focussed_csv:
    :param labels_file:
    :param j3tree_trace_measures_csv:
    :param sj2t_trace_output_file:
    :param label_feature_index:
    :rtype: object
    """
    print("Importing data...")
    featured_data, labels, constraints_names, feature_name = import_labels(labels_file,
                                                                           j3tree_trace_measures_csv,
                                                                           focussed_csv,
                                                                           label_feature_index)
    # X: [n_samples, n_features] --> featured data: for each trace put the constraint feature vector
    # Y: [n_samples] --> target: for each trace put the clusters label
    print("Building decision Tree...")
    featured_data = np.nan_to_num(np.array(featured_data), posinf=1.7976931348623157e+32,
                                  neginf=-1.7976931348623157e+32)
    labels = np.nan_to_num(np.array(labels), posinf=1.7976931348623157e+32, neginf=-1.7976931348623157e+32)
    print(f"number of labels to classify: {len(set(labels))}")
    clf = tree.DecisionTreeClassifier(
        # ccp_alpha=0.01
    )
    clf = clf.fit(featured_data, labels)
    print("Exporting decision Tree...")
    tree.plot_tree(clf)
    # dot_data = tree.export_graphviz(clf, out_file=sj2t_trace_output_file)
    dot_data = tree.export_graphviz(clf,
                                    out_file=sj2t_trace_output_file,
                                    feature_names=constraints_names,
                                    class_names=[feature_name + "_" + str(i) for i in clf.classes_],
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
            # ccp_alpha=0.001
        )
        clf = clf.fit(featured_data, current_labels)
        tree.plot_tree(clf)
        # dot_data = tree.export_graphviz(clf, out_file=sj2t_trace_output_file)
        dot_data = tree.export_graphviz(clf,
                                        out_file=sj2t_trace_output_file + "_" + feature_name + "_" + cluster + ".dot",
                                        feature_names=constraints_names,
                                        class_names=[feature_name + "_" + str(i) for i in clf.classes_],
                                        filled=True,
                                        rounded=True,
                                        # special_characters = True
                                        )


def retrieve_decision_tree_multi_perspective_for_traces_to_clusters(labels_file,
                                                                    j3tree_trace_measures_csv,
                                                                    sj2t_trace_output_file,
                                                                    focussed_csv,
                                                                    label_feature_index=1):
    """
Use existing decision tree building techniques to retrieve a decision tree for your clusters
where the splits are either declare rules, attributes, or performances

    label_index 1 is always the clusters label
    :param focussed_csv:
    :param labels_file:
    :param j3tree_trace_measures_csv:
    :param sj2t_trace_output_file:
    :param label_feature_index:
    :rtype: object
    """
    print("Importing data...")
    featured_data, labels, features_names, selected_feature_name = import_labels_multi_perspective(labels_file,
                                                                                                   j3tree_trace_measures_csv,
                                                                                                   focussed_csv,
                                                                                                   label_feature_index)
    # X: [n_samples, n_features] --> featured data: for each trace put the constraint feature vector
    # Y: [n_samples] --> target: for each trace put the clusters label
    print("Building decision Tree...")
    featured_data = np.nan_to_num(np.array(featured_data), posinf=1.7976931348623157e+100,
                                  neginf=-1.7976931348623157e+100)
    labels = np.nan_to_num(np.array(labels), posinf=1.7976931348623157e+100, neginf=-1.7976931348623157e+100)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(featured_data, labels)
    print("Exporting decision Tree...")
    tree.plot_tree(clf)
    # dot_data = tree.export_graphviz(clf, out_file=sj2t_trace_output_file)
    dot_data = tree.export_graphviz(clf,
                                    out_file=sj2t_trace_output_file,
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
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(featured_data, current_labels)
        tree.plot_tree(clf)
        # dot_data = tree.export_graphviz(clf, out_file=sj2t_trace_output_file)
        dot_data = tree.export_graphviz(clf,
                                        out_file=sj2t_trace_output_file + "_" + selected_feature_name + "_" + cluster + ".dot",
                                        feature_names=features_names,
                                        class_names=[selected_feature_name + "_" + str(i) for i in clf.classes_],
                                        filled=True,
                                        rounded=True,
                                        # special_characters = True
                                        )


def retrieve_decision_tree_attributes_for_traces_to_clusters(labels_file,
                                                             j3tree_trace_measures_csv,
                                                             sj2t_trace_output_file,
                                                             focussed_csv,
                                                             label_feature_index=1):
    """
Use existing decision tree building techniques to retrieve a decision tree for your clusters
where the splits are attributes

    label_index 1 is always the clusters label
    :param labels_file:
    :param j3tree_trace_measures_csv: useless, but kept only to be able to use the same command for all the decison tree launchers
    :param sj2t_trace_output_file:
    :param label_feature_index:
    :rtype: object
    """
    print("Importing data...")
    featured_data, labels, features_names, selected_feature_name = import_labels_attributes(labels_file,
                                                                                            focussed_csv,
                                                                                            label_feature_index)
    # X: [n_samples, n_features] --> featured data: for each trace put the constraint feature vector
    # Y: [n_samples] --> target: for each trace put the clusters label
    print("Building decision Tree...")
    featured_data = np.nan_to_num(np.array(featured_data), posinf=1.7976931348623157e+100,
                                  neginf=-1.7976931348623157e+100)
    labels = np.nan_to_num(np.array(labels), posinf=1.7976931348623157e+100, neginf=-1.7976931348623157e+100)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(featured_data, labels)
    print("Exporting decision Tree...")
    tree.plot_tree(clf)
    # dot_data = tree.export_graphviz(clf, out_file=sj2t_trace_output_file)
    dot_data = tree.export_graphviz(clf,
                                    out_file=sj2t_trace_output_file,
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
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(featured_data, current_labels)
        tree.plot_tree(clf)
        # dot_data = tree.export_graphviz(clf, out_file=sj2t_trace_output_file)
        dot_data = tree.export_graphviz(clf,
                                        out_file=sj2t_trace_output_file + "_" + selected_feature_name + "_" + cluster + ".dot",
                                        feature_names=features_names,
                                        class_names=[selected_feature_name + "_" + str(i) for i in clf.classes_],
                                        filled=True,
                                        rounded=True,
                                        # special_characters = True
                                        )


def retrieve_decision_tree_performances_for_traces_to_clusters(labels_file,
                                                               j3tree_trace_measures_csv,
                                                               sj2t_trace_output_file,
                                                               focussed_csv,
                                                               label_feature_index=1):
    """
Use existing decision tree building techniques to retrieve a decision tree for your clusters
where the splits are performance

    label_index 1 is always the clusters label
    :param labels_file:
    :param j3tree_trace_measures_csv: useless, but kept only to be able to use the same command for all the decison tree launchers
    :param sj2t_trace_output_file:
    :param label_feature_index:
    :rtype: object
    """
    print("Importing data...")
    featured_data, labels, features_names, selected_feature_name = import_labels_performances(labels_file,
                                                                                              focussed_csv,
                                                                                              label_feature_index)
    # X: [n_samples, n_features] --> featured data: for each trace put the constraint feature vector
    # Y: [n_samples] --> target: for each trace put the clusters label
    print("Building decision Tree...")
    featured_data = np.nan_to_num(np.array(featured_data), posinf=1.7976931348623157e+100,
                                  neginf=-1.7976931348623157e+100)
    labels = np.nan_to_num(np.array(labels), posinf=1.7976931348623157e+100, neginf=-1.7976931348623157e+100)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(featured_data, labels)
    print("Exporting decision Tree...")
    tree.plot_tree(clf)
    # dot_data = tree.export_graphviz(clf, out_file=sj2t_trace_output_file)
    dot_data = tree.export_graphviz(clf,
                                    out_file=sj2t_trace_output_file,
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
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(featured_data, current_labels)
        tree.plot_tree(clf)
        # dot_data = tree.export_graphviz(clf, out_file=sj2t_trace_output_file)
        dot_data = tree.export_graphviz(clf,
                                        out_file=sj2t_trace_output_file + "_" + selected_feature_name + "_" + cluster + ".dot",
                                        feature_names=features_names,
                                        class_names=[selected_feature_name + "_" + str(i) for i in clf.classes_],
                                        filled=True,
                                        rounded=True,
                                        # special_characters = True
                                        )
