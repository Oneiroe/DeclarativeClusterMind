import csv
import math
import os
import sys
from random import random
import graphviz
import pydot
import ClusterMind.IO.SJ2T_import as cmio
import ClusterMind.IO.J3Tree_import as j3tio

from pm4py.objects.log.obj import EventLog
import pm4py as pm
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import pm4py.statistics.traces.log as stats

from sklearn import tree
import numpy as np
import graphviz


class ClusterNode:
    def __init__(self, constraint=None, threshold=0.8):
        self.ok = None  # child node fulfilling the constraint
        self.nok = None  # child node not fulfilling the constraint
        self.constraint = constraint  # Constraint discriminating the current node
        self.threshold = threshold  # Constraint threshold discriminating the current node
        self.clusters = set()  # Set of cluster at the current node
        self.used_constraints = set()  #

    def insert_child(self, cluster_name, value):
        if value >= self.threshold:
            if not self.ok:
                self.ok = ClusterNode(threshold=self.threshold)
            self.ok.clusters.add(cluster_name)

        else:
            if not self.nok:
                self.nok = ClusterNode(threshold=self.threshold)
            self.nok.clusters.add(cluster_name)

    def print_node(self):
        if self.constraint:
            print("[" + self.constraint + "]")
        else:
            print("<" + str(self.clusters) + ">")

    def print_tree_dfs(self):
        if self.ok:
            print('\t', end="")
            self.ok.print_tree_dfs()
        self.print_node()
        if self.nok:
            print('\t', end="")
            self.nok.print_tree_dfs()

    def print_tree_bfs(self):
        self.print_node()
        if self.ok:
            # print('\t', end="")
            self.ok.print_tree_bfs()
        if self.nok:
            # print('\t', end="")
            self.nok.print_tree_bfs()

    def print_node_graphviz(self):
        if self.constraint:
            return "[" + self.constraint + "]"
        else:
            return "<" + str(self.clusters) + ">"


def print_tree_graphviz(graph, node):
    this_node_code = node.print_node_graphviz() + str(random())
    if node.constraint:
        this_node = graph.node(this_node_code, label=node.print_node_graphviz())
    else:
        this_node = graph.node(this_node_code, label=node.print_node_graphviz(), fillcolor="lightblue", style='filled')

    if node.ok:
        next_left = print_tree_graphviz(graph, node.ok)
        graph.edge(this_node_code, next_left, label="YES [" + str(len(node.ok.clusters)) + "]", color="green")
    if node.nok:
        next_right = print_tree_graphviz(graph, node.nok)
        graph.edge(this_node_code, next_right, label="NO [" + str(len(node.nok.clusters)) + "]", color="red")
    return this_node_code


def minimize_tree(node):
    """
    SIDE EFFECT! Remove the nodes with only one child.
    :param node:
    """
    new_node = node
    if node.ok:
        new_node.ok = minimize_tree(node.ok)
    if node.nok:
        new_node.nok = minimize_tree(node.nok)
    if node.ok and not node.nok:
        return new_node.ok
    if not node.ok and node.nok:
        return new_node.nok
    return new_node


def order_constraints_overall(clusters_file, reverse=False):
    """
    It orders the constraints from the most common across the cluster to the less one from the SJ2T results
    :param reverse: True if descending order, False for Ascending
    :param clusters:
    """
    priority_sorted_constraints = []
    constraints_map = {}
    clusters_map = {}
    with open(clusters_file, 'r') as aggregated_result:
        # cluster_csv = csv.reader(aggregated_result, delimiter=';')
        csv_map = csv.DictReader(aggregated_result, delimiter=';')
        clusters_list = set(csv_map.fieldnames)
        clusters_list.discard('Constraint')
        for cluster in clusters_list:
            clusters_map[cluster] = {}
        for line in csv_map:
            constraint = line['Constraint']
            constraints_map[constraint] = {}
            constraints_map[constraint]['SUM'] = 0
            for cluster in clusters_list:
                value = float(line[cluster])
                constraints_map[constraint][cluster] = value
                clusters_map[cluster][constraint] = value
                if not math.isnan(value):
                    constraints_map[constraint]['SUM'] += value

    # constraint names and values
    priority_sorted_constraints = sorted([(i, constraints_map[i]['SUM']) for i in constraints_map],
                                         key=lambda item: item[1], reverse=reverse)
    print(priority_sorted_constraints)
    # only constraints names
    priority_sorted_constraints = [f[0] for f in sorted([(i, constraints_map[i]['SUM']) for i in constraints_map],
                                                        key=lambda item: item[1], reverse=reverse)]
    #  TODO remove field "SUM" from each item in constraints_map
    return priority_sorted_constraints, constraints_map, clusters_map


def build_declare_tree(clusters_file, threshold, output_file, minimize=False, reverse=True):
    """
Builds the DECLARE tree according to the aggregated result of the clusters.
Constraints are used in total frequency order from the most common among the clusters to the rarest one
    :param minimize:
    :param output_file:
    :param clusters_file:
    :param threshold:
    :return:
    """
    ordered_constraints, constraints_map, clusters_map = order_constraints_overall(clusters_file, reverse)
    # root
    result_tree = ClusterNode(threshold=threshold)
    result_tree.clusters = set(clusters_map.keys())
    leaves = set()
    leaves.add(result_tree)
    for constraint in ordered_constraints:
        new_leaves = set()
        for leaf in leaves:
            if len(leaf.clusters) == 1:
                continue
            leaf.constraint = constraint
            for cluster_in_node in leaf.clusters:
                leaf.insert_child(cluster_in_node, clusters_map[cluster_in_node][constraint])
                if leaf.ok:
                    new_leaves.add(leaf.ok)
                if leaf.nok:
                    new_leaves.add(leaf.nok)
        leaves = new_leaves

    if minimize:
        minimize_tree(result_tree)

    print("### Graphviz")
    graph = graphviz.Digraph(format='svg')
    print_tree_graphviz(graph, result_tree)
    graph.render(filename=output_file)

    return result_tree


def get_clusters_table(clusters_file):
    """
    It builds a matrix from the SJ2T results

    result [constraint x cluster] with headers for column and rows, plus last column "SUM" is the sum of the row
    Thus the column 0 are the constraints names, row 0 are the clusters names, column -1 is the sum of the row

    :param clusters:
    """
    clusters_table = []
    clusters_index = {}
    constraints_index = {}
    with open(clusters_file, 'r') as aggregated_result:
        # cluster_csv = csv.reader(aggregated_result, delimiter=';')
        csv_map = csv.reader(aggregated_result, delimiter=';')
        first = True
        for line in csv_map:
            if first:
                row = line + ['SUM']
                first = False
            else:
                row = [line[0]]
                sum_temp = 0.0
                for i in line[1:]:
                    if math.isnan(float(i)):
                        # row += [float(0)]  # consider vacuous satisfaction as a violation
                        # continue  # equal to +=0
                        row += [float(1)]
                        sum_temp += float(1)
                        # it is a vacuous satisfaction, but see atMostOne problem for the consequences of skipping it
                        # e.g. atMostOne(a) was used to distinguish clusters with a and cluster without it
                    else:
                        row += [float(i)]
                        sum_temp += float(i)
                row += [sum_temp]
            clusters_table += [row]
    clusters_counter = 1
    for cluster in clusters_table[0][1:-1]:
        clusters_index[cluster] = clusters_counter
        clusters_counter += 1
    constraints_counter = 1
    for constraint in clusters_table[1:]:
        constraints_index[constraint[0]] = constraints_counter
        constraints_counter += 1
    return clusters_table, clusters_index, constraints_index


def order_clusters_table(clusters_table, reverse=True):
    """
    Given a matrix [constrain X clusters] It orders the constraints by frequency across the clusters
    :param clusters_table:
    """
    clusters_table = [clusters_table[0]] + sorted(clusters_table[1:], key=lambda item: item[-1], reverse=reverse)
    return clusters_table


def get_most_common_constraint(cluster_table, clusters, used_constraints, reverse):
    view = []
    header = True
    for row in cluster_table:
        if row[0] in used_constraints:
            continue
        view_row = [row[0]]
        sum_row = 0.0
        for cluster_i in range(len(row)):
            if cluster_table[0][cluster_i] in clusters:
                view_row += [row[cluster_i]]
                if header or math.isnan(row[cluster_i]):
                    continue
                sum_row += row[cluster_i]
        if header:
            view_row += [row[-1]]
            header = False
        else:
            view_row += [sum_row]
        view += [view_row]
    return order_clusters_table(view, reverse)[1][0]


def build_declare_tree_dynamic(clusters_file, constraint_measure_threshold, output_file, minimize=False, reverse=True):
    """
Builds the DECLARE tree according to the aggregated result of the clusters.
Constraints are reordered in each sub-branch according to the frequency in the remaining clusters.
    :param reverse:
    :param minimize:
    :param output_file:
    :param clusters_file:
    :param constraint_measure_threshold: threshold above which a constraint's measure is considered part of a cluster
    :return:
    """
    # Import initial data
    clusters_table, clusters_indices, constraints_indices = get_clusters_table(clusters_file)
    # root initialization
    result_tree = ClusterNode(threshold=constraint_measure_threshold)
    result_tree.clusters = set(clusters_table[0][1:-1])
    leaves = set()
    leaves.add(result_tree)

    # while splittable leaves
    while len(leaves) > 0:
        #   for branch
        new_leaves = set()
        for leaf in leaves:
            if len(leaf.clusters) == 1 or len(leaf.used_constraints) == len(constraints_indices):
                continue
            #       split according to most frequent constraint
            leaf.constraint = get_most_common_constraint(clusters_table, leaf.clusters, leaf.used_constraints, reverse)
            for cluster_in_node in leaf.clusters:
                leaf.insert_child(cluster_in_node, clusters_table[constraints_indices[leaf.constraint]][
                    clusters_indices[cluster_in_node]])
            if leaf.ok:
                leaf.ok.used_constraints = leaf.used_constraints.copy()
                leaf.ok.used_constraints.add(leaf.constraint)
                new_leaves.add(leaf.ok)
            if leaf.nok:
                leaf.nok.used_constraints = leaf.used_constraints.copy()
                leaf.nok.used_constraints.add(leaf.constraint)
                new_leaves.add(leaf.nok)
        leaves = new_leaves

    if minimize:
        minimize_tree(result_tree)

    print("### Graphviz")
    graph = graphviz.Digraph(format='svg')
    print_tree_graphviz(graph, result_tree)
    graph.render(filename=output_file)

    return result_tree


def split_log(log, clusters):
    """
    WIP
    Split the log into sub-logs according to the clusters, returns the list of logs
    :param log:
    :param clusters:
    """
    n_clusters = max(clusters.labels_) - min(clusters.labels_) + 1
    # sub_logs = list(range(n_clusters))
    sub_logs = dict.fromkeys(set(clusters.labels_), [])
    # initialize sublogs with original log properties
    # for i in range(n_clusters):
    for i in set(clusters.labels_):
        sub_log = EventLog()
        sub_log._attributes = log.attributes
        sub_log._classifiers = log.classifiers
        sub_log._extensions = log.extensions
        sub_log._omni = log.omni_present
        sub_logs[i] = sub_log
    trace_index = 0
    # put traces in sub-logs
    for trace in log:
        sub_logs[clusters.labels_[trace_index]].append(trace)
        trace_index += 1
    return sub_logs


def build_clusters_from_branches(tree_root, original_log_file, output_folder):
    """
    WIP
It build clusters sub-logs from the leaves of the tree
    :param tree_root:
    """
    # Define clusters
    clusters = {}
    #
    log = pm.read_xes(original_log_file)
    logs = split_log(log, clusters)
    # export clusters logs to disk
    for cluster_index in logs:
        xes_exporter.apply(logs[cluster_index],
                           output_folder + log.attributes['concept:name'] + '_cluster_' + str(
                               cluster_index) + '.xes')


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


def retrieve_decision_tree_rules_for_clusters(labels_file,
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
                                    feature_names=constraints_names,
                                    class_names=[feature_name + "_" + str(i) for i in clf.classes_],
                                    filled=True,
                                    rounded=True,
                                    # special_characters = True
                                    )

    for cluster in set(labels):
        print(f"Decision tree of cluster {cluster}", end="\r")
        current_labels = np.where(labels != cluster, 'others', labels)
        clf = tree.DecisionTreeClassifier()
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


def retrieve_decision_tree_multi_perspective_for_clusters(labels_file, j3tree_trace_measures_csv,
                                                          sj2t_trace_output_file, focussed_csv,
                                                          label_feature_index=1):
    """
Use existing decision tree building techniques to retrieve a decision tree for your clusters
where the splits are either declare rules or attributes

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


def retrieve_decision_tree_multi_perspective_for_clusters(labels_file,
                                                          j3tree_trace_measures_csv,
                                                          sj2t_trace_output_file,
                                                          focussed_csv,
                                                          label_feature_index=1):
    """
Use existing decision tree building techniques to retrieve a decision tree for your clusters
where the splits are attributes

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
    for cluster in set(labels):
        print(f"Decision tree of cluster {cluster}", end="\r")
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


def retrieve_decision_tree_attributes_for_clusters(labels_file,
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

    for cluster in set(labels):
        print(f"Decision tree of cluster {cluster}", end="\r")
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
