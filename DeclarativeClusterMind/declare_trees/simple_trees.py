""" Build a simple decision tree based only on declarative rules.

Policies supported:
    - (static) frequency/sum:   the constraints are sorted at the beginning by their support among the clusters/variant
                                and thei are used in this order for splitting
    - dynamic (frequency/sum):  for each split is chosen is the current constraints which is more
                                supported/frequent/valid among the clusters/variants
    - (dynamic) variance:   for each split is chosen is the current constraints which has more variance among the
                            clusters/variants

"""

import csv
import math
import os
import statistics
from random import random

from pm4py.objects.log.obj import EventLog
import pm4py as pm
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

import graphviz

from DeclarativeClusterMind.io.Janus3_import import extract_detailed_trace_rules_perspective_csv


class ClusterNode:
    def __init__(self, constraint=None, threshold=0.8):
        self.ok = None  # child node fulfilling the constraint
        self.nok = None  # child node not fulfilling the constraint
        self.nan = None  # child node not violating but also not activating the constraint
        self.constraint = constraint  # Constraint discriminating the current node
        self.threshold = threshold  # Constraint threshold discriminating the current node
        self.clusters = set()  # Set of cluster at the current node
        self.used_constraints = set()  #
        self.ok_max_previous_value = 0.0  # minimum value once the node is created
        self.ok_min_previous_value = 1.0  # maximum value once the node is created
        self.nok_max_previous_value = 0.0  # minimum value once the node is created
        self.nok_min_previous_value = 1.0  # maximum value once the node is created

    def insert_child(self, cluster_name, value):
        if math.isnan(value):
            if not self.nan:
                self.nan = ClusterNode(threshold=self.threshold)
            self.nan.clusters.add(cluster_name)
        elif value >= self.threshold:
            if not self.ok:
                self.ok = ClusterNode(threshold=self.threshold)
            self.ok.clusters.add(cluster_name)
            self.ok_min_previous_value = min(self.ok_min_previous_value, value)
            self.ok_max_previous_value = max(self.ok_max_previous_value, value)
        else:
            if not self.nok:
                self.nok = ClusterNode(threshold=self.threshold)
            self.nok.clusters.add(cluster_name)
            self.nok_min_previous_value = min(self.nok_min_previous_value, value)
            self.nok_max_previous_value = max(self.nok_max_previous_value, value)

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
        if self.nan:
            print('\t', end="")
            self.nan.print_tree_dfs()
        if self.nok:
            print('\t', end="")
            self.nok.print_tree_dfs()

    def print_tree_bfs(self):
        self.print_node()
        if self.ok:
            # print('\t', end="")
            self.ok.print_tree_bfs()
        if self.nan:
            print('\t', end="")
            self.nan.print_tree_dfs()
        if self.nok:
            # print('\t', end="")
            self.nok.print_tree_bfs()

    def print_node_graphviz(self):
        if self.constraint:
            return "[" + self.constraint + "]"
        else:
            if len(str(self.clusters)) > 16000:
                return "<...>"
            else:
                return "<" + str(self.clusters) + ">"


def print_tree_graphviz(graph, node, aggregate=False):
    this_node_code = str(random())
    if node.constraint:
        this_node = graph.node(this_node_code, label=node.print_node_graphviz())
    else:
        if aggregate:
            this_node = graph.node(this_node_code, label=f"[{len(node.clusters)}]", fillcolor="lightblue",
                                   style='filled')
        else:
            this_node = graph.node(this_node_code, label=node.print_node_graphviz(), fillcolor="lightblue",
                                   style='filled')

    if node.ok:
        next_left = print_tree_graphviz(graph, node.ok, aggregate)
        # graph.edge(this_node_code, next_left, label="YES [" + str(len(node.ok.clusters)) + "]", color="green")
        graph.edge(this_node_code, next_left,
                   label=f"≥{round(node.ok_min_previous_value, 2)} [{len(node.ok.clusters)}]",
                   color="green")
    if node.nan:
        next_center = print_tree_graphviz(graph, node.nan, aggregate)
        graph.edge(this_node_code, next_center, label="NA [" + str(len(node.nan.clusters)) + "]", color="gray")
    if node.nok:
        next_right = print_tree_graphviz(graph, node.nok, aggregate)
        # graph.edge(this_node_code, next_right, label="NO [" + str(len(node.nok.clusters)) + "]", color="red")
        graph.edge(this_node_code, next_right,
                   label=f"≤{round(node.nok_max_previous_value, 2)} [{len(node.nok.clusters)}]",
                   color="red")
    return this_node_code


def minimize_tree(node):
    """
    SIDE EFFECT! Remove the nodes with only one child.
    :param node:
    """
    new_node = node
    if node.ok:
        new_node.ok = minimize_tree(node.ok)
    if node.nan:
        new_node.nan = minimize_tree(node.nan)
    if node.nok:
        new_node.nok = minimize_tree(node.nok)
    if node.ok and not node.nan and not node.nok:
        return new_node.ok
    if not node.ok and node.nan and not node.nok:
        return new_node.nan
    if not node.ok and not node.nan and node.nok:
        return new_node.nok
    return new_node


def order_constraints_overall(clusters_file, reverse=False):
    """
    It orders the constraints from the most common across the cluster to the less one from the SJ2T results
    :param clusters_file:
    :param reverse: True if descending order, False for Ascending
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


def build_declare_tree_static(clusters_file, threshold, output_file, minimize=False, reverse=True):
    """
Builds the DECLARE tree according to the aggregated result of the clusters.
Constraints are used in total frequency order from the most common among the clusters to the rarest one
    :param reverse:
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
                if leaf.nan:
                    new_leaves.add(leaf.nan)
                if leaf.nok:
                    new_leaves.add(leaf.nok)
        leaves = new_leaves

    if minimize:
        minimize_tree(result_tree)

    graph = graphviz.Digraph(format='svg')
    print_tree_graphviz(graph, result_tree)
    graph.render(filename=output_file)

    return result_tree


def get_clusters_table_sum(clusters_file):
    """
    It builds a matrix from the janus results

    result [constraint x cluster] with headers for column and rows, plus last column "SUM" is the sum of the row
    Thus the column 0 are the constraints names, row 0 are the clusters names, column -1 is the sum of the row

    :param clusters_file:
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
                        # row += [float(1)]
                        row += [float(i)]
                        sum_temp += float(1)
                        # it is a vacuous satisfaction, but see atMostOne problem for the consequences of skipping it
                        # e.g. atMostOne(a) was used to distinguish clusters with a and cluster without it
                        # thus we keep the NaN and split each level in fulfilled, violated, and not activated
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


def get_clusters_table_var(clusters_file):
    """
    It builds a matrix from the janus results

    result [constraint x cluster] with headers for column and rows, plus last column "VAR" is the variance of the row
    Thus the column 0 are the constraints names, row 0 are the clusters names, column -1 is the variance of the row

    :param clusters_file:
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
                row = line + ['VAR']
                first = False
            else:
                row = [line[0]]
                for i in line[1:]:
                    if math.isnan(float(i)):
                        # row += [float(0)]  # consider vacuous satisfaction as a violation
                        # continue  # equal to +=0
                        # row += [float(1)]
                        row += [float(i)]
                        # it is a vacuous satisfaction, but see atMostOne problem for the consequences of skipping it
                        # e.g. atMostOne(a) was used to distinguish clusters with a and cluster without it
                        # thus we keep the NaN and split each level in fulfilled, violated, and not activated
                    else:
                        row += [float(i)]
                row += [statistics.variance(row[1:])]
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


def get_most_variant_constraint(cluster_table, clusters, used_constraints, reverse, grace_percent=0.02):
    view = []
    header = True
    for row in cluster_table:
        if row[0] in used_constraints:
            continue
        view_row = [row[0]]
        for cluster_i in range(len(row)):
            if cluster_table[0][cluster_i] in clusters:
                if header or not math.isnan(float(row[cluster_i])):
                    view_row += [row[cluster_i]]
                else:
                    # view_row += [1.0]  # vacuous satisfaction
                    continue
                    # used only to compute the variance, during the leaf assignment it is used its original NaN value

        if header:
            view_row += [row[-1]]
            header = False
        else:
            if len(view_row[1:]) == 0:
                view_row += [0.0]
            elif len(view_row[1:]) == 1:
                view_row += [1.0]
            else:
                view_row += [statistics.variance(view_row[1:])]
        view += [view_row]
    result = order_clusters_table(view, reverse)
    # the threshold is set to the average of the sample minus a grace percentage,
    # in this way when the clusters became very near they are not separated for very small differences
    if len(result[1][1:-1]) == 0:
        return result[1][0], 0.0
    else:
        return result[1][0], (sum(result[1][1:-1]) / len(result[1][1:-1])) - grace_percent


def build_declare_tree_dynamic(clusters_file,
                               constraint_measure_threshold,
                               branching_policy,
                               output_file=None,
                               minimize=True,
                               reverse=True,
                               min_leaf_size=0,
                               grace_percent=0.02):
    """
Builds the DECLARE tree according to the aggregated result of the clusters.
Constraints are reordered in each sub-branch according to the frequency in the remaining clusters.
    :param branching_policy:
    :param output_file:
    :param clusters_file:
    :param constraint_measure_threshold: threshold above which a constraint's measure is considered part of a cluster
    :param reverse: decreasing order if true
    :param minimize:
    :param min_leaf_size: if a node has less then or an equal amount of elements in it, then it is considered a leaf
    :param grace_percent:
    :return:
    """
    print("Data pre processing...")
    # Import initial data
    if branching_policy == "dynamic-frequency":
        clusters_table, clusters_indices, constraints_indices = get_clusters_table_sum(clusters_file)
    elif branching_policy == "dynamic-variance":
        clusters_table, clusters_indices, constraints_indices = get_clusters_table_var(clusters_file)
    else:
        print(f"ERROR! Branching policy not recognized [{branching_policy}]")
        exit(1)
    print("Building dynamic simple tree...")
    # root initialization
    result_tree = ClusterNode(threshold=constraint_measure_threshold)
    result_tree.clusters = set(clusters_table[0][1:-1])
    leaves = set()
    leaves.add(result_tree)

    # while splittable leaves
    while len(leaves) > 0:
        print(f"\rOpen leaves: {len(leaves)} ", end='')
        #   for branch
        new_leaves = set()
        for leaf in leaves:
            if len(leaf.clusters) == 1 or \
                    len(leaf.used_constraints) == len(constraints_indices) or \
                    leaf.threshold == 0.0 or \
                    len(leaf.clusters) <= min_leaf_size:
                continue
            if branching_policy == "dynamic-frequency":
                leaf.constraint = get_most_common_constraint(
                    clusters_table, leaf.clusters, leaf.used_constraints, reverse)
            else:  # elif branching_policy == "dynamic-variance":
                leaf.constraint, leaf.threshold = get_most_variant_constraint(
                    clusters_table, leaf.clusters, leaf.used_constraints, reverse, grace_percent)
                # new threshold to divide the clusters, not on their absolute adherence to the constraint, but to their relative difference
            for cluster_in_node in leaf.clusters:
                leaf.insert_child(
                    cluster_in_node,
                    clusters_table[constraints_indices[leaf.constraint]][clusters_indices[cluster_in_node]])
            if leaf.ok:
                leaf.ok.used_constraints = leaf.used_constraints.copy()
                leaf.ok.used_constraints.add(leaf.constraint)
                new_leaves.add(leaf.ok)
            if leaf.nan:
                leaf.nan.used_constraints = leaf.used_constraints.copy()
                leaf.nan.used_constraints.add(leaf.constraint)
                new_leaves.add(leaf.nan)
            if leaf.nok:
                leaf.nok.used_constraints = leaf.used_constraints.copy()
                leaf.nok.used_constraints.add(leaf.constraint)
                new_leaves.add(leaf.nok)
        leaves = new_leaves
    print("DONE")

    if minimize:
        print("Minimizing tree...")
        minimize_tree(result_tree)

    if output_file is not None:
        print("Graphviz output...")
        graph = graphviz.Digraph(format='svg')
        print_tree_graphviz(graph, result_tree, min_leaf_size > 0.0)
        graph.render(filename=output_file)

    return result_tree


def split_log(log, clusters_nodes):
    """
    Split the log into sub-logs according to the simple tree clusters nodes, returns the list of logs and trace labels
    :param log:
    :param clusters_nodes:
    """
    print(f"number of clusters : {len(clusters_nodes)}")
    traces_labels = []
    result_logs = {}
    clusters_names = {}
    # initialize sublogs with original log properties
    # for i in range(n_clusters):
    for cluster_index, cluster in enumerate(clusters_nodes):
        sub_log = EventLog()
        sub_log._attributes = log.attributes
        sub_log._classifiers = log.classifiers
        sub_log._extensions = log.extensions
        sub_log._omni = log.omni_present
        result_logs[f"Cluster_{cluster_index}"] = sub_log
        clusters_names[cluster] = f"Cluster_{cluster_index}"

    # put traces in sub-logs
    for trace_index, trace in enumerate(log):
        for cluster in clusters_nodes:
            if f'T{trace_index}' in cluster.clusters:
                current_cluster = clusters_names[cluster]
        result_logs[current_cluster].append(trace)
        traces_labels += [current_cluster]

    return result_logs, traces_labels


def get_tree_leaves(tree_root):
    """
Returns the leaves of a tree
    :param tree_root:
    :return:
    """
    leaves = set()
    open_nodes = set()
    open_nodes.add(tree_root)
    while len(open_nodes) > 0:
        new_nodes = set()
        for node in open_nodes:
            is_leaf = True
            if node.ok:
                new_nodes.add(node.ok)
                is_leaf = False
            if node.nan:
                new_nodes.add(node.nan)
                is_leaf = False
            if node.nok:
                new_nodes.add(node.nok)
                is_leaf = False
            if is_leaf:
                leaves.add(node)
        open_nodes = new_nodes
    return leaves


def build_clusters_from_traces_simple_tree(tree_root, original_log_file, output_folder):
    """
It build clusters sub-logs from the leaves of the tree

    :param output_folder:
    :param original_log_file:
    :param tree_root:
    """
    # Define clusters
    clusters_nodes = get_tree_leaves(tree_root)
    #
    log = pm.read_xes(original_log_file)
    clusters_logs, traces_labels = split_log(log, clusters_nodes)
    # export clusters logs to disk
    for cluster in clusters_logs:
        xes_exporter.apply(clusters_logs[cluster],
                           os.path.join(output_folder, f"{log.attributes['concept:name']}_cluster_{cluster}.xes"))

    # export traces labels
    with open(os.path.join(output_folder, "tree-traces-labels.csv"), 'w') as label_file:
        header = ["TRACE", "CLUSTER"]
        writer = csv.writer(label_file, delimiter=';')
        writer.writerow(header)
        writer.writerows(enumerate(traces_labels))

    # print tree with clusters
    graph = graphviz.Digraph(format='svg')
    print_tree_graphviz(graph, tree_root, True)
    graph.render(filename=os.path.join(output_folder, "Clusters_tree.dot"))


def evaluate_simple_tree_against_labeled_traces(labeled_traces_csv, traces_measures_csv, simple_tree):
    """
Given a featured event log where each trace has the constraints trace measures and the cluster label,
check the percentage of correctly classified traces

It is expected that the header of the labels csv to be like the following:
TRACE	CLUSTER

BROKEN: we cannot use a log measure to classify according to traces measures. problematic examples:
- if Resp(x,y): 0.33 means that all of all the traces containing X 33% was followed by Y,
    yet it does say nothing about the other traces without x, i.e.,
    Resp(x,y) will not classify the other traces of the same clusters without the occurrence of x


    :param labeled_traces_csv:
    :param traces_measures_csv:
    :param simple_tree:
    """

    print("Importing traces measures...")
    constraints_traces_measures, constraints_names = extract_detailed_trace_rules_perspective_csv(traces_measures_csv)

    result = 0
    traces_num = len(constraints_traces_measures)

    print("checking accuracy...")
    with open(labeled_traces_csv, 'r') as traces_file:
        csv_labels_reader = csv.DictReader(traces_file, delimiter=';')
        for trace in csv_labels_reader:
            real_class = trace["CLUSTER"]
            curr_node = simple_tree
            while True:
                if curr_node.nan is None and curr_node.ok is None and curr_node.nok is None:
                    break
                curr_constraint = curr_node.constraint
                curr_threshold = curr_node.threshold
                curr_value = constraints_traces_measures[int(trace['TRACE'])][constraints_names.index(curr_constraint)]
                if curr_value < curr_threshold:
                    if curr_node.nok is None:
                        break
                    curr_node = curr_node.nok
                elif curr_value >= curr_threshold:
                    if curr_node.ok is None:
                        break
                    curr_node = curr_node.ok
                else:
                    if curr_node.nan is None:
                        break
                    curr_node = curr_node.nan
            classified_class = curr_node.clusters
            if real_class in classified_class:
                result += 1

    print(f"Correctly classified {result}/{traces_num} thus {result / traces_num}")
