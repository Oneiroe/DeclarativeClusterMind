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
import statistics
from random import random

from pm4py.objects.log.obj import EventLog
import pm4py as pm
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

import graphviz


class ClusterNode:
    def __init__(self, constraint=None, threshold=0.8):
        self.ok = None  # child node fulfilling the constraint
        self.nok = None  # child node not fulfilling the constraint
        self.nan = None  # child node not violating but also not activating the constraint
        self.constraint = constraint  # Constraint discriminating the current node
        self.threshold = threshold  # Constraint threshold discriminating the current node
        self.clusters = set()  # Set of cluster at the current node
        self.used_constraints = set()  #

    def insert_child(self, cluster_name, value):
        if math.isnan(value):
            if not self.nan:
                self.nan = ClusterNode(threshold=self.threshold)
            self.nan.clusters.add(cluster_name)
        elif value >= self.threshold:
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
            return "<" + str(self.clusters) + ">"


def print_tree_graphviz(graph, node):
    this_node_code = node.print_node_graphviz() + str(random())
    if node.constraint:
        this_node = graph.node(this_node_code, label=node.print_node_graphviz())
    else:
        this_node = graph.node(this_node_code, label=node.print_node_graphviz(), fillcolor="lightblue", style='filled')
        # this_node = graph.node(this_node_code, label="Cluster-X", fillcolor="lightblue", style='filled')

    if node.ok:
        next_left = print_tree_graphviz(graph, node.ok)
        # graph.edge(this_node_code, next_left, label="YES [" + str(len(node.ok.clusters)) + "]", color="green")
        graph.edge(this_node_code, next_left, label=f">{round(node.threshold, 2)} [{len(node.ok.clusters)}]",
                   color="green")
    if node.nan:
        next_center = print_tree_graphviz(graph, node.nan)
        graph.edge(this_node_code, next_center, label="NA [" + str(len(node.nan.clusters)) + "]", color="gray")
    if node.nok:
        next_right = print_tree_graphviz(graph, node.nok)
        # graph.edge(this_node_code, next_right, label="NO [" + str(len(node.nok.clusters)) + "]", color="red")
        graph.edge(this_node_code, next_right, label=f"<{round(node.threshold, 2)} [{len(node.nok.clusters)}]",
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
                               output_file,
                               minimize=True, reverse=True, min_leaf_size=0):
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
    print("START Building dynamic simple tree")
    # root initialization
    result_tree = ClusterNode(threshold=constraint_measure_threshold)
    result_tree.clusters = set(clusters_table[0][1:-1])
    leaves = set()
    leaves.add(result_tree)

    # while splittable leaves
    while len(leaves) > 0:
        print(f"Open leaves: {len(leaves)}")
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
                    clusters_table, leaf.clusters, leaf.used_constraints, reverse)
                # new threshold to divide the clusters, not on their absolute adherence to the constraint, but to their relative difference
            for cluster_in_node in leaf.clusters:
                leaf.insert_child(cluster_in_node, clusters_table[constraints_indices[leaf.constraint]][
                    clusters_indices[cluster_in_node]])
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
    print("END Building dynamic simple tree")

    if minimize:
        print("Minimizing tree...")
        minimize_tree(result_tree)

    print("Graphviz output...")
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
