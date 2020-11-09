import csv
import math
import os
import sys
from random import random
import graphviz
import pydot
import ClusterMind.IO.SJ2T_import as cmio


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
        graph.edge(this_node_code, print_tree_graphviz(graph, node.ok), label="YES", color="green")
    if node.nok:
        graph.edge(this_node_code, print_tree_graphviz(graph, node.nok), label="NO", color="red")
    return this_node_code


def order_constraints_overall(clusters_file):
    """
    It orders the constraints from the most common across the cluster to the less one from the SJ2T results
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
    # priority_sorted_constraints = sorted([(i, constraints_map[i]['SUM']) for i in constraints_map],
    #                                      key=lambda item: item[1], reverse=True)
    # only constraints names
    priority_sorted_constraints = [f[0] for f in sorted([(i, constraints_map[i]['SUM']) for i in constraints_map],
                                                        key=lambda item: item[1], reverse=True)]
    #  TODO remove field "SUM" from each item in constraints_map
    return priority_sorted_constraints, constraints_map, clusters_map


def build_declare_tree(clusters_file, threshold, output_file):
    """
Builds the DECLARE tree according to the aggregated result of the clusters.
Constraints are used in total frequency order from the most common among the clusters to the rarest one
    :param output_file:
    :param clusters_file:
    :param threshold:
    :return:
    """
    ordered_constraints, constraints_map, clusters_map = order_constraints_overall(clusters_file)
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
                    row += [float(i)]
                    if i == 'nan':
                        continue
                    else:
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


def order_clusters_table(clusters_table):
    """
    Given a matrix [constrain X clusters] It orders the constraints by frequency across the clusters
    :param clusters_table:
    """
    clusters_table = [clusters_table[0]] + sorted(clusters_table[1:], key=lambda item: item[-1], reverse=True)
    return clusters_table


def get_most_common_constraint(cluster_table, clusters, used_constraints):
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
    return order_clusters_table(view)[1][0]


def build_declare_tree_dynamic(clusters_file, threshold, output_file):
    """
Builds the DECLARE tree according to the aggregated result of the clusters.
Constraints are reordered in each sub-branch according to the frequency in the remaining clusters.
    :param output_file:
    :param clusters_file:
    :param threshold:
    :return:
    """
    # Import initial data
    clusters_table, clusters_indices, constraints_indices = get_clusters_table(clusters_file)
    # root initialization
    result_tree = ClusterNode(threshold=threshold)
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
            leaf.constraint = get_most_common_constraint(clusters_table, leaf.clusters, leaf.used_constraints)
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

    print("### Graphviz")
    graph = graphviz.Digraph(format='svg')
    print_tree_graphviz(graph, result_tree)
    graph.render(filename=output_file)

    return result_tree
