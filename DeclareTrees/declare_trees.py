import csv
import math
import os
import sys
from random import random

import graphviz
import pydot


class ClusterNode:
    def __init__(self, constraint=None, threshold=0.8):
        self.ok = None  # child node fulfilling the constraint
        self.nok = None  # child node not fulfilling the constraint
        self.constraint = constraint  # Constraint discriminating the current node
        self.threshold = threshold  # Constraint threshold discriminating the current node
        self.clusters = set()  # Set of cluster at th current node

    def insert_child(self, cluster_name, cluster_data):
        value = cluster_data[self.constraint]
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


def order_constraints(clusters):
    """
It order the constraints from the most common across the cluster to the less one
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
Builds the DECLARE tree according to the aggregated result of the clusters
    :param output_file:
    :param clusters_file:
    :param threshold:
    :return:
    """
    ordered_constraints, constraints_map, clusters_map = order_constraints(clusters_file)
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
                leaf.insert_child(cluster_in_node, clusters_map[cluster_in_node])
                if leaf.ok:
                    new_leaves.add(leaf.ok)
                if leaf.nok:
                    new_leaves.add(leaf.nok)
        leaves = new_leaves

    # result_tree.print_tree_dfs()
    # result_tree.print_tree_bfs()
    graph = graphviz.Digraph(format='svg')
    print_tree_graphviz(graph, result_tree)
    graph.render(filename=output_file)

    return result_tree


if __name__ == '__main__':
    print(sys.argv)
    clusters_file = sys.argv[1]
    constraints_threshold = float(sys.argv[2])
    output_file = sys.argv[3]
    build_declare_tree(clusters_file, constraints_threshold, output_file)
