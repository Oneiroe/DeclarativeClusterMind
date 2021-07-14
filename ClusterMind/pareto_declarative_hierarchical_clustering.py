import os
import sys
import subprocess
import csv

from random import random
import graphviz

import ClusterMind.IO.J3Tree_import as j3io
import ClusterMind.utils.aggregate_clusters_measures
import utils.split_log_according_to_declare_model as splitter

from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.filtering.log.attributes import attributes_filter

from ClusterMind.cm_clustering import get_attributes_statistics_in_trace

JANUS_JAR_PATH_GLOBAL = ""
SIMPLIFICATION_FLAG = False


class ClusterNode:
    def __init__(self, log, threshold=0.8, model_path=None):
        self.ok = None  # child node fulfilling the model
        self.nok = None  # child node not fulfilling the model
        self.log_path = log  # event log at the current node
        self.model = model_path  # model discriminating the current node
        self.threshold = threshold  # model threshold discriminating the current node
        self.model_log_confidence = 0.0  # log confidence of the model
        self.node_id = str(random())[2:]  # identifier string for the current node

    def insert_child_ok(self, log, threshold):
        self.ok = ClusterNode(log, threshold)

    def insert_child_nok(self, log, threshold):
        self.nok = ClusterNode(log, threshold)

    def print_node(self):
        if self.model:
            print(f"[{self.node_id}]")
        else:
            print(f"<[{self.node_id}]>")

    def print_tree_dfs(self):
        if self.ok:
            print('\t', end="")
            self.ok.print_tree_dfs()
        self.print_node()
        if self.nok:
            print('\t', end="")
            self.nok.print_tree_dfs()

    def print_leaves_dfs(self):
        if not (self.ok and self.nok):
            self.print_node()
            return
        if self.ok:
            self.ok.print_leaves_dfs()
        if self.nok:
            self.nok.print_leaves_dfs()

    def get_leaves_dfs(self):
        # TODO WIP
        if not (self.ok and self.nok):
            return {self}
        result = set()
        if self.ok:
            result = result.union(self.ok.get_leaves_dfs())
        if self.nok:
            result = result.union(self.nok.get_leaves_dfs())
        return result

    def print_tree_bfs(self):
        self.print_node()
        if self.ok:
            # print('\t', end="")
            self.ok.print_tree_bfs()
        if self.nok:
            # print('\t', end="")
            self.nok.print_tree_bfs()

    def print_node_graphviz(self):
        if self.ok:
            # return f"[{len(self.log)}]"
            # return f"[{len(xes_importer.apply(self.log_path))}] model:{self.model_log_confidence:.2f}"
            return f"{self.node_id} [{len(xes_importer.apply(self.log_path))}] model:{self.model_log_confidence}"
        else:
            # return f"<[{len(self.log)}]>"
            # return f"<[{len(xes_importer.apply(self.log_path))}] model:{self.model_log_confidence:.2f}>"
            return f"{self.node_id} <[{len(xes_importer.apply(self.log_path))}] model:{self.model_log_confidence}>"

    def remove_intermediary_files(self, directory):
        if self.ok or self.nok:
            for file in os.listdir(directory):
                if file.__contains__(self.node_id):
                    os.remove(os.path.join(directory, file))
        if self.ok:
            self.ok.remove_intermediary_files(directory)
        if self.nok:
            self.nok.remove_intermediary_files(directory)


def print_tree_graphviz(graph, node):
    this_node_code = node.node_id
    if node.ok:
        this_node = graph.node(this_node_code, label=node.print_node_graphviz())
    else:
        this_node = graph.node(this_node_code, label=node.print_node_graphviz(), fillcolor="lightblue", style='filled')

    if node.ok:
        next_left = print_tree_graphviz(graph, node.ok)
        graph.edge(this_node_code, next_left, label="YES", color="green")
    if node.nok:
        next_right = print_tree_graphviz(graph, node.nok)
        graph.edge(this_node_code, next_right, label="NO", color="red")
    return this_node_code


JANUS_DISCOVERY_COMMAND_LINE = lambda JANUS_JAR_PATH, INPUT_LOG, CONFIDENCE, SUPPORT, MODEL: \
    f"java -cp {JANUS_JAR_PATH} minerful.JanusOfflineMinerStarter -iLF {INPUT_LOG} -iLE xes -c {CONFIDENCE} -s {SUPPORT} -i 0 -oJSON {MODEL}"
MINERFUL_SIMPLIFIER_COMMAND_LINE = lambda JANUS_JAR_PATH, MODEL: \
    f"java -cp {JANUS_JAR_PATH} minerful.MinerFulSimplificationStarter -iMF {MODEL} -iME json -oJSON {MODEL} -s 0 -c 0 -i 0 -prune hierarchyconflictredundancydouble"
JANUS_MEASUREMENT_COMMAND_LINE = lambda JANUS_JAR_PATH, INPUT_LOG, MODEL, OUTPUT_CHECK_CSV, MEASURE: \
    f"java -cp {JANUS_JAR_PATH} minerful.JanusMeasurementsStarter -iLF {INPUT_LOG} -iLE xes -iMF {MODEL} -iME json -oCSV {OUTPUT_CHECK_CSV} -d none -nanLogSkip -measure {MEASURE}"


def discover_declarative_model(log, output_model, support_threshold=0, confidence_threshold=0.8,
                               simplification_flag=False):
    """
Lauch Janus command line to retrieve a declarative model for a specific log
    :param log: path to the input event log
    :param output_model: path to the output Json model
    :param support_threshold: [0,1] support threshold for the discovery
    :param confidence_threshold: [0.1] confidence threshold for the discovery
    :param simplification_flag: apply redundancy simplification to the discovered model
    """
    command = JANUS_DISCOVERY_COMMAND_LINE(JANUS_JAR_PATH_GLOBAL, log, confidence_threshold, support_threshold,
                                           output_model)
    print(command)

    # process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE) # to suppress the stdOutput
    process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL)
    process.wait()
    print(process.returncode)

    if simplification_flag:
        command = MINERFUL_SIMPLIFIER_COMMAND_LINE(JANUS_JAR_PATH_GLOBAL, output_model)
        print(command)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL)
        process.wait()
        print(process.returncode)

    return output_model


def measure_declarative_model(log_path, model_path, output, measure):
    """
Lauch Janus command line to retrieve a declarative model measures for a specific log
    :param log: path to the input event log
    :param model: path to the input Json model
    :param output_folder:
    """
    command = JANUS_MEASUREMENT_COMMAND_LINE(JANUS_JAR_PATH_GLOBAL, log_path, model_path, output, measure)
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL)
    process.wait()
    print(process.returncode)

    event_measures = output[:-4] + "[eventsEvaluation]"
    trace_measures = output[:-4] + "[tracesMeasures].csv"
    trace_stats = output[:-4] + "[tracesMeasuresStats].csv"
    log_measures = output[:-4] + "[logMeasures].csv"

    return event_measures, trace_measures, trace_stats, log_measures


def recursive_log_split(current_node, output_folder):
    """
Recursively build the hierarchical cluster calling the function in each split cluster node
    :param current_node:
    :param output_folder:
    """
    # Discover model
    current_node.model = discover_declarative_model(current_node.log_path,
                                                    output_folder + f"model_{current_node.node_id}.json",
                                                    0, split_threshold, SIMPLIFICATION_FLAG)

    # measure model
    event_measures, trace_measures, trace_stats, log_measures = measure_declarative_model(current_node.log_path,
                                                                                          current_node.model,
                                                                                          output_folder + f"output_{current_node.node_id}.csv",
                                                                                          "Confidence")
    current_node.model_log_confidence = float(j3io.import_log_measures(log_measures)['MODEL'])

    # split_log
    output_log_80, output_log_20 = splitter.split_log_according_to_model(current_node.log_path,
                                                                         trace_measures,
                                                                         current_node.threshold)
    # halt condition check
    if len(output_log_80) == 0 or len(output_log_20) == 0:
        return

    current_node.insert_child_ok(None, current_node.threshold)
    xes_exporter.apply(output_log_80, output_folder + f"log_{current_node.ok.node_id}.xes")
    current_node.ok.log_path = output_folder + f"log_{current_node.ok.node_id}.xes"
    recursive_log_split(current_node.ok, output_folder)

    current_node.insert_child_nok(None, current_node.threshold)
    xes_exporter.apply(output_log_20, output_folder + f"log_{current_node.nok.node_id}.xes")
    current_node.nok.log_path = output_folder + f"log_{current_node.nok.node_id}.xes"
    recursive_log_split(current_node.nok, output_folder)


def export_traces_labels_multi_perspective(input_log, clusters_nodes, output_file_path):
    """
    Export a csv file containing for each trace the corresponding cluster and values of the attributes
    :param output_file_path:
    """
    print("Exporting traces cluster labels to " + output_file_path)
    log = xes_importer.apply(input_log)
    clusters_logs = [(l, xes_importer.apply(l.log_path)) for l in clusters_nodes]
    with open(output_file_path, 'w') as output_file:
        all_events_attributes = sorted(list(attributes_filter.get_all_event_attributes_from_log(log)))

        csv_writer = csv.writer(output_file, delimiter=';')
        header = [
                     "TRACE",
                     "CLUSTER"
                 ] + all_events_attributes
        csv_writer.writerow(header)

        # put traces in sub-logs
        for trace_index in range(len(log)):
            trace_attributes = get_attributes_statistics_in_trace(log[trace_index], all_events_attributes)
            for c in clusters_logs:
                if log[trace_index] in c[1]:
                    csv_writer.writerow([trace_index, c[0].node_id] + trace_attributes)
                    break


def pareto_declarative_hierarchical_clustering(input_log, output_folder, split_threshold=0.8, min_leaf_size=0):
    """
Cluster the log according recursively through declarative models:
at each step a declarative model is discovered and
the log is divided between the traces fulfilling the model and the one that do not.
The recursion ends is:
- all the traces in the node respect the model, i.e., the node is not split
- a leaf contains min_leaf_size traces or less
    :param input_log: input event log
    :param output_folder: base folder of the output
    :param split_threshold: threshold of the described behaviour by the model
    :param min_leaf_size: minimum number of trace for a cluster
    """
    # pre-phase
    # original_log = xes_importer.apply(input_log)
    root = ClusterNode(input_log, split_threshold)
    recursive_log_split(root, output_folder)

    print("### Graphviz")
    graph = graphviz.Digraph(format='svg')
    print_tree_graphviz(graph, root)
    graph.render(filename=output_folder + "TREE.dot")

    print('### Result Leaves')
    root.print_leaves_dfs()
    clusters_leaves = root.get_leaves_dfs()
    print(f"Number of clusters: {len(clusters_leaves)}")

    ## Post processing ready for decision trees
    # keep only final clusters files
    root.remove_intermediary_files(output_folder)
    # aggregate the clusters X rules results
    ClusterMind.utils.aggregate_clusters_measures.aggregate_clusters_measures(output_folder,
                                                                              "[logMeasures].csv",
                                                                              "aggregated_result.csv")
    # export traces labels
    export_traces_labels_multi_perspective(input_log, clusters_leaves, os.path.join(output_folder, "traces-labels.csv"))


if __name__ == '__main__':
    print(sys.argv)
    input_log = sys.argv[1]
    output_folder = sys.argv[2]
    split_threshold = float(sys.argv[3])
    JANUS_JAR_PATH_GLOBAL = sys.argv[4]
    SIMPLIFICATION_FLAG = sys.argv[5] == "True"

    pareto_declarative_hierarchical_clustering(input_log, output_folder, split_threshold)
