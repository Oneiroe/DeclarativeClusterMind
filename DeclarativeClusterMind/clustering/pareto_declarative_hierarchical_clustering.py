import os
import subprocess
import csv

from random import random
import graphviz

import DeclarativeClusterMind.io.Janus3_import as j3io
import DeclarativeClusterMind.utils.aggregate_clusters_measures
import DeclarativeClusterMind.utils.split_log_according_to_declare_model as splitter
from DeclarativeClusterMind.clustering.cm_clustering import get_attributes_statistics_in_trace
from DeclarativeClusterMind.evaluation.clusters_statistics import get_attributes_statistics_in_log

import pm4py as pm
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.filtering.log.attributes import attributes_filter
import pm4py.statistics.traces.generic.log as stats

# from sklearn.metrics import silhouette_score, silhouette_samples

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
    :param log_path: path to the input event log
    :param model_path: path to the input Json model
    :param output:
    :param measure:
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


def recursive_log_split(current_node, output_folder, min_leaf_len, split_threshold):
    """
Recursively build the hierarchical cluster calling the function in each split cluster node
    :param current_node:
    :param output_folder:
    """
    # Discover model
    current_node.model = discover_declarative_model(current_node.log_path,
                                                    os.path.join(output_folder, f"model_{current_node.node_id}.json"),
                                                    0, split_threshold, SIMPLIFICATION_FLAG)

    # measure model
    event_measures, trace_measures, trace_stats, log_measures = measure_declarative_model(current_node.log_path,
                                                                                          current_node.model,
                                                                                          os.path.join(output_folder,
                                                                                                       f"output_{current_node.node_id}.csv"),
                                                                                          "Confidence")
    current_node.model_log_confidence = float(j3io.import_log_measures(log_measures)['MODEL'])

    # split_log
    output_log_80, output_log_20 = splitter.split_log_according_to_model(current_node.log_path,
                                                                         trace_measures,
                                                                         current_node.threshold)
    # halt condition check
    if len(output_log_80) <= min_leaf_len or len(output_log_20) <= min_leaf_len:
        return

    current_node.insert_child_ok(None, current_node.threshold)
    xes_exporter.apply(output_log_80, os.path.join(output_folder, f"log_{current_node.ok.node_id}.xes"))
    current_node.ok.log_path = os.path.join(output_folder, f"log_{current_node.ok.node_id}.xes")
    recursive_log_split(current_node.ok, output_folder, min_leaf_len, split_threshold)

    current_node.insert_child_nok(None, current_node.threshold)
    xes_exporter.apply(output_log_20, os.path.join(output_folder, f"log_{current_node.nok.node_id}.xes"))
    current_node.nok.log_path = os.path.join(output_folder, f"log_{current_node.nok.node_id}.xes")
    recursive_log_split(current_node.nok, output_folder, min_leaf_len, split_threshold)


def export_traces_labels_multi_perspective(input_log, clusters_nodes, output_file_path):
    """
    Export a csv file containing for each trace the corresponding cluster and values of the attributes
    :param output_file_path:
    """
    print("Exporting traces cluster labels to " + output_file_path)
    log = xes_importer.apply(input_log)
    all_events_attributes = sorted(list(attributes_filter.get_all_event_attributes_from_log(log)))
    clusters_logs = [(l, xes_importer.apply(l.log_path)) for l in clusters_nodes]
    labels = []
    with open(output_file_path, 'w') as output_file:

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
                    labels += [c[0].node_id]
                    csv_writer.writerow([trace_index, c[0].node_id] + trace_attributes)
                    break
    return labels


def export_cluster_statistics_multi_perspective(input_log, clusters_leaves, output_folder):
    log = xes_importer.apply(input_log)
    all_events_attributes = sorted(list(attributes_filter.get_all_event_attributes_from_log(log)))

    with open(os.path.join(output_folder, 'clusters-stats.csv'), 'w') as output:
        csv_out = csv.writer(output, delimiter=';')
        csv_out.writerow([
                             'CLUSTER_NUM',
                             'TRACES',
                             'TRACE-LEN-AVG',
                             'TRACE-LEN-MIN',
                             'TRACE-LEN-MAX',
                             'DURATION-MEDIAN',
                             'DURATION-MIN',
                             'DURATION-MAX',
                             'CASE-ARRIVAL-AVG',
                             'TASKS-NUM',
                             'TASKS',
                             'FITNESS', 'PRECISION', 'F1'
                         ] + all_events_attributes
                         )
        f1_avg = 0
        for cluster_node in clusters_leaves:
            current_s_log = xes_importer.apply(cluster_node.log_path)
            traces_num = len(current_s_log)
            events_avg = sum((len(i) for i in current_s_log)) / len(current_s_log)
            events_min = min(len(i) for i in current_s_log)
            events_max = max(len(i) for i in current_s_log)
            unique_tasks = sorted(list(set(e['concept:name'] for t in current_s_log for e in t)))
            unique_tasks_num = len(unique_tasks)
            duration_median = stats.case_statistics.get_median_caseduration(current_s_log)
            duration_min = min(stats.case_statistics.get_all_casedurations(current_s_log))
            duration_max = max(stats.case_statistics.get_all_casedurations(current_s_log))
            case_arrival_avg = stats.case_arrival.get_case_arrival_avg(current_s_log)

            # F1 fitness et all
            # petri_net, initial_marking, final_marking = pm.discover_petri_net_heuristics(current_s_log)
            petri_net, initial_marking, final_marking = pm.discover_petri_net_inductive(current_s_log)
            # FITNESS
            # fitness_align_dictio = pm.fitness_alignments(current_s_log, petri_net, initial_marking, final_marking)
            fitness_replay_dictio = pm.fitness_token_based_replay(current_s_log, petri_net, initial_marking,
                                                                  final_marking)
            # fitness = fitness_align_dictio['averageFitness']
            fitness = fitness_replay_dictio['log_fitness']
            # PRECISION:alignment vs token replay
            # precision = pm.precision_alignments(current_s_log, petri_net, initial_marking, final_marking)
            precision = pm.precision_token_based_replay(current_s_log, petri_net, initial_marking, final_marking)
            f1 = 2 * (precision * fitness) / (precision + fitness)
            # print(fitness_align_dictio)
            # print(f"Fitness: {fitness}")
            # print(f"Precision: {prec_align}")
            # print(f"F1: {f1}")
            f1_avg += f1

            # Attributes
            events_attributes = get_attributes_statistics_in_log(current_s_log, all_events_attributes)

            csv_out.writerow(
                [cluster_node.node_id, traces_num, events_avg, events_min, events_max,
                 duration_median, duration_min, duration_max, case_arrival_avg,
                 unique_tasks_num, unique_tasks, fitness, precision, f1] + events_attributes)

    print(f"average F1: {f1_avg / len(clusters_leaves)}")


def pareto_declarative_hierarchical_clustering(input_log, output_folder, janus_jar_path_global, simplification_flag, split_threshold=0.8, min_leaf_size=0 ):
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
    global JANUS_JAR_PATH_GLOBAL, SIMPLIFICATION_FLAG
    JANUS_JAR_PATH_GLOBAL = janus_jar_path_global
    SIMPLIFICATION_FLAG = simplification_flag
    # pre-phase
    # original_log = xes_importer.apply(input_log)
    root = ClusterNode(input_log, split_threshold)
    recursive_log_split(root, output_folder, min_leaf_size, split_threshold)

    print("### Graphviz")
    graph = graphviz.Digraph(format='svg')
    print_tree_graphviz(graph, root)
    graph.render(filename=os.path.join(output_folder, "TREE.dot"))

    print('### Result Leaves')
    root.print_leaves_dfs()
    clusters_leaves = root.get_leaves_dfs()
    print(f"Number of clusters: {len(clusters_leaves)}")

    ## Post processing ready for decision trees
    # save the reference measures for the original log
    # event_measures, trace_measures, trace_stats, log_measures = measure_declarative_model(root.log_path,
    #                                                                                       root.model,
    #                                                                                       output_folder + f"output_{root.node_id}.csv",
    #                                                                                       "Confidence")
    #
    # input3D = j3io.import_trace_measures(trace_measures, 'csv', boolean_flag=True)
    # input2D = input3D.reshape((input3D.shape[0], input3D.shape[1] * input3D.shape[2]))

    # keep only final clusters files
    root.remove_intermediary_files(output_folder)

    # aggregate the clusters X rules results
    DeclarativeClusterMind.utils.aggregate_clusters_measures.aggregate_clusters_measures(output_folder,
                                                                              "[logMeasures].csv",
                                                                              "aggregated_result.csv")
    # export traces labels
    labels = export_traces_labels_multi_perspective(input_log, clusters_leaves,
                                                    os.path.join(output_folder, "traces-labels.csv"))

    # export clusters stats
    export_cluster_statistics_multi_perspective(input_log, clusters_leaves, output_folder)

    # silhouette_score
    # TODO WIP
    # mean_silhouette = silhouette_score(input2D, labels)
    # print(f'mean Silhouette Coefficient of all samples: {mean_silhouette}')
    # DeclarativeClusterMind.cm_clustering.visualize_silhouette(None, input2D, labels, mean_silhouette)

if __name__ == '__main__':
    print("Use DeclarativeClusterMind.ui_clustering.py to launch the script via CLI/GUI ")
