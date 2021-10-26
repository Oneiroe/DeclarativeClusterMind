import csv
import os

import pm4py as pm
from pm4py.objects.log.obj import EventLog


def import_traces_clusters_labels(trace_clusters_csv_path):
    """
Imports the trace labels for cluster from the csv. order of traces is assumed
    :param trace_clusters_csv_path:
    :return:
    """
    result = []
    with open(trace_clusters_csv_path, 'r') as trace_clusters_file:
        csv_reader = csv.DictReader(trace_clusters_file, delimiter=';')
        for line in csv_reader:
            # TODO beware! works only if the CSV has ordered traces
            result += [line["CLUSTER"]]
    return result


def export_traces_clusters_labels(labels, output_csv_path):
    """
Export in output the list of trace labels given a clustering
    :param labels: list
    :param output_csv_path:
    """
    with open(output_csv_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=";")
        csv_writer.writerow(["TRACE", "CLUSTER"])
        csv_writer.writerows(enumerate(labels))


def split_log_according_to_clusters(original_xes_log_path, traces_clusters, output_folder=None):
    """
Given an event log and the cluster indices for each of its traces, it is returned in output the list of singles XES logs for each cluster
    :param original_xes_log_path:
    :param traces_clusters: list of clusters labels, where each index is the index of the trace and the value is the associated cluster label
    :param output_folder: optional, folder where to store the clusters event log
    """
    original_log = pm.read_xes(original_xes_log_path)
    labels = set(traces_clusters)
    sub_logs = dict.fromkeys(labels, [])
    # initialize sublogs with original log properties
    # for i in range(n_clusters):
    for i in labels:
        sub_log = EventLog()
        sub_log._attributes = original_log.attributes
        sub_log._classifiers = original_log.classifiers
        sub_log._extensions = original_log.extensions
        sub_log._omni = original_log.omni_present
        sub_logs[i] = sub_log
    trace_index = 0
    # put traces in sub-logs
    for trace in original_log:
        sub_logs[traces_clusters[trace_index]].append(trace)
        trace_index += 1

    if output_folder is not None:
        for sub_log in sub_logs:
            pm.write_xes(sub_logs[sub_log], os.path.join(output_folder, sub_log))

    return sub_logs


def load_clusters_logs_from_indices_file(original_log_path, trace_clusters_csv_path, output_folder=None):
    """
    Loads the singles clusters log from the original log given a list of indices trace-->cluster

    :param original_log_path: path to original un-clustered log
    :param trace_clusters_csv_path: path to csv file containing the trace labels
    :param output_folder: optional, if provided the clusters logs are exported in output
    """
    return split_log_according_to_clusters(original_log_path,
                                           import_traces_clusters_labels(trace_clusters_csv_path),
                                           output_folder)


def load_clusters_logs_from_folder(folder_path):
    """
    Given a folder, it loads all the contained .xes logs.
    It is assumed the each logs belong to one cluster.

    :param folder_path: path to the folder containing the logs of the clusters
    :return: list of XES log parsers, list of names of the logs
    """
    result = []
    indices = []

    counter = 0
    for log_file in os.listdir(folder_path):
        if log_file.endswith(".xes"):
            counter += 1
            result += [pm.read_xes(os.path.join(folder_path, log_file))]
            indices += [log_file[:-4]]
    print(f"Loaded {counter} clusters logs")
    return result, indices


def log_to_2d_array(log):
    nLog = []
    for i in range(len(log)):
        nLog += [[]]
        for j in range(len(log[i])):
            nLog[i] += [log[i][j]['concept:name']]


if __name__ == '__main__':
    log = "/home/alessio/Data/Phd/Research/ClusterMind/Code-ClusterMind/input/SEPSIS-log.xes"
    cotradict = "/home/alessio/Data/Phd/Research/ClusterMind/Trace-Clustering-competitors/cotradic/results/sepsis/sepsis Fri Oct 2021 14.02.xls"
    labels = "/home/alessio/Data/Phd/Research/ClusterMind/Trace-Clustering-competitors/cotradic/results/sepsis/sepsis_labels.csv"
    out_folder = "/home/alessio/Data/Phd/Research/ClusterMind/Trace-Clustering-competitors/cotradic/results/sepsis"

    load_clusters_logs_from_indices_file(log, labels, out_folder)
