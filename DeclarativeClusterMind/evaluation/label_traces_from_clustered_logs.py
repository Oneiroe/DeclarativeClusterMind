import sys

from pm4py.algo.filtering.log.attributes import attributes_filter

from DeclarativeClusterMind.clustering.cm_clustering import get_attributes_statistics_in_trace
from DeclarativeClusterMind.evaluation.utils import load_clusters_logs_list_from_folder, export_traces_clusters_labels


def export_traces_clusters_labels_from_logs(folder, output_file="traces-labels.csv"):
    clusters_logs, clusters_indices = load_clusters_logs_list_from_folder(folder)
    labels = []
    all_events_attributes = sorted(list(attributes_filter.get_all_event_attributes_from_log(clusters_logs[0])))
    header = ["TRACE", "CLUSTER"] + all_events_attributes
    for cluster_index, log in enumerate(clusters_logs):
        for trace in log:
            labels += [[clusters_indices[cluster_index]] + get_attributes_statistics_in_trace(trace,
                                                                                             all_events_attributes)]
    export_traces_clusters_labels(labels, output_file, header)


if __name__ == '__main__':
    print(sys.argv)
    logs_folder = sys.argv[1]
    output_name = sys.argv[2]
    export_traces_clusters_labels_from_logs(logs_folder, output_name)
