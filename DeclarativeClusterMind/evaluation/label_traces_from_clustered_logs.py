import datetime
import sys

from pm4py.algo.filtering.log.attributes import attributes_filter

from DeclarativeClusterMind.evaluation.utils import load_clusters_logs_list_from_folder, export_traces_clusters_labels


def get_attributes_statistics_in_trace(current_trace, all_events_attributes):
    """
    COPY-PASTED from clustering_cm to avoid the import of WX. Move the function in a single place and call it from there.

    Returns the statistics of the given events attributes in a trace:
    - numerical attributes: [avg, min, max]
    - categorical attributes: [number of values, list of all values in cluster]
    - TimeStamp: [avg,min,max]

    :param current_trace:
    :param all_events_attributes:
    :return:
    """
    result = []
    for attribute in all_events_attributes:
        result += [[]]
        current_attribute_values = attributes_filter.get_attribute_values([current_trace], attribute)
        current_attributes_value_list = sorted(list(current_attribute_values.keys()))
        if len(current_attributes_value_list) == 0:
            continue
        if type(current_attributes_value_list[0]) is datetime.datetime:
            current_max = datetime.datetime.strftime(max(current_attributes_value_list), "%Y-%m-%d %H:%M:%S")
            current_min = datetime.datetime.strftime(min(current_attributes_value_list), "%Y-%m-%d %H:%M:%S")
            # This average is not weighted
            current_avg = datetime.datetime.strftime(datetime.datetime.fromtimestamp(
                sum(map(datetime.datetime.timestamp, current_attributes_value_list)) / len(
                    current_attributes_value_list)), "%Y-%m-%d %H:%M:%S")
            result[-1] = [current_avg, current_min, current_max]
        else:
            result[-1] = current_attributes_value_list
    return result


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
