import csv
import os

import datetime
from pm4py.algo.filtering.log.attributes import attributes_filter
import pm4py.statistics.traces.generic.log as stats
from DeclarativeClusterMind.evaluation.utils import load_clusters_logs_map_from_folder
import plotly.graph_objects as go


def get_attributes_statistics_in_log(current_log, all_events_attributes):
    """
    Returns the statistics of the given events attributes in a log:
    - numerical attributes: [avg, min, max]
    - categorical attributes: [number of values, list of all values in cluster]
    - TimeStamp: [avg,min,max]

    :param current_log:
    :param all_events_attributes:
    :return:
    """
    result = []
    for attribute in all_events_attributes:
        result += [[]]
        # map, key-value: attribute value-number of traces with that value
        current_attribute_values = attributes_filter.get_attribute_values(current_log, attribute)
        #     If attribute is numeric (int or float)
        current_attributes_value_list = sorted(list(current_attribute_values.keys()))
        if len(current_attributes_value_list) == 0:
            continue
        if type(current_attributes_value_list[0]) is int or type(current_attributes_value_list[0]) is float:
            # BEWARE sometimes INT are used for categorical encoding
            current_max = max(current_attributes_value_list)
            current_min = min(current_attributes_value_list)
            current_avg = sum([k * current_attribute_values[k] for k in current_attribute_values]) / sum(
                current_attribute_values[k] for k in current_attribute_values)
            result[-1] = [current_avg, current_min, current_max]
        #     If attribute is categorical (string)
        elif type(current_attributes_value_list[0]) is str:
            current_values = current_attributes_value_list
            current_values_num = len(current_attributes_value_list)
            result[-1] = [current_values_num, current_values]
        elif type(current_attributes_value_list[0]) is datetime.datetime:
            current_max = datetime.datetime.strftime(max(current_attributes_value_list), "%Y-%m-%d %H:%M:%S")
            current_min = datetime.datetime.strftime(min(current_attributes_value_list), "%Y-%m-%d %H:%M:%S")
            # This average is not weighted
            current_avg = datetime.datetime.strftime(datetime.datetime.fromtimestamp(
                sum(map(datetime.datetime.timestamp, current_attributes_value_list)) / len(
                    current_attributes_value_list)), "%Y-%m-%d %H:%M:%S")
            result[-1] = [current_avg, current_min, current_max]
        elif type(current_attributes_value_list[0]) is bool:
            true = 0
            false = 0
            if True in current_attribute_values:
                true = current_attribute_values[True]
            if False in current_attribute_values:
                false = current_attribute_values[False]
            # current_distribution = f"[True:{true},False:{false}]" # string with original values
            current_distribution = true / (true + false)  # proportion of true over the total number of traces
            result[-1] = current_distribution

    return result


def export_cluster_statistics_multi_perspective(logs, output_csv_file):
    """
     retrieve the statistics of the performances and attributes of the sub-logs of each clusters.
     Specifically, it retrieves for each cluster:
     - PERFORMANCES:
        - number of traces
        - average, min and max trace length
        - unique tasks in the sub-log
        - min and max timestamp (i.e. timestamp of the first and last activities of the cluster)
        + PM4Py ready to use stats
    - OTHER PERSPECTIVES (log dependent)
        - numerical attributes: max, min, avg
        - categorical attributes: number of values, list of all values in cluster

    :param logs: dict with key:label of log, value: event log reader
    :param output_csv_file:
    """
    print('>>>>>>>>>> Statistics')
    # load log
    all_events_attributes = set()
    for log in logs:
        all_events_attributes.update(attributes_filter.get_all_event_attributes_from_log(logs[log]))
    all_events_attributes = sorted(list(all_events_attributes))

    header = ['CLUSTER_NUM',
              'TRACES',
              'TRACE-LEN-AVG',
              'TRACE-LEN-MIN',
              'TRACE-LEN-MAX',
              'DURATION-MEDIAN',
              'DURATION-MIN',
              'DURATION-MAX',
              'CASE-ARRIVAL-AVG',
              'VARIANTS-NUM',
              'TASKS-NUM',
              'TASKS']
    header += all_events_attributes

    print(f"Writing stats in {output_csv_file} ...")
    # retrieve and output stats
    with open(output_csv_file, 'w') as output:
        csv_out = csv.writer(output, delimiter=';')
        csv_out.writerow(header)
        for cluster_index in logs:
            current_s_log = logs[cluster_index]
            traces_num = len(current_s_log)
            events_avg = sum((len(i) for i in current_s_log)) / len(current_s_log)
            events_min = min(len(i) for i in current_s_log)
            events_max = max(len(i) for i in current_s_log)
            unique_tasks = sorted(list(set(e['concept:name'] for t in current_s_log for e in t)))
            unique_tasks_num = len(unique_tasks)
            duration_median = stats.case_statistics.get_median_case_duration(current_s_log)
            duration_min = min(stats.case_statistics.get_all_case_durations(current_s_log))
            duration_max = max(stats.case_statistics.get_all_case_durations(current_s_log))
            case_arrival_avg = stats.case_arrival.get_case_arrival_avg(current_s_log)
            variants_num = len(stats.case_statistics.get_variant_statistics(current_s_log))

            # Attributes
            events_attributes = get_attributes_statistics_in_log(current_s_log, all_events_attributes)

            row_to_write = [cluster_index, traces_num, events_avg, events_min, events_max,
                            duration_median, duration_min, duration_max, case_arrival_avg, variants_num,
                            unique_tasks_num, unique_tasks]

            row_to_write += events_attributes
            csv_out.writerow(row_to_write)


def plot_clusters_performances_box_plots(clusters_logs, output_file=None, immediate_visualization=False):
    """
Plot the boxplot of the performance (execution time) of each cluster

    :param clusters_logs: map of pm4py xes [key:log_label]:[value:xes log]
    :param output_file: map
    :param immediate_visualization:
    """
    print("plotting boxplots of clusters performances...")
    data = {}
    for cluster_index in clusters_logs:
        current_data = [(trace[-1]['time:timestamp'] - trace[0]['time:timestamp']).total_seconds()
                        for trace in clusters_logs[cluster_index]]
        data[sum(current_data) / len(current_data)] = [cluster_index, current_data]

    fig = go.Figure()
    for cluster in sorted(data):
        fig.add_trace(go.Box(y=data[cluster][1], name=f"{data[cluster][0]}"))
    # fig = px.box(df)
    fig.update_layout(
        title="Clusters performances",
        yaxis_title="Seconds",
    )

    if output_file is None or immediate_visualization:
        fig.show()

    if output_file is not None:
        print(f"Saving boxplot in {output_file}")
        fig.write_html(f"{output_file}.html")
        fig.write_image(f"{output_file}.svg")


# if __name__ == '__main__':
#     logs_folder = "/home/alessio/Data/Phd/my_code/ClusterMind/experiments/REAL-LIFE-EXPLANATION/SEPSIS_age/1-clustered-logs"
#     output_csv_stats = os.path.join(logs_folder, 'clusters-stats.csv')
#     performances_plot = os.path.join(logs_folder, 'performances_boxplot.svg')
#
#     logs = load_clusters_logs_map_from_folder(logs_folder)
#
#     export_cluster_statistics_multi_perspective(logs, output_csv_stats)
#
#     plot_clusters_performances_box_plots(logs, performances_plot, True)
