import csv
import os

import pm4py as pm


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


def load_clusters_logs_from_indices(original_log_path, clusters_indices_path):
    """
    Loads the singles clusters log from the original log given a list of indices trace-->cluster

    :param original_log_path: path to original un-clustered log
    :param clusters_indices_path: path to csv file containing the trace labels
    """
    print("NOT YET IMPLEMENTED")


def compute_f1(clusters_logs, indices_logs, output_csv_file_path):
    """
    Compute the F1 score, along with the fitness and precision, of all the clusters.
    The details are stored in the desired output file and the average is returned in output.

    :param clusters_logs:
    :param indices_logs:
    :param output_csv_file_path:
    :return: fitness_avg, precision_avg, f1_avg
    """
    header = ['CLUSTER', 'TRACES_NUM', 'FITNESS', 'PRECISION', 'F1']

    # retrieve and output stats
    with open(output_csv_file_path, 'w') as detailed_output:
        csv_detailed_out = csv.writer(detailed_output, delimiter=';')
        csv_detailed_out.writerow(header)

        fitness_avg = 0
        precision_avg = 0
        f1_avg = 0

        current_index = 0
        for current_s_log in clusters_logs:
            traces_num = len(current_s_log)

            # Model discovery
            # petri_net, initial_marking, final_marking = pm.discover_petri_net_heuristics(current_s_log)
            petri_net, initial_marking, final_marking = pm.discover_petri_net_inductive(current_s_log, 0.3)

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

            fitness_avg += fitness
            precision_avg += precision
            f1_avg += f1

            row_to_write = [indices_logs[current_index], traces_num, fitness, precision, f1]
            csv_detailed_out.writerow(row_to_write)

            current_index += 1

        fitness_avg = fitness_avg / len(clusters_logs)
        precision_avg = precision_avg / len(clusters_logs)
        f1_avg = f1_avg / len(clusters_logs)

        csv_detailed_out.writerow(["AVERAGE", "", fitness_avg, precision_avg, f1_avg])

    print(f"average Fitness: {fitness_avg}")
    print(f"average Precision: {precision_avg}")
    print(f"average F1: {f1_avg}")

    return fitness_avg, precision_avg, f1_avg


if __name__ == '__main__':
    output_file_path = "/home/alessio/Data/Phd/Research/ClusterMind/Code-ClusterMind/experiments/REAL-LIFE-EXPLORATION/SEPSIS/clusters_rules-treeSplit_rules/2-clustered-logs/f1-results.csv"
    clusters_logs, indices_logs = load_clusters_logs_from_folder(
        "/home/alessio/Data/Phd/Research/ClusterMind/Code-ClusterMind/experiments/REAL-LIFE-EXPLORATION/SEPSIS/clusters_rules-treeSplit_rules/2-clustered-logs")
    compute_f1(clusters_logs, indices_logs, output_file_path)

#
# def retrieve_clusters_statistics_multi_perspective(
#         input_logs_folder,
#         output_folder,
#         discovery_algorithm
# ):
#     """
#
#     :param base_logs_name:
#     :param input_logs_folder:
#     :param output_folder:
#     :param discovery_algorithm:
#     :return:
#     """
#     print('>>>>>>>>>> Statistics')
#     # load log
#     log = pm.read_xes(log_file_path)
#     all_events_attributes = sorted(list(attributes_filter.get_all_event_attributes_from_log(log)))
#     logs = split_log_according_to_clusters(log, clusters)
#     # export_traces_labels(log, clusters, os.path.join(output_folder, log.attributes['concept:name'] + '_traces-labels.csv'))
#     export_traces_labels_multi_perspective(log, clusters,
#                                            os.path.join(output_folder,
#                                                         f"{log.attributes['concept:name']}_traces-labels.csv"))
#     # export clusters logs to disk
#     for cluster_index in logs:
#         xes_exporter.apply(logs[cluster_index],
#                            os.path.join(output_folder, f"{log.attributes['concept:name']}_cluster_{cluster_index}.xes"))
#     header = ['CLUSTER_NUM',
#               'TRACES',
#               'TRACE-LEN-AVG',
#               'TRACE-LEN-MIN',
#               'TRACE-LEN-MAX',
#               'DURATION-MEDIAN',
#               'DURATION-MIN',
#               'DURATION-MAX',
#               'CASE-ARRIVAL-AVG',
#               'TASKS-NUM',
#               'TASKS']
#     if compute_f1:
#         header += ['FITNESS', 'PRECISION', 'F1']
#     header += all_events_attributes
#
#     # retrieve and output stats
#     with open(os.path.join(output_folder, f"{log.attributes['concept:name']}_clusters-stats.csv"), 'w') as output:
#         csv_out = csv.writer(output, delimiter=';')
#         csv_out.writerow(header)
#         f1_avg = 0
#         for cluster_index in logs:
#             current_s_log = logs[cluster_index]
#             traces_num = len(current_s_log)
#             events_avg = sum((len(i) for i in current_s_log)) / len(current_s_log)
#             events_min = min(len(i) for i in current_s_log)
#             events_max = max(len(i) for i in current_s_log)
#             unique_tasks = sorted(list(set(e['concept:name'] for t in current_s_log for e in t)))
#             unique_tasks_num = len(unique_tasks)
#             duration_median = stats.case_statistics.get_median_caseduration(current_s_log)
#             duration_min = min(stats.case_statistics.get_all_casedurations(current_s_log))
#             duration_max = max(stats.case_statistics.get_all_casedurations(current_s_log))
#             case_arrival_avg = stats.case_arrival.get_case_arrival_avg(current_s_log)
#
#             if compute_f1:
#                 # F1 fitness et all
#                 # petri_net, initial_marking, final_marking = pm.discover_petri_net_heuristics(current_s_log)
#                 petri_net, initial_marking, final_marking = pm.discover_petri_net_inductive(current_s_log, 0.3)
#                 # FITNESS
#                 # fitness_align_dictio = pm.fitness_alignments(current_s_log, petri_net, initial_marking, final_marking)
#                 fitness_replay_dictio = pm.fitness_token_based_replay(current_s_log, petri_net, initial_marking,
#                                                                       final_marking)
#                 # fitness = fitness_align_dictio['averageFitness']
#                 fitness = fitness_replay_dictio['log_fitness']
#                 # PRECISION:alignment vs token replay
#                 # precision = pm.precision_alignments(current_s_log, petri_net, initial_marking, final_marking)
#                 precision = pm.precision_token_based_replay(current_s_log, petri_net, initial_marking, final_marking)
#                 f1 = 2 * (precision * fitness) / (precision + fitness)
#                 # print(fitness_align_dictio)
#                 # print(f"Fitness: {fitness}")
#                 # print(f"Precision: {prec_align}")
#                 # print(f"F1: {f1}")
#                 f1_avg += f1
#
#             # Attributes
#             events_attributes = get_attributes_statistics_in_log(current_s_log, all_events_attributes)
#
#             row_to_write = [cluster_index, traces_num, events_avg, events_min, events_max,
#                             duration_median, duration_min, duration_max, case_arrival_avg,
#                             unique_tasks_num, unique_tasks]
#             if compute_f1:
#                 row_to_write += [fitness, precision, f1]
#             row_to_write += events_attributes
#             csv_out.writerow(row_to_write)
#
#     if compute_f1:
#         print(f"average F1: {f1_avg / len(logs)}")
#
#     return logs
