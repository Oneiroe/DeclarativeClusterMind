import csv

import pm4py as pm
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
import utils
import numpy as np
import matplotlib.cm as cm


# import ClusterMind.cm_clustering as cm


def compute_silhouette(log_input_2D, traces_clusters_labels):
    mean_silhouette = silhouette_score(log_input_2D, traces_clusters_labels)
    print(f'mean Silhouette Coefficient of all samples: {mean_silhouette}')
    return mean_silhouette


def compute_f1(clusters_logs, traces_clusters_labels, output_csv_file_path):
    """
    Compute the F1 score, along with the fitness and precision, of all the clusters.
    The details are stored in the desired output file and the average is returned in output.

    :param clusters_logs: list of XES log parsers
    :param traces_clusters_labels: list of clusters labels, where each index is the index of the trace and the value is the associated cluster label
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

            row_to_write = [traces_clusters_labels[current_index], traces_num, fitness, precision, f1]
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


def visualize_silhouette(input2D, traces_cluster_labels, silhouette_avg):
    """
Visualize the silhouette score of each cluster and trace

    :param input2D:
    :param silhouette_avg:
    :param traces_cluster_labels:
    """
    sample_silhouette_values = silhouette_samples(input2D, traces_cluster_labels)
    n_clusters = len(set(traces_cluster_labels))

    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([0, len(input2D) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in sorted(set(traces_cluster_labels)):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[traces_cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()


if __name__ == '__main__':
    log_path = "/home/alessio/Data/Phd/Research/ClusterMind/Code-ClusterMind/input/SEPSIS-log.xes"
    labels_path = "/home/alessio/Data/Phd/Research/ClusterMind/Code-ClusterMind/experiments/REAL-LIFE-EXPLORATION/SEPSIS/clusters_rules-treeSplit_rules/3-results/traces-labels.csv"
    out_folder = "/home/alessio/Data/Phd/Research/ClusterMind/Trace-Clustering-competitors/cotradic/results/sepsis"

    logs_folder = "/home/alessio/Data/Phd/Research/ClusterMind/Trace-Clustering-competitors/ActiTrac/sepsis"
    clusters_logs, indices_logs = utils.load_clusters_logs_from_folder(logs_folder)

    # labels = np.array(indices_logs)
    labels = []
    # log = pm.read_xes(log_path)
    nLog = []
    l = 0
    for log in clusters_logs:
        for i in range(len(log)):
            labels += [l]
            nLog += [[]]
            for j in range(len(log[i])):
                nLog[i] += [hash(log[i][j]['concept:name'])]
        l += 1

    labels = np.array(labels)

    max_len = max(len(z) for z in nLog)

    nnLog = np.zeros((len(nLog), max_len))

    for i in range(len(nLog)):
        for j in range(len(nLog[i])):
            nnLog[i][j] = nLog[i][j]

    avg_silhouette = compute_silhouette(nnLog, labels)

    visualize_silhouette(nnLog, labels, avg_silhouette)

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
