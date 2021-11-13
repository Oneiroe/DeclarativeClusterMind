import csv
import os
import sys

import pm4py as pm
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
import utils
import numpy as np
import matplotlib.cm as cm


# import DeclarativeClusterMind.cm_clustering as cm


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
        tot_clusters = len(clusters_logs)

        fitness_weighted_avg = 0
        precision_weighted_avg = 0
        f1_weighted_avg = 0
        tot_taces = 0

        current_index = 0
        for current_s_log in clusters_logs:
            traces_num = len(current_s_log)
            tot_taces += traces_num

            # Model discovery
            petri_net, initial_marking, final_marking = pm.discover_petri_net_heuristics(current_s_log)
            # petri_net, initial_marking, final_marking = pm.discover_petri_net_inductive(current_s_log, 0.3)

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

            fitness_weighted_avg += fitness * traces_num
            precision_weighted_avg += precision * traces_num
            f1_weighted_avg += f1 * traces_num

            row_to_write = [traces_clusters_labels[current_index], traces_num, fitness, precision, f1]
            csv_detailed_out.writerow(row_to_write)

            current_index += 1

        fitness_avg = fitness_avg / tot_clusters
        precision_avg = precision_avg / tot_clusters
        f1_avg = f1_avg / tot_clusters

        fitness_weighted_avg = fitness_weighted_avg / tot_taces
        precision_weighted_avg = precision_weighted_avg / tot_taces
        f1_weighted_avg = f1_weighted_avg / tot_taces

        csv_detailed_out.writerow(["AVERAGE", "", fitness_avg, precision_avg, f1_avg])
        csv_detailed_out.writerow(
            ["WEIGHTED-AVERAGE", "", fitness_weighted_avg, precision_weighted_avg, f1_weighted_avg])

    print(f"average Fitness: {fitness_avg}, weighted average:{fitness_weighted_avg}")
    print(f"average Precision: {precision_avg}, weighted average:{precision_weighted_avg}")
    print(f"average F1: {f1_avg}, weighted average:{f1_weighted_avg}")

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


def aggregate_f1_results(base_result_folder, output_file_path):
    with open(output_file_path, 'w') as out_file:
        header = ['TECHNIQUE', 'CLUSTERS_NUM', 'FITNESS', 'PRECISION', 'F1']
        header_results = ['CLUSTER', 'TRACES_NUM', 'FITNESS', 'PRECISION', 'F1']
        result_writer = csv.DictWriter(out_file, header, delimiter=';')
        result_writer.writeheader()
        for dirName, subdirList, fileList in os.walk(base_result_folder):
            for file in fileList:
                if file.endswith(".csv") and "f1" in file:
                    with open(os.path.join(dirName, file), 'r') as curr_result:
                        result_reader = csv.DictReader(curr_result, header_results, delimiter=';')
                        clusters_num = 0
                        for line in result_reader:
                            if line['CLUSTER'] == 'AVERAGE':
                                continue
                            if line['CLUSTER'] == 'CLUSTER':
                                continue
                            if line['CLUSTER'] == 'WEIGHTED-AVERAGE':
                                result_writer.writerow({
                                    'TECHNIQUE': os.path.basename(dirName),
                                    'CLUSTERS_NUM': clusters_num,
                                    'FITNESS': line['FITNESS'],
                                    'PRECISION': line['PRECISION'],
                                    'F1': line['F1']
                                })
                                continue
                            clusters_num += 1


if __name__ == '__main__':
    result_path = "/Trace-Clustering-Competitors/TraCluSi/TraCluSi-executable/output/SEPSIS"
    out_file = "/home/alessio/Data/Phd/my_code/ClusterMind/Trace-Clustering-Competitors/TraCluSi/TraCluSi-executable/output/SEPSIS/f1-score-aggregated.csv"

    aggregate_f1_results(result_path, out_file)
