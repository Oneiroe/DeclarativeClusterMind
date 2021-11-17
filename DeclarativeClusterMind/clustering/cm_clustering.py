import os
import sys
import csv
import datetime

import DeclarativeClusterMind.io.Janus3_import as j3io
from DeclarativeClusterMind.evaluation import utils, clusters_statistics

import pm4py as pm

from pm4py.objects.log.util import get_log_representation
from pm4py.algo.filtering.log.attributes import attributes_filter

import pandas as pd
import numpy as np
import sklearn.cluster as cluster
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

from matplotlib import pyplot as plt
import matplotlib.cm as cm

import wx


def cluster_traces(input2D, apply_pca_flag, features, algorithm, nc=None):
    """
Cluster the traces according to the selected algorithm
    :param nc:
    :param input2D:
    :param apply_pca_flag:
    :param features:
    :param algorithm:
    :return:
    """
    ## CLUSTERING

    #  number of clusters
    if nc is None or nc <= 1:
        nc = input2D.shape[1]  # number of features actually used
        # nc = features  # total number of constraints
        # nc = 10  # fixed number, for debugging

        if nc > len(input2D):
            # when the number of clusters is greater than the number of traces algorithms like k-means trow exceptions
            nc = len(input2D)
        if nc == 1:
            if apply_pca_flag:
                nc = features
            else:
                nc = pd.DataFrame(input2D).nunique()[0]  # works only if PCA is not enabled

    if algorithm == 'kmeans':
        # K-means
        try:
            print("K-Kmeans...")
            kmeans = cluster.KMeans(n_clusters=nc).fit(input2D)
            print("Kmeans: \t\t" + str(kmeans.labels_))
            return kmeans
        except:
            print("K-Means error:", sys.exc_info()[0])
    elif algorithm == 'affinity':
        # Affinity
        try:
            affinity = cluster.AffinityPropagation(random_state=0).fit(input2D)
            print("Affinity: \t\t" + str(affinity.labels_))
            return affinity
        except:
            print("affinity error:", sys.exc_info()[0])
    elif algorithm == 'meanshift':
        # mean-shift
        try:
            mean_shift = cluster.MeanShift().fit(input2D)
            print("mean_shift: \t" + str(mean_shift.labels_))
            return mean_shift
        except:
            print("mean_shift error:", sys.exc_info()[0])
    elif algorithm == 'agglomerative':
        # Agglomerative hierarchical
        try:
            agglomerative = cluster.AgglomerativeClustering(n_clusters=nc).fit(input2D)
            print("Agglomerative: \t" + str(agglomerative.labels_))
            return agglomerative
        except:
            print("Agglomerative error:", sys.exc_info()[0])
    elif algorithm == 'spectral':
        # Spectral
        try:
            spectral = cluster.SpectralClustering(n_clusters=nc).fit(input2D)
            print("Spectral: \t\t" + str(spectral.labels_))
            return spectral
        except:
            print("Spectral error:", sys.exc_info()[0])
    elif algorithm == 'dbscan':
        # DBSCAN
        try:
            dbscan = cluster.DBSCAN().fit(input2D)
            print("DBSCAN: \t\t" + str(dbscan.labels_))
            return dbscan
        except:
            print("DBSCAN error:", sys.exc_info()[0])
    elif algorithm == 'optics':
        # OPTICS
        try:
            optics = cluster.OPTICS().fit(input2D)
            print("OPTICS: \t\t" + str(optics.labels_))
            return optics
        except:
            print("OPTICS error:", sys.exc_info()[0])
    elif algorithm == 'birch':
        # birch
        try:
            birch = cluster.Birch(n_clusters=nc).fit(input2D)
            # birch = cluster.Birch().fit(input2D)
            print("birch: \t\t\t" + str(birch.labels_))
            return birch
        except:
            print("birch error:", sys.exc_info()[0])
    elif algorithm == 'gaussian':  # DO NOT USE THIS!
        # gaussian
        gaussian = GaussianMixture(n_components=nc).fit(input2D)
        print("gaussian: \t\t" + str(gaussian.labels_))
    else:
        print("Algorithm not recognized")
        #     TODO rise exception and close
        return None


def cluster_traces_from_rules_trace_measures(trace_measures_file_path, algorithm='optics', boolean_confidence=True,
                                             apply_pca=True, number_of_clusters=0):
    """
Cluster traces according to declarative rules measurements evaluated on them
    :param number_of_clusters:
    :param trace_measures_file_path:
    :param algorithm:
    :param boolean_confidence:
    :param apply_pca:
    :return:
    """
    # INPUT IMPORT
    file_format = trace_measures_file_path.split(".")[-1]
    # input3D = cmio.import_SJ2T(file_path, file_format, boolean=boolean_confidence)
    input3D = j3io.import_trace_measures(trace_measures_file_path, file_format, boolean_flag=boolean_confidence)
    input2D = input3D.reshape((input3D.shape[0], input3D.shape[1] * input3D.shape[2]))
    print("2D shape:" + str(input2D.shape))

    # PRE-PROCESSING

    # Clean NaN and infinity
    input2D = np.nan_to_num(input2D, posinf=1.7976931348623157e+100, neginf=-1.7976931348623157e+100)
    # input2D = np.nan_to_num(np.power(input2D, -10), posinf=1.7976931348623157e+100, neginf=-1.7976931348623157e+100)
    # input2D = np.nan_to_num(input2D, posinf=100, neginf=-100)

    # reduce dimensions with PCA
    # pca = None
    pca_variance = 0.98
    pca = PCA(pca_variance)
    pca.fit(input2D)
    if apply_pca:
        input2D = pca.transform(input2D)
        print('Dimension of data PCA= ' + str(input2D.shape))

    # CLUSTERING
    print("Clustering...")
    traces = input3D.shape[0]
    constraints = input3D.shape[1]
    measures = input3D.shape[2]

    clusters = cluster_traces(input2D, apply_pca, constraints, algorithm, number_of_clusters)

    return clusters, pca, input2D


def get_attributes_statistics_in_trace(current_trace, all_events_attributes):
    """
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


def export_traces_labels_multi_perspective(log, clusters, output_file_path):
    """
    Export a csv file containing for each trace the corresponding cluster and values of the attributes
    :param output_file_path:
    """
    print("Exporting traces cluster labels to " + output_file_path)
    with open(output_file_path, 'w') as output_file:
        all_events_attributes = sorted(list(attributes_filter.get_all_event_attributes_from_log(log)))
        performances = ["case-duration", "case-length", "case-unique-tasks"]

        csv_writer = csv.writer(output_file, delimiter=';')
        header = [
                     "TRACE",
                     "CLUSTER"
                 ] + all_events_attributes + performances
        csv_writer.writerow(header)

        # put traces in sub-logs
        for trace_index in range(len(log)):
            trace = log[trace_index]
            trace_attributes = get_attributes_statistics_in_trace(trace, all_events_attributes)
            trace_performances = [
                (trace[-1]['time:timestamp'] - trace[0]['time:timestamp']).total_seconds(),
                len(trace),
                len(set(e['concept:name'] for e in trace))
            ]
            csv_writer.writerow([trace_index, clusters.labels_[trace_index]] + trace_attributes + trace_performances)


def export_centroids_constraints(clusters, pca, threshold, measures_num, constraints_names, output_folder):
    print(">>>>>visualize centroids constraints")
    try:
        res_matrix = [list() for i in range(len(clusters.cluster_centers_))]
        for centroid_index in range(len(clusters.cluster_centers_)):
            centroid = clusters.cluster_centers_[centroid_index]
            c = pca.inverse_transform(centroid)
            for i in range(len(constraints_names)):
                if c[1 + measures_num * i] > threshold:
                    # confidence>threshold, it is the 2nd measure
                    res_matrix[centroid_index] += [1]
                else:
                    res_matrix[centroid_index] += [0]
        # export to csv
        with open(os.path.join(output_folder, 'centroids-constraints.csv'), 'w') as output:
            csv_output = csv.writer(output, delimiter=';')
            # header
            csv_output.writerow(constraints_names)
            # values
            csv_output.writerows(res_matrix)
    except:
        print("ERROR >>> Centroid export error:", sys.exc_info()[0])


def export_pca_relevant_feature(pca, feature_names, output_folder):
    print(">>>>>visualize PCA selected features")
    with open(os.path.join(output_folder, 'pca-features.csv'), 'w') as output:
        n_pcs = pca.components_.shape[0]
        pca_features = pd.DataFrame(pca.components_, columns=feature_names)
        # Print all features importance
        csv_output = csv.writer(output, delimiter=';')
        csv_output.writerow(feature_names)
        for i in range(pca_features.shape[0]):
            csv_output.writerow(pca_features.transpose()[i])

        # Correlation Matrix between constraints
        # plt.matshow(pca_features.corr())
        # plt.show()

        # most important features
        most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
        most_important_names = [feature_names[most_important[i]] for i in range(n_pcs)]
        dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}
        print(pd.DataFrame(dic.items()))


def visualize_silhouette(input2D, traces_cluster_labels, silhouette_avg, output_folder=None,
                         immediate_visualization=False):
    """
Visualize the silhouette score of each cluster and trace

    :param output_folder:
    :param silhouette_avg:
    :param clusters:
    :param input2D:
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

    if output_folder is not None:
        output_file = os.path.join(output_folder, "silhouette")
        print(f"Saving silhouette plot in {output_file}")
        plt.savefig(f"{output_file}.svg")

    if output_folder is None or immediate_visualization:
        plt.show()


def behavioural_clustering(trace_measures_csv_file_path, log_file_path, clustering_algorithm, boolean_confidence,
                           output_folder,
                           visualization_flag, apply_pca_flag, number_of_clusters):
    """
    Cluster the traces of a log according to a set of declarative rules and their trace measurements

    :param number_of_clusters:
    :param trace_measures_csv_file_path:
    :param log_file_path:
    :param clustering_algorithm:
    :param boolean_confidence:
    :param output_folder:
    :param visualization_flag:
    :param apply_pca_flag:
    """
    print(clustering_algorithm)

    # traces_num, constraints_num, measures_num, constraints_names = cmio.retrieve_SJ2T_csv_data(sj2t_csv_file_path)
    traces_num, constraints_num, measures_num, constraints_names = j3io.retrieve_trace_measures_metadata(
        trace_measures_csv_file_path)

    # CLUSTERING
    clusters, pca, input2D = cluster_traces_from_rules_trace_measures(trace_measures_csv_file_path,
                                                                      clustering_algorithm,
                                                                      boolean_confidence,
                                                                      apply_pca_flag,
                                                                      number_of_clusters)
    log = pm.read_xes(log_file_path)
    clustering_postprocessing_and_visualization(log, output_folder, input2D, clusters, measures_num,
                                                constraints_names, visualization_flag,
                                                apply_pca_flag, pca)


def clustering_postprocessing_and_visualization(log, output_folder, input2D, clusters, measures_num,
                                                features_names, visualization_flag,
                                                apply_pca_flag, pca):
    """
Retrieve and export cluster statistics and visualization if enabled

    :param output_folder:
    :param log:
    :param visualization_flag:
    :param input2D: 
    :param clusters: 
    :param measures_num: 
    :param features_names: 
    :param apply_pca_flag: 
    :param pca: 
    """
    # STATS
    #
    print('>>>>>>>>>> Statistics')
    # load log
    export_traces_labels_multi_perspective(log,
                                           clusters,
                                           os.path.join(output_folder,
                                                        f"{log.attributes['concept:name']}_traces-labels.csv"))
    clusters_logs = utils.split_log_according_to_clusters(log, clusters.labels_, output_folder)
    # export clusters logs to disk
    # for cluster_index in logs:
    #     xes_exporter.apply(logs[cluster_index],
    #                        os.path.join(output_folder, f"{log.attributes['concept:name']}_cluster_{cluster_index}.xes"))

    clusters_statistics.export_cluster_statistics_multi_perspective(
        clusters_logs,
        os.path.join(output_folder, f"{log.attributes['concept:name']}_clusters-stats.csv"))

    if apply_pca_flag:
        if measures_num > 0:
            constraint_measures_names = [c + "_m" + str(m) for c in features_names for m in range(measures_num)]
            export_pca_relevant_feature(pca, constraint_measures_names, output_folder)
        else:
            export_pca_relevant_feature(pca, features_names, output_folder)

    # average Silhouette coefficient
    traces_cluster_labels = clusters.fit_predict(input2D)
    mean_silhouette = silhouette_score(input2D, traces_cluster_labels)
    print(f'mean Silhouette Coefficient of all samples: {mean_silhouette}')

    # VISUALIZATION
    if visualization_flag:
        print(">>>>>>>>>>>> Visualization")
        # plot_clusters_imperative_models(clusters_logs)

        clusters_statistics.plot_clusters_performances_box_plots(clusters_logs,
                                                                 os.path.join(output_folder,
                                                                              "performances_boxplot.svg"))

        visualize_silhouette(input2D, traces_cluster_labels, mean_silhouette, output_folder)

        # plot_tSNE_3d(input2D, clusters)
        # visualize_heatmap(input2D, clusters)

        # threshold = 0.95
        # labels, traces_index = j3io.import_trace_labels(trace_measures_csv_file_path, constraints_num, threshold)
        # plot_constraints_in_clusters(clusters, labels, traces_index)

        # export_centroids_constraints(clusters, pca, threshold, measures_num, constraints_names, output_folder)
    else:
        print(">>>>>>>>>>>> Visualization SKIPPED")


def attribute_clustering(log_file_path, clustering_algorithm, output_folder, visualization_flag, apply_pca_flag,
                         number_of_clusters):
    """
    Cluster the traces of a log according to the log categorical attributes

    :param log_file_path:
    :param clustering_algorithm:
    :param output_folder:
    :param visualization_flag:
    :param apply_pca_flag:
    """
    log = pm.read_xes(log_file_path)

    data, feature_names = get_log_representation.get_default_representation(log)
    # 1-hot encoding
    input2D = pd.DataFrame(data, columns=feature_names)

    traces = input2D.shape[0]
    attributes = input2D.shape[1]

    print(clustering_algorithm)
    print("Traces: " + str(traces))
    print("Attributes: " + str(attributes))
    print(feature_names)

    # Clean NaN and infinity
    input2D = np.nan_to_num(input2D, posinf=1.7976931348623157e+100, neginf=-1.7976931348623157e+100)
    # input2D = np.nan_to_num(np.power(input2D, -10), posinf=1.7976931348623157e+100, neginf=-1.7976931348623157e+100)
    # input2D = np.nan_to_num(input2D, posinf=100, neginf=-100)

    # reduce dimensions with PCA
    # pca = None
    pca_variance = 0.98
    pca = PCA(pca_variance)
    if apply_pca_flag:
        pca.fit(input2D)
        input2D = pca.transform(input2D)
        print('Dimension of data PCA= ' + str(input2D.shape))

    # CLUSTERING
    print("Clustering...")
    clusters = cluster_traces(input2D, apply_pca_flag, attributes, clustering_algorithm, number_of_clusters)

    clustering_postprocessing_and_visualization(log, output_folder, input2D, clusters, 0, feature_names,
                                                visualization_flag,
                                                apply_pca_flag, pca)


def performances_clustering(log_file_path, clustering_algorithm, output_folder, visualization_flag, apply_pca_flag,
                            number_of_clusters):
    """
    Cluster the traces of a log according to their performances, i.e.,
        - duration time
        - trace length
        - tasks number

    :param number_of_clusters:
    :param log_file_path:
    :param clustering_algorithm:
    :param output_folder:
    :param visualization_flag:
    :param apply_pca_flag:
    """

    log = pm.read_xes(log_file_path)

    # data = [[trace[-1]['time:timestamp'] - trace[0]['time:timestamp'], len(trace), len(set(e['concept:name'] for e in trace))] for trace in log]
    data = [[
        (trace[-1]['time:timestamp'] - trace[0]['time:timestamp']).total_seconds(),
        len(trace),
        len(set(e['concept:name'] for e in trace))
    ] for trace in log]
    feature_names = [
        "case-duration",
        "case-length",
        "case-unique-tasks"
    ]
    # 1-hot encoding
    input2D = pd.DataFrame(data, columns=feature_names)

    traces = input2D.shape[0]
    attributes = input2D.shape[1]

    print(clustering_algorithm)
    print("Traces: " + str(traces))
    print("Attributes: " + str(attributes))
    print(feature_names)

    # Clean NaN and infinity
    input2D = np.nan_to_num(input2D, posinf=1.7976931348623157e+100, neginf=-1.7976931348623157e+100)
    # input2D = np.nan_to_num(np.power(input2D, -10), posinf=1.7976931348623157e+100, neginf=-1.7976931348623157e+100)
    # input2D = np.nan_to_num(input2D, posinf=100, neginf=-100)

    # reduce dimensions with PCA
    # pca = None
    pca_variance = 0.98
    pca = PCA(pca_variance)
    if apply_pca_flag:
        pca.fit(input2D)
        input2D = pca.transform(input2D)
        print('Dimension of data PCA= ' + str(input2D.shape))

    # CLUSTERING
    print("Clustering...")
    clusters = cluster_traces(input2D, apply_pca_flag, attributes, clustering_algorithm, number_of_clusters)

    clustering_postprocessing_and_visualization(log, output_folder, input2D, clusters, 0, feature_names,
                                                visualization_flag,
                                                apply_pca_flag, pca)


def select_attribute_GUI(options_list):
    """
Open a WX window to select from the available attributes of the event log
    :rtype: object
    """
    dlg = wx.SingleChoiceDialog(None,
                                caption='Select attribute',
                                message="Select the desired attribute upon which basing the clustering of the log"'Select attribute',
                                choices=options_list)
    # dlg = wx.MessageDialog(None, 'App Title', style=wx.CB_READONLY)
    dlg.ShowModal()

    return dlg.GetStringSelection()


def specific_attribute_clustering(log_file_path, clustering_algorithm, output_folder, visualization_flag,
                                  number_of_clusters):
    """
    Cluster the traces of a log according to a single specific categorical attributes of the log (selected by user on input

    :param number_of_clusters:
    :param log_file_path:
    :param clustering_algorithm:
    :param output_folder:
    :param visualization_flag:
    """
    log = pm.read_xes(log_file_path)

    data, feature_names = get_log_representation.get_default_representation(log)
    # 1-hot encoding
    input2D = pd.DataFrame(data, columns=feature_names)

    # Get attribute to use
    while True:
        try:
            app = wx.App()
            print([f"{i},{e}" for i, e in enumerate(feature_names)])
            # selected_attribute_index = int(input("Select feature index: "))
            # feature_name = feature_names[selected_attribute_index]
            feature_name = select_attribute_GUI(feature_names)
            print(f"Selected feature: {feature_name}")
            break
        except ValueError:
            print("This is not a valid number, try again.")
        except IndexError:
            print("Index out of range, try again.")

    input2D = input2D[[feature_name]]

    traces = input2D.shape[0]
    attributes = input2D.shape[1]

    print(clustering_algorithm)
    print("Traces: " + str(traces))
    print("Attributes: " + str(attributes))
    print(feature_name)

    # Clean NaN and infinity
    input2D = np.nan_to_num(input2D, posinf=1.7976931348623157e+100, neginf=-1.7976931348623157e+100)
    # input2D = np.nan_to_num(np.power(input2D, -10), posinf=1.7976931348623157e+100, neginf=-1.7976931348623157e+100)
    # input2D = np.nan_to_num(input2D, posinf=100, neginf=-100)

    # CLUSTERING
    print("Clustering...")
    clusters = cluster_traces(input2D, False, attributes, clustering_algorithm, number_of_clusters)

    clustering_postprocessing_and_visualization(log, output_folder, input2D, clusters, 0, feature_names,
                                                visualization_flag,
                                                False, None)


def mixed_clustering(trace_measures_csv_file_path, log_file_path, clustering_algorithm, boolean_confidence,
                     output_folder, visualization_flag, apply_pca_flag, number_of_clusters):
    """
    Cluster the traces of a log according to both a set of declarative rules and the log attributes

    TODO BEWARE using performances in the clustering may cover all the other effects, especially using PCA. Consider to have a selection of parameters to use

    :param number_of_clusters:
    :param trace_measures_csv_file_path:
    :param log_file_path:
    :param clustering_algorithm:
    :param boolean_confidence:
    :param output_folder:
    :param visualization_flag:
    :param apply_pca_flag:
    """
    # INPUT IMPORT
    traces_num, constraints_num, measures_num, constraints_names = j3io.retrieve_trace_measures_metadata(
        trace_measures_csv_file_path)
    file_format = trace_measures_csv_file_path.split(".")[-1]
    input3D = j3io.import_trace_measures(trace_measures_csv_file_path, file_format, boolean_flag=boolean_confidence)
    input2D_rules = pd.DataFrame(input3D.reshape((input3D.shape[0], input3D.shape[1] * input3D.shape[2])),
                                 columns=constraints_names)

    print("2D shape rules:" + str(input2D_rules.shape))

    log = pm.read_xes(log_file_path)

    attributes_data, attributes_names = get_log_representation.get_default_representation(log)
    # 1-hot encoding
    input2D_attributes = pd.DataFrame(attributes_data, columns=attributes_names)

    traces_num = input2D_attributes.shape[0]
    attributes_num = input2D_attributes.shape[1]
    print("Attributes: " + str(attributes_num))
    print(attributes_names)

    # data_performances = [[
    #     (trace[-1]['time:timestamp'] - trace[0]['time:timestamp']).total_seconds(),
    #     len(trace),
    #     len(set(e['concept:name'] for e in trace))
    # ] for trace in log]
    # performances_names = [
    #     "case-duration",
    #     "case-length",
    #     "case-unique-tasks"
    # ]
    # # 1-hot encoding
    # input2D_performances = pd.DataFrame(data_performances, columns=performances_names)

    # input2D = pd.concat([input2D_rules, input2D_attributes, input2D_performances], axis=1)
    input2D = pd.concat([input2D_rules, input2D_attributes], axis=1)
    # features_names = np.concatenate([constraints_names, attributes_names, performances_names])
    features_names = np.concatenate([constraints_names, attributes_names])
    features_num = features_names.shape[0]

    print(
        # f"FEATURES  rules:{len(constraints_names)} attributes:{len(attributes_names)} performances:{len(performances_names)}"
        f"FEATURES  rules:{len(constraints_names)} attributes:{len(attributes_names)}"
    )

    # Clean NaN and infinity
    input2D = np.nan_to_num(input2D, posinf=1.7976931348623157e+100, neginf=-1.7976931348623157e+100)
    # input2D = np.nan_to_num(np.power(input2D, -10), posinf=1.7976931348623157e+100, neginf=-1.7976931348623157e+100)
    # input2D = np.nan_to_num(input2D, posinf=100, neginf=-100)

    print('Dimension of data= ' + str(input2D.shape))
    # reduce dimensions with PCA
    # pca = None
    pca_variance = 0.98
    pca = PCA(pca_variance)
    if apply_pca_flag:
        pca.fit(input2D)
        input2D = pca.transform(input2D)
        print('Dimension of data PCA= ' + str(input2D.shape))

    # CLUSTERING
    print("Clustering...")
    print(clustering_algorithm)
    print("Traces: " + str(traces_num))
    print("Features: " + str(features_num))
    clusters = cluster_traces(input2D, apply_pca_flag, features_num, clustering_algorithm, number_of_clusters)

    clustering_postprocessing_and_visualization(log, output_folder, input2D, clusters, measures_num,
                                                features_names, visualization_flag,
                                                apply_pca_flag, pca)


if __name__ == '__main__':
    print("Use DeclarativeClusterMind.ui_clustering.py to launch the script via CLI/GUI")
