# import sklearn.cluster as clustering
import sys
import csv
from collections import Counter
import datetime

# import ClusterMind.IO.SJ2T_import as cmio
import ClusterMind.IO.J3Tree_import as j3io

import pm4py as pm
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.importer.xes import importer as xes_importer
import pm4py.statistics.traces.log as stats
from pm4py.objects.log.util import get_log_representation
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualization

import pandas as pd
import numpy as np
import sklearn.cluster as cluster
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy

import plotly.express as px
from matplotlib import pyplot as plt
import seaborn as sns


def block_diag_einsum(arr, num):
    rows, cols = arr.shape
    result = np.zeros((num, rows, num, cols), dtype=arr.dtype)
    diag = np.einsum('ijik->ijk', result)
    diag[:] = arr
    return result.reshape(rows * num, cols * num)


def initialize_centroids(measures_num, centroids_num):
    """
    Create diagonal matrix of 1s representing centroids in which all the measure of a centroids
    are equal to 1 for just a specific constraint and 0 elsewhere.
    
    Example. Given 3 centroids and 4 measures the result will be
    1111 0000 0000
    0000 1111 0000
    0000 0000 1111
    
    :param measures_num:
    :param centroids_num:
    :return:
    """
    a = np.zeros((1, measures_num))
    a.fill(1)
    return block_diag_einsum(a, centroids_num)


def cluster_traces(input2D, traces, constraints, measures, algorithm):
    ## CLUSTERING

    nc = constraints  # number of clusters
    # nc = 10  # number of clusters

    if nc > len(input2D):
        # when the number of clusters is greater than the number of traces algorithms like k-means trow exceptions
        nc = len(input2D)

    if (algorithm == 'kmeans'):
        # K-means
        try:
            print("K-Kmeans...")
            # centroids_init = initialize_centroids(measures, nc)
            # kmeans = cluster.KMeans(n_clusters=nc, init=centroids_init).fit(input2D)
            kmeans = cluster.KMeans(n_clusters=nc).fit(input2D)
            print("Kmeans: \t\t" + str(kmeans.labels_))
            return kmeans
        except:
            print("K-Means error:", sys.exc_info()[0])
    elif (algorithm == 'affinity'):
        # Affinity
        try:
            affinity = cluster.AffinityPropagation(random_state=0).fit(input2D)
            print("Affinity: \t\t" + str(affinity.labels_))
            return affinity
        except:
            print("affinity error:", sys.exc_info()[0])
    elif (algorithm == 'meanshift'):
        # mean-shift
        try:
            mean_shift = cluster.MeanShift().fit(input2D)
            print("mean_shift: \t" + str(mean_shift.labels_))
            return mean_shift
        except:
            print("mean_shift error:", sys.exc_info()[0])
    elif (algorithm == 'agglomerative'):
        # Agglomerative hierarchical
        try:
            agglomerative = cluster.AgglomerativeClustering(n_clusters=nc).fit(input2D)
            print("Agglomerative: \t" + str(agglomerative.labels_))
            return agglomerative
        except:
            print("Agglomerative error:", sys.exc_info()[0])
    elif (algorithm == 'spectral'):
        # Spectral
        try:
            spectral = cluster.SpectralClustering(n_clusters=nc).fit(input2D)
            print("Spectral: \t\t" + str(spectral.labels_))
            return spectral
        except:
            print("Spectral error:", sys.exc_info()[0])
    elif (algorithm == 'dbscan'):
        # DBSCAN
        try:
            dbscan = cluster.DBSCAN().fit(input2D)
            print("DBSCAN: \t\t" + str(dbscan.labels_))
            return dbscan
        except:
            print("DBSCAN error:", sys.exc_info()[0])
    elif (algorithm == 'optics'):
        # OPTICS
        try:
            optics = cluster.OPTICS().fit(input2D)
            print("OPTICS: \t\t" + str(optics.labels_))
            return optics
        except:
            print("OPTICS error:", sys.exc_info()[0])
    elif (algorithm == 'birch'):
        # birch
        try:
            birch = cluster.Birch(n_clusters=nc).fit(input2D)
            # birch = cluster.Birch().fit(input2D)
            print("birch: \t\t\t" + str(birch.labels_))
            return birch
        except:
            print("birch error:", sys.exc_info()[0])
    elif (algorithm == 'gaussian'):  # DO NOT USE THIS!
        # gaussian
        gaussian = GaussianMixture(n_components=nc).fit(input2D)
        print("gaussian: \t\t" + str(gaussian.labels_))
    else:
        print("Algorithm not recognized")
        #     TODO rise exception and close
        return None


def visualize_matrices(input2D, clusters):
    print(">>>> Visualize heatmap of MMM")
    try:
        centorids = clusters.cluster_centers_
        fig = px.imshow(centorids, title='Centroids')
        fig.show()
    except:
        print("ERROR >>> Centroid visualization error:", sys.exc_info()[0])

    fig2 = px.imshow(input2D, title='MMM-2D')
    fig2.show()


def visualize_constraints_in_clusters(clusters, labels, traces_index):
    print(">>>>> Visualize constraints present in the clusters")
    res_df = pd.DataFrame()
    res_df_naive = pd.DataFrame()
    res = {}
    i = 0
    n_clusters = max(clusters.labels_) - min(clusters.labels_) + 1
    for c in clusters.labels_:
        res.setdefault(c, Counter())
        for label in labels[traces_index[i]]:
            # beware: a trace with multiple active constraints will count +1 for each label
            res[c].setdefault(label, 0)
            res[c][label] += 1
        i += 1
    for cluster in res.keys():
        for rule in res[cluster]:
            res_df = res_df.append({'cluster': cluster, 'rule': rule, 'amount': res[cluster][rule]}, ignore_index=True)

    for cluster in res.keys():
        # in case there are "empty" cluster, this cycle makes sure that they are included in the bar chart.
        # note. an "empty" cluster is a cluster in which no rule is satisfied, thus it still brings information
        if len(res[cluster]) == 0:
            # Beware! Using the variable "rule" implies that in the previous cycle at least one rule is present in a cluster
            res_df = res_df.append({'cluster': cluster, 'rule': rule, 'amount': 0}, ignore_index=True)
    print("construct fig")
    fig = px.bar(res_df,
                 # fig = px.bar(res_df[(res_df['cluster'] > 10) & (res_df['cluster'] < 20)],
                 # fig = px.bar(res_df[res_df['cluster'].isin([12])],
                 # barmode='group',
                 title='Result check',
                 x='rule', y='amount', facet_col='cluster', color='rule',
                 facet_col_wrap=10, facet_row_spacing=0.01, facet_col_spacing=0.01)
    print("render fig")
    fig.show()

    # NAIVE show the constraints present in the clusters weighted for their frequency in other clusters
    rules_cluster_frequency = Counter()
    for cluster in res:
        rules_cluster_frequency.update(res[cluster].keys())
    for cluster in res:
        for rule in res[cluster]:
            value = n_clusters - rules_cluster_frequency[rule]
            res_df_naive = res_df_naive.append({'cluster': cluster, 'rule': rule, 'amount': value},
                                               ignore_index=True)
    for cluster in res:
        if len(res[cluster]) == 0:
            # Beware! Using the variable "rule" implies that in the previous cycle at least one rule is present in a cluster
            res_df_naive = res_df_naive.append({'cluster': cluster, 'rule': rule, 'amount': 0}, ignore_index=True)
    fig_naive = px.bar(res_df_naive,
                       # fig = px.bar(res_df[(res_df['cluster'] > 10) & (res_df['cluster'] < 20)],
                       # fig = px.bar(res_df[res_df['cluster'].isin([12])],
                       # barmode='group',
                       title='Naive: rules in clusters weighted for the inverse of their frequency in other clusters (i.e. rule in just few clusters-Z high bar',
                       x='rule', y='amount', facet_col='cluster', color='rule',
                       facet_col_wrap=10, facet_row_spacing=0.01, facet_col_spacing=0.01)
    fig_naive.show()


def plot_3d(df, title='t-SNE 3D Clusters visualization', name='labels'):
    fig = px.scatter_3d(df, x='x', y='y', z='z', color=name, opacity=0.5, title=title)

    fig.update_traces(marker=dict(size=3))
    fig.show()


def plot_tSNE_3d(input2D, clusters):
    # 3d plot of clusters through t-SNE
    print(">>>>> tSNE 3D visualization")
    names = ['x', 'y', 'z']
    # Default perplexity=30 perplexity suggested [5,50], n_iter=1000,
    matrix = TSNE(n_components=3, perplexity=30, n_iter=50000).fit_transform(input2D)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.rename({i: names[i] for i in range(3)}, axis=1, inplace=True)
    df_matrix['labels'] = clusters.labels_
    plot_3d(df_matrix)
    if -1 in df_matrix.labels.array:
        # if the cluster algorithm has a "-1" cluster for unclusterable elements, this line removes these elements form the 3D visualization
        plot_3d(df_matrix[df_matrix.labels != -1])


def cluster_traces_from_rules_trace_measures(trace_measures_file_path, algorithm='dbscan', boolean_confidence=True,
                                             apply_pca=True):
    """
Cluster traces according to declarative rules measurements evaluated on them
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

    clusters = cluster_traces(input2D, traces, constraints, measures, algorithm)

    return clusters, pca, input2D


def split_log(log, clusters):
    """
    Split the log into sub-logs according to the clusters, returns the list of logs
    :param log:
    :param clusters:
    """
    n_clusters = max(clusters.labels_) - min(clusters.labels_) + 1
    # sub_logs = list(range(n_clusters))
    sub_logs = dict.fromkeys(set(clusters.labels_), [])
    # initialize sublogs with original log properties
    # for i in range(n_clusters):
    for i in set(clusters.labels_):
        sub_log = EventLog()
        sub_log._attributes = log.attributes
        sub_log._classifiers = log.classifiers
        sub_log._extensions = log.extensions
        sub_log._omni = log.omni_present
        sub_logs[i] = sub_log
    trace_index = 0
    # put traces in sub-logs
    for trace in log:
        sub_logs[clusters.labels_[trace_index]].append(trace)
        trace_index += 1
    return sub_logs


def retrieve_cluster_statistics(clusters, log_file_path, output_folder):
    """
     retrieve the statistics of the sub-logs of each clusters.
     Specifically, it retrieves for each cluster:
        - number of traces
        - average, min and max trace length
        - unique tasks in the sub-log
        - min and max timestamp (i.e. timestamp of the first and last activities of the cluster)
        + PM4Py ready to use stats
    :param clusters:
    :param log_file_path:
    """
    print('>>>>>>>>>> Statistics')
    # load log
    log = pm.read_xes(log_file_path)
    logs = split_log(log, clusters)
    # export_traces_labels(log, clusters, output_folder + log.attributes['concept:name'] + '_traces-labels.csv')
    export_traces_labels_multi_perspective(log, clusters,
                                           output_folder + log.attributes['concept:name'] + '_traces-labels.csv')
    # export clusters logs to disk
    for cluster_index in logs:
        xes_exporter.apply(logs[cluster_index],
                           output_folder + log.attributes['concept:name'] + '_cluster_' + str(
                               cluster_index) + '.xes')
    # retrieve and output stats
    with open(output_folder + log.attributes['concept:name'] + '_clusters-stats.csv', 'w') as output:
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
            'TASKS'
        ])
        for cluster_index in logs:
            current_s_log = logs[cluster_index]
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
            csv_out.writerow(
                [cluster_index, traces_num, events_avg, events_min, events_max,
                 duration_median, duration_min, duration_max, case_arrival_avg,
                 unique_tasks_num, unique_tasks])

        return logs


def get_attributes_statistics_in_log(current_s_log, all_events_attributes):
    """
    Returns the statistics of the given events attributes in a log:
    - numerical attributes: [avg, min, max]
    - categorical attributes: [number of values, list of all values in cluster]
    - TimeStamp: [avg,min,max]

    :param current_s_log:
    :param all_events_attributes:
    :return:
    """
    result = []
    for attribute in all_events_attributes:
        result += [[]]
        # map, key-value: attribute value-number of traces with that value
        current_attribute_values = attributes_filter.get_attribute_values(current_s_log, attribute)
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

    for trace in current_s_log:
        for event in trace:
            pass
    return result


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


def retrieve_cluster_statistics_multi_perspective(clusters, log_file_path, output_folder):
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

    :param clusters:
    :param log_file_path:
    :param output_folder:
    """
    print('>>>>>>>>>> Statistics')
    # load log
    log = pm.read_xes(log_file_path)
    all_events_attributes = sorted(list(attributes_filter.get_all_event_attributes_from_log(log)))
    logs = split_log(log, clusters)
    # export_traces_labels(log, clusters, output_folder + log.attributes['concept:name'] + '_traces-labels.csv')
    export_traces_labels_multi_perspective(log, clusters,
                                           output_folder + log.attributes['concept:name'] + '_traces-labels.csv')
    # export clusters logs to disk
    for cluster_index in logs:
        xes_exporter.apply(logs[cluster_index],
                           output_folder + log.attributes['concept:name'] + '_cluster_' + str(
                               cluster_index) + '.xes')
    # retrieve and output stats
    with open(output_folder + log.attributes['concept:name'] + '_clusters-stats.csv', 'w') as output:
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
                             'TASKS'
                         ] + all_events_attributes
                         )
        for cluster_index in logs:
            current_s_log = logs[cluster_index]
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
            events_attributes = get_attributes_statistics_in_log(current_s_log, all_events_attributes)
            csv_out.writerow(
                [cluster_index, traces_num, events_avg, events_min, events_max,
                 duration_median, duration_min, duration_max, case_arrival_avg,
                 unique_tasks_num, unique_tasks] + events_attributes)

        return logs


def export_traces_labels(log, clusters, output_file_path):
    """
Export a csv file containing for each trace the corresponding cluster
    :param output_file_path:
    """
    print("Exporting traces cluster labels to " + output_file_path)
    with open(output_file_path, 'w') as output_file:
        csv_writer = csv.writer(output_file, delimiter=';')
        header = ["TRACE", "CLUSTER"]
        csv_writer.writerow(header)

        # put traces in sub-logs
        for trace_index in range(len(log)):
            csv_writer.writerow([trace_index, clusters.labels_[trace_index]])


def export_traces_labels_multi_perspective(log, clusters, output_file_path):
    """
    Export a csv file containing for each trace the corresponding cluster and values of the attributes
    :param output_file_path:
    """
    print("Exporting traces cluster labels to " + output_file_path)
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
            csv_writer.writerow([trace_index, clusters.labels_[trace_index]] + trace_attributes)


def plot_clusters_imperative_models(clusters_logs, model='DFG'):
    """
        Plot the desired imperative model of each cluster
    :param clusters_logs:
    :param model:
    """
    # Imperative models
    print("clusters imperative models...")
    for cluster_index in clusters_logs:
        # PROCESS TREE
        if model == 'process-tree':
            tree = inductive_miner.apply_tree(clusters_logs[cluster_index])
            gviz = pt_visualizer.apply(tree)
            pt_visualizer.view(gviz)
        # PETRI-NET
        elif model == 'petrinet':
            net, initial_marking, final_marking = inductive_miner.apply(clusters_logs[cluster_index])
            gviz = pn_visualizer.apply(net)
            pt_visualizer.view(gviz)
        ## HEURISTIC-NET
        elif model == 'heuristic-net':
            heu_net = heuristics_miner.apply_heu(clusters_logs[cluster_index], parameters={
                heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.99})
            gviz = hn_visualizer.apply(heu_net)
            hn_visualizer.view(gviz)
        ## Directly Follow Graph
        elif model == 'DFG':
            dfg = dfg_discovery.apply(clusters_logs[cluster_index], variant=dfg_discovery.Variants.PERFORMANCE)
            gviz = dfg_visualization.apply(dfg, log=clusters_logs[cluster_index],
                                           variant=dfg_visualization.Variants.PERFORMANCE)
            dfg_visualization.view(gviz)


def visualize_centroids_constraints(clusters, pca, threshold, measures_num, constraints_names, output_folder):
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
        with open(output_folder + '/centroids-constraints.csv', 'w') as output:
            csv_output = csv.writer(output, delimiter=';')
            # header
            csv_output.writerow(constraints_names)
            # values
            csv_output.writerows(res_matrix)
    except:
        print("ERROR >>> Centroid export error:", sys.exc_info()[0])


def visualize_pca_relevant_constraints(clusters, pca, measures_num, constraints_names, output_folder):
    print(">>>>>visualize PCA selected constraints")
    with open(output_folder + '/pca-features.csv', 'w') as output:
        constraint_measures_names = [c + "_m" + str(m) for c in constraints_names for m in range(measures_num)]
        n_pcs = pca.components_.shape[0]
        pca_features = pd.DataFrame(pca.components_, columns=constraint_measures_names)
        # Print all features importance
        csv_output = csv.writer(output, delimiter=';')
        csv_output.writerow(constraint_measures_names)
        for i in range(pca_features.shape[0]):
            csv_output.writerow(pca_features.transpose()[i])

        # Correlation Matrix between constraints
        # plt.matshow(pca_features.corr())
        # plt.show()

        # most important features
        most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
        most_important_names = [constraint_measures_names[most_important[i]] for i in range(n_pcs)]
        dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}
        print(pd.DataFrame(dic.items()))


def visualize_pca_relevant_feature(pca, feature_names, output_folder):
    print(">>>>>visualize PCA selected features")
    with open(output_folder + '/pca-features.csv', 'w') as output:
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


def behavioural_clustering(trace_measures_csv_file_path, log_file_path, clustering_algorithm, boolean_confidence,
                           output_folder,
                           visualization_flag, apply_pca):
    """
    Cluster the traces of a log according to a set of declarative rules and their trace measurements

    :param trace_measures_csv_file_path:
    :param log_file_path:
    :param clustering_algorithm:
    :param boolean_confidence:
    :param output_folder:
    :param visualization_flag:
    :param apply_pca:
    """
    print(clustering_algorithm)

    # traces_num, constraints_num, measures_num, constraints_names = cmio.retrieve_SJ2T_csv_data(sj2t_csv_file_path)
    traces_num, constraints_num, measures_num, constraints_names = j3io.retrieve_trace_measures_metadata(
        trace_measures_csv_file_path)

    # CLUSTERING
    clusters, pca, input2D = cluster_traces_from_rules_trace_measures(trace_measures_csv_file_path,
                                                                      clustering_algorithm,
                                                                      boolean_confidence, apply_pca)

    # STATS
    # clusters_logs = retrieve_cluster_statistics(clusters, log_file_path, output_folder)
    clusters_logs = retrieve_cluster_statistics_multi_perspective(clusters, log_file_path, output_folder)
    if apply_pca:
        visualize_pca_relevant_constraints(clusters, pca, measures_num, constraints_names, output_folder)

    # VISUALIZATION
    if visualization_flag:
        print(">>>>>>>>>>>> Visualization")
        # plot_clusters_imperative_models(clusters_logs)

        plot_tSNE_3d(input2D, clusters)
        # visualize_matrices(input2D, clusters)

        threshold = 0.95
        # labels, traces_index = j3io.import_trace_labels(trace_measures_csv_file_path, constraints_num, threshold)
        # visualize_constraints_in_clusters(clusters, labels, traces_index)

        visualize_centroids_constraints(clusters, pca, threshold, measures_num, constraints_names, output_folder)
    else:
        print(">>>>>>>>>>>> Visualization SKIPPED")


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    plt.title('Hierarchical Clustering Dendrogram')

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    # linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    linkage_matrix = hierarchy.linkage(model.children_, 'ward')

    # Plot the corresponding dendrogram
    dendrogram(
        linkage_matrix,
        p=len(set(model.labels_)), truncate_mode='lastp',
        # show_leaf_counts=True,
        show_contracted=True,
        color_threshold=0.5 * max(linkage_matrix[:, 2])
    )
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()


def attribute_clustering(log_file_path, clustering_algorithm, output_folder, visualization_flag, apply_pca_flag):
    """
    Cluster the traces of a log according to the log categorical attributes

    :param log_file_path:
    :param clustering_algorithm:
    :param output_folder:
    :param visualization_flag:
    :param apply_pca_flag:
    """

    data, feature_names = get_log_representation.get_default_representation(xes_importer.apply(log_file_path))
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
    clusters = cluster_traces(input2D, traces, attributes, 0, clustering_algorithm)

    # STATS
    # clusters_logs = retrieve_cluster_statistics(clusters, log_file_path, output_folder)
    clusters_logs = retrieve_cluster_statistics_multi_perspective(clusters, log_file_path, output_folder)
    if apply_pca_flag:
        visualize_pca_relevant_feature(pca, feature_names, output_folder)

    # VISUALIZATION
    if visualization_flag:
        print(">>>>>>>>>>>> Visualization")
        # plot_clusters_imperative_models(clusters_logs)

        plot_tSNE_3d(input2D, clusters)
        # visualize_matrices(input2D, clusters)

        # plot_dendrogram(clusters.fit(input2D))

        # threshold = 0.95
        # labels, traces_index = j3io.import_trace_labels(trace_measures_csv_file_path, constraints_num, threshold)
        # visualize_constraints_in_clusters(clusters, labels, traces_index)

        # visualize_centroids_constraints(0, pca, 0, measures_num, constraints_names, output_folder)
    else:
        print(">>>>>>>>>>>> Visualization SKIPPED")


def mixed_clustering(trace_measures_csv_file_path, log_file_path, clustering_algorithm, boolean_confidence,
                     output_folder, visualization_flag, apply_pca_flag):
    """
    Cluster the traces of a log according to both a set of declarative rules and the log attributes

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

    attributes_data, attributes_names = get_log_representation.get_default_representation(
        xes_importer.apply(log_file_path))
    # 1-hot encoding
    input2D_attributes = pd.DataFrame(attributes_data, columns=attributes_names)

    traces_num = input2D_attributes.shape[0]
    attributes_num = input2D_attributes.shape[1]
    print("Attributes: " + str(attributes_num))
    print(attributes_names)

    input2D = pd.concat([input2D_rules, input2D_attributes], axis=1)
    features_names = np.concatenate([constraints_names, attributes_names])
    features_num = features_names.shape[0]

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
    print(clustering_algorithm)
    print("Traces: " + str(traces_num))
    print("Features: " + str(features_num))
    clusters = cluster_traces(input2D, traces_num, features_num, 0, clustering_algorithm)

    # STATS
    # clusters_logs = retrieve_cluster_statistics(clusters, log_file_path, output_folder)
    clusters_logs = retrieve_cluster_statistics_multi_perspective(clusters, log_file_path, output_folder)
    if apply_pca_flag:
        visualize_pca_relevant_feature(pca, features_names, output_folder)

    # VISUALIZATION
    if visualization_flag:
        print(">>>>>>>>>>>> Visualization")
        # plot_clusters_imperative_models(clusters_logs)

        plot_tSNE_3d(input2D, clusters)
        # visualize_matrices(input2D, clusters)

        # plot_dendrogram(clusters.fit(input2D))

        # threshold = 0.95
        # labels, traces_index = j3io.import_trace_labels(trace_measures_csv_file_path, constraints_num, threshold)
        # visualize_constraints_in_clusters(clusters, labels, traces_index)

        # visualize_centroids_constraints(0, pca, 0, measures_num, constraints_names, output_folder)
    else:
        print(">>>>>>>>>>>> Visualization SKIPPED")


if __name__ == '__main__':
    clustering_policy = sys.argv[1]
    print("Clustering policy: " + str(clustering_policy))
    # 'rules'
    # 'attributes'
    # 'specific-attribute'  TODO
    # 'mixed' TODO
    if clustering_policy == 'rules':
        # if len(sys.argv) != 9:
        #     print(f"ERROR: wrong number of arguments for {clustering_policy}-based clustering")
        # BEHAVIOURAL CLUSTERING
        log_file_path = sys.argv[2]
        clustering_algorithm = sys.argv[3]
        output_folder = sys.argv[4]
        visualization_flag = sys.argv[5] == "True"
        apply_pca_flag = sys.argv[6] == "True"
        trace_measures_csv_file_path = sys.argv[7]
        boolean_confidence = sys.argv[8] == "True"

        behavioural_clustering(trace_measures_csv_file_path, log_file_path, clustering_algorithm, boolean_confidence,
                               output_folder,
                               visualization_flag, apply_pca_flag)
    elif clustering_policy == 'attributes':
        # if len(sys.argv) != 7:
        #     print(f"ERROR: wrong number of arguments for {clustering_policy}-based clustering")
        # ATTRIBUTE CLUSTERING
        log_file_path = sys.argv[2]
        clustering_algorithm = sys.argv[3]
        output_folder = sys.argv[4]
        visualization_flag = sys.argv[5] == "True"
        apply_pca_flag = sys.argv[6] == "True"

        attribute_clustering(log_file_path, clustering_algorithm, output_folder, visualization_flag, apply_pca_flag)

    elif clustering_policy == 'specific-attribute':
        # SPECIFIC attribute CLUSTERING
        print("Clustering based on a single specific attribute is not yet implemented")

    elif clustering_policy == 'mixed':
        # if len(sys.argv) != 9:
        #     print(f"ERROR: wrong number of arguments for {clustering_policy}-based clustering")
        # mixed rules-attributes CLUSTERING
        log_file_path = sys.argv[2]
        clustering_algorithm = sys.argv[3]
        output_folder = sys.argv[4]
        visualization_flag = sys.argv[5] == "True"
        apply_pca_flag = sys.argv[6] == "True"
        trace_measures_csv_file_path = sys.argv[7]
        boolean_confidence = sys.argv[8] == "True"

        mixed_clustering(trace_measures_csv_file_path, log_file_path, clustering_algorithm, boolean_confidence,
                         output_folder,
                         visualization_flag, apply_pca_flag)
    else:
        print("Clustering policy not recognized: " + str(clustering_policy))
