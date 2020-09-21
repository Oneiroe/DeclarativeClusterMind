# import sklearn.cluster as clustering
import sys
from collections import Counter

import sklearn.cluster as cluster
import pandas as pd
from pm4py.objects.log.log import EventLog
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import ClusterMind.IO.SJ2T_import as cmio
import numpy as np
import plotly.express as px
import pm4py as pm
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import pm4py.statistics.traces.log as stats
import csv
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer


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


def cluster_traces(input2D, traces, constraints, measures):
    ## CLUSTERING

    # nc = constraints  # number of clusters
    nc = 10  # number of clusters
    # K-means
    # centroids_init = initialize_centroids(measures, nc)
    # kmeans = cluster.KMeans(n_clusters=nc, init=centroids_init).fit(input2D)
    try:
        print("K-Kmeans...")
        kmeans = cluster.KMeans(n_clusters=nc).fit(input2D)
        print("Kmeans: \t\t" + str(kmeans.labels_))
    except:
        print("K-Means error:", sys.exc_info()[0])
    # Affinity
    # try:
    #     affinity = cluster.AffinityPropagation(random_state=0).fit(input2D)
    #     print("Affinity: \t\t" + str(affinity.labels_))
    # except:
    #     print("affinity error:", sys.exc_info()[0])
    # # mean-shift
    # try:
    #     mean_shift = cluster.MeanShift().fit(input2D)
    #     print("mean_shift: \t" + str(mean_shift.labels_))
    # except:
    #     print("mean_shift error:", sys.exc_info()[0])
    # # Agglomerative
    # try:
    #     agglomerative = cluster.AgglomerativeClustering(n_clusters=nc).fit(input2D)
    #     print("Agglomerative: \t" + str(agglomerative.labels_))
    # except:
    #     print("Agglomerative error:", sys.exc_info()[0])
    # # Spectral
    # try:
    #     spectral = cluster.SpectralClustering(n_clusters=nc).fit(input2D)
    #     print("Spectral: \t\t" + str(spectral.labels_))
    # except:
    #     print("Spectral error:", sys.exc_info()[0])
    # # DBSCAN
    # try:
    #     dbscan = cluster.DBSCAN().fit(input2D)
    #     print("DBSCAN: \t\t" + str(dbscan.labels_))
    # except:
    #     print("DBSCAN error:", sys.exc_info()[0])
    # # OPTICS
    # try:
    #     optics = cluster.OPTICS(min_samples=nc).fit(input2D)
    #     print("OPTICS: \t\t" + str(optics.labels_))
    # except:
    #     print("OPTICS error:", sys.exc_info()[0])
    # # birch
    # try:
    #     birch = cluster.Birch(n_clusters=nc).fit(input2D)
    #     # birch = cluster.Birch().fit(input2D)
    #     print("birch: \t\t\t" + str(birch.labels_))
    # except:
    #     print("birch error:", sys.exc_info()[0])
    # # gaussian
    # # gaussian = GaussianMixture(n_components=nc).fit(input2D)
    # # print("gaussian: \t\t" + str(gaussian.labels_))

    print(">>>>>>>>>>>>K-Means labels validation")
    return kmeans
    # return birch


def visualize_matrices(input2D, clusters):
    centorids = clusters.cluster_centers_

    fig = px.imshow(centorids, title='Centroids')
    fig.show()
    fig2 = px.imshow(input2D, title='MMM-2D')
    fig2.show()


def visualize_results(clusters, labels, traces_index):
    res_df = pd.DataFrame()
    res = {}
    i = 0
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
    # print(res)
    fig = px.bar(res_df,
                 # fig = px.bar(res_df[(res_df['cluster'] > 10) & (res_df['cluster'] < 20)],
                 # fig = px.bar(res_df[res_df['cluster'].isin([12])],
                 # barmode='group',
                 title='Result check',
                 x='rule', y='amount', facet_col='cluster', color='rule',
                 facet_col_wrap=10, facet_row_spacing=0.01, facet_col_spacing=0.01)
    fig.show()


def plot_3d(df, name='labels'):
    iris = px.data.iris()
    fig = px.scatter_3d(df, x='x', y='y', z='z',color=name, opacity=0.5)

    fig.update_traces(marker=dict(size=3))
    fig.show()

def cluster_traces_from_file(file_path):
    # INPUT IMPORT
    file_format = file_path.split(".")[-1]
    input3D = cmio.import_SJ2T(file_path, file_format)
    input2D = input3D.reshape((input3D.shape[0], input3D.shape[1] * input3D.shape[2]))
    print("2D shape:" + str(input2D.shape))

    # PRE-PROCESSING

    # Clean NaN and infinity
    input2D = np.nan_to_num(input2D, posinf=1.7976931348623157e+100, neginf=-1.7976931348623157e+100)
    # input2D = np.nan_to_num(np.power(input2D, -10), posinf=1.7976931348623157e+100, neginf=-1.7976931348623157e+100)
    # input2D = np.nan_to_num(input2D, posinf=100, neginf=-100)

    # reduce dimensions with PCA
    pca_variance = 0.98
    pca = PCA(pca_variance)
    pca.fit(input2D)
    input2D = pca.transform(input2D)
    print('Dimension of data PCA= ' + str(input2D.shape))

    # CLUSTERING
    print("Clustering...")
    traces = input3D.shape[0]
    constraints = input3D.shape[1]
    measures = input3D.shape[2]

    clusters = cluster_traces(input2D, traces, constraints, measures)

    # 3d plot of data through t-SNE
    names = ['x', 'y', 'z']
    matrix = TSNE(n_components=3).fit_transform(input2D)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.rename({i: names[i] for i in range(3)}, axis=1, inplace=True)
    df_matrix['labels'] = clusters.labels_
    plot_3d(df_matrix)

    visualize_matrices(input2D, clusters)

    return clusters, pca


def split_log(log, clusters):
    """
    Split the log into sub-logs according to the clusters, returns the list of logs
    :param log:
    :param clusters:
    """
    sub_logs = list(range(clusters.n_clusters))
    # initialize sublogs with original log properties
    for i in range(clusters.n_clusters):
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


def retrieve_cluster_statistics(clusters, log_file_path):
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
    # export clusters logs to disk
    for cluster_index in range(clusters.n_clusters):
        xes_exporter.apply(logs[cluster_index],
                           './clustered-logs/' + log.attributes['concept:name'] + '_cluster-' + str(
                               cluster_index) + '.xes')
    # retrieve and output stats
    with open('./clustered-logs/' + log.attributes['concept:name'] + '_clusters-stats.csv', 'w') as output:
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
        cluster_index = 0
        for s_log in logs:
            traces_num = len(s_log)
            events_avg = sum((len(i) for i in s_log)) / len(s_log)
            events_min = min(len(i) for i in s_log)
            events_max = max(len(i) for i in s_log)
            unique_tasks = set(e['concept:name'] for t in s_log for e in t)
            unique_tasks_num = len(unique_tasks)
            duration_median = stats.case_statistics.get_median_caseduration(s_log)
            duration_min = min(stats.case_statistics.get_all_casedurations(s_log))
            duration_max = max(stats.case_statistics.get_all_casedurations(s_log))
            case_arrival_avg = stats.case_arrival.get_case_arrival_avg(s_log)
            csv_out.writerow(
                [cluster_index, traces_num, events_avg, events_min, events_max,
                 duration_median, duration_min, duration_max, case_arrival_avg,
                 unique_tasks_num, unique_tasks])
            cluster_index += 1

    # Imperative models
    imperative = False
    if imperative:
        print("imperative models")
        for s_log in logs:
            # net, initial_marking, final_marking = inductive_miner.apply(s_log)
            # tree = inductive_miner.apply_tree(s_log)
            heu_net = heuristics_miner.apply_heu(s_log, parameters={
                heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.99})
            # gviz = pn_visualizer.apply(net)
            # pt_visualizer.view(gviz)
            gviz = hn_visualizer.apply(heu_net)
            hn_visualizer.view(gviz)

    # retrieve centroids constraints
    pass


def visualize_centroids_constraints(clusters, pca, threshold, measures, constraints):
    print(">>>>>visualize centroids constraints")
    res_matrix = [list() for i in range(len(clusters.cluster_centers_))]
    for centroid_index in range(len(clusters.cluster_centers_)):
        centroid = clusters.cluster_centers_[centroid_index]
        c = pca.inverse_transform(centroid)
        for i in range(len(constraints)):
            if c[1 + measures * i] > threshold:
                # confidence>threshold, it is the 2nd measure
                res_matrix[centroid_index] += [1]
            else:
                res_matrix[centroid_index] += [0]
    # export to csv
    with open('./clustered-logs/centroids-constraints.csv', 'w') as output:
        csv_output = csv.writer(output, delimiter=';')
        # header
        csv_output.writerow(constraints)
        # values
        csv_output.writerows(res_matrix)


if __name__ == '__main__':
    # file_path = sys.argv[1]
    # file_path = "/home/alessio/Data/Phd/my_code/ClusterMind/test/result_m02_t05.csv"
    sj2t_file_path = "/home/alessio/Data/Phd/my_code/ClusterMind/input/SEPSIS-output.csv"
    log_file_path = "/home/alessio/Data/Phd/my_code/ClusterMind/input/SEPSIS-log.xes"

    traces, constraints_num, measures, constraints = cmio.retrieve_SJ2T_csv_data(sj2t_file_path)

    # CLUSTERING
    clusters, pca = cluster_traces_from_file(sj2t_file_path)

    # VISUALIZATION
    threshold = 0.95
    # labels, traces_index = cmio.import_SJ2T_labels(sj2t_file_path, threshold)
    # visualize_results(clusters, labels, traces_index)
    visualize_centroids_constraints(clusters, pca, threshold, measures, constraints)

    # STATS
    retrieve_cluster_statistics(clusters, log_file_path)
