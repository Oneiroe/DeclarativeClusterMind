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
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt


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
    try:
        centorids = clusters.cluster_centers_
        fig = px.imshow(centorids, title='Centroids')
        fig.show()
    except:
        print("ERROR >>> Centroid visualization error:", sys.exc_info()[0])

    fig2 = px.imshow(input2D, title='MMM-2D')
    fig2.show()


def visualize_results(clusters, labels, traces_index):
    # Visualize the contraints present in the clusters
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
    # print(res)
    fig = px.bar(res_df,
                 # fig = px.bar(res_df[(res_df['cluster'] > 10) & (res_df['cluster'] < 20)],
                 # fig = px.bar(res_df[res_df['cluster'].isin([12])],
                 # barmode='group',
                 title='Result check',
                 x='rule', y='amount', facet_col='cluster', color='rule',
                 facet_col_wrap=10, facet_row_spacing=0.01, facet_col_spacing=0.01)
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


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

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

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def cluster_traces_from_file(file_path, algorithm='dbscan', boolean_confidence=True):
    # INPUT IMPORT
    file_format = file_path.split(".")[-1]
    input3D = cmio.import_SJ2T(file_path, file_format, boolean=boolean_confidence)
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
    input2D = pca.transform(input2D)
    print('Dimension of data PCA= ' + str(input2D.shape))

    # CLUSTERING
    print("Clustering...")
    traces = input3D.shape[0]
    constraints = input3D.shape[1]
    measures = input3D.shape[2]

    clusters = cluster_traces(input2D, traces, constraints, measures, algorithm)

    # 3d plot of clusters through t-SNE
    print(">>>>>>>>>>>> Visualization")
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

    visualize_matrices(input2D, clusters)

    return clusters, pca


def split_log(log, clusters):
    """
    Split the log into sub-logs according to the clusters, returns the list of logs
    :param log:
    :param clusters:
    """
    n_clusters = max(clusters.labels_) - min(clusters.labels_) + 1
    sub_logs = list(range(n_clusters))
    # initialize sublogs with original log properties
    for i in range(n_clusters):
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
    n_clusters = max(clusters.labels_) - min(clusters.labels_) + 1
    # TODO export cluster label, not an incremental number, in order to have a precise match between these stats and the images
    for cluster_index in range(n_clusters):
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
    try:
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
    except:
        print("ERROR >>> Centroid export error:", sys.exc_info()[0])


if __name__ == '__main__':
    logs = ('BPIC12',  # 0
            'BPIC13_cp',  # 1
            'BPIC13_i',  # 2
            'BPIC14_f',  # 3
            'BPIC15_1f',  # 4
            'BPIC15_2f',  # 5
            'BPIC15_3f',  # 6
            'BPIC15_4f',  # 7
            'BPIC15_5f',  # 8
            'BPIC17_f',  # 9
            'RTFMP',  # 10
            'SEPSIS'  # 11
            )
    log_name = logs[11]
    clustering_algs = (
        'kmeans',  # 0
        'affinity',  # 1
        'meanshift',  # 2
        'agglomerative',  # 3
        'spectral',  # 4
        'dbscan',  # 5
        'optics',  # 6
        'birch',  # 7
        'gaussian',  # 8 DO NOT USE THIS!
    )
    # sj2t_csv_file_path = "./input/" + log_name + "-output.csv"
    # log_file_path = "./input/" + log_name + "-log.xes"
    # clustering_algorithm = clustering_algs[6]
    # boolean_confidence = True

    sj2t_csv_file_path = sys.argv[1]
    log_file_path = sys.argv[2]
    clustering_algorithm = sys.argv[3]
    boolean_confidence = sys.argv[4] == "True"

    print(log_name)
    print(clustering_algorithm)

    traces, constraints_num, measures, constraints = cmio.retrieve_SJ2T_csv_data(sj2t_csv_file_path)

    # CLUSTERING
    clusters, pca = cluster_traces_from_file(sj2t_csv_file_path, clustering_algorithm, boolean_confidence)

    # VISUALIZATION
    threshold = 0.95
    labels, traces_index = cmio.import_SJ2T_labels(sj2t_csv_file_path, threshold)
    visualize_results(clusters, labels, traces_index)
    visualize_centroids_constraints(clusters, pca, threshold, measures, constraints)

    # STATS
    retrieve_cluster_statistics(clusters, log_file_path)
