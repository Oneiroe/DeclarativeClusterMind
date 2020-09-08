# import sklearn.cluster as clustering
import sys
from collections import Counter

import sklearn.cluster as cluster
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import ClusterMind.IO.SJ2T_import as cmio
import numpy as np
import plotly.express as px


def visualize_matrices(input2D, clusters):
    centorids = clusters.cluster_centers_

    fig = px.imshow(centorids, title='Centroids')
    fig.show()
    fig2 = px.imshow(input2D, title='MMM-2D')
    fig2.show()


def block_diag_einsum(arr, num):
    rows, cols = arr.shape
    result = np.zeros((num, rows, num, cols), dtype=arr.dtype)
    diag = np.einsum('ijik->ijk', result)
    diag[:] = arr
    return result.reshape(rows * num, cols * num)


def create_centroids(measures, constraints):
    a = np.zeros((1, measures))
    a.fill(1)
    return block_diag_einsum(a, constraints)


def cluster_traces(input2D, traces, constraints, measures):
    ## CLUSTERING

    nc = constraints  # number of clusters
    # K-means
    centroids_init = create_centroids(measures, constraints)
    # kmeans = cluster.KMeans(n_clusters=nc, init=centroids_init).fit(input2D)
    try:
        kmeans = cluster.KMeans(n_clusters=nc).fit(input2D)
        print("Kmeans: \t\t" + str(kmeans.labels_))
    except:
        print("K-Means error:", sys.exc_info()[0])
    # Affinity
    try:
        affinity = cluster.AffinityPropagation(random_state=0).fit(input2D)
        print("Affinity: \t\t" + str(affinity.labels_))
    except:
        print("affinity error:", sys.exc_info()[0])
    # mean-shift
    try:
        mean_shift = cluster.MeanShift().fit(input2D)
        print("mean_shift: \t" + str(mean_shift.labels_))
    except:
        print("mean_shift error:", sys.exc_info()[0])
    # Agglomerative
    try:
        agglomerative = cluster.AgglomerativeClustering(n_clusters=nc).fit(input2D)
        print("Agglomerative: \t" + str(agglomerative.labels_))
    except:
        print("Agglomerative error:", sys.exc_info()[0])
    # Spectral
    try:
        spectral = cluster.SpectralClustering(n_clusters=nc).fit(input2D)
        print("Spectral: \t\t" + str(spectral.labels_))
    except:
        print("Spectral error:", sys.exc_info()[0])
    # DBSCAN
    try:
        dbscan = cluster.DBSCAN().fit(input2D)
        print("DBSCAN: \t\t" + str(dbscan.labels_))
    except:
        print("DBSCAN error:", sys.exc_info()[0])
    # OPTICS
    try:
        optics = cluster.OPTICS(min_samples=nc).fit(input2D)
        print("OPTICS: \t\t" + str(optics.labels_))
    except:
        print("OPTICS error:", sys.exc_info()[0])
    # birch
    try:
        birch = cluster.Birch(n_clusters=nc).fit(input2D)
        # birch = cluster.Birch().fit(input2D)
        print("birch: \t\t\t" + str(birch.labels_))
    except:
        print("birch error:", sys.exc_info()[0])
    # gaussian
    # gaussian = GaussianMixture(n_components=nc).fit(input2D)
    # print("gaussian: \t\t" + str(gaussian.labels_))

    print(">>>>>>>>>>>>K-Means labels validation")
    return kmeans


def retrieve_labels(file_path, threshold=0.95):
    # INPUT IMPORT
    file_format = file_path.split(".")[-1]
    labels = cmio.import_SJ2T_labels(file_path, file_format, threshold)
    return labels


def check_results(clusters, labels, traces_index):
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
    print(res)
    # graph['counts'] =
    fig = px.bar(res_df, title='Result check', barmode='group', x='rule', y='amount', facet_col='cluster', color='rule')
    fig.show()


def cluster_traces_from_file(file_path):
    # INPUT IMPORT
    file_format = file_path.split(".")[-1]
    input3D = cmio.import_SJ2T(file_path, file_format)
    input2D = input3D.reshape((input3D.shape[0], input3D.shape[1] * input3D.shape[2]))
    print("2D shape:" + str(input2D.shape))

    # PRE-PROCESSING

    # Clean NaN and infinity
    # input2D = np.nan_to_num(input2D, posinf=1.7976931348623157e+100, neginf=-1.7976931348623157e+100)
    input2D = np.nan_to_num(np.power(input2D, -10), posinf=1.7976931348623157e+100, neginf=-1.7976931348623157e+100)
    # input2D = np.nan_to_num(input2D, posinf=100, neginf=-100)

    variance = 0.98
    pca = PCA(variance)
    pca.fit(input2D)
    input2D_pca = pca.transform(input2D)
    print('Dimension of data PCA= ' + str(input2D_pca.shape))
    # CLUSTERING
    traces = input3D.shape[0]
    constraints = input3D.shape[1]
    measures = input3D.shape[2]

    clusters = cluster_traces(input2D, traces, constraints, measures)

    visualize_matrices(input2D, clusters)

    return clusters


if __name__ == '__main__':
    file_path = sys.argv[1]
    clusters = cluster_traces_from_file(file_path)

    labels, traces_index = retrieve_labels(file_path)
    check_results(clusters, labels, traces_index)
