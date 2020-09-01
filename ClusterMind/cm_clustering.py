# import sklearn.cluster as clustering
import sys

import sklearn.cluster as cluster
from sklearn.mixture import GaussianMixture
import ClusterMind.IO.SJ2T_import as cmio
import numpy as np
import io


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
    kmeans = cluster.KMeans(n_clusters=nc).fit(input2D)
    # Affinity
    affinity = cluster.AffinityPropagation().fit(input2D)
    # mean-shift
    mean_shift = cluster.MeanShift().fit(input2D)
    # Agglomerative
    agglomerative = cluster.AgglomerativeClustering(n_clusters=nc).fit(input2D)
    # Spectral
    spectral = cluster.SpectralClustering(n_clusters=nc).fit(input2D)
    # DBSCAN
    dbscan = cluster.DBSCAN().fit(input2D)
    # OPTICS
    optics = cluster.OPTICS(min_samples=nc).fit(input2D)
    # birch
    birch = cluster.Birch(n_clusters=nc).fit(input2D)
    # # # gaussian
    # # gaussian = GaussianMixture(n_components=nc).fit(input2D)

    print("Kmeans: \t\t" + str(kmeans.labels_))
    print("Affinity: \t\t" + str(affinity.labels_))
    print("mean_shift: \t" + str(mean_shift.labels_))
    print("Agglomerative: \t" + str(agglomerative.labels_))
    print("Spectral: \t\t" + str(spectral.labels_))
    print("DBSCAN: \t\t" + str(dbscan.labels_))
    print("OPTICS: \t\t" + str(optics.labels_))
    print("birch: \t\t\t" + str(birch.labels_))
    # print("gaussian: \t\t" + str(gaussian.labels_))


def cluster_traces_from_file(file_path):
    file_format = file_path.split(".")[-1]
    input3D = cmio.import_SJ2T(file_path, file_format)
    input2D = input3D.reshape((input3D.shape[0], input3D.shape[1] * input3D.shape[2]))
    print("2D shape:" + str(input2D.shape))

    # Clean NaN and infinity
    input2D = np.nan_to_num(input2D, posinf=1.7976931348623157e+100, neginf=-1.7976931348623157e+100)

    traces = input3D.shape[0]
    constraints = input3D.shape[1]
    measures = input3D.shape[2]

    cluster_traces(input2D, traces, constraints, measures)


if __name__ == '__main__':
    file_path = sys.argv[1]
    cluster_traces_from_file(file_path)
