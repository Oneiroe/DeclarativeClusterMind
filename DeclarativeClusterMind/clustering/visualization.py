""" Collection of function for various plot related to cm_clustering results
"""
import sys
from collections import Counter

import pandas as pd
import numpy as np

from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy

import plotly.express as px
from matplotlib import pyplot as plt


def visualize_heatmap(input2D, clusters):
    print(">>>> Visualize heatmap of MMM")
    try:
        centorids = clusters.cluster_centers_
        fig = px.imshow(centorids, title='Centroids')
        fig.show()
    except:
        print("ERROR >>> Centroid visualization error:", sys.exc_info()[0])

    fig2 = px.imshow(input2D, title='MMM-2D')
    fig2.show()


def plot_constraints_in_clusters(clusters, labels, traces_index):
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


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    plt.title('>>>>> Hierarchical Clustering Dendrogram')

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
