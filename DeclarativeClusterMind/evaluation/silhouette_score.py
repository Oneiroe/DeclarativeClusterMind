from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import DeclarativeClusterMind.io.Janus3_import as j3tio


def compute_silhouette_from_trace_measures_files(trace_measures_csv_file, labels_file, plot_file=None):
    input_2d, constraints_names = j3tio.extract_detailed_trace_rules_perspective_csv(trace_measures_csv_file)
    input_2d = pd.DataFrame(input_2d, columns=constraints_names)
    input_2d = np.nan_to_num(input_2d, posinf=1.7976931348623157e+100, neginf=-1.7976931348623157e+100)
    labels = pd.read_csv(labels_file, delimiter=';').CLUSTER
    mean_silhouette = compute_silhouette(input_2d, labels)
    visualize_silhouette(input_2d, labels, mean_silhouette, plot_file)


def compute_silhouette(input_2D, labels):
    """
Compute the silhouette score of a given tabular featured data.
The score is computed according to the chosen label

    :param input_2D:
    :param labels: the classification label for each entry in the featured data
    :return:
    """
    mean_silhouette = silhouette_score(input_2D, labels)
    print(f'mean Silhouette Coefficient of all samples: {mean_silhouette}')
    return mean_silhouette


def visualize_silhouette(input2D, traces_cluster_labels, silhouette_avg, output_file=None, immediate_visualization=False):
    """
Visualize the silhouette score of each cluster and trace

    :param output_file:
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
    for index, i in enumerate(sorted(set(traces_cluster_labels))):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[traces_cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(index) / n_clusters)
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

    if output_file is not None:
        print(f"Saving silhouette plot in {output_file}")
        plt.savefig(output_file)
    if immediate_visualization:
        plt.show()


if __name__ == '__main__':
    input_2D_transposed_file = "/home/alessio/Data/Phd/my_code/ClusterMind/experiments/REAL-LIFE-EXPLANATION/COVID/2-merged-log/COVID-output[tracesMeasures].csv"
    labels_file = "/home/alessio/Data/Phd/my_code/ClusterMind/experiments/REAL-LIFE-EXPLANATION/COVID/3-results/traces-labels.csv"

    compute_silhouette_from_trace_measures_files(input_2D_transposed_file, labels_file)
