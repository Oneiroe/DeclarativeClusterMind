import sys

from DeclarativeClusterMind.evaluation.utils import load_clusters_logs_list_from_folder, export_traces_clusters_labels


def export_traces_clusters_labels_from_logs(folder, output_file="traces-labels.csv"):
    result, clusters_indices = load_clusters_logs_list_from_folder(folder)
    labels = []
    for cluster_index, log in enumerate(result):
        for t in log:
            labels += [clusters_indices[cluster_index]]
    export_traces_clusters_labels(labels, output_file)


if __name__ == '__main__':
    print(sys.argv)
    logs_folder = sys.argv[1]
    output_name = sys.argv[2]
    export_traces_clusters_labels_from_logs(logs_folder, output_name)
