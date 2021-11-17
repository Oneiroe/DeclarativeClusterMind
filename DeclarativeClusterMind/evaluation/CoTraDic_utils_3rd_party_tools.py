import argparse

import xlrd

from DeclarativeClusterMind.evaluation.utils import export_traces_clusters_labels, \
    split_log_according_to_clusters
import pm4py as pm


def CoTraDic_import_clusters_indices(xls_file_path, output_csv_path=None):
    """
imports the trace labels from the clustering results of CoTraDict
    :param xls_file_path:
    :param output_csv_path: optional, if provided the labels are stored in a csv file
    :return: list of clusters labels, where each index is the index of the trace and the value is the associated cluster label
    """
    book = xlrd.open_workbook(xls_file_path)
    print("The number of worksheets is {0}".format(book.nsheets))
    print("Worksheet name(s): {0}".format(book.sheet_names()))
    sh = book.sheet_by_index(0)
    print(f"{sh.name}, rows:{sh.nrows}, cols:{sh.ncols}")
    clusters_names = [int(str(i).replace('text:', '').replace("'", "").replace("Cluster:", "")) for i in sh.row(0)]
    print(clusters_names)
    traces_indices_per_cluster = []
    for i in sh.row(2):
        indices = str(i).replace(' ', '').replace('text:', '').replace("'", "").replace("[", "").replace("]", "").split(
            ",")
        if len(indices) == 1 and indices[0] == '':
            traces_indices_per_cluster += [[]]
        else:
            traces_indices_per_cluster += [[int(j) for j in indices]]
    print(traces_indices_per_cluster)

    result = []
    i = 0
    empty = False
    while not empty:
        for c, cluster in enumerate(traces_indices_per_cluster):
            if len(cluster) > 0 and cluster[0] == i:
                result += [clusters_names[c]]
                traces_indices_per_cluster[c] = traces_indices_per_cluster[c][1:]
                i += 1
                break
        empty = True
        for cluster in traces_indices_per_cluster:
            if len(cluster) > 0:
                empty = False
                break
    print(result)

    if output_csv_path is not None:
        export_traces_clusters_labels(result, output_csv_path)

    return result


def CoTraDic_export_clusters_log_from_result(original_log_file, xls_file_path, output_folder):
    labels = CoTraDic_import_clusters_indices(xls_file_path)
    split_log_according_to_clusters(pm.read_xes(original_log_file), labels, output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CoTraDic results importer')

    parser.add_argument('-l', '--log-file', type=str, help='log file path', required=True)
    parser.add_argument('-x', '--xls-result', type=str, help='xls result file', required=True)
    parser.add_argument('-o', '--output-folder', type=str, help='output folder path', required=True)

    args = parser.parse_args()

    CoTraDic_export_clusters_log_from_result(args.log_file, args.xls_result, args.output_folder)
