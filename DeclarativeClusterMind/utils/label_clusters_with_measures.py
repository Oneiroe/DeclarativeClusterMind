import sys
import csv
import os


def label_clusters_with_measures(folder, files_suffix, output_file="clusters-labels.csv"):
    """
JUST FOR SINGLE MEASURES RESULTS
returns a label file where each row refer to a cluster and each column is a constraints containing its log measure in that cluster
    :param folder:
    :param files_suffix:
    :param output_file:
    """
    result_map = {}
    for file in os.listdir(folder):
        if file.endswith(files_suffix):
            with open(os.path.join(folder, file), 'r') as cluster_file:
                cluster_csv = csv.reader(cluster_file, delimiter=';')
                cluster_name = ""
                for line in cluster_csv:
                    if line[0] == "Constraint":
                        # TODO too adhoc naming
                        if "xes" in file:
                            cluster_name = str(file.split(".xes" + files_suffix)[0].split('cluster_')[-1])
                        elif "csv" in file:
                            cluster_name = str(file.split("[logMeasures]")[0].split('output_')[-1])
                        else:
                            cluster_name = str(file)
                        result_map[cluster_name] = {}
                    else:
                        result_map[cluster_name][line[0]] = [line[1]]

    with open(os.path.join(folder, output_file), 'w') as result:
        csv_result = csv.writer(result, delimiter=';')
        temp_cluster, temp_line = result_map.popitem()
        header = ["CLUSTER"] + sorted(list(temp_line.keys()))
        csv_result.writerow(header)

        line = [temp_cluster]
        for key in header[1:]:
            line += temp_line[key]
        csv_result.writerow(line)

        while len(result_map.keys()) > 0:
            temp_cluster, temp_line = result_map.popitem()
            line = [temp_cluster]
            for key in header[1:]:
                line += temp_line[key]
            csv_result.writerow(line)


if __name__ == '__main__':
    print(sys.argv)
    label_clusters_with_measures(sys.argv[1], sys.argv[2], sys.argv[3])
