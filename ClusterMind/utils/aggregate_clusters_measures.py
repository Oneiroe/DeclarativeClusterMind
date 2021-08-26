import sys
import csv
import os


def aggregate_clusters_measures(folder, files_suffix, output_file="aggregated_result.csv"):
    result_map = {}

    for file in os.listdir(folder):
        if file.endswith(files_suffix):
            with open(os.path.join(folder, file), 'r') as cluster_file:
                cluster_csv = csv.reader(cluster_file, delimiter=';')
                for line in cluster_csv:
                    result_map.setdefault(line[0], [])
                    if line[0] == "Constraint":
                        # TODO too adhoc naming
                        if "xes" in file:
                            result_map[line[0]] += [
                                "Cluster_" + str(file.split(".xes" + files_suffix)[0].split('cluster_')[-1])]
                        elif "csv" in file:
                            result_map[line[0]] += [
                                "Cluster_" + str(file.split("[logMeasures]")[0].split('output_')[-1])]
                        else:
                            result_map[line[0]] += ["Cluster_" + str(file)]
                    else:
                        result_map[line[0]] += [line[1]]

    with open(os.path.join(folder, output_file), 'w') as result:
        csv_result = csv.writer(result, delimiter=';')
        for key in result_map.keys():
            # print([key] + result_map[key])
            csv_result.writerow([key] + result_map[key])


if __name__ == '__main__':
    print(sys.argv)
    aggregate_clusters_measures(sys.argv[1], sys.argv[2], sys.argv[3])
