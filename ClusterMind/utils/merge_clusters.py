import sys
import csv
import os


def merge_clusters(folder, files_suffix, output_file="aggregated_result.csv"):
    result_map = {}

    for file in os.listdir(folder):
        if file.endswith(files_suffix):
            with open(folder + file, 'r') as cluster_file:
                cluster_csv = csv.reader(cluster_file, delimiter=';')
                for line in cluster_csv:
                    result_map.setdefault(line[0], [])
                    if line[0] == "Constraint":
                        result_map[line[0]] += ["Cluster-" + str(file.split(".xes-output[MEAN].csv")[0].split('-')[-1])]
                    else:
                        result_map[line[0]] += [line[1]]

    with open(folder + output_file, 'w') as result:
        csv_result = csv.writer(result, delimiter=';')
        for key in result_map.keys():
            # print([key] + result_map[key])
            csv_result.writerow([key] + result_map[key])


if __name__ == '__main__':
    print(sys.argv)
    merge_clusters(sys.argv[1], sys.argv[2], sys.argv[3])
