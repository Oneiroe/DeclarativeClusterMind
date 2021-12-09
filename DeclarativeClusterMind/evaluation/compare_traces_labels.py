import csv
import os.path
import sys


def compare_traces_labels(labels_original_file, labels_resulting_file, output_matrix):
    print("Comparing trace labels...")
    # TRACE;CLUSTER
    confusion_matrix = {}

    original_names = set()
    resulting_names = set()

    # TRACE;CLUSTER
    with open(labels_original_file, 'r') as labels_original, open(labels_resulting_file, 'r') as labels_resulting:
        original_reader = csv.DictReader(labels_original, delimiter=';')
        resulting_reader = csv.DictReader(labels_resulting, delimiter=';')
        for (original_line, resulting_line) in zip(original_reader, resulting_reader):
            original_names.add(original_line['CLUSTER'])
            resulting_names.add(resulting_line['CLUSTER'])
            confusion_matrix.setdefault(resulting_line['CLUSTER'], {})
            confusion_matrix[resulting_line['CLUSTER']].setdefault(original_line['CLUSTER'], set())

            confusion_matrix[resulting_line['CLUSTER']][original_line['CLUSTER']].add(original_line['TRACE'])

    original_names = sorted(list(original_names))
    resulting_names = sorted(list(resulting_names))

    with open(output_matrix, 'w', newline='') as out_file:
        w = csv.writer(out_file, delimiter=';')
        header = ["clusters"] + [x for x in sorted(original_names)]
        w.writerow(header)
        for cluster in resulting_names:
            row = [cluster] + [len(confusion_matrix[cluster].setdefault(x,set())) for x in original_names]
            w.writerow(row)


if __name__ == '__main__':
    print(sys.argv)
    labels_original_file = sys.argv[1]
    labels_resulting_file = sys.argv[2]
    output_matrix = sys.argv[3]

    compare_traces_labels(labels_original_file, labels_resulting_file, output_matrix)
