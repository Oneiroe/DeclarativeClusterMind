import sys
import csv
import os

from pm4py.objects.log.log import EventLog
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import pm4py as pm


@DeprecationWarning
def legacy_merge_logs(folder, files_prefix, output_path):
    """
Deprecated function for result aggregation (see aggregate_clusters_measures.py now)
    :param folder:
    :param files_prefix:
    :param output_path:
    """
    with open(output_path + "-labels.csv", 'w') as output_file:
        csv_writer = csv.writer(output_file, delimiter=';')
        header = ["TRACE", "CLUSTER"]
        csv_writer.writerow(header)
        result_log = EventLog()
        trace_index = 0

        for file in os.listdir(folder):
            if file.startswith(files_prefix) and file.endswith("xes") and ("merged" not in file):
                print(file)
                log = xes_importer.apply(folder + file)
                result_log._attributes.update(log._attributes)
                result_log._classifiers.update(log._classifiers)
                result_log._extensions.update(log._extensions)
                result_log._omni.update(log._omni)

                for t in log:
                    result_log.append(t)
                    csv_writer.writerow([trace_index, file])
                    trace_index += 1

        xes_exporter.apply(result_log, output_file)
    print("Output here: " + output_path)


def merge_logs(output_log_file_path, logs_files_paths):
    """
Merge the input log into one unique xes event log
    :param output_log:
    :param logs_files_paths:
    """
    result_log = EventLog()
    for log_file in logs_files_paths:
        if log_file.endswith("xes"):
            print(log_file)
            log = pm.read_xes(log_file)
            result_log._attributes.update(log._attributes)
            result_log._classifiers.update(log._classifiers)
            result_log._extensions.update(log._extensions)
            result_log._omni.update(log._omni)

            for trace in log:
                result_log.append(trace)
    pm.write_xes(result_log, output_log_file_path)
    print(f"Log merged in {output_log_file_path}")


if __name__ == '__main__':
    print(sys.argv)
    output_log_file_path = sys.argv[1]
    logs_files_paths = sys.argv[2:]
    merge_logs(output_log_file_path, logs_files_paths)
