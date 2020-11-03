import sys
import csv
import os

from pm4py.objects.log.log import EventLog
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter


def merge_logs(folder, files_prefix, output_path):
    result_log = EventLog()

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

    xes_exporter.apply(result_log, output)
    print("Output here: " + output)


if __name__ == '__main__':
    print(sys.argv)
    folder = sys.argv[1]
    prefix = sys.argv[2]
    output = sys.argv[3]
    # folder = "clustered-logs/"
    # prefix = "Synthetic"
    # output = folder + prefix + "-MERGED-log.xes"
    merge_logs(folder, prefix, output)
