import sys
import csv

from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter


def split_log_according_to_model(input_log_file, trace_measures_csv, split_threshold, output_log_80_fit_file=None,
                                 output_log_20_divergent_file=None):
    """
Divide the log between the traces fulfilling the given model (within a certain threshold) and the one who does not.
    :param input_log_file:bad
    :param trace_measures_csv:
    :param split_threshold:
    :param output_log_80_fit_file:
    :param output_log_20_divergent_file:
    :return:
    """
    input_log = xes_importer.apply(input_log_file)
    output_log_80 = EventLog()
    output_log_80._attributes.update(input_log._attributes)
    output_log_80._classifiers.update(input_log._classifiers)
    output_log_80._extensions.update(input_log._extensions)
    output_log_80._omni.update(input_log._omni)
    output_log_20 = EventLog()
    output_log_20._attributes.update(input_log._attributes)
    output_log_20._classifiers.update(input_log._classifiers)
    output_log_20._extensions.update(input_log._extensions)
    output_log_20._omni.update(input_log._omni)

    with open(trace_measures_csv, 'r') as measures_file:
        csv_reader = csv.reader(measures_file, delimiter=';')
        trace_index = 0

        for line in csv_reader:
            if line[1] == 'MODEL':
                if float(line[2]) >= split_threshold:
                    output_log_80.append(input_log[trace_index])
                else:
                    output_log_20.append(input_log[trace_index])
                trace_index += 1

        print(f"Number of Traces:{len(input_log)}")
        if len(output_log_80) != 0 and output_log_80_fit_file != None:
            xes_exporter.apply(output_log_80, output_log_80_fit_file)
            print(f"Traces in 80 sublog: {len(output_log_80)} ")
        if len(output_log_20) != 0 and output_log_20_divergent_file != None:
            xes_exporter.apply(output_log_20, output_log_20_divergent_file)
            print(f"Traces in 20 sublog: {len(output_log_20)} ")

    return output_log_80, output_log_20


if __name__ == '__main__':
    print(sys.argv)
    input_log = sys.argv[1]
    trace_measures_csv = sys.argv[2]
    split_threshold = float(sys.argv[3])
    output_log_80_good = sys.argv[4]
    output_log_20_bad = sys.argv[5]

    split_log_according_to_model(input_log, trace_measures_csv, split_threshold, output_log_80_good, output_log_20_bad)
