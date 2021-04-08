import csv
import json

import numpy as np


def retrieve_csv_trace_measures_metadata(input_file_path):
    """
    Retrieve metadata from CSV trace measures file
    :param input_file_path:
    """
    print("Retrieving results data...")
    traces_num = 0
    constraints_num = 0
    measures_num = 0
    constraints_names = []

    with open(input_file_path, 'r') as input_file:
        csv_reader = csv.reader(input_file, delimiter=';')
        header = 1
        lines = 0
        c = set()
        for line in csv_reader:
            # First line
            if header > 0:
                # Skip the header line
                header -= 1
                measures_num = len(line[2:])
                continue
            lines += 1
            c.add(line[1])
            if line[1] not in constraints_names:
                constraints_names += [line[1]]
        constraints_num = len(c)
        traces_num = int(lines / constraints_num)

    print("traces:" + str(traces_num) + ",constraints:" + str(constraints_num) + ",measures:" + str(measures_num))
    return traces_num, constraints_num, measures_num, constraints_names


def retrieve_json_trace_measures_metadata(input_file_path):
    """
    Retrieve metadata from JSON trace measures file

    :param input_file_path:
    """
    print("Retrieving results data...")
    traces_num = 0
    constraints_num = 0
    measures_num = 0
    constraints_names = []

    print("traces:" + str(traces_num) + ",constraints:" + str(constraints_num) + ",measures:" + str(measures_num))
    return traces_num, constraints_num, measures_num, constraints_names


def retrieve_trace_measures_metadata(input_file_path: str):
    """
    Retrieve the information regarding the Janus results. Specifically the number of traces, constraints, and measures

    :param input_file_path:
    """

    if input_file_path.endswith("csv"):
        return retrieve_csv_trace_measures_metadata(input_file_path)
    elif input_file_path.endswith("json"):
        return retrieve_json_trace_measures_metadata(input_file_path)
    else:
        print("File extension not recognized for Janus Results")


def extract_detailed_trace_perspective_csv(trace_measures_csv_file_path, output_path, measure="Confidence"):
    """
    From the trace measures, given a specific measure, transpose the results for that one measure for each trace,
    i.e. a matrix where the rows are the constraints and the columns are the traces, and
    each cell contains the measure of the constraint in that trace

    :param trace_measures_csv_file_path:
    :param output_path:
    :param measure:
    """
    temp_res = {}
    traces_mapping = {}
    trace_index = 0
    featured_data = []
    features_names = []
    temp_pivot = ""
    stop_flag = 2
    with open(trace_measures_csv_file_path, 'r') as file:
        csv_file = csv.DictReader(file, delimiter=';')
        if len(csv_file.fieldnames) == 3:
            measure = csv_file.fieldnames[-1]
        for line in csv_file:
            if temp_pivot == "":
                temp_pivot = line['Constraint']
            temp_res.setdefault(line['Constraint'], {})
            if traces_mapping.setdefault(line['Trace'], "T" + str(trace_index)) == "T" + str(trace_index):
                trace_index += 1
            if line['Constraint'] == temp_pivot:
                featured_data += [[]]
                stop_flag -= 1
            if stop_flag >= 1:
                features_names += [line['Constraint']]

            temp_res[line['Constraint']][traces_mapping[line['Trace']]] = line[measure]
            featured_data[-1] += [float(line[measure])]

        header = ["Constraint"]
        for trace in temp_res[list(temp_res.keys())[0]].keys():
            header += [trace]

        with open(output_path, 'w') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=header, delimiter=';')
            writer.writeheader()
            for constraint in temp_res:
                temp_res[constraint].update({"Constraint": constraint})
                writer.writerow(temp_res[constraint])
    return featured_data, features_names
