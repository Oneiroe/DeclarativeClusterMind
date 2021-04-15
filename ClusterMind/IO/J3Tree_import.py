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
            else:
                break
        constraints_num = len(c)
        traces_num = int((len(open(input_file_path).readlines()) - 1) / constraints_num)

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


def import_trace_measures_from_csv(input_file_path, traces_num, constraints_num, measures_num):
    """
        Import the result from SJ2T csv containing the measurement of every constraint in every trace.
        Performances note: Knowing the dimension of the matrix in advance make the process way more fast
    :param input_file_path:
    :param traces_num:
    :param constraints_num:
    :param measures_num:
    :return:
    """
    print("Importing data...")
    result = np.zeros((traces_num, constraints_num, measures_num))
    # result = np.ndarray(shape=(1, 1, len(line) - 4)) # shape of the result ndarray
    with open(input_file_path, 'r') as input_file:
        csv_reader = csv.reader(input_file, delimiter=';')
        header = 1
        it = 0
        ic = 0
        i = 0
        for line in csv_reader:
            # print(i / (1050 * 260))
            i += 1
            # First line
            if header > 0:
                # Skip the header line
                header -= 1
                continue
            if ic == constraints_num:
                ic = 0
                it += 1

            # result[it][ic] = np.nan_to_num(np.array(line[2:])) # in case NaN and +-inf is a problem
            result[it][ic] = np.array(line[2:])
            ic += 1
    print("3D shape:" + str(result.shape))
    return result


def import_boolean_trace_measures_from_csv(input_file_path, traces_num, constraints_num, measures_num, threshold=0.9):
    """
    Import the result from SJ2T csv containing only if a constraint is satisfied in a trace (conf>threshold).
    Performances note: Knowing the dimension of the matrix in advance make the process way more fast

    :param threshold:
    :param input_file_path:
    :param traces_num:
    :param constraints_num:
    :param measures_num:
    :return:
    """
    print("Importing data...")
    result = np.zeros((traces_num, constraints_num, measures_num))
    # result = np.ndarray(shape=(1, 1, len(line) - 4)) # shape of the result ndarray
    with open(input_file_path, 'r') as input_file:
        csv_reader = csv.reader(input_file, delimiter=';')
        header = 1
        it = 0
        ic = 0
        i = 0
        for line in csv_reader:
            # print(i / (1050 * 260))
            i += 1
            # First line
            if header > 0:
                # Skip the header line
                header -= 1
                continue
            if ic == constraints_num:
                ic = 0
                it += 1

            # result[it][ic] = np.nan_to_num(np.array(line[2:])) # in case NaN and +-inf is a problem
            result[it][ic] = np.array(int(float(line[2]) > threshold)) # TODO WARNING when more measures are used
            ic += 1
    print("3D shape:" + str(result.shape))
    return result


def import_trace_measures(input_file_path, input_file_format, boolean_flag=False):
    """
    Interface to import the SJ2T results. it calls the appropriate function given the file format.

    :param boolean_flag:
    :param input_file_path:
    :param input_file_format:
    """
    if input_file_format == 'csv':
        traces, constraints_num, measures, constraints = retrieve_csv_trace_measures_metadata(input_file_path)
        if boolean_flag:
            return import_boolean_trace_measures_from_csv(input_file_path, traces, constraints_num, measures)
        else:
            return import_trace_measures_from_csv(input_file_path, traces, constraints_num, measures)
    elif input_file_format == 'json':
        print("Json import not yet implemented")
    else:
        print("[" + str(input_file_format) + "]Format not recognised")


def import_trace_labels_csv(trace_measures_csv_file_path, constraints_num, threshold=0.95):
    """
        Import the labels of the trace measures csv containing the measurement of every constraint in every trace.
        Performances note: Knowing the dimension of the matrix in advance make the process way more fast
    :param constraints_num:
    :param threshold:
    :param trace_measures_csv_file_path:
    :return:
    """
    print("Importing labels...")
    result = {}
    trace_index = []
    repetition = 0
    # result = np.ndarray(shape=(1, 1, len(line) - 4)) # shape of the result ndarray
    with open(trace_measures_csv_file_path, 'r') as input_file:
        csv_reader = csv.reader(input_file, delimiter=';')
        header = 1
        for line in csv_reader:
            # First line
            if header > 0:
                # Skip the header line
                header -= 1
                continue
            result.setdefault(line[0], set())
            if repetition == 0:
                trace_index += [line[0]]
                repetition = constraints_num
            repetition -= 1
            if float(line[2]) > threshold:
                result[line[0]].add(line[1]) # TODO WARNING when more measures are used

    return result, trace_index


def import_trace_labels(input_file_path, constraints_num, threshold):
    """
    Interface to import the labels of SJ2T results. it calls the appropriate function given the file format.

    :param threshold:
    :param input_file_path:
    """
    input_file_format = input_file_path.split(".")[-1]
    if input_file_format == 'csv':
        labels, traces_index = import_trace_labels_csv(input_file_path, constraints_num, threshold)
        return labels, traces_index
    elif input_file_format == 'json':
        print("Json import not yet implemented")
    else:
        print("[" + str(input_file_format) + "]Format not recognised")