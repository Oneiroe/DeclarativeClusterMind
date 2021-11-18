""" Import Janus v2.X measurement results """

import csv

import numpy as np
import pandas as pd


def retrieve_csv_trace_measures_metadata(input_file_path):
    """
    Retrieve metadata from CSV trace measures file
    :param input_file_path:
    """
    print("Retrieving janus results data...")
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


def extract_detailed_trace_rules_perspective_csv(trace_measures_csv_file_path, output_path=None, measure="Confidence",
                                                 skip_model_measures=True):
    """
    From the trace measures, given a specific measure, transpose the results for that one measure for each trace,
    i.e. a matrix where the rows are the constraints and the columns are the traces, and
    each cell contains the measure of the constraint in that trace

    :param trace_measures_csv_file_path:
    :param output_path:
    :param measure:
    :param skip_model_measures: do not import the measures related to the entire model
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
            if skip_model_measures and line['Constraint'] == 'MODEL':
                continue
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

        if output_path is not None:
            with open(output_path, 'w') as out_file:
                writer = csv.DictWriter(out_file, fieldnames=header, delimiter=';')
                writer.writeheader()
                for constraint in temp_res:
                    temp_res[constraint].update({"Constraint": constraint})
                    writer.writerow(temp_res[constraint])
    return featured_data, features_names


def extract_detailed_trace_multi_perspective_csv(trace_measures_csv_file_path,
                                                 trace_labels_file_path,
                                                 output_path,
                                                 label_feature_index=1,
                                                 performances_index=-3,
                                                 measure="Confidence"):
    """
    From the trace measures, given a specific measure, transpose the results for that one measure for each trace,
    i.e. a matrix where the rows are the constraints and attributes and the columns are the traces, and
    each cell contains the measure of the constraint in that trace or the value of the attribute

    :param trace_measures_csv_file_path:
    :param trace_labels_file_path:
    :param output_path:
    :param label_feature_index:
    :param measure:
    :param performances_index:
    """
    # RULES
    featured_data_rules, features_names_rules = extract_detailed_trace_rules_perspective_csv(
        trace_measures_csv_file_path,
        output_path,
        measure)
    # ATTRIBUTES
    featured_data_attributes, features_names_attributes = extract_detailed_attributes_csv(trace_labels_file_path,
                                                                                          label_feature_index,
                                                                                          performances_index
                                                                                          )
    # PERFORMANCES : performances hide the other features, thus they are commented out for now
    # featured_data_performances, features_names_performances = extract_detailed_performances_csv(trace_labels_file_path,
    #                                                                                             label_feature_index,
    #                                                                                             performances_index
    #                                                                                             )

    # MERGE
    featured_data = pd.concat([pd.DataFrame(featured_data_rules, columns=features_names_rules),
                               featured_data_attributes
                                  # , featured_data_performances
                               ], axis=1)
    features_names = np.concatenate([features_names_rules,
                                     features_names_attributes
                                        # , features_names_performances
                                     ])
    return featured_data, features_names


def extract_detailed_attributes_csv(trace_labels_file_path,
                                    label_feature_index=1,
                                    performances_index=-3,
                                    clean_attributes=True):
    """
    Return a matrix where the rows are the attributes and the columns are the traces, and
    each cell contains the value of the attribute

    :param trace_labels_file_path:
    :param label_feature_index:
    :param clean_attributes:
    :param output_path:
    :param performances_index: index from which starts the performances attributes
    :return:
    """
    # ATTRIBUTES
    featured_data_attributes = []
    features_names_attributes = []
    with open(trace_labels_file_path, 'r') as file:
        csv_file = csv.reader(file, delimiter=';')
        header = True
        for line in csv_file:
            if header:
                # BEWARE if index goes out of range it does not rise exception
                if performances_index < 0:
                    features_names_attributes = line[2:label_feature_index] + line[
                                                                              label_feature_index + 1:performances_index]
                else:
                    if label_feature_index <= performances_index:
                        features_names_attributes = line[performances_index:]
                    else:
                        features_names_attributes = line[performances_index:label_feature_index] + line[
                                                                                                   label_feature_index + 1:]
                header = False
            else:
                if performances_index < 0:
                    featured_data_attributes += [
                        line[2:label_feature_index] + line[label_feature_index + 1:performances_index]]
                else:
                    if label_feature_index <= performances_index:
                        featured_data_attributes += [line[performances_index:]]
                    else:
                        featured_data_attributes += [line[performances_index:label_feature_index] + line[
                                                                                                    label_feature_index + 1:]]

    featured_data_attributes = pd.DataFrame(featured_data_attributes, columns=features_names_attributes)
    if clean_attributes:
        # non-numerical attributes and sets cannot be used for decision tree construction
        featured_data_attributes = featured_data_attributes.replace({'\[': '', '\]': ''}, regex=True)
        for i in featured_data_attributes:
            featured_data_attributes[i] = pd.to_numeric(featured_data_attributes[i], errors='coerce')

    # TODO output labelled data for debugging

    return featured_data_attributes, features_names_attributes


def extract_detailed_performances_csv(trace_labels_file_path,
                                      label_feature_index=1,
                                      performances_index=-3,
                                      clean_attributes=True):
    """
    Return a matrix where the rows are the performances and the columns are the traces, and
    each cell contains the value of the performance

    :param trace_labels_file_path:
    :param label_feature_index:
    :param clean_attributes:
    :param output_path:
    :return:
    """
    # ATTRIBUTES
    featured_data_performances = []
    features_names_performances = []
    with open(trace_labels_file_path, 'r') as file:
        csv_file = csv.reader(file, delimiter=';')
        header = True
        for line in csv_file:
            if header:
                # BEWARE if index goes out of range it does not rise exception
                if performances_index < 0:
                    features_names_performances = line[performances_index:]
                else:
                    if performances_index > label_feature_index > 2:
                        features_names_performances = line[2:label_feature_index] + line[
                                                                                    label_feature_index + 1:performances_index]
                    else:
                        features_names_performances = line[2:performances_index]
                header = False
            else:
                if performances_index < 0:
                    featured_data_performances += [line[performances_index:]]
                else:
                    if performances_index > label_feature_index > 2:
                        featured_data_performances += [line[2:label_feature_index] + line[
                                                                                     label_feature_index + 1:performances_index]]
                    else:
                        featured_data_performances += [line[2:performances_index]]

    featured_data_performances = pd.DataFrame(featured_data_performances, columns=features_names_performances)
    if clean_attributes:
        # non-numerical attributes and sets cannot be used for decision tree construction
        featured_data_performances = featured_data_performances.replace({'\[': '', '\]': ''}, regex=True)
        for i in featured_data_performances:
            featured_data_performances[i] = pd.to_numeric(featured_data_performances[i], errors='coerce')

    # TODO output labelled data for debugging

    return featured_data_performances, features_names_performances


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


def import_boolean_trace_measures_from_csv(input_file_path, traces_num, constraints_num, measures_num, threshold=1.0):
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
            result[it][ic] = np.array(int(float(line[2]) >= threshold))  # TODO WARNING when more measures are used
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


def import_log_measures_from_csv(input_file_path):
    """
    interface to import the log measures from Janus3 log measures result CSV file.

    :param input_file_path: Janus3 log measures result file
    :return: HashMap mapping constraint->measure->value
    """
    result = {}
    with open(input_file_path, 'r') as input_file:
        reader = csv.DictReader(input_file, delimiter=';')
        for line in reader:
            for measure in line:
                if measure == 'Constraint':
                    continue
                result[line['Constraint']] = line[measure]

    return result


def import_log_measures(input_file_path):
    """
    interface to import the log measures from Janus3 log measures result file.
    It calls the appropriate function given the file format.

    :param input_file_path: Janus3 log measures result file
    :return: HashMap mapping constraint->measure->value
    """
    input_file_format = input_file_path.split('.')[-1]
    if input_file_format == 'csv':
        return import_log_measures_from_csv(input_file_path)
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
                result[line[0]].add(line[1])  # TODO WARNING when more measures are used

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
