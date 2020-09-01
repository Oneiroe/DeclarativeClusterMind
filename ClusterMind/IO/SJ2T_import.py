import csv
import numpy as np


# Trace;Constraint;Events-Evaluation;Support;Confidence;Recall;.....
# 0;    1;         2;                [3:]
@DeprecationWarning
def import_SJ2T_csv(input_file_path):
    """
        SLOW: the creation/resizing of the ndarrays is super slow
    :param input_file_path:
    :return:
    """
    result = np.array([])
    # result = np.ndarray(shape=(1, 1, len(line) - 4)) # shape of the result ndarray
    with open(input_file_path, 'r') as input_file:
        csv_reader = csv.reader(input_file, delimiter=';')
        header = 1
        first_trace = ''
        c = 0
        i = 0
        for line in csv_reader:
            print(i / (1050 * 260))
            i += 1
            # First line
            if header > 0:
                # Skip the header line
                header -= 1
                continue
            # First trace initialization
            m = np.array(line[3:])
            if result.size == 0:
                result = np.reshape(m, (1, 1, len(line) - 4))
                first_trace = line[0]
                pass
            # First trace
            elif line[0] == first_trace:
                result = np.append(result, np.reshape(m, (1, 1, len(line) - 4)), axis=1)
            # Other traces
            elif c < result.shape[1]:
                result[-1][c] = m
                c += 1
            # first time other traces
            else:
                result = np.resize(result, (result.shape[0] + 1, result.shape[1], result.shape[2]))
                c = 0
                result[-1][c] = m
                c += 1
    print(result.shape)
    return result


def retrieve_SJ2T_csv_data(input_file_path):
    """
    retrieve the information regarding the SJ2T csv results. Specifically the number of traces, constraints, and measures

    :param input_file_path:
    :return:
    """
    print("Retrieving results data...")
    traces = 0
    constraints = 0
    measures = 0
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
                measures = len(line[3:])
                continue
            lines += 1
            c.add(line[1])
        constraints = len(c)
        traces = int(lines / constraints)
    print("traces:" + str(traces) + ",constraints:" + str(constraints) + ",measures:" + str(measures))
    return traces, constraints, measures


def import_SJ2T_csv_known(input_file_path, traces, constraints, measures):
    """
        Import the result from SJ2T csv containing the measurement of every constraint in every trace.
        Performances note: Knowing the dimension of the matrix in advance make the process way more fast
    :param input_file_path:
    :param traces:
    :param constraints:
    :param measures:
    :return:
    """
    print("Importing data...")
    result = np.zeros((traces, constraints, measures))
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
            if ic == constraints:
                ic = 0
                it += 1

            # result[it][ic] = np.nan_to_num(np.array(line[3:])) # in case NaN and +-inf is a problem
            result[it][ic] = np.array(line[3:])
            ic += 1
    print("3D shape:" + str(result.shape))
    return result


def import_SJ2T(input_file_path, input_file_format):
    """
    Interface to import the SJ2T results. it calls the appropriate function given the file format.

    :param input_file_path:
    :param input_file_format:
    """
    if input_file_format == 'csv':
        traces, constraints, measures = retrieve_SJ2T_csv_data(input_file_path)
        result = import_SJ2T_csv_known(input_file_path, traces, constraints, measures)
        return result
    elif input_file_format == 'json':
        print("Json inport not yet implemented")
    else:
        print("[" + str(input_file_format) + "]Format not recognised")
