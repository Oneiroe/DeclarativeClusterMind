import csv
import numpy as np


# Trace;Constraint;Events-Evaluation;Support;Confidence;Recall;.....
# 0;    1;         2;                [3:-1]
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
            m = np.array(line[3:-1])
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


def import_SJ2T_csv_known(input_file_path, traces, constraints, measures):
    """
        Knowing the dimension of the matrix in advance make the process way more fast
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

            result[it][ic] = np.array(line[3:-1])
            ic += 1
    print("result shape:" + str(result.shape))
    return result


def retrieve_csv_data(input_file_path):
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
                measures = len(line[3:-1])
                continue
            lines += 1
            c.add(line[1])
        constraints = len(c)
        traces = int(lines / constraints)
    print("traces:" + str(traces) + ",constraints:" + str(constraints) + ",measures:" + str(measures))
    return traces, constraints, measures


file_path = "/home/alessio/Data/Phd/my_code/ClusterMind/input/SEPSIS-output.csv"
# result = import_SJ2T_csv(file_path)
traces, constraints, measures = retrieve_csv_data(file_path)
result = import_SJ2T_csv_known(file_path, traces, constraints, measures)
# result = import_SJ2T_csv_known(file_path, 1050, 260, 36)
pass
