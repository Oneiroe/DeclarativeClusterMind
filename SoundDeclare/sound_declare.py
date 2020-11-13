import csv
import json
import sys
import ClusterMind.IO.SJ2T_import as cmio
import numpy as np

# Default threshold for relevant measures
MEASURES_THRESHOLDS = {
    "Confidence": 0.8,  # Confidence must be high
    "Piatetsky-Shapiro": 0.05,
    # 0 is equivalent to statistical independence, >0 positive dependence, <0 negative dependence
    "Lift": 1.1  # 1 is equivalent to statistical independence, >1 positive dependence, <1 negative dependence
}


def retrieve_mutual_information(sj2t_3d):
    """
Retrieve the mutual information measure of all the constraints constraint

    MEMO. shape input (trace, constraint, measure)
    :param sj2t_3d:
    """
    constraints_n = range(sj2t_3d.shape[1])
    result = dict.fromkeys(constraints_n, 0)
    for trace in sj2t_3d:
        for constraint in constraints_n:
            support = trace[constraint][0]
            information_gain = trace[constraint][31]  # check the index
            current = support * information_gain
            if np.isnan(current):
                continue
            result[constraint] += current

    return result


def retrieve_chi_squared(sj2t_3d):
    """
Retrieve the chi-squared measure of all the constraints constraint

    MEMO. shape input (trace, constraint, measure)
    :param sj2t_3d:
    """
    constraints_n = range(sj2t_3d.shape[1])
    result = dict.fromkeys(constraints_n, 0)
    for trace in sj2t_3d:
        for constraint in constraints_n:
            # n= ???
            pA = trace[constraint][12]
            pT = trace[constraint][13]
            piatetsky_shapiro = trace[constraint][29]  # check the index
            # current = n * (piatetsky_shapiro * piatetsky_shapiro) / (pA * pT)
            current = (piatetsky_shapiro * piatetsky_shapiro) / (pA * pT)
            if np.isnan(current):
                continue
            result[constraint] += current

    return result


def statistical_trim(sj2t_focused_csv, measures_thresholds):
    result = set()
    original_counter = 0
    with open(sj2t_focused_csv, 'r') as input_file:
        csv_reader = csv.DictReader(input_file, delimiter=';')
        for constraint in csv_reader:
            original_counter += 1
            result.add(constraint['Constraint'])
            for measure in measures_thresholds:
                if float(constraint[measure]) < measures_thresholds[measure]:
                    result.remove(constraint['Constraint'])
                    break

    print("Original Constraints: " + str(original_counter))
    print("Remaining Constriants: " + str(len(result)))
    print("Trimmed " + str((1 - len(result) / original_counter) * 100) + "%")
    return result


def export_declare_model_csv(constraints, output_file):
    print("Exporting model into csv...")
    header = ['Constraint', 'Template', 'Activation', 'Target']
    with open(output_file, 'w') as file:
        csv_writer = csv.writer(file, delimiter=";")
        csv_writer.writerow(header)
        for constraint in constraints:
            template = constraint.split('(')[0]
            var1 = constraint.split('(')[1].strip(')').split(',')[0]
            if len(constraint.split('(')[1].strip(')').split(',')) > 1:
                var2 = constraint.split('(')[1].strip(')').split(',')[1].strip()
                if 'precedence' in constraint:
                    csv_writer.writerow([constraint, template, var2, var1])
                else:
                    csv_writer.writerow([constraint, template, var1, var2])
            else:
                csv_writer.writerow([constraint, template, var1, ""])


def export_declare_model_json(constraints, output_file, log_name="sound model"):
    print("Exporting model into json...")
    result = {}
    with open(output_file, 'w') as file:
        result["name"] = log_name
        result["tasks"] = set()
        result["constraints"] = []
        for constraint in constraints:
            template = constraint.split('(')[0]
            var1 = constraint.split('(')[1].strip(')').split(',')[0]
            result["tasks"].add(var1)
            if len(constraint.split('(')[1].strip(')').split(',')) > 1:
                var2 = constraint.split('(')[1].strip(')').split(',')[1].strip()
                result["constraints"] += [
                    {
                        "template": template,
                        "parameters": [
                            [
                                var1
                            ],
                            [
                                var2
                            ]
                        ],
                        "support": 1.0,
                        "confidence": 1.0,
                        "interestFactor": 1.0
                    }
                ]
                result["tasks"].add(var2)
            else:
                result["constraints"] += [
                    {
                        "template": template,
                        "parameters": [
                            [
                                var1
                            ]
                        ],
                        "support": 1.0,
                        "confidence": 1.0,
                        "interestFactor": 1.0
                    }
                ]
        result["tasks"] = list(result["tasks"])
        json.dump(result, file, indent=4)


if __name__ == '__main__':
    sj2t_json_aggregated_file_path = sys.argv[1]
    focussed_csv = sys.argv[2]
    output_model_file_path = sys.argv[3]

    cmio.extract_aggregated_perspective(sj2t_json_aggregated_file_path,
                                        focussed_csv,
                                        "Mean",
                                        MEASURES_THRESHOLDS.keys())

    new_constraints = statistical_trim(focussed_csv, MEASURES_THRESHOLDS)
    export_declare_model_csv(new_constraints, output_model_file_path + ".csv")
    export_declare_model_json(new_constraints, output_model_file_path + ".json")

    ## Retrieve chi-square and information gain
    # sj2t_csv_detailed_file_path = sys.argv[1]
    # boolean_confidence = sys.argv[2]
    # # Import SJ2T traces results
    # traces, constraints_num, measures, constraints = cmio.retrieve_SJ2T_csv_data(sj2t_csv_detailed_file_path)
    # file_format = sj2t_csv_detailed_file_path.split(".")[-1]
    # boolean_confidence = False
    # input3D = cmio.import_SJ2T(sj2t_csv_detailed_file_path, file_format, boolean=boolean_confidence)
    # input2D = input3D.reshape((input3D.shape[0], input3D.shape[1] * input3D.shape[2]))
    #
    # print("CHI-SQUARED")
    # chi = retrieve_chi_squared(input3D)
    # print(chi)
    # print("MUTUAL-INFORMATION")
    # mi = retrieve_mutual_information(input3D)
    # print(mi)
