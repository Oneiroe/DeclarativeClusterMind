import csv
import json
import sys


def make_json_model_from_constraints_list(constraints_list_file_path, json_model_file_path):
    """
Given a list of constraints (one per line), the function builds a valid declare Json model.
It is expected an header line with a "Constraint" column for the constraints names.
WARNING: the measures associate to each constraints are set by default to 1

    :param constraints_list_file_path:
    :param json_model_file_path:
    """
    with open(json_model_file_path, 'w') as json_file:
        #       'Constraint';'Template';'Activation';'Target';'Support';'Confidence level';'Interest factor'
        data = {
            "name": "Model",
            "tasks": set(),
            "constraints": []
        }
        with open(constraints_list_file_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file, fieldnames=['Constraint'], delimiter=';')
            for line in csv_reader:
                if line['Constraint'] == 'Constraint' or line['Constraint'] == "MODEL":
                    continue

                template = line['Constraint'].split('(')[0]
                tasks = line['Constraint'].split('(')[1].replace(')', '')  # it may be not, but who cares now
                if "," in tasks:
                    activator = tasks.split(",")[0]
                    target = tasks.split(",")[1]
                    data['tasks'].add(activator)
                    data['tasks'].add(target)
                    data["constraints"] += [
                        {
                            "template": template,
                            "parameters": [
                                [
                                    activator if ("Precedence" not in template) else target
                                ],
                                [
                                    target if ("Precedence" not in template) else activator
                                ]
                            ],
                            "support": 1.0,
                            "confidence": 1.0,
                            "interestFactor": 1.0
                        }
                    ]
                else:
                    data['tasks'].add(tasks)
                    data["constraints"] += [
                        {
                            "template": template,
                            "parameters": [
                                [
                                    tasks
                                ]
                            ],
                            "support": 1.0,
                            "confidence": 1.0,
                            "interestFactor": 1.0
                        }
                    ]
            data["tasks"] = list(data["tasks"])

            print("Serializing JSON...")
            json.dump(data, json_file)


if __name__ == '__main__':
    constraints_list_file_path = sys.argv[1]
    json_model_file_path = sys.argv[2]

    make_json_model_from_constraints_list(constraints_list_file_path, json_model_file_path)
