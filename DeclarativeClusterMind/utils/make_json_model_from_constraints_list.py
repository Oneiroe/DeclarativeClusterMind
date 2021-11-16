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
            data["tasks"] = sorted(list(data["tasks"]))

            print("Serializing JSON...")
            json.dump(data, json_file, indent=4)


def make_json_model_from_single_pattern(tasks_list_file_path, json_model_file_path, template, n_tasks_per_rule=2):
    """
Given a list of tasks (one per line), the function builds a valid model
with all the combination of the tasks given the desired template.
It is expected an header line with a "Task" column for the tasks names.

    :param tasks_list_file_path:
    :param json_model_file_path:
    :param template:
    :param n_tasks_per_rule:
    """
    tasks = set()
    with open(tasks_list_file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file, fieldnames=['Task'], delimiter=';')
        for line in csv_reader:
            if line['Task'] == 'Task':
                continue
            tasks.add(line['Task'])
    with open(json_model_file_path, 'w') as json_file:
        #       'Constraint';'Template';'Activation';'Target';'Support';'Confidence level';'Interest factor'
        data = {
            "name": f"Model {template}",
            "tasks": sorted(list(tasks)),
            "constraints": []
        }
        if n_tasks_per_rule == 1:
            for task in tasks:
                data["constraints"] += [
                    {
                        "template": template,
                        "parameters": [
                            [
                                task
                            ]
                        ],
                        "support": 1.0,
                        "confidence": 1.0,
                        "interestFactor": 1.0
                    }
                ]

        elif n_tasks_per_rule == 2:
            for activator in tasks:
                for target in tasks:
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

        print("Serializing JSON...")
        json.dump(data, json_file, indent=4)


if __name__ == '__main__':
    list_file_path = sys.argv[1]
    json_model_file_path = sys.argv[2]

    with open(list_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for line in csv_reader:
            if line[0] == 'Task':
                pattern = sys.argv[3]
                n_tasks_per_rule = int(sys.argv[4])
                make_json_model_from_single_pattern(list_file_path, json_model_file_path, pattern)
            elif line[0] == 'Constraint':
                make_json_model_from_constraints_list(list_file_path, json_model_file_path)
            else:
                print("Error! List not recognized")
            break
