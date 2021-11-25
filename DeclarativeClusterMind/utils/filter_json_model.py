import csv
import json
import sys


def filter_model_of_specific_templates(input_model, constraints_templates_black_list, output_model):
    """
Given a Json model, remove from it all the constraints from the given templates list.
The list is expected to have a "template" column in the header
    :param input_model: json declare model to filter
    :param constraints_templates_black_list: list of declare templates to filter out
    :param output_model: final output json model
    """
    with open(input_model, 'r') as json_file:
        #       'Constraint';'Template';'Activation';'Target';'Support';'Confidence level';'Interest factor'
        data = json.load(json_file)

        black_listed_templates = set()
        with open(constraints_templates_black_list, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file, fieldnames=['template'], delimiter=';')
            for line in csv_reader:
                if line['template'] == 'template' or line['template'] == "MODEL":
                    continue
                black_listed_templates.add(line['template'])
        filtered_constraints = []
        for constraint in data['constraints']:
            if constraint['template'] not in black_listed_templates:
                filtered_constraints += [constraint]

        data['constraints'] = filtered_constraints
        with open(output_model, 'w') as output_file:
            print("Serializing JSON...")
            json.dump(data, output_file, indent=4)


def filter_model_of_specific_tasks(input_model, tasks_black_list, output_model):
    """
Given a Json model, remove from it all the constraints involving tasks from the given list.
The list is expected to have a "tasks" column in the header
    :param input_model: json declare model to filter
    :param tasks_black_list: list of declare templates to filter out
    :param output_model: final output json model
    """
    with open(input_model, 'r') as json_file:
        #       'Constraint';'Template';'Activation';'Target';'Support';'Confidence level';'Interest factor'
        data = json.load(json_file)

        black_listed_tasks = set()
        with open(tasks_black_list, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file, fieldnames=['tasks'], delimiter=';')
            for line in csv_reader:
                if line['tasks'] == 'tasks' or line['tasks'] == "MODEL":
                    continue
                black_listed_tasks.add(line['tasks'])
        filtered_constraints = []
        for constraint in data['constraints']:
            add = True
            for param in constraint['parameters']:
                if param[0] in black_listed_tasks:
                    add = False
            if add:
                filtered_constraints += [constraint]

        data['constraints'] = filtered_constraints
        with open(output_model, 'w') as output_file:
            print("Serializing JSON...")
            json.dump(data, output_file, indent=4)


if __name__ == '__main__':
    input_model = sys.argv[1]
    black_list = sys.argv[2]
    output_model = sys.argv[3]

    if "tasks" in black_list:
        filter_model_of_specific_tasks(input_model, black_list, output_model)
    else:
        filter_model_of_specific_templates(input_model, black_list, output_model)
