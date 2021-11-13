import csv
import json
import sys


def filter_model_of_specific_templates(input_model, constraints_templates_black_list, output_model):
    """
Given a Jason model, remove from it all the constraints from the given templates list.
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


if __name__ == '__main__':
    input_model = sys.argv[1]
    constraints_black_list = sys.argv[2]
    output_model = sys.argv[3]

    filter_model_of_specific_templates(input_model, constraints_black_list, output_model)
