import json
import os
import sys


def merge_json_models(models_folder, models_suffix, output_json_model):
    """
Merge all the declarative JSON models contained in a folder.
This method does not care about the measures associated with the constrains, it use the first one encountered.
    :param models_folder: folder containing the json models to merge
    :param models_suffix: shared suffix in all models files names
    :param output_model: output model file path
    """
    with(open(output_json_model, 'w')) as out_file:
        data = {
            "name": "Merged Model",
            "tasks": set(),
            "constraints": []
        }
        visited_constraints = set()
        for file in os.listdir(models_folder):
            if file.endswith(models_suffix):
                with(open(os.path.join(models_folder, file), 'r')) as current_model_file:
                    reader = json.load(current_model_file)
                    for task in reader["tasks"]:
                        data["tasks"].add(task)
                    for constraint in reader['constraints']:
                        par = []
                        for x in constraint['parameters']:
                            for y in x:
                                par += [y]
                        if tuple([constraint['template']] + par) not in visited_constraints:
                            visited_constraints.add(tuple([constraint['template']] + par))
                            data['constraints'] += [constraint]

        data["tasks"] = sorted(list(data["tasks"]))
        json.dump(data, out_file, indent=4)


if __name__ == '__main__':
    print(sys.argv)
    models_folder = sys.argv[1]
    models_suffix = sys.argv[2]
    output_json_model = sys.argv[3]
    merge_json_models(models_folder, models_suffix, output_json_model)
