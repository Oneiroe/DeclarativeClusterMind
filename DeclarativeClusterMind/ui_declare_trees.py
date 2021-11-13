""" GUI/CLI interface for Declarative Decision Trees

Trees supported:
- simple tree
- decision tree

Supported details levels:
- traces to clusters
- logs to clusters

perspectives supported:
- declare rules (trace/log measures)
- attributes (trace/log)
- performances (traces)
"""

import os

from gooey import Gooey, GooeyParser

from DeclarativeClusterMind.declare_trees.decision_trees import retrieve_decision_tree_rules_for_logs_to_clusters, \
    retrieve_decision_tree_multi_perspective_for_traces_to_clusters, \
    retrieve_decision_tree_attributes_for_traces_to_clusters, \
    retrieve_decision_tree_performances_for_traces_to_clusters, retrieve_decision_tree_rules_for_traces_to_clusters
from DeclarativeClusterMind.declare_trees.simple_trees import build_declare_tree_static, build_declare_tree_dynamic, \
    build_declare_tree_variance


@Gooey(
    program_name='Declarative Decision Trees',
    program_description='Explanation of clusters differences via decision trees based on declarative rules, log attributes and performances'
)
def main():
    """
Use --ignore-gooey option in the terminal to suppress the GUI and use the CLI
    """
    # Common parameters among decision trees
    parent_parser = GooeyParser(add_help=False)
    parent_parser.add_argument('-i', '--input-featured-data',
                               help='Path to the input featured data CSV file',
                               type=str, widget='FileChooser', required=True)
    parent_parser.add_argument('-o', '--output-file',
                               help='Path to the output DOT/SVG tree file',
                               type=str, widget='FileSaver', required=True)

    parser = GooeyParser(description="Clusters log distinctions explanation via decision trees")
    parser.add_argument('-v', '--version', action='version', version='1.0.0', gooey_options={'visible': False})
    parser.add_argument('--ignore-gooey', help='use the CLI instead of the GUI', action='store_true',
                        gooey_options={'visible': False})
    subparsers = parser.add_subparsers(help='Available decision tree options', dest='tree_technique')
    subparsers.required = True

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>> DECISION TREE LOGS 2 CLUSTERS PARSER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    parser_decision_tree_logs_to_clusters = subparsers.add_parser(
        "decision-tree-logs-to-clusters",
        description="decision tree via SciKit CART implementation mapping logs to clusters",
        help="decision tree via SciKit CART implementation mapping logs to clusters",
        parents=[parent_parser])

    parser_decision_tree_logs_to_clusters.add_argument(
        '-fi', '--classification-feature-index',
        help='index of the feature, among the ones in the input featured data, that should be used for the classification (by default CLUSTER column 0)',
        type=int,
        widget='IntegerField', default=0)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>> DECISION TREE TRACES 2 CLUSTERS PARSER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    parser_decision_tree_traces_to_clusters = subparsers.add_parser(
        "decision-tree-traces-to-clusters",
        description="decision tree via SciKit CART implementation mapping traces to clusters",
        help="decision tree via SciKit CART implementation mapping traces to clusters",
        parents=[parent_parser])

    parser_decision_tree_traces_to_clusters.add_argument(
        '-fi', '--classification-feature-index',
        help='index of the feature, among the ones in the input featured data, that should be used for the classification (by default CLUSTER column 0)',
        type=int,
        widget='IntegerField', default=0)
    parser_decision_tree_traces_to_clusters.add_argument(
        '-p', '--split-perspective',
        help='Perspective upon which splitting the nodes of the tree',
        type=str, widget='Dropdown',
        choices=['rules',
                 'attributes',
                 # 'specific-attribute',
                 'performances',
                 'mixed'],
        default='rules')
    parser_decision_tree_traces_to_clusters.add_argument(
        '-m', '--rules-measures',
        help='Path to the Janus CSV file trace/log measures', type=str,
        widget='FileChooser', required=True)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>> SIMPLE TREE LOGS 2 CLUSTERS PARSER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    parser_simple_tree_logs_to_clusters = subparsers.add_parser(
        "simple-tree-logs-to-clusters",
        description="simple declarative-rules-only decision tree guiding to clusters logs differences",
        help="simple declarative-rules-only decision tree guiding to clusters logs differences",
        parents=[parent_parser])

    parser_simple_tree_logs_to_clusters.add_argument(
        '-t', '--constraints-threshold',
        help='Measure threshold where to split the node',
        type=float,
        widget='DecimalField', default=0.95
        # , gooey_options={'min': 0.0, 'max': 1.0}
    )
    parser_simple_tree_logs_to_clusters.add_argument(
        '-p', '--branching-policy',
        help='Policy for splitting the nodes of the tree',
        type=str, widget='Dropdown',
        choices=['static-frequency',
                 'dynamic-frequency',
                 'dynamic-variance'],
        default='dynamic-variance')
    parser_simple_tree_logs_to_clusters.add_argument(
        '-decreasing', '--decreasing-order',
        help='Use a decreasing order to select the rule given the chosen branching policy measure, ascending otherwise',
        action="store_true", widget='BlockCheckbox')
    parser_simple_tree_logs_to_clusters.add_argument(
        '-min', '--minimize-tree',
        help='Flag to enable the minimization of the tree: collapsing of singles choice transitions',
        action="store_true", widget='BlockCheckbox')

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>> SIMPLE TREE TRACES PARSER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # NOT YET UPDATED TO JANUS3!
    # parser_simple_tree_traces = subparsers.add_parser(
    #     "simple-tree-traces",
    #     description="simple declarative-rules-only decision tree guiding to traces differences",
    #     help="simple declarative-rules-only decision tree guiding to traces differences",
    #     parents=[parent_parser])

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    args = parser.parse_args()
    print(args)

    tree_technique = args.tree_technique
    print(f"Decision tree technique: {str(tree_technique)}")
    # 'simple-tree-traces'
    # 'simple-tree-logs-to-clusters'
    # 'decision-tree-traces-to-clusters'
    # 'decision-tree-logs-to-clusters'

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    if tree_technique == 'simple-tree-traces':
        print("WARNING! Legacy script still using SJ2T measures")
        pass

    elif tree_technique == 'simple-tree-logs-to-clusters':
        branching_policy = args.branching_policy
        if branching_policy == "static-frequency":
            build_declare_tree_static(args.input_featured_data,
                                      args.constraints_threshold,
                                      args.output_file,
                                      args.minimize_tree,
                                      args.decreasing_order)
        elif branching_policy == "dynamic-frequency":
            build_declare_tree_dynamic(args.input_featured_data,
                                       args.constraints_threshold,
                                       args.output_file,
                                       args.minimize_tree,
                                       args.decreasing_order)
        elif branching_policy == "dynamic-variance":
            build_declare_tree_variance(args.input_featured_data,
                                        args.constraints_threshold,
                                        args.output_file,
                                        args.minimize_tree,
                                        args.decreasing_order)
        else:
            print("Branching policy not recognized. Supported policies: [static-frequency, dynamic-frequency, dynamic-variance] ")

    elif tree_technique == 'decision-tree-traces-to-clusters':
        split_policy = args.split_perspective
        focussed_csv = os.path.join(os.path.dirname(args.output_file), "focus.csv")
        if split_policy == 'mixed':
            # MIXED
            print("multi-perspective decision tree")
            retrieve_decision_tree_multi_perspective_for_traces_to_clusters(args.input_featured_data,
                                                                            args.rules_measures,
                                                                            args.output_file,
                                                                            focussed_csv,
                                                                            args.classification_feature_index)
        elif split_policy == 'attributes':
            # ATTRIBUTES
            print("attributes-only decision tree")
            retrieve_decision_tree_attributes_for_traces_to_clusters(args.input_featured_data,
                                                                     args.rules_measures,
                                                                     args.output_file,
                                                                     focussed_csv,
                                                                     args.classification_feature_index)
        elif split_policy == 'specific-attribute':
            # SPECIFIC ATTRIBUTE
            print("split on single specific attribute not yet implemented")
            pass
        elif split_policy == 'performances':
            # PERFORMANCES
            print("performances-only decision tree")
            retrieve_decision_tree_performances_for_traces_to_clusters(args.input_featured_data,
                                                                       args.rules_measures,
                                                                       args.output_file,
                                                                       focussed_csv,
                                                                       args.classification_feature_index)
        elif split_policy == 'rules':
            # RULES
            print("rules-only decision tree")
            retrieve_decision_tree_rules_for_traces_to_clusters(args.input_featured_data,
                                                                args.rules_measures,
                                                                args.output_file,
                                                                focussed_csv,
                                                                args.classification_feature_index)
        else:
            print("ERROR: Decision tree split policy not recognized")

    elif tree_technique == 'decision-tree-logs-to-clusters':
        retrieve_decision_tree_rules_for_logs_to_clusters(args.input_featured_data,
                                                          args.output_file,
                                                          args.classification_feature_index)
    else:
        print("Decision tree technique not recognized: " + str(tree_technique))


if __name__ == '__main__':
    main()
