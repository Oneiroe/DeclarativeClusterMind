""" GUI/CLI interface for Declarative Clustering"""

import os

from gooey import Gooey, GooeyParser
import DeclarativeClusterMind.clustering.cm_clustering as cm_clustering
import DeclarativeClusterMind.clustering.pareto_declarative_hierarchical_clustering as pareto_clustering


@Gooey(
    program_name='Declarative Trace Clustering',
    program_description='Trace clustering based on Declarative Specifications',  # Defaults to ArgParse Description

)
def main():
    """
Use --ignore-gooey option in the terminal to suppress the GUI and use the CLI
    """
    # Common parameters among clusterings
    parent_parser = GooeyParser(add_help=False)
    parent_parser.add_argument('-iL', '--log-file-path',
                               help='Path to input Event Log File',
                               type=str, widget='FileChooser', required=True)
    parent_parser.add_argument('-a', '--clustering-algorithm',
                               help='Algorithm with which clustering the featured traces',
                               type=str, widget='Dropdown',
                               choices=['kmeans',
                                        'affinity',
                                        'meanshift',
                                        'agglomerative',
                                        'spectral',
                                        'dbscan',
                                        'optics',
                                        'birch'],
                               default='optics')
    parent_parser.add_argument('-o', '--output-folder', help='Path to folder where to save the output', type=str,
                               widget='DirChooser', required=True)
    parent_parser.add_argument('-vf', '--visualization-flag',
                               help='Flag to enable the visualization features (CPU heavy): 3D tSNE, ...',
                               action="store_true", widget='BlockCheckbox')
    parent_parser.add_argument('-nc', '--number-of-clusters',
                               help='manuals setting of clusters number (for k-means like) or maximal number of clusters for non-parametrized ones (for density based)',
                               type=int,
                               widget='IntegerField', default=None)

    parser = GooeyParser(description="Behavioural trace clustering based on declarative rules.")
    parser.add_argument('-v', '--version', action='version', version='1.0.0', gooey_options={'visible': False})
    parser.add_argument('--ignore-gooey', help='use the CLI instead of the GUI', action='store_true',
                        gooey_options={'visible': False})
    subparsers = parser.add_subparsers(help='Available clustering policies', dest='clustering_policy')
    subparsers.required = True

    # RULES PARSER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    parser_rules = subparsers.add_parser("rules",
                                         description="clustering based on rules",
                                         help="clustering based on rules", parents=[parent_parser])
    parser_rules.add_argument('-pca', '--apply-pca-flag',
                              help='Flag to enable features reduction through PCA',
                              action="store_true", widget='BlockCheckbox')
    parser_rules.add_argument('-tm', '--trace-measures-csv-file-path',
                              help='Path to the Janus trace measures CSV output',
                              type=str, widget='FileChooser', required=True)
    parser_rules.add_argument('-b', '--boolean-confidence',
                              help='Flag to consider the rules true/false if not enough compliant on a trace or to keep the specific measurements',
                              action="store_true", widget='BlockCheckbox')

    # ATTRIBUTES PARSER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    parser_attributes = subparsers.add_parser("attributes",
                                              description="clustering based on all event log attributes",
                                              help="clustering based on all event log attributes",
                                              parents=[parent_parser])
    parser_attributes.add_argument('-pca', '--apply-pca-flag',
                                   help='Flag to enable features reduction through PCA',
                                   action="store_true", widget='BlockCheckbox')

    # SPECIFIC ATTRIBUTE PARSER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    parser_specific_attribute = subparsers.add_parser("specific-attribute",
                                                      description="clustering based on one event log attribute",
                                                      help="clustering based on one event log attribute",
                                                      parents=[parent_parser])

    # ATTRIBUTES PARSER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    parser_performances = subparsers.add_parser("performances",
                                                description="clustering based on trace performances",
                                                help="clustering based on trace performances",
                                                parents=[parent_parser])
    parser_performances.add_argument('-pca', '--apply-pca-flag',
                                     help='Flag to enable features reduction through PCA',
                                     action="store_true", widget='BlockCheckbox')

    # MIXED PARSER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    parser_mixed = subparsers.add_parser("mixed",
                                         description="clustering based on rules, attributes, and performances",
                                         help="clustering based on rules, attributes, and performances",
                                         parents=[parent_parser])
    parser_mixed.add_argument('-pca', '--apply-pca-flag',
                              help='Flag to enable features reduction through PCA',
                              action="store_true", widget='BlockCheckbox')
    parser_mixed.add_argument('-tm', '--trace-measures-csv-file_path',
                              help='Path to the Janus trace measures CSV output',
                              type=str, widget='FileChooser', required=True)
    parser_mixed.add_argument('-b', '--boolean-confidence',
                              help='Flag to consider the rules true/false if not enough compliant on a trace or to keep the specific measurements',
                              action="store_true", widget='BlockCheckbox')

    # PARETO PARSER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    pareto_parser = subparsers.add_parser("pareto",
                                          description="Hierarchical clustering of a log according to its compliance to a declarative model.",
                                          help="hierarchical clustering based on declarative models")

    pareto_parser.add_argument('-iL', '--input-log', help='Path to input Event Log File', type=str,
                               widget='FileChooser')
    pareto_parser.add_argument('-o', '--output-folder', help='Path to folder where to save the output', type=str,
                               widget='DirChooser')
    pareto_parser.add_argument('-t', '--split-threshold', help='Measure threshold where to split the clusters',
                               type=float,
                               widget='DecimalField', default=0.95, gooey_options={'min': 0.0, 'max': 1.0})
    pareto_parser.add_argument('-j', '--janus-jar-path-global', help='Path to Janus JAR executable', type=str,
                               widget='FileChooser')
    pareto_parser.add_argument('-s', '--simplification-flag',
                               help='Flag to enable the simplification of the models at each step',
                               action="store_true", widget='BlockCheckbox')
    pareto_parser.add_argument('-min', '--min-leaf-size',
                               help='Minimum size of the leaves/clusters below which the recursion is stopped',
                               type=int,
                               widget='IntegerField')

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    args = parser.parse_args()
    print(args)

    clustering_policy = args.clustering_policy
    print("Clustering policy: " + str(clustering_policy))
    # 'rules'
    # 'attributes'
    # 'specific-attribute'
    # 'performances'
    # 'mixed'
    # 'pareto'

    os.makedirs(args.output_folder, exist_ok=True)

    if clustering_policy == 'rules':
        cm_clustering.behavioural_clustering(args.trace_measures_csv_file_path,
                                             args.log_file_path,
                                             args.clustering_algorithm,
                                             args.boolean_confidence,
                                             args.output_folder,
                                             args.visualization_flag,
                                             args.apply_pca_flag,
                                             args.number_of_clusters)
    elif clustering_policy == 'attributes':
        cm_clustering.attribute_clustering(args.log_file_path,
                                           args.clustering_algorithm,
                                           args.output_folder,
                                           args.visualization_flag,
                                           args.apply_pca_flag,
                                           args.number_of_clusters)
    elif clustering_policy == 'specific-attribute':
        cm_clustering.specific_attribute_clustering(args.log_file_path,
                                                    args.clustering_algorithm,
                                                    args.output_folder,
                                                    args.visualization_flag,
                                                    args.number_of_clusters)
    elif clustering_policy == 'performances':
        cm_clustering.performances_clustering(args.log_file_path,
                                              args.clustering_algorithm,
                                              args.output_folder,
                                              args.visualization_flag,
                                              args.apply_pca_flag,
                                              args.number_of_clusters)
    elif clustering_policy == 'mixed':
        cm_clustering.mixed_clustering(args.trace_measures_csv_file_path,
                                       args.log_file_path,
                                       args.clustering_algorithm,
                                       args.boolean_confidence,
                                       args.output_folder,
                                       args.visualization_flag,
                                       args.apply_pca_flag,
                                       args.number_of_clusters)
    elif clustering_policy == 'pareto':
        pareto_clustering.pareto_declarative_hierarchical_clustering(args.input_log,
                                                                     args.output_folder,
                                                                     args.janus_jar_path_global,
                                                                     args.simplification_flag,
                                                                     args.split_threshold,
                                                                     args.min_leaf_size)
    else:
        print("Clustering policy not recognized: " + str(clustering_policy))


if __name__ == '__main__':
    main()
