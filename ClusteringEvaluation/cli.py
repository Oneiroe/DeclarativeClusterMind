from gooey import Gooey, GooeyParser

from ClusteringEvaluation import f1_score, utils


@Gooey(
    program_name='Clustering independent Evaluation',
    program_description='Evaluation of clustering independent from the technique used',
    # Defaults to ArgParse Description
)
def main():
    parent_parser = GooeyParser(add_help=False)

    parent_parser.add_argument('-iLf', '--input-logs-folder',
                               help='Path to the folder containing the clusters event logs', type=str,
                               widget='DirChooser', required=True)

    parent_parser.add_argument('-o', '--output-file', help='Path to file where to save the output', type=str,
                               widget='FileChooser', required=True)

    parser = GooeyParser(
        description="Evaluation of clustering results independent form the techniques used. It takes in input only the resulting clustered event logs.")
    parser.add_argument('-v', '--version', action='version', version='1.0.0', gooey_options={'visible': False})
    parser.add_argument('--ignore-gooey', help='use the CLI instead of the GUI', action='store_true',
                        gooey_options={'visible': False})
    subparsers = parser.add_subparsers(help='Available evaluation metrics', dest='metric')
    subparsers.required = True

    # F1-SCORE parser >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    parser_f1 = subparsers.add_parser("f1",
                                      description="Average Precision/Recall measure of the clusters",
                                      help="Precision/Recall measure of the clusters", parents=[parent_parser])
    parser_f1.add_argument('-a', '--discovery-algorithm',
                           help='Discovery algorithm to be used for the discovery of clusters models',
                           type=str, widget='Dropdown',
                           choices=['inductiveMiner',
                                    'heuristicMiner'],
                           default='inductiveMiner')

    # SILHOUETTE parser >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    parser_silhouette = subparsers.add_parser("silhouette",
                                      description="Silhouette measure of the clusters",
                                      help="Silhouette measure of the clusters", parents=[parent_parser])

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    args = parser.parse_args()
    print(args)

    metric = args.metric
    print("Clustering policy: " + str(metric))
    # 'f1'
    # 'silhouette'

    if metric == 'f1':
        clusters_logs, indices_logs = utils.load_clusters_logs_from_folder(args.input_logs_folder)
        f1_score.compute_f1(
            clusters_logs,
            indices_logs,
            args.output_file
        )
    elif metric == 'silhouette':
        print("Not yet Implemented")


if __name__ == '__main__':
    main()