import csv
import os

import pm4py as pm

import plotly.express as px
import pandas as pd

from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.visualization.dfg import visualizer as dfg_visualization


def compute_f1(clusters_logs, traces_clusters_labels, output_csv_file_path,
               discovery_algorithm="heuristics",
               fitness_precision_algorithm="token",
               plot_petrinets=False):
    """
    Compute the F1 score, along with the fitness and precision, of all the clusters.
    The details are stored in the desired output file and the average is returned in output.

    :param plot_petrinets:
    :param clusters_logs: list of XES log parsers
    :param traces_clusters_labels: list of clusters labels, where each index is the index of the trace and the value is the associated cluster label
    :param output_csv_file_path:
    :return: fitness_avg, precision_avg, f1_avg
    :param discovery_algorithm: [heuristics, inductive]
    :param fitness_precision_algorithm:
    """
    header = ['CLUSTER', 'TRACES_NUM', 'FITNESS', 'PRECISION', 'F1']

    # retrieve and output stats
    with open(output_csv_file_path, 'w') as detailed_output:
        csv_detailed_out = csv.writer(detailed_output, delimiter=';')
        csv_detailed_out.writerow(header)

        fitness_avg = 0
        precision_avg = 0
        f1_avg = 0
        tot_clusters = len(clusters_logs)

        fitness_weighted_avg = 0
        precision_weighted_avg = 0
        f1_weighted_avg = 0
        tot_taces = 0

        current_index = 0
        for current_s_log in clusters_logs:
            traces_num = len(current_s_log)
            tot_taces += traces_num

            # Model discovery
            if discovery_algorithm == 'heuristics':
                petri_net, initial_marking, final_marking = pm.discover_petri_net_heuristics(current_s_log)
            elif discovery_algorithm == 'inductive':
                petri_net, initial_marking, final_marking = pm.discover_petri_net_inductive(current_s_log, 0.3)
            else:
                print(f"ERROR: discovery algorithm not recognized: {discovery_algorithm}")
                exit(1)

            if plot_petrinets:
                gviz = pn_visualizer.apply(petri_net, initial_marking, final_marking)
                pt_visualizer.view(gviz)
                pt_visualizer.save(gviz, os.path.join(os.path.dirname(output_csv_file_path),
                                                      f"{traces_clusters_labels[current_index]}-petrinet.dot"))

            # FITNESS & PRECISION
            if fitness_precision_algorithm == 'token':
                fitness_replay_dictio = pm.fitness_token_based_replay(current_s_log, petri_net, initial_marking,
                                                                      final_marking)
                fitness = fitness_replay_dictio['log_fitness']
                precision = pm.precision_token_based_replay(current_s_log, petri_net, initial_marking, final_marking)
            elif fitness_precision_algorithm == 'alignments':
                fitness_align_dictio = pm.fitness_alignments(current_s_log, petri_net, initial_marking, final_marking)
                fitness = fitness_align_dictio['averageFitness']
                precision = pm.precision_alignments(current_s_log, petri_net, initial_marking, final_marking)
            else:
                print(f"ERROR: fitness/precision algorithm not recognized: {fitness_precision_algorithm}")
                exit(1)

            f1 = 2 * (precision * fitness) / (precision + fitness)

            fitness_avg += fitness
            precision_avg += precision
            f1_avg += f1

            fitness_weighted_avg += fitness * traces_num
            precision_weighted_avg += precision * traces_num
            f1_weighted_avg += f1 * traces_num

            row_to_write = [traces_clusters_labels[current_index], traces_num, fitness, precision, f1]
            csv_detailed_out.writerow(row_to_write)

            current_index += 1

        fitness_avg = fitness_avg / tot_clusters
        precision_avg = precision_avg / tot_clusters
        f1_avg = f1_avg / tot_clusters

        fitness_weighted_avg = fitness_weighted_avg / tot_taces
        precision_weighted_avg = precision_weighted_avg / tot_taces
        f1_weighted_avg = f1_weighted_avg / tot_taces

        csv_detailed_out.writerow(["AVERAGE", "", fitness_avg, precision_avg, f1_avg])
        csv_detailed_out.writerow(
            ["WEIGHTED-AVERAGE", "", fitness_weighted_avg, precision_weighted_avg, f1_weighted_avg])

    print(f"average Fitness: {fitness_avg}, weighted average:{fitness_weighted_avg}")
    print(f"average Precision: {precision_avg}, weighted average:{precision_weighted_avg}")
    print(f"average F1: {f1_avg}, weighted average:{f1_weighted_avg}")

    return fitness_avg, precision_avg, f1_avg


def aggregate_f1_results(base_result_folder, output_file_path, plot_output_path=None):
    """
Aggregate the results of different F1-score tests. It is used the WEIGHTED average foreach technique.

It is expected that:
    - base_result_folder containing sub-folders containing f1-scores.
    - the f1-scores results are CSV files with "f1" in the name

The label for each entry-technique is given by the name of the sub-folder

    :param base_result_folder:
    :param output_file_path:
    :param plot_output_path: if given, plot the result in this file as bar-plot
    """
    print("Aggregating F1-scores results...")
    with open(output_file_path, 'w') as out_file:
        header = ['TECHNIQUE', 'CLUSTERS_NUM', 'FITNESS', 'PRECISION', 'F1']
        header_results = ['CLUSTER', 'TRACES_NUM', 'FITNESS', 'PRECISION', 'F1']
        result_writer = csv.DictWriter(out_file, header, delimiter=';')
        result_writer.writeheader()
        for dirName, subdirList, fileList in os.walk(base_result_folder):
            for file in fileList:
                if file.endswith(".csv") and "f1" in file:
                    with open(os.path.join(dirName, file), 'r') as curr_result:
                        result_reader = csv.DictReader(curr_result, header_results, delimiter=';')
                        clusters_num = 0
                        for line in result_reader:
                            if line['CLUSTER'] == 'AVERAGE':
                                continue
                            if line['CLUSTER'] == 'CLUSTER':
                                continue
                            if line['CLUSTER'] == 'WEIGHTED-AVERAGE':
                                result_writer.writerow({
                                    'TECHNIQUE': os.path.basename(dirName),
                                    'CLUSTERS_NUM': clusters_num,
                                    'FITNESS': line['FITNESS'],
                                    'PRECISION': line['PRECISION'],
                                    'F1': line['F1']
                                })
                                continue
                            clusters_num += 1
    if plot_output_path is not None:
        visualize_bar_plot_aggregated_f1_scores(output_file_path, plot_output_path)


def visualize_bar_plot_aggregated_f1_scores(aggregated_f1_csv_file_path,
                                            ouput_graph_file_path,
                                            popup_immediate_visualization=False):
    """
Plot a bar-plot for the aggregated results of the F1-scores
    :param aggregated_f1_csv_file_path:
    :param ouput_graph_file_path:
    :param popup_immediate_visualization:
    """
    print("Plotting aggregated F1-scores results...")
    data = pd.read_csv(aggregated_f1_csv_file_path, delimiter=';')
    data = data.sort_values(by=['F1'])

    colour_map = {}
    for index, x in enumerate(data["CLUSTERS_NUM"].sort_values()):
        if str(x) in colour_map:
            continue
        colour_map[str(x)] = px.colors.sequential.Plasma_r[index]

    data["CLUSTERS_NUM"] = data["CLUSTERS_NUM"].astype(str)
    fig = px.bar(data, x='TECHNIQUE', y='F1',
                 text='F1',
                 color='CLUSTERS_NUM',
                 # color_discrete_sequence=px.colors.sequential.Plasma_r
                 color_discrete_map=colour_map,
                 )
    # fig.update_layout(legend_traceorder="normal")
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    # BASE-LINE AT unclustered results
    # fig.add_shape(  # add a horizontal "target" line
    #     type="line", line_color="salmon", line_width=3, opacity=1, line_dash="dot",
    #     x0=0, x1=1, xref="paper", y0=950, y1=950, yref="y"
    # )

    if popup_immediate_visualization:
        fig.show()
    fig.write_image(ouput_graph_file_path)
    fig.write_html(f"{ouput_graph_file_path}.html")


def plot_clusters_imperative_models(clusters_logs, model='DFG'):
    """
        Plot the desired imperative model of each cluster
    :param clusters_logs:
    :param model:
    """
    # Imperative models
    print("clusters imperative models...")
    for cluster_index in clusters_logs:
        # PROCESS TREE
        if model == 'process-tree':
            tree = inductive_miner.apply_tree(clusters_logs[cluster_index])
            gviz = pt_visualizer.apply(tree)
            pt_visualizer.view(gviz)
        # PETRI-NET
        elif model == 'petrinet':
            net, initial_marking, final_marking = inductive_miner.apply(clusters_logs[cluster_index])
            gviz = pn_visualizer.apply(net)
            pt_visualizer.view(gviz)
        ## HEURISTIC-NET
        elif model == 'heuristic-net':
            heu_net = heuristics_miner.apply_heu(clusters_logs[cluster_index], parameters={
                heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.99})
            gviz = hn_visualizer.apply(heu_net)
            hn_visualizer.view(gviz)
        ## Directly Follow Graph
        elif model == 'DFG':
            dfg = dfg_discovery.apply(clusters_logs[cluster_index], variant=dfg_discovery.Variants.PERFORMANCE)
            gviz = dfg_visualization.apply(dfg, log=clusters_logs[cluster_index],
                                           variant=dfg_visualization.Variants.PERFORMANCE)
            dfg_visualization.view(gviz)


if __name__ == '__main__':
    result_path = "/Trace-Clustering-Competitors/TraCluSi/TraCluSi-executable/output/SEPSIS"
    out_file_csv = "/home/alessio/Data/Phd/my_code/ClusterMind/Trace-Clustering-Competitors/TraCluSi/TraCluSi-executable/output/SEPSIS/f1-score_SEPSIS(copy).csv"
    plot_path = "/home/alessio/Data/Phd/my_code/ClusterMind/Trace-Clustering-Competitors/TraCluSi/TraCluSi-executable/output/SEPSIS/plot.svg"
    # aggregate_f1_results(result_path, out_file_csv)
    visualize_bar_plot_aggregated_f1_scores(out_file_csv, plot_path, True)
