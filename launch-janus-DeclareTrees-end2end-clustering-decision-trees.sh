#!/bin/bash

# WHAT: Create a classic decision tree from the clustering output
# WHY: hopefully the decision trees models in a concise and meaningful way the discriminants between the clusters

##################################################################
# PARAMETERS
##################################################################
# experiment folders
EXPERIMENT_NAME="experiments/DECISION-TREE-CLUSTERS"
INPUT_FOLDER=$EXPERIMENT_NAME"/0-input"
PREPROCESSED_DATA_FOLDER=$EXPERIMENT_NAME"/1-generated-logs-and-models"
PROCESSED_DATA_FOLDER=$EXPERIMENT_NAME"/2-clustered-logs"
RESULTS_FOLDER=$EXPERIMENT_NAME"/3-results"
mkdir -p $EXPERIMENT_NAME $INPUT_FOLDER $PREPROCESSED_DATA_FOLDER $PROCESSED_DATA_FOLDER $RESULTS_FOLDER

# Janus main classes
LOG_MAINCLASS="minerful.MinerFulLogMakerStarter"
SIMPLIFIER_MAINCLASS="minerful.MinerFulSimplificationStarter"
ERROR_MAINCLASS="minerful.MinerFulErrorInjectedLogMakerStarter"
JANUS_DISCOVERY_MAINCLASS="minerful.JanusOfflineMinerStarter"
JANUS_CHECK_MAINCLASS="minerful.JanusMeasurementsStarter"

# Input log
LOG_NAME="SEPSIS"
#LOG_NAME="RTFMP"
#LOG_NAME="BPIC13"
INPUT_LOG=$INPUT_FOLDER"/"$LOG_NAME"-log.xes"
LOG_ENCODING="xes"

# Discovery & Measurements
SUPPORT=0.0
CONFIDENCE=0.5
MODEL=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME".xes-model[s_"$SUPPORT"_c_"$CONFIDENCE"].json"
#MODEL=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-model[GROUND-TRUTH].json"
#MODEL=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-model[PARTICIPATION].json"
MODEL_ENCODING="json"

OUTPUT_CHECK_CSV=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output.csv"
OUTPUT_CHECK_JSON=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output.json"
OUTPUT_TRACE_MEASURES_CSV=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output[tracesMeasures].csv"
OUTPUT_TRACE_MEASURES_STATS_CSV=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output[tracesMeasuresStats].csv"
OUTPUT_LOG_MEASURES_CSV=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output[logMeasures].csv"

# CLustering
CLUSTERING_POLICY="specific-attribute"
# 'rules'
# 'attributes'
# 'specific-attribute'
# 'mixed'
CLUSTERING_ALGORITHM="optics"
#        'kmeans',  # 0
#        'affinity',  # 1
#        'meanshift',  # 2
#        'agglomerative',  # 3
#        'spectral',  # 4
#        'dbscan',  # 5
#        'optics',  # 6
#        'birch',  # 7
#        'gaussian',  # 8 DO NOT USE THIS!
BOOLEAN_RULES="True"
VISUALIZATION_FLAG="False"
APPLY_PCA_FLAG="True"

# DECLRE-Tree
CONSTRAINTS_THRESHOLD=0.8
PROCESSED_OUTPUT_CHECK_CSV=$PROCESSED_DATA_FOLDER"/"$LOG_NAME"-output.csv"
RESULT_TREE=$RESULTS_FOLDER"/"$LOG_NAME"-DeclareTree.dot"
BRANCHING_POLICY="dynamic"
MINIMIZATION_FLAG="True"
BRANCHING_ORDER_DECREASING_FLAG="True"

MULTI_PERSPECTIVE_FLAG="True"

##################################################################
# SCRIPT
##################################################################
#
# Discover process model (if not existing)
echo "${MODEL}"
echo "################################ DISCOVERY"
if test -f "${MODEL}"; then
  echo "$FILE already exists."
else
  java -cp Janus.jar $JANUS_DISCOVERY_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -c $CONFIDENCE -s $SUPPORT -i 0 -oJSON ${MODEL}
#  java -cp Janus.jar $JANUS_DISCOVERY_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -c $CONFIDENCE -s $SUPPORT -i 0 -keep -oJSON ${MODEL}
fi

# Simplify model, i.e., remove redundant constraints
echo "################################ SIMPLIFICATION"
#java -cp Janus.jar $SIMPLIFIER_MAINCLASS -iMF $MODEL -iME $MODEL_ENCODING -oJSON $MODEL -s 0 -c 0 -i 0 -prune hierarchyconflictredundancydouble

# Retrieve measure
echo "################################ MEASURE"
if test -f "${OUTPUT_TRACE_MEASURES_CSV}"; then
  echo "$OUTPUT_TRACE_MEASURES_CSV already exists."
else
  java -cp Janus.jar $JANUS_CHECK_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -iMF $MODEL -iME $MODEL_ENCODING -oCSV $OUTPUT_CHECK_CSV -d none -nanLogSkip -measure Confidence
#  java -cp Janus.jar $JANUS_CHECK_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -iMF $MODEL -iME $MODEL_ENCODING -oCSV $OUTPUT_CHECK_CSV -oJSON $OUTPUT_CHECK_JSON -d none -nanLogSkip
fi

# Launch clustering
echo "################################ CLUSTERING"
python3 -m ClusterMind.cm_clustering $CLUSTERING_POLICY $INPUT_LOG $CLUSTERING_ALGORITHM $PROCESSED_DATA_FOLDER"/" $VISUALIZATION_FLAG $APPLY_PCA_FLAG "$OUTPUT_TRACE_MEASURES_CSV" $BOOLEAN_RULES

# Retrieve measures for each cluster
echo "################################ CLUSTERS MEASURES and POSTPROCESSING"
for INPUT_LOG in $PROCESSED_DATA_FOLDER"/"*.xes; do
  echo $INPUT_LOG
  OUTPUT_CHECK_CSV="${INPUT_LOG}""-output.csv"
  OUTPUT_CHECK_JSON="${INPUT_LOG}""-output.json"
  java -cp Janus.jar $JANUS_CHECK_MAINCLASS -iLF "${INPUT_LOG}" -iLE $LOG_ENCODING -iMF "$MODEL" -iME $MODEL_ENCODING -oCSV "$OUTPUT_CHECK_CSV" -d none -detailsLevel log -measure Confidence
  #  java -cp Janus.jar $JANUS_CHECK_MAINCLASS -iLF "${INPUT_LOG}" -iLE $LOG_ENCODING -iMF "$MODEL" -iME $MODEL_ENCODING -oCSV "$OUTPUT_CHECK_CSV" -oJSON "$OUTPUT_CHECK_JSON" -d none -detailsLevel log -measure Confidence

  #  -nanLogSkip,--nan-log-skip                            Flag to skip or not NaN values when computing log measures
  #  -nanTraceSubstitute,--nan-trace-substitute            Flag to substitute or not the NaN values when computing trace measures
  #  -nanTraceValue,--nan-trace-value <number>

  #  keep only mean
  #  python3 singleAggregationPerspectiveFocusCSV.py "${OUTPUT_CHECK_JSON}AggregatedMeasures.json" "${INPUT_LOG}""-output[MEAN].csv"
done

# merge results
python3 -m ClusterMind.utils.aggregate_clusters_measures $PROCESSED_DATA_FOLDER"/" "-output[logMeasures].csv" "aggregated_result.csv"

cp $PROCESSED_DATA_FOLDER"/aggregated_result.csv" $RESULTS_FOLDER"/aggregated_result.csv"
cp ${PROCESSED_DATA_FOLDER}/*stats.csv $RESULTS_FOLDER"/clusters-stats.csv"
cp ${PROCESSED_DATA_FOLDER}/*labels.csv $RESULTS_FOLDER"/traces-labels.csv"
if test -f $PROCESSED_DATA_FOLDER"/aggregated_result.csv"; then
  cp $PROCESSED_DATA_FOLDER"/pca-features.csv" $RESULTS_FOLDER"/pca-features.csv"
fi

# Build decision-Tree
echo "################################ DECLARE TREES"
python3 -m DeclareTrees.declare_trees_for_clusters $PROCESSED_DATA_FOLDER"/aggregated_result.csv" $CONSTRAINTS_THRESHOLD $RESULT_TREE $BRANCHING_POLICY $MINIMIZATION_FLAG $BRANCHING_ORDER_DECREASING_FLAG

echo "################################ DECISION TREES"
python3 -m DeclareTrees.decision_trees_for_clusters \
  ${RESULTS_FOLDER}"/traces-labels.csv" \
  "$OUTPUT_TRACE_MEASURES_CSV" \
  ${RESULTS_FOLDER}"/decision_tree.dot" \
  1 \
  ${MULTI_PERSPECTIVE_FLAG}
