#!/bin/bash

# Create a dynamic declare tree from the clusters derived from a log

##################################################################
# PARAMETERS
##################################################################
# experiment folders
EXPERIMENT_NAME="experiments/DECLARE-TREE-DYNAMIC"
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
JANUS_CHECK_MAINCLASS="minerful.JanusModelCheckStarter"

LOG_NAME="SEPSIS"
INPUT_LOG=$INPUT_FOLDER"/"$LOG_NAME"-log.xes"
LOG_ENCODING="xes"
MODEL=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME".xes-model[s_0.05_c_0.8].json"
#MODEL=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME".xes-model[s_0.00_c_0.5].json"
MODEL_ENCODING="json"
OUTPUT_CHECK_CSV=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output.csv"
OUTPUT_CHECK_JSON=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output.json"

#        'kmeans',  # 0
#        'affinity',  # 1
#        'meanshift',  # 2
#        'agglomerative',  # 3
#        'spectral',  # 4
#        'dbscan',  # 5
#        'optics',  # 6
#        'birch',  # 7
#        'gaussian',  # 8 DO NOT USE THIS!
CLUSTERING_ALGORITHM="dbscan"
BOOLEAN="True"
VISUALIZATION_FLAG="False"

# Discover process model (if not existing)
echo "################################ DISCOVERY"
if test -f "${MODEL}"; then
  echo "$FILE already exists."
else
  java -cp Janus.jar $JANUS_DISCOVERY_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -c 0.8 -s 0.05 -i 0 -keep -oJSON ${MODEL}
fi

# Simplify model, i.e., remove redundant constraints
echo "################################ SIMPLIFICATION"
#java -cp Janus.jar $SIMPLIFIER_MAINCLASS -iMF $MODEL -iME $MODEL_ENCODING -oJSON $MODEL -s 0 -c 0 -i 0 -prune hierarchyconflictredundancydouble

# Retrieve measure
echo "################################ MEASURE"
java -cp Janus.jar $JANUS_CHECK_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -iMF $MODEL -iME $MODEL_ENCODING -oCSV $OUTPUT_CHECK_CSV -d none -nanLogSkip

# Launch clustering
echo "################################ CLUSTERING"
python3 -m ClusterMind.cm_clustering $OUTPUT_CHECK_CSV $INPUT_LOG $CLUSTERING_ALGORITHM $BOOLEAN $PROCESSED_DATA_FOLDER"/" $VISUALIZATION_FLAG

# Retrieve measures for each cluster
echo "################################ CLUSTERS MEASURES and POSTPROCESSING"
for INPUT_LOG in $PROCESSED_DATA_FOLDER"/"*.xes; do
  echo $INPUT_LOG
  OUTPUT_CHECK_CSV="${INPUT_LOG}""-output.csv"
  OUTPUT_CHECK_JSON="${INPUT_LOG}""-output.json"
  java -cp Janus.jar minerful.JanusModelCheckStarter -iLF "${INPUT_LOG}" -iLE $LOG_ENCODING -iMF "$MODEL" -iME $MODEL_ENCODING -oCSV "$OUTPUT_CHECK_CSV" -oJSON "$OUTPUT_CHECK_JSON" -d none -nanLogSkip

#  -nanLogSkip,--nan-log-skip                            Flag to skip or not NaN values when computing log measures
#  -nanTraceSubstitute,--nan-trace-substitute            Flag to substitute or not the NaN values when computing trace measures
#  -nanTraceValue,--nan-trace-value <number>

  #  keep only mean
  python3 singleAggregationPerspectiveFocusCSV.py "${OUTPUT_CHECK_JSON}AggregatedMeasures.json" "${INPUT_LOG}""-output[MEAN].csv"
done

# merge results
python3 -m ClusterMind.utils.merge_clusters $PROCESSED_DATA_FOLDER"/" "-output[MEAN].csv" "aggregated_result.csv"

# Build DECLARE-Tree
echo "################################ DECLARE TREES"
python3 -m DeclareTrees.trees_for_clusters $PROCESSED_DATA_FOLDER"/aggregated_result.csv" 0.8 $RESULTS_FOLDER"/DeclareTree.dot" "dynamic"
cp $PROCESSED_DATA_FOLDER"/aggregated_result.csv" $RESULTS_FOLDER"/aggregated_result.csv"
cp ${PROCESSED_DATA_FOLDER}/*stats.csv $RESULTS_FOLDER"/clusters-stats.csv"
