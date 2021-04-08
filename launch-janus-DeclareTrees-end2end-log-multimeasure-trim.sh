#!/bin/bash

# Create a dynamic declare tree from the clusters derived from a log

##################################################################
# PARAMETERS
##################################################################
# experiment folders
EXPERIMENT_NAME="experiments/STATISTICAL-TRIM"
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

# DECLRE-Tree
CONSTRAINTS_THRESHOLD=0.8
PROCESSED_OUTPUT_CHECK_CSV=$PROCESSED_DATA_FOLDER"/"$LOG_NAME"-output.csv"
RESULT_TREE=$RESULTS_FOLDER"/"$LOG_NAME"-DeclareTree.dot"
BRANCHING_POLICY="dynamic"
MINIMIZATION_FLAG="True"
BRANCHING_ORDER_FLAG="True"

# Sound DECLARE
FOCUSSED_CSV=${OUTPUT_CHECK_CSV}AggregatedMeasures-FOCUS.csv
SOUND_MODEL=$RESULTS_FOLDER"/"$LOG_NAME".xes-model[StaTrim]"
SOUND_MODEL_CSV=$SOUND_MODEL".csv"
SOUND_MODEL_JSON=$SOUND_MODEL".json"

# Discover process model (if not existing)
echo "################################ DISCOVERY"
if test -f "${MODEL}"; then
  echo "$FILE already exists."
else
  #  java -cp Janus.jar $JANUS_DISCOVERY_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -c 0.8 -s 0.05 -i 0 -oJSON ${MODEL} -oCSV ${MODEL}.csv
  java -cp Janus.jar $JANUS_DISCOVERY_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -c 0.8 -s 0.001 -i 0 -keep -oJSON ${MODEL} -oCSV ${MODEL}.csv

  # Retrieve measure
  echo "################################ MEASURES"
  java -cp Janus.jar $JANUS_CHECK_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -iMF $MODEL -iME $MODEL_ENCODING -oJSON $OUTPUT_CHECK_JSON -oCSV $OUTPUT_CHECK_CSV -d none -nanLogSkip

  # Trim constraints
  echo "################################ STATISTICAL TRIM"
  python3 -m SoundDeclare.sound_declare ${OUTPUT_CHECK_JSON}AggregatedMeasures.json $FOCUSSED_CSV $SOUND_MODEL

  # Simplify model, i.e., remove redundant constraints
  echo "################################ SIMPLIFICATION"
  #java -cp Janus.jar $SIMPLIFIER_MAINCLASS -iMF $SOUND_MODEL_JSON -iME $MODEL_ENCODING -oCSV $SOUND_MODEL_CSV -oJSON $SOUND_MODEL_JSON -s 0 -c 0 -i 0 -prune hierarchyconflictredundancydouble
  java -cp Janus.jar $SIMPLIFIER_MAINCLASS -iMF $SOUND_MODEL_JSON -iME $MODEL_ENCODING -oJSON $SOUND_MODEL_JSON -s 0 -c 0 -i 0 -prune hierarchyconflictredundancydouble
  java -cp Janus.jar $JANUS_CHECK_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -iMF $SOUND_MODEL_JSON -iME $MODEL_ENCODING -oJSON $OUTPUT_CHECK_JSON -oCSV $OUTPUT_CHECK_CSV -d none -nanLogSkip
fi

# Build DECLARE-Tree
echo "################################ DECLARE TREES"
python3 -m DeclareTrees.trees_for_logs $OUTPUT_CHECK_CSV $CONSTRAINTS_THRESHOLD $PROCESSED_OUTPUT_CHECK_CSV $RESULT_TREE $BRANCHING_POLICY $MINIMIZATION_FLAG $BRANCHING_ORDER_FLAG

#cp $PROCESSED_DATA_FOLDER"/aggregated_result.csv" $RESULTS_FOLDER"/aggregated_result.csv"
#cp ${PROCESSED_DATA_FOLDER}/*stats.csv $RESULTS_FOLDER"/clusters-stats.csv"

# Clustering from DECLARE-Tree
echo "################################ CLUSTERS FROM LOG TREE"
echo "build clusters TO-DO"
echo "add stats to declare tree TO-DO"

# Retrieve measures for each cluster
#echo "################################ CLUSTERS MEASURES and POSTPROCESSING"
#for INPUT_LOG in $PROCESSED_DATA_FOLDER"/"*.xes; do
#  echo $INPUT_LOG
#  OUTPUT_CHECK_CSV="${INPUT_LOG}""-output.csv"
#  OUTPUT_CHECK_JSON="${INPUT_LOG}""-output.json"
#  java -cp Janus.jar minerful.JanusModelCheckStarter -iLF "${INPUT_LOG}" -iLE $LOG_ENCODING -iMF "$MODEL" -iME $MODEL_ENCODING -oCSV "$OUTPUT_CHECK_CSV" -oJSON "$OUTPUT_CHECK_JSON" -d none -nanLogSkip

#  -nanLogSkip,--nan-log-skip                            Flag to skip or not NaN values when computing log measures
#  -nanTraceSubstitute,--nan-trace-substitute            Flag to substitute or not the NaN values when computing trace measures
#  -nanTraceValue,--nan-trace-value <number>

#  keep only mean
#  python3 singleAggregationPerspectiveFocusCSV.py "${OUTPUT_CHECK_JSON}AggregatedMeasures.json" "${INPUT_LOG}""-output[MEAN].csv"
#done

# merge results
#python3 -m ClusterMind.utils.aggregate_clusters_measures $PROCESSED_DATA_FOLDER"/" "-output[MEAN].csv" "aggregated_result.csv"
