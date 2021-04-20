#!/bin/bash

# create synthetic logs based on mutually exclusive declarative models and check if they are correctly split based on the expected rules
# Model 1: succession(a,b)
# Model 2: notSuccession(a,b)

##################################################################
# PARAMETERS
##################################################################
# experiment folders
EXPERIMENT_NAME="experiments/SYNTHETIC-OPPOSITE"
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

# Logs
LOG_NAME="SYNTHETIC-OPPOSITE"
INPUT_LOG=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME
INPUT_LOG_MERGED=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-merged-log.xes"
LOG_ENCODING="xes"

## Log generation settings
MIN_STRLEN=10
MAX_STRLEN=30
TESTBED_SIZE=100
MEMORY_MAX="2048m"
LOG_ENCODING="xes"
GENERATOR_MODEL_1=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-1-model-original.json"
GENERATOR_MODEL_2=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-2-model-original.json"

# discovery
SUPPORT=0.05
CONFIDENCE=0.5

# Model
HAND_MODEL=$INPUT_FOLDER"/"$LOG_NAME"-hand-model-original.json"
MODEL=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-merged-model.json"
MODEL_ENCODING="json"
OUTPUT_CHECK_CSV=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output.csv"
OUTPUT_CHECK_JSON=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output.json"

# Clustering
#        'kmeans',  # 0
#        'affinity',  # 1
#        'meanshift',  # 2
#        'agglomerative',  # 3
#        'spectral',  # 4
#        'dbscan',  # 5
#        'optics',  # 6
#        'birch',  # 7
#        'gaussian',  # 8 DO NOT USE THIS!
CLUSTERING_ALGORITHM="optics"
BOOLEAN="Tree"
VISUALIZATION_FLAG="False"

##################################################################
# SCRIPT
##################################################################

# Simulate processes
echo "################################ Generate Logs"
### GENERATE LOG with MinerFulLogMakerStarter ****
for LOG_NUM in {1..2..1}; do
  #  Generate a model
  GENERATOR_MODEL=$INPUT_FOLDER"/"$LOG_NAME"-"${LOG_NUM}"-model-original.json"
  echo "generating log from...."${GENERATOR_MODEL}
  #  TODO generate a random model from an alphabet and a set of constraints

  #  Simulate model
  java -Xmx$MEMORY_MAX -cp Janus.jar $LOG_MAINCLASS \
    --input-model-file $GENERATOR_MODEL \
    --input-model-encoding $MODEL_ENCODING \
    --size $TESTBED_SIZE \
    --minlen $MIN_STRLEN \
    --maxlen $MAX_STRLEN \
    --out-log-encoding $LOG_ENCODING \
    --out-log-file ${INPUT_LOG}"-"${LOG_NUM}"-log.xes" \
    -d none
done
# Merge logs
echo "################################ MERGE"
python -m ClusterMind.utils.merge_logs $PREPROCESSED_DATA_FOLDER"/" $LOG_NAME $INPUT_LOG_MERGED

# Discover process models (if not existing)
#echo "################################ DISCOVERY"
#if test -f "${MODEL}"; then
#  echo "$FILE already exists."
#else
#  java -cp Janus.jar $JANUS_DISCOVERY_MAINCLASS -iLF $INPUT_LOG_MERGED -iLE $LOG_ENCODING -c $CONFIDENCE -s $SUPPORT -i 0 -oJSON ${MODEL} -keep -prune none
##  java -cp Janus.jar $JANUS_DISCOVERY_MAINCLASS -iLF $INPUT_LOG_MERGED -iLE $LOG_ENCODING -c $CONFIDENCE -s $SUPPORT -i 0 -oJSON ${MODEL}
#fi

# Overwrite discovered model with a prefixed one hand written by you
MODEL=$HAND_MODEL

# Simplify model, i.e., remove redundant constraints
#echo "################################ SIMPLIFICATION"
#java -cp Janus.jar $SIMPLIFIER_MAINCLASS -iMF $MODEL -iME $MODEL_ENCODING -oJSON $MODEL -s 0 -c 0 -i 0 -prune hierarchyconflictredundancydouble

# Retrieve measures
echo "################################ MEASURE"
java -cp Janus.jar $JANUS_CHECK_MAINCLASS -iLF $INPUT_LOG_MERGED -iLE $LOG_ENCODING -iMF $MODEL -iME $MODEL_ENCODING -oCSV $OUTPUT_CHECK_CSV -d none -nanLogSkip

# Launch clustering
echo "################################ CLUSTERING"
python3 -m ClusterMind.cm_clustering $OUTPUT_CHECK_CSV $INPUT_LOG_MERGED $CLUSTERING_ALGORITHM $BOOLEAN $PROCESSED_DATA_FOLDER"/" $VISUALIZATION_FLAG

# Retrieve measures for each cluster
echo "################################ CLUSTERS MEASURES and POSTPROCESSING"
for INPUT_LOG_MERGED in $PROCESSED_DATA_FOLDER"/"*.xes; do
  echo $INPUT_LOG_MERGED
  OUTPUT_CHECK_CSV="${INPUT_LOG_MERGED}""-output.csv"
  OUTPUT_CHECK_JSON="${INPUT_LOG_MERGED}""-output.json"
  java -cp Janus.jar minerful.JanusModelCheckStarter -iLF "${INPUT_LOG_MERGED}" -iLE $LOG_ENCODING -iMF "$MODEL" -iME $MODEL_ENCODING -oCSV "$OUTPUT_CHECK_CSV" -oJSON "$OUTPUT_CHECK_JSON" -d none -nanLogSkip

  #  -nanLogSkip,--nan-log-skip                            Flag to skip or not NaN values when computing log measures
  #  -nanTraceSubstitute,--nan-trace-substitute            Flag to substitute or not the NaN values when computing trace measures
  #  -nanTraceValue,--nan-trace-value <number>

  #  keep only mean
  python3 singleAggregationPerspectiveFocusCSV.py "${OUTPUT_CHECK_JSON}AggregatedMeasures.json" "${INPUT_LOG_MERGED}""-output[MEAN].csv"
done

# merge results
python3 -m ClusterMind.utils.merge_clusters $PROCESSED_DATA_FOLDER"/" "-output[MEAN].csv" "aggregated_result.csv"

# Build DECLARE-Tree
python3 -m DeclareTrees.trees_for_clusters $PROCESSED_DATA_FOLDER"/aggregated_result.csv" 0.8 $RESULTS_FOLDER"/DeclareTree.dot"
cp $PROCESSED_DATA_FOLDER"/aggregated_result.csv" $RESULTS_FOLDER"/aggregated_result.csv"
cp ${PROCESSED_DATA_FOLDER}/*stats.csv $RESULTS_FOLDER"/clusters-stats.csv"
