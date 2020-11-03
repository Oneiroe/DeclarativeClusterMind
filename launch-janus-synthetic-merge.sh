#!/bin/bash

##################################################################
# PARAMETERS
##################################################################
# Janus main classes
LOG_MAINCLASS="minerful.MinerFulLogMakerStarter"
SIMPLIFIER_MAINCLASS="minerful.MinerFulSimplificationStarter"
ERROR_MAINCLASS="minerful.MinerFulErrorInjectedLogMakerStarter"
JANUS_DISCOVERY_MAINCLASS="minerful.JanusOfflineMinerStarter"
JANUS_CHECK_MAINCLASS="minerful.JanusModelCheckStarter"

# Logs
LOG_NAME="SYNTHETIC"
INPUT_LOG="input/"$LOG_NAME
INPUT_LOG_MERGED="input/"$LOG_NAME"-merged-log.xes"
LOG_ENCODING="xes"

## Log generation settings
MIN_STRLEN=10
MAX_STRLEN=30
TESTBED_SIZE=50
MEMORY_MAX="2048m"
LOG_ENCODING="xes"
GENERATOR_MODEL_1="input/"$LOG_NAME"-1-model-original.json"
GENERATOR_MODEL_2="input/"$LOG_NAME"-2-model-original.json"


# discovery
SUPPORT=0.05
CONFIDENCE=0.9

# Model
MODEL="input/"$LOG_NAME"-0-model.json"
MODEL_ENCODING="json"
OUTPUT_CHECK_CSV="input/"$LOG_NAME"-output.csv"
OUTPUT_CHECK_JSON="input/"$LOG_NAME"-output.json"

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
BOOLEAN="True"

##################################################################
# SCRIPT
##################################################################

# Simulate processes
echo "################################ Generate Logs"
### GENERATE LOG with MinerFulLogMakerStarter ****
for LOG_NUM in {1..2..1}; do
#  Generate a model
  GENERATOR_MODEL="input/"$LOG_NAME"-"${LOG_NUM}"-model-original.json"
  echo "generating log from...."${GENERATOR_MODEL}
#  TODO generate a random model from an alphabet and a set of constraints

#  Simulate model
  java -Xmx$MEMORY_MAX -cp Janus.jar $LOG_MAINCLASS \
      --input-model-file $GENERATOR_MODEL \
      --input-model-encoding $MODEL_ENCODING  \
      --size $TESTBED_SIZE \
      --minlen $MIN_STRLEN \
      --maxlen $MAX_STRLEN \
      --out-log-encoding $LOG_ENCODING \
      --out-log-file ${INPUT_LOG}"-"${LOG_NUM}"-log.xes" \
      -d none
done
# Merge logs
echo "################################ MERGE"
python -m ClusterMind.utils.merge_logs "input/" $LOG_NAME $INPUT_LOG_MERGED

# Discover process models (if not existing)
echo "################################ DISCOVERY"
if test -f "${MODEL}"; then
  echo "$FILE already exists."
else
  java -cp Janus.jar $JANUS_DISCOVERY_MAINCLASS -iLF $INPUT_LOG_MERGED -iLE $LOG_ENCODING -c $CONFIDENCE -s $SUPPORT -i 0 -oJSON ${MODEL} -keep
fi

# Simplify model, i.e., remove redundant constraints
#echo "################################ SIMPLIFICATION"
#java -cp Janus.jar $SIMPLIFIER_MAINCLASS -iMF $MODEL -iME $MODEL_ENCODING -oJSON $MODEL -s 0 -c 0 -i 0 -prune hierarchyconflictredundancydouble

# Retrieve measures
echo "################################ MEASURE"
java -cp Janus.jar $JANUS_CHECK_MAINCLASS -iLF $INPUT_LOG_MERGED -iLE $LOG_ENCODING -iMF $MODEL -iME $MODEL_ENCODING -oCSV $OUTPUT_CHECK_CSV -d none

# Launch clustering
echo "################################ CLUSTERING"
python3 -m ClusterMind.cm_clustering $OUTPUT_CHECK_CSV $INPUT_LOG_MERGED $CLUSTERING_ALGORITHM $BOOLEAN

# Retrieve measures for each cluster
echo "################################ CLUSTERS MEASURES and POSTPROCESSING"
for INPUT_LOG_MERGED in "clustered-logs/"*.xes ; do
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
python3 -m ClusterMind.utils.merge_clusters "clustered-logs/" "-output[MEAN].csv" "aggregated_result.csv"

# Build DECLARE-Tree
python3 -m DeclareTrees.trees_for_clusters "clustered-logs/aggregated_result.csv" 0.8 "clustered-logs/DeclareTree.dot"
