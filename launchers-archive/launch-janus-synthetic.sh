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

# Log
LOG_NAME="SYNTHETIC"
INPUT_LOG="input/"$LOG_NAME"-log.xes"
LOG_ENCODING="xes"

## Log generation settings
MIN_STRLEN=10
MAX_STRLEN=100
TESTBED_SIZE=100
MEMORY_MAX="2048m"
LOG_ENCODING="xes"
GENERATOR_MODEL="input/"$LOG_NAME"-model-original.json"
TEMP_TEXT_FILE=input/${LOG_NAME}"-log-original.txt"


# discovery
SUPPORT=0
CONFIDENCE=0.5

# Model
MODEL="input/"$LOG_NAME"-model.json"
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

# Simulate process
echo "################################ Generate Log"
### GENERATE LOG with MinerFulLogMakerStarter ****
java -Xmx$MEMORY_MAX -cp Janus.jar $LOG_MAINCLASS \
    --input-model-file $GENERATOR_MODEL \
    --input-model-encoding $MODEL_ENCODING  \
    --size $TESTBED_SIZE \
    --minlen $MIN_STRLEN \
    --maxlen $MAX_STRLEN \
    --out-log-encoding $LOG_ENCODING \
    --out-log-file $INPUT_LOG \
    -d none

# Discover process model (if not existing)
echo "################################ DISCOVERY"
if test -f "${MODEL}"; then
  echo "$FILE already exists."
else
  java -cp Janus.jar $JANUS_DISCOVERY_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -c $CONFIDENCE -s $SUPPORT -i 0 -oJSON ${MODEL} -keep
fi

# Simplify model, i.e., remove redundant constraints
#echo "################################ SIMPLIFICATION"
#java -cp Janus.jar $SIMPLIFIER_MAINCLASS -iMF $MODEL -iME $MODEL_ENCODING -oJSON $MODEL -s 0 -c 0 -i 0 -prune hierarchyconflictredundancydouble

# Retrieve measure
echo "################################ MEASURE"
java -cp Janus.jar $JANUS_CHECK_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -iMF $MODEL -iME $MODEL_ENCODING -oCSV $OUTPUT_CHECK_CSV -d none

# Launch clustering
echo "################################ CLUSTERING"
python3 -m ClusterMind.cm_clustering $OUTPUT_CHECK_CSV $INPUT_LOG $CLUSTERING_ALGORITHM $BOOLEAN

# Retrieve measures for each cluster
echo "################################ CLUSTERS MEASURES and POSTPROCESSING"
for INPUT_LOG in "clustered-logs/"*.xes ; do
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
python3 -m ClusterMind.utils.merge_clusters "clustered-logs/" "-output[MEAN].csv" "aggregated_result.csv"

