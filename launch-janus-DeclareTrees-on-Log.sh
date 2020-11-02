#!/bin/bash

# Janus main classes
LOG_MAINCLASS="minerful.MinerFulLogMakerStarter"
SIMPLIFIER_MAINCLASS="minerful.MinerFulSimplificationStarter"
ERROR_MAINCLASS="minerful.MinerFulErrorInjectedLogMakerStarter"
JANUS_DISCOVERY_MAINCLASS="minerful.JanusOfflineMinerStarter"
JANUS_CHECK_MAINCLASS="minerful.JanusModelCheckStarter"

#INPUT_LOG="test/log_m"$M"_t"$T".txt"
LOG_NAME="SEPSIS"
INPUT_LOG="input/"$LOG_NAME"-log.xes"
LOG_ENCODING="xes"
MODEL="input/"$LOG_NAME".xes-model[s_0.05_c_0.8].json"
MODEL_ENCODING="json"
OUTPUT_CHECK_CSV="input/"$LOG_NAME"-output.csv"
OUTPUT_CHECK_JSON="input/"$LOG_NAME"-output.json"

BOOLEAN="True"

# Discover process model (if not existing)
echo "################################ DISCOVERY"
if test -f "${MODEL}"; then
  echo "$FILE already exists."
else
  java -cp Janus.jar $JANUS_DISCOVERY_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -c 0.8 -s 0.05 -i 0 -keep -oJSON ${MODEL}
fi

# Simplify model, i.e., remove redundant constraints
echo "################################ SIMPLIFICATION"
java -cp Janus.jar $SIMPLIFIER_MAINCLASS -iMF $MODEL -iME $MODEL_ENCODING -oJSON $MODEL -s 0 -c 0 -i 0 -prune hierarchyconflictredundancydouble

# Retrieve measure
echo "################################ MEASURE"
java -cp Janus.jar $JANUS_CHECK_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -iMF $MODEL -iME $MODEL_ENCODING -oCSV $OUTPUT_CHECK_CSV -oJSON $OUTPUT_CHECK_JSON -d none -nanLogSkip

#  -nanLogSkip,--nan-log-skip                            Flag to skip or not NaN values when computing log measures
#  -nanTraceSubstitute,--nan-trace-substitute            Flag to substitute or not the NaN values when computing trace measures
#  -nanTraceValue,--nan-trace-value <number>

echo "################################ DATA PROCESSING"
# post process result? like merge clusters
#python3 -m ClusterMind.utils.merge_clusters "clustered-logs/" "-output[MEAN].csv" "aggregated_result.csv"

echo "################################ DECLARE TREE"
# Build DECLARE-Tree
python3 -m DeclareTrees.trees_for_logs $OUTPUT_CHECK_CSV 0.8 "results-declare-trees/temp.csv" "results-declare-trees/DeclareTree.dot"