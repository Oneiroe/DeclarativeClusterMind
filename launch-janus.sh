#!/bin/bash

M="03"
T="SS"
#INPUT_LOG="test/log_m"$M"_t"$T".txt"
LOG_NAME="RTFMP"
INPUT_LOG="input/"$LOG_NAME"-log.xes"
LOG_ENCODING="xes"
MODEL="input/"$LOG_NAME".xes-model[s_0.05_c_0.8].json"
MODEL_ENCODING="json"
OUTPUT_CHECK_CSV="input/"$LOG_NAME"-output.csv"
OUTPUT_CHECK_JSON="input/"$LOG_NAME"-output.json"
SIMPLE_MODEL="input/"$LOG_NAME"-model[simple].json"

# Simplify model, i.e., remove redundant constraints
java -cp Janus.jar minerful.MinerFulSimplificationStarter -iMF $MODEL -iME $MODEL_ENCODING -oJSON $SIMPLE_MODEL -s 0 -c 0 -i 0 -prune hierarchyconflictredundancydouble

# Retrieve measure
java -cp Janus.jar minerful.JanusModelCheckStarter -iLF $INPUT_LOG -iLE $LOG_ENCODING -iMF $SIMPLE_MODEL -iME $MODEL_ENCODING -oCSV $OUTPUT_CHECK_CSV -d none
