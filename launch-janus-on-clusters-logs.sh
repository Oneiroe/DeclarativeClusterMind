#!/bin/bash

M="03"
T="SS"
#INPUT_LOG="test/log_m"$M"_t"$T".txt"
LOG_BASE_NAME="SEPSIS"
LOG_ENCODING="xes"
MODEL="input/"$LOG_BASE_NAME".xes-model[s_0.05_c_0.8].json"
MODEL_ENCODING="json"
SIMPLE_MODEL="input/"$LOG_BASE_NAME"-model[simple].json"

# Retrieve measure
for INPUT_LOG in "clustered-logs/"*.xes ; do
  echo $INPUT_LOG
  OUTPUT_CHECK_CSV="${INPUT_LOG}""-output.csv"
  OUTPUT_CHECK_JSON="${INPUT_LOG}""-output.json"
  java -cp Janus.jar minerful.JanusModelCheckStarter -iLF "${INPUT_LOG}" -iLE $LOG_ENCODING -iMF "$SIMPLE_MODEL" -iME $MODEL_ENCODING -oCSV "$OUTPUT_CHECK_CSV" -oJSON "$OUTPUT_CHECK_JSON" -d none -nanLogSkip

#  -nanLogSkip,--nan-log-skip                            Flag to skip or not NaN values when computing log measures
#  -nanTraceSubstitute,--nan-trace-substitute            Flag to substitute or not the NaN values when computing trace measures
#  -nanTraceValue,--nan-trace-value <number>

  #  keep only mean
  python3 singleAggregationPerspectiveFocusCSV.py "${OUTPUT_CHECK_JSON}AggregatedMeasures.json" "${INPUT_LOG}""-output[MEAN].csv"
done

# merge results
python3 utils.py "clustered-logs/" "-output[MEAN].csv"

