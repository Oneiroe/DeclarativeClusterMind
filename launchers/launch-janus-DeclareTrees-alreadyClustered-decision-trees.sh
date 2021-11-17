#!/bin/bash

# WHAT: Create a classic decision tree from the clustering output
# WHY: hopefully the decision trees models in a concise and meaningful way the discriminants between the clusters

##################################################################
# PARAMETERS
##################################################################
WORKING_DIR="/home/alessio/Data/Phd/my_code/ClusterMind"
cd $WORKING_DIR

# Janus main classes
LOG_MAINCLASS="minerful.MinerFulLogMakerStarter"
SIMPLIFIER_MAINCLASS="minerful.MinerFulSimplificationStarter"
ERROR_MAINCLASS="minerful.MinerFulErrorInjectedLogMakerStarter"
JANUS_DISCOVERY_MAINCLASS="minerful.JanusOfflineMinerStarter"
JANUS_CHECK_MAINCLASS="minerful.JanusMeasurementsStarter"

JAVA_BIN="/home/alessio/Software/jdk/jdk-11.0.10/bin/java"
DISCOVERY_JAR="/home/alessio/Data/Phd/code_3rd_party/MINERful/MINERful.jar"
DISCOVERY_MAINCLASS="minerful.MinerFulMinerStarter"
DISCOVERY_SUPPORT=0.9    # support threshold used for the initial discovery of the constraints of the variances
DISCOVERY_CONFIDENCE=0.0 # confidence threshold used for the initial discovery of the constraints of the variances

LOG_NAME="WSVX"
# "MANUAL"
# "SEPSIS_age"
# "BPIC15_f"
# "BPIC15_f_participation"
# "WSVX"
# "COVID"
#
# "BPIC12"
# "BPIC13"
# "SEPSIS"
# "RTFMP"
# "BPIC17_f"

SPLIT_POLICY="rules"
# 'rules'
# 'attributes'
# 'specific-attribute'
# 'performances'
# 'mixed'

# experiment folders
EXPERIMENT_NAME="experiments/REAL-LIFE-EXPLANATION/"${LOG_NAME}
PROCESSED_DATA_FOLDER=$EXPERIMENT_NAME"/1-clustered-logs"
MERGED_FOLDER=$EXPERIMENT_NAME"/2-merged-log"
PREPROCESSED_DATA_FOLDER=$MERGED_FOLDER
RESULTS_FOLDER=$EXPERIMENT_NAME"/3-results"
mkdir -p $EXPERIMENT_NAME $MERGED_FOLDER $PREPROCESSED_DATA_FOLDER $PROCESSED_DATA_FOLDER $RESULTS_FOLDER

# DECLRE-Tree
CONSTRAINTS_THRESHOLD=0.95
PROCESSED_OUTPUT_CHECK_CSV=$PROCESSED_DATA_FOLDER"/"$LOG_NAME"-output.csv"
BRANCHING_POLICY="dynamic-variance" # "static-frequency" "dynamic-frequency" "dynamic-variance"
RESULT_DECLARE_TREE_TRACES=$RESULTS_FOLDER"/"$LOG_NAME"-DeclareTree-TRACES-"${BRANCHING_POLICY}".dot"
RESULT_DECLARE_TREE_CLUSTERS=$RESULTS_FOLDER"/"$LOG_NAME"-DeclareTree-CLUSTERS-"${BRANCHING_POLICY}".dot"
#MINIMIZATION_FLAG="True"
MINIMIZATION_FLAG="-min"
#MINIMIZATION_FLAG=""
#BRANCHING_ORDER_DECREASING_FLAG="True"
BRANCHING_ORDER_DECREASING_FLAG="-decreasing"
#BRANCHING_ORDER_DECREASING_FLAG=""

#SPLIT_POLICY="mixed"
## 'rules'
## 'attributes'
## 'specific-attribute'
## 'mixed'
## 'performances'

# Input log
MERGED_LOG=$MERGED_FOLDER"/"$LOG_NAME"-merged-log.xes"
LOG_ENCODING="xes"

# Discovery & Measurements
SUPPORT=0.0
CONFIDENCE=0.9
MODEL=$MERGED_FOLDER"/"$LOG_NAME".xes-model[s_"$SUPPORT"_c_"$CONFIDENCE"].json"
#MODEL=$MERGED_FOLDER"/"$LOG_NAME".xes-model[s_"$SUPPORT"_c_"$CONFIDENCE"]-SIMPLIFIED.json"
#MODEL=$MERGED_FOLDER"/"$LOG_NAME"-model[GROUND-TRUTH].json"
#MODEL=$MERGED_FOLDER"/"$LOG_NAME"-model[PARTICIPATION].json"
#MODEL=$MERGED_FOLDER"/"$LOG_NAME"-model[ABSENCE].json"
#MODEL=$MERGED_FOLDER"/"$LOG_NAME"-model[ALL].json"
MODEL_ENCODING="json"

OUTPUT_CHECK_CSV=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output.csv"
OUTPUT_CHECK_JSON=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output.json"
OUTPUT_TRACE_MEASURES_CSV=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output[tracesMeasures].csv"
OUTPUT_TRACE_MEASURES_STATS_CSV=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output[tracesMeasuresStats].csv"
OUTPUT_LOG_MEASURES_CSV=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output[logMeasures].csv"

CONSTRAINTS_TEMPLATE_BLACKLIST=${PROCESSED_DATA_FOLDER}"/blacklist.csv"

##################################################################
# SCRIPT
##################################################################
#
# Discover process model (if not existing)
# IDEA: discover the process models out of each cluster, then merge them. afterward you compute the measures given the unique model
echo "################################ CLUSTERS MODEL DISCOVERY"
for INPUT_LOG in $PROCESSED_DATA_FOLDER"/"*.xes; do
  echo $INPUT_LOG
  CURRENT_MODEL=${INPUT_LOG}"_model.json"
  if test -f "${CURRENT_MODEL}"; then
    echo "$FILE already exists."
  else
    java -cp Janus.jar $JANUS_DISCOVERY_MAINCLASS -iLF "${INPUT_LOG}" -iLE $LOG_ENCODING -c $CONFIDENCE -s $SUPPORT -i 0 -oJSON "${CURRENT_MODEL}"
#    $JAVA_BIN -cp $DISCOVERY_JAR $DISCOVERY_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -c $DISCOVERY_CONFIDENCE -s $DISCOVERY_SUPPORT -oJSON ${CURRENT_MODEL} -vShush

    # Filter undesired templates, e.g., NotSuccession or NotChainSuccession
    if test -f "${CONSTRAINTS_TEMPLATE_BLACKLIST}"; then
      python3 -m DeclarativeClusterMind.utils.filter_json_model ${CURRENT_MODEL} ${CONSTRAINTS_TEMPLATE_BLACKLIST} ${CURRENT_MODEL}
    fi
    #    # Simplify model, i.e., remove redundant constraints
    #    echo "################################ SIMPLIFICATION"
    #    java -cp Janus.jar $SIMPLIFIER_MAINCLASS -iMF $CURRENT_MODEL -iME $MODEL_ENCODING -oJSON $CURRENT_MODEL -s 0 -c 0 -i 0 -prune hierarchyconflictredundancydouble
  fi
done

## merge process models
python3 -m DeclarativeClusterMind.utils.merge_models $PROCESSED_DATA_FOLDER "_model.json" ${MODEL}

# Retrieve measures for each cluster
echo "################################ CLUSTERS MEASURES and POSTPROCESSING"
for INPUT_LOG in $PROCESSED_DATA_FOLDER"/"*.xes; do
  echo $INPUT_LOG
  CURRENT_OUTPUT_CHECK_CSV="${INPUT_LOG}""-output.csv"
  OUTPUT_CHECK_JSON="${INPUT_LOG}""-output.json"
  TEMP_OUT_MESURES_FILE="${INPUT_LOG}""-output[logMeasures].csv"
  echo $TEMP_OUT_MESURES_FILE
  if test -f $TEMP_OUT_MESURES_FILE; then
    echo "${TEMP_OUT_MESURES_FILE} already exists."
  else
    java -cp Janus.jar $JANUS_CHECK_MAINCLASS -iLF "${INPUT_LOG}" -iLE $LOG_ENCODING -iMF "$MODEL" -iME $MODEL_ENCODING -oCSV "$CURRENT_OUTPUT_CHECK_CSV" -d none -detailsLevel log -measure Confidence
  fi
  #  java -cp Janus.jar $JANUS_CHECK_MAINCLASS -iLF "${INPUT_LOG}" -iLE $LOG_ENCODING -iMF "$MODEL" -iME $MODEL_ENCODING -oCSV "$CURRENT_OUTPUT_CHECK_CSV" -oJSON "$OUTPUT_CHECK_JSON" -d none -detailsLevel log -measure Confidence

  #  -nanLogSkip,--nan-log-skip                            Flag to skip or not NaN values when computing log measures
  #  -nanTraceSubstitute,--nan-trace-substitute            Flag to substitute or not the NaN values when computing trace measures
  #  -nanTraceValue,--nan-trace-value <number>

  #  keep only mean
  #  python3 pySupport/singleAggregationPerspectiveFocusCSV_confidence-only.py "${OUTPUT_CHECK_JSON}AggregatedMeasures.json" "${INPUT_LOG}""-output[MEAN].csv"
done

# Retrieve measure for trace decision tree
if test -f "${MERGED_LOG}"; then
  echo "$MERGED_LOG already exists."
else
  python3 -m DeclarativeClusterMind.utils.merge_logs $MERGED_LOG $PROCESSED_DATA_FOLDER"/"*.xes
fi
echo "################################ MEASURE"
if test -f "${OUTPUT_TRACE_MEASURES_CSV}"; then
  echo "$OUTPUT_TRACE_MEASURES_CSV already exists."
else
  # -Xmx12000m in case of extra memory required add this
  java -cp Janus.jar $JANUS_CHECK_MAINCLASS -iLF $MERGED_LOG -iLE $LOG_ENCODING -iMF $MODEL -iME $MODEL_ENCODING -oCSV $OUTPUT_CHECK_CSV -d none -nanLogSkip -measure "Confidence" -detailsLevel trace
#  java -cp Janus.jar $JANUS_CHECK_MAINCLASS -iLF $MERGED_LOG -iLE $LOG_ENCODING -iMF $MODEL -iME $MODEL_ENCODING -oCSV $OUTPUT_CHECK_CSV -oJSON $OUTPUT_CHECK_JSON -d none -nanLogSkip
# 'Lift','Confidence','Relative Risk'
# 'Support','all','Compliance,'Added Value','J Measure'
# 'Recall','Lovinger','Specificity','Accuracy','Leverage','Odds Ratio', 'Gini Index','Certainty factor','Coverage','Prevalence',
# 'Jaccard','Ylue Q','Ylue Y','Klosgen','Conviction','Interestingness Weighting Dependency','Collective Strength','Laplace Correction',
# 'One-way Support','Two-way Support','Two-way Support Variation',
# 'Linear Correlation Coefficient','Piatetsky-Shapiro','Cosine','Information Gain','Sebag-Schoenauer','Least Contradiction','Odd Multiplier','Example and Counterexample Rate','Zhang'}.
fi

# merge results
python3 -m DeclarativeClusterMind.utils.aggregate_clusters_measures $PROCESSED_DATA_FOLDER "-output[logMeasures].csv" "aggregated_result.csv"
python3 -m DeclarativeClusterMind.utils.label_clusters_with_measures $PROCESSED_DATA_FOLDER "-output[logMeasures].csv" "clusters-labels.csv"
python3 -m DeclarativeClusterMind.evaluation.label_traces_from_clustered_logs $PROCESSED_DATA_FOLDER

cp ${PROCESSED_DATA_FOLDER}/*traces-labels.csv $RESULTS_FOLDER"/traces-labels.csv"
cp $PROCESSED_DATA_FOLDER"/aggregated_result.csv" $RESULTS_FOLDER"/aggregated_result.csv"
cp ${PROCESSED_DATA_FOLDER}/*stats.csv $RESULTS_FOLDER"/clusters-stats.csv"
cp ${PROCESSED_DATA_FOLDER}"/clusters-labels.csv" $RESULTS_FOLDER"/clusters-labels.csv"
cp ${PROCESSED_DATA_FOLDER}/performances_boxplot* $RESULTS_FOLDER
cp ${PROCESSED_DATA_FOLDER}/silhouette* $RESULTS_FOLDER
if test -f $PROCESSED_DATA_FOLDER"/pca-features.csv"; then
  cp $PROCESSED_DATA_FOLDER"/pca-features.csv" $RESULTS_FOLDER"/pca-features.csv"
fi

# Build decision-Tree
echo "################################ SIMPLE TREES Clusters"
python3 -m DeclarativeClusterMind.ui_declare_trees --ignore-gooey simple-tree-logs-to-clusters\
  -i $PROCESSED_DATA_FOLDER"/aggregated_result.csv" \
  -o $RESULT_DECLARE_TREE_CLUSTERS"-Decreasing.dot" \
  -t $CONSTRAINTS_THRESHOLD \
  -p $BRANCHING_POLICY \
  $MINIMIZATION_FLAG \
  -decreasing

python3 -m DeclarativeClusterMind.ui_declare_trees --ignore-gooey simple-tree-logs-to-clusters\
  -i $PROCESSED_DATA_FOLDER"/aggregated_result.csv" \
  -o $RESULT_DECLARE_TREE_CLUSTERS"-Increasing.dot" \
  -t $CONSTRAINTS_THRESHOLD \
  -p $BRANCHING_POLICY \
  $MINIMIZATION_FLAG

echo "################################ DECISION TREES clusters"
python3 -m DeclarativeClusterMind.ui_declare_trees --ignore-gooey decision-tree-logs-to-clusters \
  -i ${RESULTS_FOLDER}"/clusters-labels.csv" \
  -o ${RESULTS_FOLDER}"/decision_tree_clusters.dot" \
  -fi 0

echo "################################ DECISION TREES traces"
python3 -m DeclarativeClusterMind.ui_declare_trees --ignore-gooey decision-tree-traces-to-clusters \
  -i ${RESULTS_FOLDER}"/traces-labels.csv" \
  -o ${RESULTS_FOLDER}"/decision_tree_traces.dot" \
  -fi 1 \
  -m "$OUTPUT_TRACE_MEASURES_CSV" \
  -p ${SPLIT_POLICY}

echo "################################ SIMPLE TREES Traces"
python3 -m DeclarativeClusterMind.ui_declare_trees --ignore-gooey simple-tree-traces \
  -i ${OUTPUT_TRACE_MEASURES_CSV} \
  -o $RESULT_DECLARE_TREE_TRACES \
  -t $CONSTRAINTS_THRESHOLD \
  -p $BRANCHING_POLICY \
  $MINIMIZATION_FLAG \
  $BRANCHING_ORDER_DECREASING_FLAG \
  -mls 100