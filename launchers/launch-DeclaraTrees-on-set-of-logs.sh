#!/bin/bash

# WHAT: Create a classic decision tree from the clustering output
# WHY: hopefully the decision trees models in a concise and meaningful way the discriminants between the clusters

##################################################################
# PARAMETERS
##################################################################
INPUT_LOGS_FOLDER="PATH_TO_INPUT_LOGS_FOLDER"
RESULTS_FOLDER="PATH_TO_RESULTS_FOLDER"

CONSTRAINTS_TEMPLATE_BLACKLIST=${INPUT_LOGS_FOLDER}"/blacklist.csv"
CONSTRAINTS_TASKS_BLACKLIST=${INPUT_LOGS_FOLDER}"/blacklist-tasks.csv"

##################################################################

WORKING_DIR="/home/alessio/Data/Phd/my_code/DeclarativeClusterMind" # path containing DeclarativeClusterMind package
cd $WORKING_DIR || exit

# Janus/MINERful main classes
JAVA_BIN="/home/alessio/Software/jdk/jdk-11.0.10/bin/java"
JANUS_JAR="/home/alessio/Data/Phd/my_code/DeclarativeClusterMind/Janus.jar"
MINERFUL_JAR="/home/alessio/Data/Phd/code_3rd_party/MINERful/MINERful.jar"
JANUS_MEASURES_MAINCLASS="minerful.JanusMeasurementsStarter"
JANUS_DISCOVERY_MAINCLASS="minerful.JanusOfflineMinerStarter"
JANUS_DISCOVERY_SUPPORT=0.0
JANUS_DISCOVERY_CONFIDENCE=0.9
MINERFUL_DISCOVERY_MAINCLASS="minerful.MinerFulMinerStarter"
MINERFUL_DISCOVERY_SUPPORT=0.9    # support threshold for the initial discovery of the constraints of the variances
MINERFUL_DISCOVERY_CONFIDENCE=0.0 # confidence threshold for the initial discovery of the constraints of the variances
SIMPLIFIER_MAINCLASS="minerful.MinerFulSimplificationStarter"

# Discovery & Measurements
LOG_ENCODING="xes"
MODEL=${RESULTS_FOLDER}"/model.json"
MODEL_ENCODING="json"

OUTPUT_CHECK_CSV=$RESULTS_FOLDER"/measures.csv"
OUTPUT_TRACE_MEASURES_CSV=$RESULTS_FOLDER"/measures[tracesMeasures].csv"

# DECLRE-Tree
MINIMIZATION_FLAG="-min"
BRANCHING_ORDER_DECREASING_FLAG="-decreasing"
CONSTRAINTS_THRESHOLD=0.9
BRANCHING_POLICY="dynamic-variance" # "static-frequency" "dynamic-frequency" "dynamic-variance"
MULTI_PERSPECTIVE_FEATURES="mixed"
# 'rules'
# 'attributes'
# 'specific-attribute'
# 'performances'
# 'mixed'

MERGED_LOG="${RESULTS_FOLDER}/merged-log.xes"

##################################################################
# SCRIPT
##################################################################
#
# experiment folders
mkdir -p $RESULTS_FOLDER

# Logs attributes and Performances
python3 -m DeclarativeClusterMind.cli_evaluation stats \
  -iLf $INPUT_LOGS_FOLDER \
  -o $RESULTS_FOLDER"/clusters-stats.csv"

# Discover process models for each input log (if not existing)
#   Models are stored in the input logs folder
echo "################################ CLUSTERS MODEL DISCOVERY"
for INPUT_LOG in "${INPUT_LOGS_FOLDER}"/*.xes; do
  echo $INPUT_LOG
  CURRENT_MODEL=${INPUT_LOG}"_model.json"
  if test -f "${CURRENT_MODEL}"; then
    echo "${CURRENT_MODEL} already exists."
  else
    #    Discovery with Janus
    #    $JAVA_BIN -cp $JANUS_JAR $JANUS_DISCOVERY_MAINCLASS -iLF "${INPUT_LOG}" -iLE $LOG_ENCODING -c $JANUS_DISCOVERY_CONFIDENCE -s $JANUS_DISCOVERY_SUPPORT -i 0 -oJSON "${CURRENT_MODEL}" -vShush
    #    Discovery with MINERful
    $JAVA_BIN -cp $MINERFUL_JAR $MINERFUL_DISCOVERY_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -c $MINERFUL_DISCOVERY_CONFIDENCE -s $MINERFUL_DISCOVERY_SUPPORT -oJSON ${CURRENT_MODEL} -vShush

    # Filter undesired templates,
    if test -f "${CONSTRAINTS_TEMPLATE_BLACKLIST}"; then
      python3 -m DeclarativeClusterMind.utils.filter_json_model ${CURRENT_MODEL} ${CONSTRAINTS_TEMPLATE_BLACKLIST} ${CURRENT_MODEL}
    fi
    # Filter any rule involving undesired tasks
    if test -f "${CONSTRAINTS_TASKS_BLACKLIST}"; then
      python3 -m DeclarativeClusterMind.utils.filter_json_model ${CURRENT_MODEL} ${CONSTRAINTS_TASKS_BLACKLIST} ${CURRENT_MODEL}
    fi

    #    # Simplify model, i.e., remove redundant constraints
    #    echo "################################ SIMPLIFICATION"
    #    $JAVA_BIN -cp $JANUS_JAR $SIMPLIFIER_MAINCLASS -iMF $CURRENT_MODEL -iME $MODEL_ENCODING -oJSON $CURRENT_MODEL -s 0 -c 0 -i 0 -prune hierarchyconflictredundancydouble
  fi
done

# Retrieve measures for each cluster
echo "################################ LOGS MEASURES"
# merge process models
python3 -m DeclarativeClusterMind.utils.merge_models ${INPUT_LOGS_FOLDER} "_model.json" ${MODEL}
#python3 -m DeclarativeClusterMind.utils.intersect_models $INPUT_LOGS_FOLDER "_model.json" ${MODEL}
#python3 -m DeclarativeClusterMind.utils.intersect_alphabet_models $INPUT_LOGS_FOLDER "_model.json" ${MODEL}

for INPUT_LOG in "${INPUT_LOGS_FOLDER}"/*.xes; do
  echo $INPUT_LOG
  CURRENT_OUTPUT_CHECK_CSV="${INPUT_LOG}-output.csv"
  CURRENT_LOG_MESURES_FILE="${INPUT_LOG}-output[logMeasures].csv"
  echo $CURRENT_LOG_MESURES_FILE
  if test -f $CURRENT_LOG_MESURES_FILE; then
    echo "${CURRENT_LOG_MESURES_FILE} already exists."
  else
    $JAVA_BIN -cp $JANUS_JAR $JANUS_MEASURES_MAINCLASS -iLF "${INPUT_LOG}" -iLE $LOG_ENCODING -iMF "$MODEL" -iME $MODEL_ENCODING -oCSV "$CURRENT_OUTPUT_CHECK_CSV" -d none -detailsLevel log -measure "Confidence"
  fi
done
# Aggregate obtained measure in one unique matrix
python3 -m DeclarativeClusterMind.utils.aggregate_clusters_measures ${INPUT_LOGS_FOLDER} "-output[logMeasures].csv" "${RESULTS_FOLDER}/aggregated_result.csv"
python3 -m DeclarativeClusterMind.utils.label_clusters_with_measures ${INPUT_LOGS_FOLDER} "-output[logMeasures].csv" "${RESULTS_FOLDER}/clusters-labels.csv"


# Retrieve measure for trace decision tree
echo "################################ TRACES MEASURES"
# Merge the input logs
if test -f $MERGED_LOG; then
  echo "${MERGED_LOG} already exists."
else
  python3 -m DeclarativeClusterMind.utils.merge_logs $MERGED_LOG $INPUT_LOGS_FOLDER"/"*.xes
fi
# compute the trace measures of the joint log
if test -f "${OUTPUT_TRACE_MEASURES_CSV}"; then
  echo "$OUTPUT_TRACE_MEASURES_CSV already exists."
else
  # -Xmx12000m in case of extra memory required add this (adjusting the required memory)
  $JAVA_BIN -cp $JANUS_JAR $JANUS_MEASURES_MAINCLASS -iLF $MERGED_LOG -iLE $LOG_ENCODING -iMF $MODEL -iME $MODEL_ENCODING -oCSV $OUTPUT_CHECK_CSV -d none -nanLogSkip -measure "Confidence" -detailsLevel trace
fi
# Label traces
python3 -m DeclarativeClusterMind.evaluation.label_traces_from_clustered_logs $INPUT_LOGS_FOLDER ${RESULTS_FOLDER}/traces-labels.csv

# Build decision-Trees
echo "################################ DeclaraTrees Clusters"
python3 -m DeclarativeClusterMind.cli_decision_trees simple-tree-logs-to-clusters \
  -i ${RESULTS_FOLDER}"/aggregated_result.csv" \
  -o ${RESULTS_FOLDER}"/DeclareTree-LOGS.dot" \
  -t $CONSTRAINTS_THRESHOLD \
  -p $BRANCHING_POLICY \
  $MINIMIZATION_FLAG \
  $BRANCHING_ORDER_DECREASING_FLAG

echo "################################ CART DECISION TREES clusters"
# If rules: -i clusters-labels.csv and -m None
# If attributes/performance: -i clusters-stats.csv and -m None
# If mixed: -i clusters-stats.csv and -m clusters-labels.csv
python3 -m DeclarativeClusterMind.cli_decision_trees decision-tree-logs-to-clusters \
  -i ${RESULTS_FOLDER}"/clusters-stats.csv" \
  -o ${RESULTS_FOLDER}"/decision_tree_logs.dot" \
  -p ${MULTI_PERSPECTIVE_FEATURES} \
  -m ${RESULTS_FOLDER}"/clusters-labels.csv" \
  -fi 0

echo "################################ DeclaraTrees Traces"
python3 -m DeclarativeClusterMind.cli_decision_trees simple-tree-traces \
  -i ${OUTPUT_TRACE_MEASURES_CSV} \
  -o ${RESULTS_FOLDER}"/DeclareTree-TRACES.dot" \
  -t $CONSTRAINTS_THRESHOLD \
  -p $BRANCHING_POLICY \
  $MINIMIZATION_FLAG \
  $BRANCHING_ORDER_DECREASING_FLAG \
  -mls 100

echo "################################ CART DECISION TREES traces"
python3 -m DeclarativeClusterMind.cli_decision_trees decision-tree-traces-to-clusters \
  -i ${RESULTS_FOLDER}"/traces-labels.csv" \
  -o ${RESULTS_FOLDER}"/decision_tree_traces.dot" \
  -fi 1 \
  -m "$OUTPUT_TRACE_MEASURES_CSV" \
  -p ${MULTI_PERSPECTIVE_FEATURES}

echo "################################ DESCRIPTIVE Stats"
# STATS
python3 -m DeclarativeClusterMind.cli_evaluation performances \
  -iLf $INPUT_LOGS_FOLDER \
  -o $RESULTS_FOLDER"/performances_boxplot.svg"
