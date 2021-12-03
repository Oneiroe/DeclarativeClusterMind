#!/bin/bash

##################################################################
# PARAMETERS
##################################################################
#INPUT_LOG="PATH_TO_INPUT_LOG"
#RESULTS_CLUSTERS_FOLDER="PATH_TO_CLUSTERING_RESULTS_FOLDER"
#RESULTS_TREES_FOLDER="PATH_TO_DECISION_TREES_RESULTS_FOLDER"
INPUT_LOG="/home/alessio/Data/Phd/my_code/DeclarativeClusterMind/experiments/test3/input/SEPSIS-log.xes"
RESULTS_CLUSTERS_FOLDER="/home/alessio/Data/Phd/my_code/DeclarativeClusterMind/experiments/test3/clusters"
RESULTS_TREES_FOLDER="/home/alessio/Data/Phd/my_code/DeclarativeClusterMind/experiments/test3/trees"

CONSTRAINTS_TEMPLATE_BLACKLIST=$(dirname -- "$INPUT_LOG")"/blacklist.csv"
CONSTRAINTS_TASKS_BLACKLIST=$(dirname -- "$INPUT_LOG")"/blacklist-tasks.csv"

# PATHs
WORKING_DIR="/home/alessio/Data/Phd/my_code/DeclarativeClusterMind" # path containing DeclarativeClusterMind package
cd $WORKING_DIR || exit
JAVA_BIN="/home/alessio/Software/jdk/jdk-11.0.10/bin/java"
JANUS_JAR="/home/alessio/Data/Phd/my_code/DeclarativeClusterMind/Janus.jar"
MINERFUL_JAR="/home/alessio/Data/Phd/code_3rd_party/MINERful/MINERful.jar"

##################################################################

# Janus/MINERful main classes
JANUS_MEASURES_MAINCLASS="minerful.JanusMeasurementsStarter"
JANUS_DISCOVERY_MAINCLASS="minerful.JanusOfflineMinerStarter"
JANUS_DISCOVERY_SUPPORT=0.0
JANUS_DISCOVERY_CONFIDENCE=0.9
MINERFUL_DISCOVERY_MAINCLASS="minerful.MinerFulMinerStarter"
MINERFUL_DISCOVERY_SUPPORT=0.9    # support threshold for the initial discovery of the constraints of the variances
MINERFUL_DISCOVERY_CONFIDENCE=0.0 # confidence threshold for the initial discovery of the constraints of the variances
SIMPLIFIER_MAINCLASS="minerful.MinerFulSimplificationStarter"

# Clustering
CLUSTERING_PERSPECTIVE="rules"
# 'rules'
# 'attributes'
# 'specific-attribute'
# 'performances'
# 'mixed'
CLUSTERING_ALGORITHM="optics"
#        'kmeans',  # 0
#        'affinity',  # 1
#        'meanshift',  # 2
#        'agglomerative',  # 3
#        'spectral',  # 4
#        'dbscan',  # 5
#        'optics',  # 6
#        'birch',  # 7
#        'gaussian',  # 8 DO NOT USE THIS!
BOOLEAN_RULES="-b"
VISUALIZATION_FLAG="-vf"
APPLY_PCA_FLAG="-pca"
CLUSTERS_NUMBER=10 # not all the clustering algorithm requires a specific number

# Discovery & Measurements
LOG_ENCODING="xes"
MODEL=${INPUT_LOG}"-model.json"
MODEL_ENCODING="json"

MEASURES_BASE_CSV=$RESULTS_CLUSTERS_FOLDER"/measures.csv"
TRACE_MEASURES_CSV=$RESULTS_CLUSTERS_FOLDER"/measures[tracesMeasures].csv"
TRACE_MEASURES_STATS_CSV=$RESULTS_CLUSTERS_FOLDER"/"$LOG_NAME"-output[tracesMeasuresStats].csv"
LOG_MEASURES_CSV=$RESULTS_CLUSTERS_FOLDER"/"$LOG_NAME"-output[logMeasures].csv"

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
MINIMUM_LEAF_SIZE=100

UNION_MODEL=${RESULTS_TREES_FOLDER}"/union-model.json"

##################################################################
# SCRIPT
##################################################################
#
# experiment folders
mkdir -p $RESULTS_CLUSTERS_FOLDER
mkdir -p $RESULTS_TREES_FOLDER

# Discover process model (if not existing)
echo "${MODEL}"
echo "################################ DISCOVERY"
if test -f "${MODEL}"; then
  echo "${MODEL} already exists."
else
  #    Discovery with Janus
  #  $JAVA_BIN -cp $JANUS_JAR $JANUS_DISCOVERY_MAINCLASS -iLF "${INPUT_LOG}" -iLE $LOG_ENCODING -c $JANUS_DISCOVERY_CONFIDENCE -s $JANUS_DISCOVERY_SUPPORT -i 0 -oJSON "${MODEL}" -vShush
  #    Discovery with MINERful
  $JAVA_BIN -cp $MINERFUL_JAR $MINERFUL_DISCOVERY_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -c $MINERFUL_DISCOVERY_CONFIDENCE -s $MINERFUL_DISCOVERY_SUPPORT -oJSON ${MODEL} -vShush

  # Filter undesired templates,
  if test -f "${CONSTRAINTS_TEMPLATE_BLACKLIST}"; then
    python3 -m DeclarativeClusterMind.utils.filter_json_model ${MODEL} ${CONSTRAINTS_TEMPLATE_BLACKLIST} ${MODEL}
  fi
  # Filter any rule involving undesired tasks
  if test -f "${CONSTRAINTS_TASKS_BLACKLIST}"; then
    python3 -m DeclarativeClusterMind.utils.filter_json_model ${MODEL} ${CONSTRAINTS_TASKS_BLACKLIST} ${MODEL}
  fi

  #    # Simplify model, i.e., remove redundant constraints
  #    echo "################################ SIMPLIFICATION"
  #    $JAVA_BIN -cp $JANUS_JAR $SIMPLIFIER_MAINCLASS -iMF $MODEL -iME $MODEL_ENCODING -oJSON $MODEL -s 0 -c 0 -i 0 -prune hierarchyconflictredundancydouble
fi

# Retrieve measures
echo "################################ MEASURES"
if test -f "${TRACE_MEASURES_CSV}"; then
  echo "$TRACE_MEASURES_CSV already exists."
else
  $JAVA_BIN -cp $JANUS_JAR $JANUS_MEASURES_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -iMF $MODEL -iME $MODEL_ENCODING -oCSV $MEASURES_BASE_CSV -d none -nanLogSkip -measure "Confidence" -detailsLevel allTrace
fi

# Launch clustering
echo "################################ CLUSTERING"
if [ $CLUSTERING_PERSPECTIVE == "rules" ] || [ $CLUSTERING_PERSPECTIVE == "mixed" ]; then
  python3 -m DeclarativeClusterMind.cli_clustering $CLUSTERING_PERSPECTIVE -iL $INPUT_LOG -a $CLUSTERING_ALGORITHM -o $RESULTS_CLUSTERS_FOLDER $VISUALIZATION_FLAG -nc $CLUSTERS_NUMBER $APPLY_PCA_FLAG -tm "$TRACE_MEASURES_CSV" $BOOLEAN_RULES
else
  python3 -m DeclarativeClusterMind.cli_clustering $CLUSTERING_PERSPECTIVE -iL $INPUT_LOG -a $CLUSTERING_ALGORITHM -o $RESULTS_CLUSTERS_FOLDER $VISUALIZATION_FLAG -nc $CLUSTERS_NUMBER $APPLY_PCA_FLAG
fi

###########################################################################

# Discover process models for each input log (if not existing)
#   Models are stored in the input logs folder
echo "################################ CLUSTERS MODEL DISCOVERY"
for CLUSTER_LOG in "${RESULTS_CLUSTERS_FOLDER}"/*.xes; do
  echo $CLUSTER_LOG
  CURRENT_MODEL=${CLUSTER_LOG}"_model.json"
  if test -f "${CURRENT_MODEL}"; then
    echo "${CURRENT_MODEL} already exists."
  else
    #    Discovery with Janus
    #    $JAVA_BIN -cp $JANUS_JAR $JANUS_DISCOVERY_MAINCLASS -iLF "${CLUSTER_LOG}" -iLE $LOG_ENCODING -c $JANUS_DISCOVERY_CONFIDENCE -s $JANUS_DISCOVERY_SUPPORT -i 0 -oJSON "${CURRENT_MODEL}" -vShush
    #    Discovery with MINERful
    $JAVA_BIN -cp $MINERFUL_JAR $MINERFUL_DISCOVERY_MAINCLASS -iLF "${CLUSTER_LOG}" -iLE $LOG_ENCODING -c $MINERFUL_DISCOVERY_CONFIDENCE -s $MINERFUL_DISCOVERY_SUPPORT -oJSON "${CURRENT_MODEL}" -vShush

    # Filter undesired templates,
    if test -f "${CONSTRAINTS_TEMPLATE_BLACKLIST}"; then
      python3 -m DeclarativeClusterMind.utils.filter_json_model "${CURRENT_MODEL}" ${CONSTRAINTS_TEMPLATE_BLACKLIST} "${CURRENT_MODEL}"
    fi
    # Filter any rule involving undesired tasks
    if test -f "${CONSTRAINTS_TASKS_BLACKLIST}"; then
      python3 -m DeclarativeClusterMind.utils.filter_json_model "${CURRENT_MODEL}" ${CONSTRAINTS_TASKS_BLACKLIST} "${CURRENT_MODEL}"
    fi

    #    # Simplify model, i.e., remove redundant constraints
    #    echo "################################ SIMPLIFICATION"
    #    $JAVA_BIN -cp $JANUS_JAR $SIMPLIFIER_MAINCLASS -iMF $CURRENT_MODEL -iME $MODEL_ENCODING -oJSON $CURRENT_MODEL -s 0 -c 0 -i 0 -prune hierarchyconflictredundancydouble
  fi
done

# Retrieve measures for each cluster
echo "################################ LOGS MEASURES"
# merge process models
python3 -m DeclarativeClusterMind.utils.merge_models ${RESULTS_CLUSTERS_FOLDER} "_model.json" ${UNION_MODEL}
#python3 -m DeclarativeClusterMind.utils.intersect_models $INPUT_LOGS_FOLDER "_model.json" ${UNION_MODEL}
#python3 -m DeclarativeClusterMind.utils.intersect_alphabet_models $INPUT_LOGS_FOLDER "_model.json" ${UNION_MODEL}

for CLUSTER_LOG in "${RESULTS_CLUSTERS_FOLDER}"/*.xes; do
  echo $CLUSTER_LOG
  CURRENT_OUTPUT_CHECK_CSV="${CLUSTER_LOG}-output.csv"
  CURRENT_LOG_MESURES_FILE="${CLUSTER_LOG}-output[logMeasures].csv"
  echo $CURRENT_LOG_MESURES_FILE
  if test -f $CURRENT_LOG_MESURES_FILE; then
    echo "${CURRENT_LOG_MESURES_FILE} already exists."
  else
    $JAVA_BIN -cp $JANUS_JAR $JANUS_MEASURES_MAINCLASS -iLF "${CLUSTER_LOG}" -iLE $LOG_ENCODING -iMF "${MODEL}" -iME $MODEL_ENCODING -oCSV "${CURRENT_OUTPUT_CHECK_CSV}" -d none -detailsLevel log -measure "Confidence"
  fi
done
# Aggregate obtained measure in one unique matrix
python3 -m DeclarativeClusterMind.utils.aggregate_clusters_measures ${RESULTS_CLUSTERS_FOLDER} "-output[logMeasures].csv" "${RESULTS_TREES_FOLDER}/aggregated_result.csv"
python3 -m DeclarativeClusterMind.utils.label_clusters_with_measures ${RESULTS_CLUSTERS_FOLDER} "-output[logMeasures].csv" "${RESULTS_TREES_FOLDER}/clusters-labels.csv"

# Build decision-Trees
echo "################################ DeclaraTrees Clusters"
python3 -m DeclarativeClusterMind.cli_decision_trees simple-tree-logs-to-clusters \
  -i ${RESULTS_TREES_FOLDER}"/aggregated_result.csv" \
  -o ${RESULTS_TREES_FOLDER}"/DeclareTree-LOGS.dot" \
  -t $CONSTRAINTS_THRESHOLD \
  -p $BRANCHING_POLICY \
  $MINIMIZATION_FLAG \
  $BRANCHING_ORDER_DECREASING_FLAG

echo "################################ CART DECISION TREES clusters"
# If rules: -i clusters-labels.csv and -m None
# If attributes/performance: -i clusters-stats.csv and -m None
# If mixed: -i clusters-stats.csv and -m clusters-labels.csv
python3 -m DeclarativeClusterMind.cli_decision_trees decision-tree-logs-to-clusters \
  -i ${RESULTS_CLUSTERS_FOLDER}"/"*"clusters-stats.csv" \
  -o ${RESULTS_TREES_FOLDER}"/decision_tree_logs.dot" \
  -p ${MULTI_PERSPECTIVE_FEATURES} \
  -m ${RESULTS_TREES_FOLDER}"/clusters-labels.csv" \
  -fi 0

echo "################################ DeclaraTrees Traces"
python3 -m DeclarativeClusterMind.cli_decision_trees simple-tree-traces \
  -i ${TRACE_MEASURES_CSV} \
  -o ${RESULTS_TREES_FOLDER}"/DeclareTree-TRACES.dot" \
  -t $CONSTRAINTS_THRESHOLD \
  -p $BRANCHING_POLICY \
  $MINIMIZATION_FLAG \
  $BRANCHING_ORDER_DECREASING_FLAG \
  -mls $MINIMUM_LEAF_SIZE

echo "################################ CART DECISION TREES traces"
python3 -m DeclarativeClusterMind.cli_decision_trees decision-tree-traces-to-clusters \
  -i ${RESULTS_CLUSTERS_FOLDER}"/"*"traces-labels.csv" \
  -o ${RESULTS_TREES_FOLDER}"/decision_tree_traces.dot" \
  -fi 1 \
  -m "$TRACE_MEASURES_CSV" \
  -p ${MULTI_PERSPECTIVE_FEATURES}
