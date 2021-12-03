#!/bin/bash

##################################################################
# PARAMETERS
##################################################################
INPUT_LOG="PATH_TO_INPUT_LOG"
RESULTS_FOLDER="PATH_TO_RESULTS_FOLDER"

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

MEASURES_BASE_CSV=$RESULTS_FOLDER"/measures.csv"
TRACE_MEASURES_CSV=$RESULTS_FOLDER"/measures[tracesMeasures].csv"
TRACE_MEASURES_STATS_CSV=$RESULTS_FOLDER"/"$LOG_NAME"-output[tracesMeasuresStats].csv"
LOG_MEASURES_CSV=$RESULTS_FOLDER"/"$LOG_NAME"-output[logMeasures].csv"

##################################################################
# SCRIPT
##################################################################
#
# experiment folders
mkdir -p $RESULTS_FOLDER

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
  python3 -m DeclarativeClusterMind.cli_clustering $CLUSTERING_PERSPECTIVE -iL $INPUT_LOG -a $CLUSTERING_ALGORITHM -o $RESULTS_FOLDER $VISUALIZATION_FLAG -nc $CLUSTERS_NUMBER $APPLY_PCA_FLAG -tm "$TRACE_MEASURES_CSV" $BOOLEAN_RULES
else
  python3 -m DeclarativeClusterMind.cli_clustering $CLUSTERING_PERSPECTIVE -iL $INPUT_LOG -a $CLUSTERING_ALGORITHM -o $RESULTS_FOLDER $VISUALIZATION_FLAG -nc $CLUSTERS_NUMBER $APPLY_PCA_FLAG
fi

