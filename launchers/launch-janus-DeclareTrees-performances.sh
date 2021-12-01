#!/bin/bash

# WHAT: Create a classic decision tree from the clustering output
# WHY: hopefully the decision trees models in a concise and meaningful way the discriminants between the clusters

##################################################################
# PARAMETERS
##################################################################
cd "/home/alessio/Data/Phd/my_code/ClusterMind"

SECONDS=0
BASE_DIR="experiments/PERFORMANCES/"
EXEC_LOG=${BASE_DIR}"log.log"
TIMES_FILE=${BASE_DIR}"times.csv"
HEADER="LOG;DISCOVERY;MEASUREMENT;PRE-PROCESSING;RULE-TREE;MULTI-TREE;TOT"
echo $HEADER >$TIMES_FILE
date >$EXEC_LOG

# Janus main classes
SIMPLIFIER_MAINCLASS="minerful.MinerFulSimplificationStarter"
JANUS_DISCOVERY_MAINCLASS="minerful.JanusOfflineMinerStarter"
JANUS_CHECK_MAINCLASS="minerful.JanusMeasurementsStarter"

JAVA_BIN="/home/alessio/Software/jdk/jdk-11.0.10/bin/java"
DISCOVERY_JAR="/home/alessio/Data/Phd/code_3rd_party/MINERful/MINERful.jar"
DISCOVERY_MAINCLASS="minerful.MinerFulMinerStarter"
DISCOVERY_SUPPORT=0.9    # support threshold used for the initial discovery of the constraints of the variances
DISCOVERY_CONFIDENCE=0.0 # confidence threshold used for the initial discovery of the constraints of the variances

#LOGS_NAMES=("SEPSIS_age" "SEPSIS_intensiveCare" "BPIC15en" "WSV_YCL_2019")
LOGS_NAMES=("LOG_NAME" "SEPSIS_intensiveCare" "WSV_YCL_2019")

ITERATIONS=2

for ITERATION in $(seq 1 ${ITERATIONS}); do
  for LOG_NAME in "${LOGS_NAMES[@]}"; do
    echo $LOG_NAME
    BEGINNING=$SECONDS
    echo -n ${LOG_NAME}";" >>${TIMES_FILE}
    # experiment folders
    EXPERIMENT_NAME=${BASE_DIR}${LOG_NAME}
    PROCESSED_DATA_FOLDER=$EXPERIMENT_NAME"/1-clustered-logs"
    MERGED_FOLDER=$EXPERIMENT_NAME"/2-merged-log"
    PREPROCESSED_DATA_FOLDER=$MERGED_FOLDER
    RESULTS_FOLDER=$EXPERIMENT_NAME"/3-results"
    mkdir -p $EXPERIMENT_NAME $MERGED_FOLDER $PREPROCESSED_DATA_FOLDER $PROCESSED_DATA_FOLDER $RESULTS_FOLDER

    # DECLRE-Tree
    CONSTRAINTS_THRESHOLD=0.95
    BRANCHING_POLICY="dynamic-variance" # "static-frequency" "dynamic-frequency" "dynamic-variance"
    RESULT_DECLARE_TREE_CLUSTERS=$RESULTS_FOLDER"/"${LOG_NAME}"-DeclareTree-CLUSTERS-"${BRANCHING_POLICY}".dot"

    # Input log
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
    CONSTRAINTS_TASKS_BLACKLIST=${PROCESSED_DATA_FOLDER}"/blacklist-tasks.csv"

    ##################################################################
    # SCRIPT
    ##################################################################
    #
    # Discover process model (if not existing)
    # IDEA: discover the process models out of each cluster, then merge them. afterward you compute the measures given the unique model
    START=$SECONDS
    echo "################################ CLUSTERS MODEL DISCOVERY"
    for INPUT_LOG in $PROCESSED_DATA_FOLDER"/"*.xes; do
      echo $INPUT_LOG >>$EXEC_LOG
      CURRENT_MODEL=${INPUT_LOG}"_model.json"
      if test -f "${CURRENT_MODEL}"; then
        echo "$FILE already exists." >>$EXEC_LOG
      else
        java -cp Janus.jar $JANUS_DISCOVERY_MAINCLASS -iLF "${INPUT_LOG}" -iLE $LOG_ENCODING -c $CONFIDENCE -s $SUPPORT -i 0 -oJSON "${CURRENT_MODEL}" -vShush >>$EXEC_LOG
        #    $JAVA_BIN -cp $DISCOVERY_JAR $DISCOVERY_MAINCLASS -iLF "${INPUT_LOG}" -iLE $LOG_ENCODING -c $DISCOVERY_CONFIDENCE -s $DISCOVERY_SUPPORT -oJSON ${CURRENT_MODEL} -vShush

        # Filter undesired templates, e.g., NotSuccession or NotChainSuccession
        if test -f "${CONSTRAINTS_TEMPLATE_BLACKLIST}"; then
          python3 -m DeclarativeClusterMind.utils.filter_json_model ${CURRENT_MODEL} ${CONSTRAINTS_TEMPLATE_BLACKLIST} ${CURRENT_MODEL} >>$EXEC_LOG
        fi
        # Filter all the rules involving undesired tasks
        if test -f "${CONSTRAINTS_TASKS_BLACKLIST}"; then
          python3 -m DeclarativeClusterMind.utils.filter_json_model ${CURRENT_MODEL} ${CONSTRAINTS_TASKS_BLACKLIST} ${CURRENT_MODEL} >>$EXEC_LOG
        fi

        #    # Simplify model, i.e., remove redundant constraints
        #    echo "################################ SIMPLIFICATION"
        #    java -cp Janus.jar $SIMPLIFIER_MAINCLASS -iMF $CURRENT_MODEL -iME $MODEL_ENCODING -oJSON $CURRENT_MODEL -s 0 -c 0 -i 0 -prune hierarchyconflictredundancydouble
      fi
    done

    ## merge process models
    python3 -m DeclarativeClusterMind.utils.merge_models $PROCESSED_DATA_FOLDER "_model.json" ${MODEL} >>$EXEC_LOG
    #python3 -m DeclarativeClusterMind.utils.intersect_models $PROCESSED_DATA_FOLDER "_model.json" ${MODEL}

    DURATION=$((SECONDS - START))
    echo -n ${DURATION}";" >>${TIMES_FILE}
    START=$SECONDS
    # Retrieve measures for each cluster
    echo "################################ CLUSTERS MEASURES and POSTPROCESSING"
    for INPUT_LOG in $PROCESSED_DATA_FOLDER"/"*.xes; do
      echo $INPUT_LOG >>$EXEC_LOG
      CURRENT_OUTPUT_CHECK_CSV="${INPUT_LOG}""-output.csv"
      OUTPUT_CHECK_JSON="${INPUT_LOG}""-output.json"
      TEMP_OUT_MESURES_FILE="${INPUT_LOG}""-output[logMeasures].csv"
      echo $TEMP_OUT_MESURES_FILE >>$EXEC_LOG
      if test -f $TEMP_OUT_MESURES_FILE; then
        echo "${TEMP_OUT_MESURES_FILE} already exists." >>$EXEC_LOG
      else
        java -cp Janus.jar $JANUS_CHECK_MAINCLASS -iLF "${INPUT_LOG}" -iLE $LOG_ENCODING -iMF "$MODEL" -iME $MODEL_ENCODING -oCSV "$CURRENT_OUTPUT_CHECK_CSV" -d none -detailsLevel log -measure Confidence >>$EXEC_LOG
      fi
      #  java -cp Janus.jar $JANUS_CHECK_MAINCLASS -iLF "${INPUT_LOG}" -iLE $LOG_ENCODING -iMF "$MODEL" -iME $MODEL_ENCODING -oCSV "$CURRENT_OUTPUT_CHECK_CSV" -oJSON "$OUTPUT_CHECK_JSON" -d none -detailsLevel log -measure Confidence

      #  -nanLogSkip,--nan-log-skip                            Flag to skip or not NaN values when computing log measures
      #  -nanTraceSubstitute,--nan-trace-substitute            Flag to substitute or not the NaN values when computing trace measures
      #  -nanTraceValue,--nan-trace-value <number>

      #  keep only mean
      #  python3 pySupport/singleAggregationPerspectiveFocusCSV_confidence-only.py "${OUTPUT_CHECK_JSON}AggregatedMeasures.json" "${INPUT_LOG}""-output[MEAN].csv"
    done
    DURATION=$((SECONDS - START))
    echo -n ${DURATION}";" >>${TIMES_FILE}
    START=$SECONDS
    # merge results
    python3 -m DeclarativeClusterMind.utils.aggregate_clusters_measures $PROCESSED_DATA_FOLDER "-output[logMeasures].csv" "aggregated_result.csv" >>$EXEC_LOG
    python3 -m DeclarativeClusterMind.utils.label_clusters_with_measures $PROCESSED_DATA_FOLDER "-output[logMeasures].csv" "clusters-labels.csv" >>$EXEC_LOG
#    python3 -m DeclarativeClusterMind.evaluation.label_traces_from_clustered_logs $PROCESSED_DATA_FOLDER ${PROCESSED_DATA_FOLDER}/traces-labels.csv >>$EXEC_LOG

#    cp ${PROCESSED_DATA_FOLDER}/traces-labels.csv $RESULTS_FOLDER"/traces-labels.csv"
    cp $PROCESSED_DATA_FOLDER"/aggregated_result.csv" $RESULTS_FOLDER"/aggregated_result.csv"
    cp ${PROCESSED_DATA_FOLDER}"/clusters-labels.csv" $RESULTS_FOLDER"/clusters-labels.csv"

    echo "################################ DESCRIPTIVE Stats"
    # STATS
#    python3 -m DeclarativeClusterMind.ui_evaluation --ignore-gooey performances \
#      -iLf $PROCESSED_DATA_FOLDER \
#      -o $RESULTS_FOLDER"/performances_boxplot.svg" >>$EXEC_LOG
    # Performances
    python3 -m DeclarativeClusterMind.ui_evaluation --ignore-gooey stats \
      -iLf $PROCESSED_DATA_FOLDER \
      -o $RESULTS_FOLDER"/clusters-stats.csv" >>$EXEC_LOG

    DURATION=$((SECONDS - START))
    echo -n ${DURATION}";" >>${TIMES_FILE}
    START=$SECONDS

    # Build decision-Tree
    echo "################################ SIMPLE TREES Clusters"
    python3 -m DeclarativeClusterMind.ui_declare_trees --ignore-gooey simple-tree-logs-to-clusters \
      -i $PROCESSED_DATA_FOLDER"/aggregated_result.csv" \
      -o $RESULT_DECLARE_TREE_CLUSTERS"-Decreasing.dot" \
      -t $CONSTRAINTS_THRESHOLD \
      -p $BRANCHING_POLICY \
      -min \
      -decreasing >>$EXEC_LOG

    DURATION=$((SECONDS - START))
    echo -n ${DURATION}";" >>${TIMES_FILE}
    START=$SECONDS

    #  echo "################################ DECISION TREES clusters rules"
    #  # If mixed: -i clusters-stats.csv and -m clusters-labels.csv
    #  python3 -m DeclarativeClusterMind.ui_declare_trees --ignore-gooey decision-tree-logs-to-clusters \
    #    -i ${RESULTS_FOLDER}"/clusters-labels.csv" \
    #    -o ${RESULTS_FOLDER}"/decision_tree_clusters_rules.dot" \
    #    -p rules \
    #    -m None \
    #    -fi 0 >>$EXEC_LOG
    #
    #  echo "################################ DECISION TREES clusters attributes"
    #  # If mixed: -i clusters-stats.csv and -m clusters-labels.csv
    #  python3 -m DeclarativeClusterMind.ui_declare_trees --ignore-gooey decision-tree-logs-to-clusters \
    #    -i ${RESULTS_FOLDER}"/clusters-stats.csv" \
    #    -o ${RESULTS_FOLDER}"/decision_tree_clusters_attributed.dot" \
    #    -p attributes \
    #    -m None \
    #    -fi 0 >>$EXEC_LOG

    echo "################################ DECISION TREES clusters mixed"
    # If mixed: -i clusters-stats.csv and -m clusters-labels.csv
    python3 -m DeclarativeClusterMind.ui_declare_trees --ignore-gooey decision-tree-logs-to-clusters \
      -i ${RESULTS_FOLDER}"/clusters-stats.csv" \
      -o ${RESULTS_FOLDER}"/decision_tree_clusters_multi.dot" \
      -p mixed \
      -m ${RESULTS_FOLDER}"/clusters-labels.csv" \
      -fi 0 >>$EXEC_LOG

    DURATION=$((SECONDS - START))
    echo -n ${DURATION}";" >>${TIMES_FILE}
    DURATION=$((SECONDS - BEGINNING))
    echo ${DURATION} >>${TIMES_FILE}

    #  CLEAN FOLDER FOR SUBSEQUENT TESTS
    rm -r ${RESULTS_FOLDER}
    rm -r ${MERGED_FOLDER}
    find ${PROCESSED_DATA_FOLDER} -type f ! -name "*.xes" -delete
  done
done
