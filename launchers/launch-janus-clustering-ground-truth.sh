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

LOG_NAME="BPIC15_f"
# "BPIC15_f"
# "BPIC15_1f"
# "BPIC12"
# "BPIC13"
# "SEPSIS"
# "RTFMP"
# "BPIC17_f"
# "WSVX"

CLUSTERING_POLICY="rules"
# 'rules'
# 'attributes'
# 'specific-attribute'
# 'performances'
# 'mixed'
SPLIT_POLICY="rules"
# 'rules'
# 'attributes'
# 'specific-attribute'
# 'performances'
# 'mixed'
CLUSTERING_ALGORITHM="affinity"
#        'kmeans',  # 0
#        'affinity',  # 1
#        'meanshift',  # 2
#        'agglomerative',  # 3
#        'spectral',  # 4
#        'dbscan',  # 5
#        'optics',  # 6
#        'birch',  # 7
#        'gaussian',  # 8 DO NOT USE THIS!

# experiment folders
EXPERIMENT_NAME="experiments/GROUND-TRUTH/"${LOG_NAME}"/clusters_"${CLUSTERING_POLICY}"-algorithm_"${CLUSTERING_ALGORITHM}
INPUT_FOLDER="experiments/GROUND-TRUTH/00-INPUT-LOGS-MODELS"
PREPROCESSED_DATA_FOLDER="experiments/GROUND-TRUTH/00-MERGED-LOGS"
PROCESSED_DATA_FOLDER=$EXPERIMENT_NAME"/2-clustered-logs"
RESULTS_FOLDER=$EXPERIMENT_NAME"/3-results"
mkdir -p $EXPERIMENT_NAME $INPUT_FOLDER $PREPROCESSED_DATA_FOLDER $PROCESSED_DATA_FOLDER $RESULTS_FOLDER

# Clustering
#CLUSTERING_POLICY="attributes"
# 'rules'
# 'attributes'
# 'specific-attribute'
# 'mixed'

#BOOLEAN_RULES="True"
BOOLEAN_RULES="-b"
#BOOLEAN_RULES=""
#VISUALIZATION_FLAG="True"
VISUALIZATION_FLAG="-vf"
#VISUALIZATION_FLAG=""
#APPLY_PCA_FLAG="True"
APPLY_PCA_FLAG="-pca"
#APPLY_PCA_FLAG=""
CLUSTERS_NUMBER=5

# DECLRE-Tree
CONSTRAINTS_THRESHOLD=0.95
PROCESSED_OUTPUT_CHECK_CSV=$PROCESSED_DATA_FOLDER"/"$LOG_NAME"-output.csv"
BRANCHING_POLICY="dynamic-variance" # "static-frequency" "dynamic-frequency" "dynamic-variance"
RESULT_DECLARE_TREE_CLUSTERS=$RESULTS_FOLDER"/"$LOG_NAME"-DeclareTree-CLUSTERS-"${BRANCHING_POLICY}".dot"
RESULT_DECLARE_TREE_TRACES=$RESULTS_FOLDER"/"$LOG_NAME"-DeclareTree-TRACES-"${BRANCHING_POLICY}".dot"
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
#LOG_NAME="BPIC13"
INPUT_LOG=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-log.xes"
LOG_ENCODING="xes"

# Discovery & Measurements
SUPPORT=0.0
CONFIDENCE=0.9
#MODEL=$INPUT_FOLDER"/"$LOG_NAME".xes-model[s_"$SUPPORT"_c_"$CONFIDENCE"].json"
MODEL=$INPUT_FOLDER"/"$LOG_NAME".xes-model[s_"$SUPPORT"_c_"$CONFIDENCE"]-SIMPLIFIED.json"
#MODEL=$INPUT_FOLDER"/"$LOG_NAME"-model[GROUND-TRUTH].json"
#MODEL=$INPUT_FOLDER"/"$LOG_NAME"-model[PARTICIPATION].json"
#MODEL=$INPUT_FOLDER"/"$LOG_NAME"-model[ABSENCE].json"
#MODEL=$INPUT_FOLDER"/"$LOG_NAME"-model[ALL].json"
MODEL_ENCODING="json"

OUTPUT_CHECK_CSV=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output.csv"
OUTPUT_CHECK_JSON=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output.json"
OUTPUT_TRACE_MEASURES_CSV=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output[tracesMeasures].csv"
OUTPUT_TRACE_MEASURES_STATS_CSV=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output[tracesMeasuresStats].csv"
OUTPUT_LOG_MEASURES_CSV=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output[logMeasures].csv"

ORIGINAL_TRACES_LABELS=$INPUT_FOLDER"/original-traces-labels.csv"

##################################################################
# SCRIPT
##################################################################
#
# label traces according to real clusters
if test -f "${ORIGINAL_TRACES_LABELS}"; then
    echo "$ORIGINAL_TRACES_LABELS already exists. Traces already labeled"
else
  python3 -m DeclarativeClusterMind.evaluation.label_traces_from_clustered_logs $INPUT_FOLDER $ORIGINAL_TRACES_LABELS
fi
# merge logs
if test -f "${INPUT_LOG}"; then
  echo "$INPUT_LOG already exists. Logs already merged"
else
  python3 DeclarativeClusterMind/utils/merge_logs.py $INPUT_LOG $INPUT_FOLDER"/"*.xes
fi

# Discover process model (if not existing)
echo "${MODEL}"
echo "################################ DISCOVERY"
if test -f "${MODEL}"; then
  echo "$FILE already exists."
else
  java -cp Janus.jar $JANUS_DISCOVERY_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -c $CONFIDENCE -s $SUPPORT -i 0 -oJSON ${MODEL}
  #  java -cp Janus.jar $JANUS_DISCOVERY_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -c $CONFIDENCE -s $SUPPORT -i 0 -keep -oJSON ${MODEL}

  # Simplify model, i.e., remove redundant constraints
  echo "################################ SIMPLIFICATION"
#  java -cp Janus.jar $SIMPLIFIER_MAINCLASS -iMF $MODEL -iME $MODEL_ENCODING -oJSON $MODEL -s 0 -c 0 -i 0 -prune hierarchyconflictredundancydouble
fi

# Retrieve measure
echo "################################ MEASURE"
if test -f "${OUTPUT_TRACE_MEASURES_CSV}"; then
  echo "$OUTPUT_TRACE_MEASURES_CSV already exists."
else
  java -cp Janus.jar $JANUS_CHECK_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -iMF $MODEL -iME $MODEL_ENCODING -oCSV $OUTPUT_CHECK_CSV -d none -nanLogSkip -measure "Confidence" -detailsLevel trace
#  java -cp Janus.jar $JANUS_CHECK_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -iMF $MODEL -iME $MODEL_ENCODING -oCSV $OUTPUT_CHECK_CSV -oJSON $OUTPUT_CHECK_JSON -d none -nanLogSkip
# 'Lift','Confidence','Relative Risk'
# 'Support','all','Compliance,'Added Value','J Measure'
# 'Recall','Lovinger','Specificity','Accuracy','Leverage','Odds Ratio', 'Gini Index','Certainty factor','Coverage','Prevalence',
# 'Jaccard','Ylue Q','Ylue Y','Klosgen','Conviction','Interestingness Weighting Dependency','Collective Strength','Laplace Correction',
# 'One-way Support','Two-way Support','Two-way Support Variation',
# 'Linear Correlation Coefficient','Piatetsky-Shapiro','Cosine','Information Gain','Sebag-Schoenauer','Least Contradiction','Odd Multiplier','Example and Counterexample Rate','Zhang'}.
fi

# Launch clustering
echo "################################ CLUSTERING"
if [ $CLUSTERING_POLICY == "rules" ] || [ $CLUSTERING_POLICY == "mixed" ]; then
  python3 -m DeclarativeClusterMind.ui_clustering --ignore-gooey $CLUSTERING_POLICY -iL $INPUT_LOG -a $CLUSTERING_ALGORITHM -o $PROCESSED_DATA_FOLDER $VISUALIZATION_FLAG $APPLY_PCA_FLAG -nc $CLUSTERS_NUMBER -tm "$OUTPUT_TRACE_MEASURES_CSV" $BOOLEAN_RULES
else
  python3 -m DeclarativeClusterMind.ui_clustering --ignore-gooey $CLUSTERING_POLICY -iL $INPUT_LOG -a $CLUSTERING_ALGORITHM -o $PROCESSED_DATA_FOLDER $VISUALIZATION_FLAG $APPLY_PCA_FLAG -nc $CLUSTERS_NUMBER
fi

# compare traces labels
echo "################################ COMPARING LABELS"
cp ${PROCESSED_DATA_FOLDER}/*traces-labels.csv $RESULTS_FOLDER"/traces-labels.csv"
cp $ORIGINAL_TRACES_LABELS $RESULTS_FOLDER"/traces-labels-original.csv"

python3 -m DeclarativeClusterMind.evaluation.compare_traces_labels $RESULTS_FOLDER"/traces-labels-original.csv" $RESULTS_FOLDER"/traces-labels.csv" $RESULTS_FOLDER"/matrix-result.csv"

