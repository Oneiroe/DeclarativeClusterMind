#!/bin/bash

# WHAT: clusters the log in a hierarchical way dividing at each split the sub log according to 80:20 on a declarative process model
# WHY: hopefully the clusters will make more sense from a declarative rules point of view

##################################################################
# PARAMETERS
##################################################################

# Janus main classes
LOG_MAINCLASS="minerful.MinerFulLogMakerStarter"
SIMPLIFIER_MAINCLASS="minerful.MinerFulSimplificationStarter"
ERROR_MAINCLASS="minerful.MinerFulErrorInjectedLogMakerStarter"
JANUS_DISCOVERY_MAINCLASS="minerful.JanusOfflineMinerStarter"
JANUS_CHECK_MAINCLASS="minerful.JanusMeasurementsStarter"

LOG_NAME="SEPSIS"
# "BPIC13"
# "SEPSIS"
# "RTFMP"
# "BPIC15_1f"
# "BPIC17_f"

CLUSTERING_POLICY="rules"
# 'rules'
# 'attributes'
# 'specific-attribute'
# 'mixed'
SPLIT_POLICY="rules"
# 'rules'
# 'attributes'
# 'specific-attribute'
# 'mixed'

# Pareto Split parameters
SIMPLIFICATION_FLAG="False"
PARETO_SPLIT_THRESHOLD=0.95
INPUT_LOG="experiments/PARETO-SPLIT/00-INPUT-LOGS-MODELS/SEPSIS-log.xes"
MIN_LEAF_SIZE=5

# experiment folders
EXPERIMENT_NAME="experiments/PARETO-SPLIT/"${LOG_NAME}"/clusters_"${PARETO_SPLIT_THRESHOLD}"-treeSplit_"${SPLIT_POLICY}
INPUT_FOLDER="experiments/PARETO-SPLIT/00-INPUT-LOGS-MODELS"
PREPROCESSED_DATA_FOLDER=$EXPERIMENT_NAME"/entire-log-measurements"
RESULTS_FOLDER=$EXPERIMENT_NAME"/clusters"
mkdir -p $EXPERIMENT_NAME $INPUT_FOLDER $PREPROCESSED_DATA_FOLDER $RESULTS_FOLDER

# Clustering
#CLUSTERING_POLICY="attributes"
# 'rules'
# 'attributes'
# 'specific-attribute'
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
BOOLEAN_RULES="True"
VISUALIZATION_FLAG="True"
APPLY_PCA_FLAG="True"

# DECLRE-Tree
CONSTRAINTS_THRESHOLD=0.95
PROCESSED_OUTPUT_CHECK_CSV=$PROCESSED_DATA_FOLDER"/"$LOG_NAME"-output.csv"
RESULT_DECLARE_TREE=$RESULTS_FOLDER"/"$LOG_NAME"-DeclareTree.dot"
BRANCHING_POLICY="dynamic"
MINIMIZATION_FLAG="True"
BRANCHING_ORDER_DECREASING_FLAG="True"
#SPLIT_POLICY="mixed"
## 'rules'
## 'attributes'
## 'specific-attribute'
## 'mixed'

# Input log
#LOG_NAME="BPIC13"
INPUT_LOG=$INPUT_FOLDER"/"$LOG_NAME"-log.xes"
LOG_ENCODING="xes"

# Discovery & Measurements
SUPPORT=0.0
CONFIDENCE=0.95
MODEL=$INPUT_FOLDER"/"$LOG_NAME".xes-model[s_"$SUPPORT"_c_"$CONFIDENCE"].json"
#MODEL=$INPUT_FOLDER"/"$LOG_NAME"-model[GROUND-TRUTH].json"
#MODEL=$INPUT_FOLDER"/"$LOG_NAME"-model[PARTICIPATION].json"
#MODEL=$INPUT_FOLDER"/"$LOG_NAME"-model[ABSENCE].json"
MODEL_ENCODING="json"

SPLIT_THRESHOLD=1.0
LOG_80_FIT=$INPUT_FOLDER"/"$LOG_NAME"-log[80-fit].xes"
LOG_20_DIVERGENT=$INPUT_FOLDER"/"$LOG_NAME"-log[20-divergent].xes"

OUTPUT_CHECK_CSV=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output.csv"
OUTPUT_CHECK_JSON=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output.json"
OUTPUT_TRACE_MEASURES_CSV=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output[tracesMeasures].csv"
OUTPUT_TRACE_MEASURES_STATS_CSV=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output[tracesMeasuresStats].csv"
OUTPUT_LOG_MEASURES_CSV=$PREPROCESSED_DATA_FOLDER"/"$LOG_NAME"-output[logMeasures].csv"

##################################################################
# SCRIPT
##################################################################
#
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
  java -cp Janus.jar $SIMPLIFIER_MAINCLASS -iMF $MODEL -iME $MODEL_ENCODING -oJSON $MODEL -s 0 -c 0 -i 0 -prune hierarchyconflictredundancydouble

fi
# Retrieve measure
echo "################################ MEASURE"
if test -f "${OUTPUT_TRACE_MEASURES_CSV}"; then
  echo "$OUTPUT_TRACE_MEASURES_CSV already exists."
else
  java -cp Janus.jar $JANUS_CHECK_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -iMF $MODEL -iME $MODEL_ENCODING -oCSV $OUTPUT_CHECK_CSV -d none -nanLogSkip -measure "Confidence"
#  java -cp Janus.jar $JANUS_CHECK_MAINCLASS -iLF $INPUT_LOG -iLE $LOG_ENCODING -iMF $MODEL -iME $MODEL_ENCODING -oCSV $OUTPUT_CHECK_CSV -oJSON $OUTPUT_CHECK_JSON -d none -nanLogSkip
# 'Lift','Confidence','Relative Risk'
# 'Support','all','Compliance,'Added Value','J Measure'
# 'Recall','Lovinger','Specificity','Accuracy','Leverage','Odds Ratio', 'Gini Index','Certainty factor','Coverage','Prevalence',
# 'Jaccard','Ylue Q','Ylue Y','Klosgen','Conviction','Interestingness Weighting Dependency','Collective Strength','Laplace Correction',
# 'One-way Support','Two-way Support','Two-way Support Variation',
# 'Linear Correlation Coefficient','Piatetsky-Shapiro','Cosine','Information Gain','Sebag-Schoenauer','Least Contradiction','Odd Multiplier','Example and Counterexample Rate','Zhang'}.
fi

# PARETO 80:20 split of the log according to the model
echo "################################ PARETO PRE-SPLIT"
#python3 -m ClusterMind.utils.split_log_according_to_declare_model $INPUT_LOG $OUTPUT_TRACE_MEASURES_CSV $SPLIT_THRESHOLD $LOG_80_FIT $LOG_20_DIVERGENT
python3 -m ClusterMind.pareto_declarative_hierarchical_clustering $INPUT_LOG $RESULTS_FOLDER"/" $PARETO_SPLIT_THRESHOLD "Janus.jar" $SIMPLIFICATION_FLAG $MIN_LEAF_SIZE

# Build decision-Tree
echo "################################ DECLARE TREES"
python3 -m DeclareTrees.declare_trees_for_clusters "$RESULTS_FOLDER""/aggregated_result.csv" $CONSTRAINTS_THRESHOLD "$RESULT_DECLARE_TREE" $BRANCHING_POLICY $MINIMIZATION_FLAG $BRANCHING_ORDER_DECREASING_FLAG

echo "################################ DECISION TREES"
python3 -m DeclareTrees.decision_trees_for_clusters \
  "$RESULTS_FOLDER"/traces-labels.csv \
  "$OUTPUT_TRACE_MEASURES_CSV" \
  "$RESULTS_FOLDER"/decision_tree.dot \
  1 \
  ${SPLIT_POLICY}
