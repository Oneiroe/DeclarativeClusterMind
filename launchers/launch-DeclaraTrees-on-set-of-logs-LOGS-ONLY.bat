@echo off

REM E:  #REM if the drive where the data is different form the default one
REM ##################################################################
REM # PARAMETERS
REM ##################################################################
set INPUT_LOGS_FOLDER="PATH_TO_INPUT_LOGS_FOLDER"
set RESULTS_FOLDER="PATH_TO_RESULTS_FOLDER"

set CONSTRAINTS_TEMPLATE_BLACKLIST=%INPUT_LOGS_FOLDER%\blacklist.csv
set CONSTRAINTS_TASKS_BLACKLIST=%INPUT_LOGS_FOLDER%\blacklist-tasks.csv

REM # PATHs
set WORKING_DIR=E:\Phd\my_code\DeclarativeClusterMind
cd %WORKING_DIR% || exit
set JAVA_BIN=java
set JANUS_JAR=E:\Phd\my_code\DeclarativeClusterMind\Janus.jar
set MINERFUL_JAR=E:\Phd\my_code\DeclarativeClusterMind\MINERful.jar
set PYTHON_BIN=python

REM ##################################################################

REM # Janus\MINERful main classes
set JANUS_MEASURES_MAINCLASS=minerful.JanusMeasurementsStarter
set JANUS_DISCOVERY_MAINCLASS=minerful.JanusOfflineMinerStarter
set JANUS_DISCOVERY_SUPPORT=0.0
set JANUS_DISCOVERY_CONFIDENCE=0.9
set MINERFUL_DISCOVERY_MAINCLASS=minerful.MinerFulMinerStarter
set MINERFUL_DISCOVERY_SUPPORT=0.9
set MINERFUL_DISCOVERY_CONFIDENCE=0.0
set SIMPLIFIER_MAINCLASS=minerful.MinerFulSimplificationStarter

REM # Discovery & Measurements
set LOG_ENCODING=xes
set MODEL=%RESULTS_FOLDER%\model.json
set MODEL_ENCODING=json

REM # DECLRE-Tree
set MINIMIZATION_FLAG=-min
set BRANCHING_ORDER_DECREASING_FLAG=-decreasing
set CONSTRAINTS_THRESHOLD=0.9
set BRANCHING_POLICY=dynamic-variance
set MULTI_PERSPECTIVE_FEATURES=mixed
REM # 'rules'
REM # 'attributes'
REM # 'specific-attribute'
REM # 'performances'
REM # 'mixed'

REM ##################################################################
REM # SCRIPT
REM ##################################################################
REM #
REM #experiment folders
md %RESULTS_FOLDER%

REM #Logs attributes and Performances
REM #OPT visualize performance boxplot
echo ################################ DESCRIPTIVE Stats
%PYTHON_BIN% -m DeclarativeClusterMind.cli_evaluation stats   -iLf %INPUT_LOGS_FOLDER%   -o %RESULTS_FOLDER%\clusters-stats.csv
%PYTHON_BIN% -m DeclarativeClusterMind.cli_evaluation performances   -iLf %INPUT_LOGS_FOLDER%   -o %RESULTS_FOLDER%\performance_boxplot.svg

REM #Discover process models for each input log (if not existing)
REM #  Models are stored in the input logs folder
echo ################################ CLUSTERS MODEL DISCOVERY
for %%I in (%INPUT_LOGS_FOLDER%\*.xes) do (
  echo %%I
  if exist %%I_model.json (
    echo %%I_model.json already exists.
  ) else (
    REM #   Discovery with Janus
    REM %JAVA_BIN% -cp %JANUS_JAR% %JANUS_DISCOVERY_MAINCLASS% -iLF %%I -iLE %LOG_ENCODING% -c %JANUS_DISCOVERY_CONFIDENCE% -s %JANUS_DISCOVERY_SUPPORT% -i 0 -oJSON %%I_model.json -vShush
    REM #   Discovery with MINERful
    %JAVA_BIN% -cp %MINERFUL_JAR% %MINERFUL_DISCOVERY_MAINCLASS% -iLF %%I -iLE %LOG_ENCODING% -c %MINERFUL_DISCOVERY_CONFIDENCE% -s %MINERFUL_DISCOVERY_SUPPORT% -oJSON %%I_model.json -vShush

    REM #Filter undesired templates,
    if exist %CONSTRAINTS_TEMPLATE_BLACKLIST% (
      %PYTHON_BIN% -m DeclarativeClusterMind.utils.filter_json_model %%I_model.json %CONSTRAINTS_TEMPLATE_BLACKLIST% %%I_model.json
    )
    REM #Filter any rule involving undesired tasks
    if exist %CONSTRAINTS_TASKS_BLACKLIST% (
      %PYTHON_BIN% -m DeclarativeClusterMind.utils.filter_json_model %%I_model.json %CONSTRAINTS_TASKS_BLACKLIST% %%I_model.json
    )

    REM #   # Simplify model, i.e., remove redundant constraints
    REM #   echo ################################ SIMPLIFICATION
    REM #   %JAVA_BIN% -cp %JANUS_JAR% %SIMPLIFIER_MAINCLASS% -iMF %%I_model.json -iME %MODEL_ENCODING% -oJSON %%I_model.json -s 0 -c 0 -i 0 -prune hierarchyconflictredundancydouble
  )
)

REM #Retrieve measures for each cluster
echo ################################ LOGS MEASURES
REM #merge process models
%PYTHON_BIN% -m DeclarativeClusterMind.utils.merge_models %INPUT_LOGS_FOLDER% _model.json %MODEL%

for %%I in (%INPUT_LOGS_FOLDER%\*.xes) do (
  echo %%I
  echo %%I-output[logMeasures].csv
  if exist %%I-output[logMeasures].csv (
    echo %%I-output[logMeasures].csv already exists.
  ) else (
    %JAVA_BIN% -cp %JANUS_JAR% %JANUS_MEASURES_MAINCLASS% -iLF %%I -iLE %LOG_ENCODING% -iMF %MODEL% -iME %MODEL_ENCODING% -oCSV %%I-output.csv -d none -detailsLevel log -measure Confidence
  )
)
REM #Aggregate obtained measure in one unique matrix
%PYTHON_BIN% -m DeclarativeClusterMind.utils.aggregate_clusters_measures %INPUT_LOGS_FOLDER% -output[logMeasures].csv %RESULTS_FOLDER%\aggregated_result.csv
%PYTHON_BIN% -m DeclarativeClusterMind.utils.label_clusters_with_measures %INPUT_LOGS_FOLDER% -output[logMeasures].csv %RESULTS_FOLDER%\clusters-labels.csv

REM #Build decision-Trees
echo ################################ DeclaraTrees Clusters
%PYTHON_BIN% -m DeclarativeClusterMind.cli_decision_trees simple-tree-logs-to-clusters -i "%RESULTS_FOLDER%\aggregated_result.csv" -o "%RESULTS_FOLDER%\DeclareTree-LOGS.dot" -t "%CONSTRAINTS_THRESHOLD%" -p %BRANCHING_POLICY% %MINIMIZATION_FLAG% %BRANCHING_ORDER_DECREASING_FLAG%

echo ################################ CART DECISION TREES clusters
REM #If rules: -i clusters-labels.csv and -m None
REM #If attributes\performance: -i clusters-stats.csv and -m None
REM #If mixed: -i clusters-stats.csv and -m clusters-labels.csv
%PYTHON_BIN% -m DeclarativeClusterMind.cli_decision_trees decision-tree-logs-to-clusters -i %RESULTS_FOLDER%\clusters-stats.csv -o %RESULTS_FOLDER%\decision_tree_logs.dot -p %MULTI_PERSPECTIVE_FEATURES% -m %RESULTS_FOLDER%\clusters-labels.csv -fi 0
