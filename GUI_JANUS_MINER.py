from gooey import Gooey, GooeyParser

JANUS_DISCOVERY_COMMAND_LINE = lambda JANUS_JAR_PATH, INPUT_LOG, CONFIDENCE, SUPPORT, MODEL: \
    f"java -cp {JANUS_JAR_PATH} minerful.JanusOfflineMinerStarter -iLF {INPUT_LOG} -iLE xes -c {CONFIDENCE} -s {SUPPORT} -i 0 -oJSON {MODEL}"
MINERFUL_SIMPLIFIER_COMMAND_LINE = lambda JANUS_JAR_PATH, MODEL: \
    f"java -cp {JANUS_JAR_PATH} minerful.MinerFulSimplificationStarter -iMF {MODEL} -iME json -oJSON {MODEL} -s 0 -c 0 -i 0 -prune hierarchyconflictredundancydouble"
JANUS_MEASUREMENT_COMMAND_LINE = lambda JANUS_JAR_PATH, INPUT_LOG, MODEL, OUTPUT_CHECK_CSV, MEASURE: \
    f"java -cp {JANUS_JAR_PATH} minerful.JanusMeasurementsStarter -iLF {INPUT_LOG} -iLE xes -iMF {MODEL} -iME json -oCSV {OUTPUT_CHECK_CSV} -d none -nanLogSkip -measure {MEASURE}"


@Gooey(target="java", program_name='Janus Miner', suppress_gooey_flag=True, requires_shell=True, clear_before_run=True,
       progress_regex=r"^Traces: (?P<current>\d+)/(?P<total>\d+)$", progress_expr="current / total * 100",
       hide_progress_msg=False
       )
def main():
    parser = GooeyParser(description="Janus declarative rules discovery")
    miner = parser.add_argument_group('Offline Miner')
    miner.add_argument('-cp',
                       metavar='Jar',
                       help='Janus JAR path',
                       # nargs='?',
                       widget='FileChooser',
                       default='Janus.jar'
                       )
    miner.add_argument(metavar='module',
                       help='Janus module',
                       default='minerful.JanusOfflineMinerStarter',
                       action='append',
                       nargs='?',
                       dest='cp',
                       widget='FileChooser',
                       # gooey_options={'visible': False}
                       )
    miner.add_argument('-iLF',
                       metavar='Input Log',
                       help='Path to input event log file',
                       required=True,  # Look at me!
                       widget='FileChooser'
                       )
    miner.add_argument('-oJSON',
                       metavar='Ouput Model',
                       help='Path for output JSON model',
                       required=True,  # Look at me!
                       widget='FileSaver'
                       )
    miner.add_argument('-iLE',
                       metavar='iLE',
                       help='Input log encoding',
                       # required=True,  # Look at me!
                       type=str,
                       default='xes',
                       gooey_options={'visible': False}
                       )
    miner.add_argument('-s',
                       metavar='Support',
                       default=0.0,
                       # required=True,  # Look at me!
                       help='Support threshold',
                       widget='DecimalField',
                       gooey_options={'min': 0.0, 'max': 1.0})
    miner.add_argument('-c',
                       metavar='Confidence',
                       default=0.8,
                       # required=True,  # Look at me!
                       help='Confidence threshold',
                       widget='DecimalField',
                       gooey_options={'min': 0.0, 'max': 1.0})

    parser.parse_args()
    print(parser.parse_args())


if __name__ == '__main__':
    main()
