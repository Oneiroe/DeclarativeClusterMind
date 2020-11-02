import sys
import ClusterMind.IO.SJ2T_import as cmio
from DeclareTrees.declare_trees import build_declare_tree

if __name__ == '__main__':
    print(sys.argv)
    sj2t_file = sys.argv[1]
    constraints_threshold = float(sys.argv[2])
    preprocessed_file = sys.argv[3]
    output_file = sys.argv[4]

    # Pre-prcessing
    cmio.extract_detailed_perspective(sj2t_file, preprocessed_file)

    # Declare tree
    build_declare_tree(preprocessed_file, constraints_threshold, output_file)
