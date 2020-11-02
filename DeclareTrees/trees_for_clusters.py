import sys
from DeclareTrees.declare_trees import build_declare_tree

if __name__ == '__main__':
    print(sys.argv)
    clusters_file = sys.argv[1]
    constraints_threshold = float(sys.argv[2])
    output_file = sys.argv[3]
    build_declare_tree(clusters_file, constraints_threshold, output_file)
