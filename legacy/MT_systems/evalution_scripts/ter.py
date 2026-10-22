import argparse
from metrics import Ter

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='ESTIMATE SacreBLEU SCORE')
    parser.add_argument("hypothesis",
                        help="Provide The Path To The Hypothesis file", type=str)
    parser.add_argument("reference",
                        help="Provide The Path To The Reference file", type=str)

    args = parser.parse_args()
    ter_score= Ter()
    print(ter_score.get_ter(args.hypothesis, args.reference))
