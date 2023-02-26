import argparse
from metrics import Sacrebleu

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='ESTIMATE SacreBLEU SCORE')
    parser.add_argument("hypothesis",
                        help="Provide The Path To The Hypothesis file", type=str)
    parser.add_argument("reference",
                        help="Provide The Path To The Reference file", type=str)

    args = parser.parse_args()
    bleu = Sacrebleu()
    print(bleu.get_bleuscore(args.hypothesis, args.reference))
