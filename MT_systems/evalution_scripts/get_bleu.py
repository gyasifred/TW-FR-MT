import argparse
from metrics import BLEU, AzunreBleu
from nltk.translate.bleu_score import SmoothingFunction

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='ESTIMATE BLEU SCORE')
    parser.add_argument("hypothesis",
                        help="Provide The Path To The Hypothesis file", type=str)
    parser.add_argument("reference",
                        help="Provide The Path To The Reference file", type=str)
    parser.add_argument("--azunre", default=False,
                        help="Specify if  you want to estimate AzunreBLEU, default estimate regular n-grams BLEU Score", type=bool)

    args = parser.parse_args()
    smooth = SmoothingFunction()

    if args.azunre == True:
        bleu = AzunreBleu()
        print(bleu.get_bleuscore(args.hypothesis, args.reference, smooth.method7))
    else:
        bleu = BLEU()
        print(bleu.get_bleuscore(args.hypothesis, args.reference, smooth.method7))
