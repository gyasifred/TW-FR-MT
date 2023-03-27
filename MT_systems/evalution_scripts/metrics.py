import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import sacrebleu
import evaluate

ter = evaluate.load("ter")


class BLEU():
    '''This class help estimate the BLEU Score (Papineni et al.,2002) using the NLTK library
    Args:
    param hypothesisfile: Path to the hypothesis document
    param referencefile: Path to the reference document(s)
    param smothingfunction: specify the smothing function to use. Default is none 
    '''

    def __init__(self):
        pass

    def get_bleuscore(self, hypothesisfile, referencefile, smothingfunction=None):
        if type(hypothesisfile) == str and type(referencefile) == str:
            # Open test file and read lines
            f = open(hypothesisfile, "r")
            hypothesis = f.readlines()
            f.close()
            # open refernce file and read lines
            f = open(referencefile, "r")
            reference = f.readlines()
            f.close()
        elif type(hypothesisfile) == list and type(referencefile) == list:
            hypothesis = hypothesisfile
            reference = referencefile
        else:
            print(f'File must be txt or python list')

        # check the length of our input sentence
        length = len(hypothesis)

        bleu_total = np.array([0., 0., 0.])
        # initialised the weights for 1-gram, 2-grams, 3-grams, and 4-grams
        weights = [(1./2, 1./2), (1./3, 1./3, 1./3),
                   (1./4, 1./4, 1./4, 1./4)]
        for i in range(length):
            hypothesis[i] = hypothesis[i]
            reference[i] = reference[i]
            groundtruth = reference[i].strip().lower().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                .replace(" ) ", ") ").replace(" , ", ", ").split()
            groundtruth = [groundtruth]

            candidate = hypothesis[i].strip().lower().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                .replace(" ) ", ") ").replace(" , ", ", ").split()
            # print("Translated Text: ", candidate)
            # print("Ground Truth: ", groundtruth)
            bleu = np.array(sentence_bleu(
                groundtruth, candidate, weights, auto_reweigh=True, smoothing_function=smothingfunction))
            bleu_total += bleu

        return f'2-GRAMS: {bleu_total[0]/length:.2f}', f'3-GRAMS: {bleu_total[1]/length:.2f}', f'4-GRAMS: {bleu_total[2]/length:.2f}'


class AzunreBleu():
    '''This class help estimate the BLEU Score (Papineni et al.,2002) using the NLTK library as implemented by Azunre et al. (2021)
    Args:
    param hypothesisfile: Path to the hypothesis document
    param referencefile: Path to the reference document(s)
    param smothingfunction: specify the smothing function to use. Default is none 
    '''

    def __init__(self):
        pass

    def get_bleuscore(self, hypothesisfile, referencefile, smothingfunction=None):
        if type(hypothesisfile) == str and type(referencefile) == str:
            # Open test file and read lines
            f = open(hypothesisfile, "r")
            hypothesis = f.readlines()
            f.close()
            # open refernce file and read lines
            f = open(referencefile, "r")
            reference = f.readlines()
            f.close()
        elif type(hypothesisfile) == list and type(referencefile) == list:
            hypothesis = hypothesisfile
            reference = referencefile
        else:
            print(f'File must be text file or python list')

        # check the length of our input sentence
        length = len(hypothesis)
        bleu_total = 0
        weights = (0.58, 0, 0, 0)
        for i in range(length):
            hypothesis[i] = hypothesis[i]
            reference[i] = reference[i]
            groundtruth = reference[i].strip().lower().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                .replace(" ) ", ") ").replace(" , ", ", ").split()
            groundtruth = [groundtruth]

            candidate = hypothesis[i].strip().lower().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                .replace(" ) ", ") ").replace(" , ", ", ").split()
            # print("Translated Text: ",candidate)
            # print("Ground Truth: ",groundtruth)
            bleu = sentence_bleu(
                groundtruth, candidate, weights, smoothing_function=smothingfunction,
                auto_reweigh=True)
            bleu_total += bleu

        return f'BLEU SCORE: {bleu_total/length:.2f}'


class Sacrebleu():
    '''This class help estimate the BLEU Score (Post,2018) using the sacrebleu package
    Args:
    param hypothesisfile: Path to the hypothesis document
    param referencefile: Path to the reference document(s)
    '''

    def __init__(self) -> None:
        pass

    def get_bleuscore(self, hypothesisfile, referencefile):

        # Open test file and read translate
        preds = []
        with open(hypothesisfile) as pred:
            for line in pred:
                line = line.strip().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                    .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                    .replace(" ) ", ") ").replace(" , ", ", ")
                preds.append(line.strip().lower())

        refs = []
        with open(referencefile) as test:
            for line in test:
                line = line.strip().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                    .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                    .replace(" ) ", ") ").replace(" , ", ", ")
                refs.append(line.strip().lower())
        refs = [refs]
        bleu = sacrebleu.corpus_bleu(preds, refs, smooth_method="add-k",
                                     force=False,
                                     lowercase=True,
                                     tokenize="intl",
                                     use_effective_order=True)
        return f'SacreBLEU SCORE: {bleu.score:.2f}'


class Ter():
    '''Estimate the Translation Error Rate (TER) (Snover et al., 2006)
    Args:
    param hypothesisfile: Path to the hypothesis document
    param referencefile: Path to the reference document(s)
    '''

    def __init__(self) -> None:
        pass

    def get_ter(self, hypothesisfile, referencefile):

        # Open test file and read translate
        preds = []
        with open(hypothesisfile) as pred:
            for line in pred:
                line = line.strip().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                    .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                    .replace(" ) ", ") ").replace(" , ", ", ")
                preds.append(line.strip().lower())

        refs = []
        with open(referencefile) as test:
            for line in test:
                line = line.strip().replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                    .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                    .replace(" ) ", ") ").replace(" , ", ", ")
                refs.append(line.strip().lower())
        refs = [refs]
        result = ter.compute(predictions=preds, references=refs)
        return f'TER SCORE: {result.score:.2f}'
