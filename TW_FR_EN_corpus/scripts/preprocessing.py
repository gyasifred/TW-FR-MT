# This code was adapted from https://github.com/GhanaNLP/kasa/blob/master/Kasa/Preprocessing.py
# A subclass of the kasafrench for preprocessing data
# import required library
import re
import unicodedata


class Preprocessing:
    # dummy initialization method
    def __init__(self):
        # initialize with some default parameters here later
        pass

    # read in parallel twi - english dataset
    def read_parallel_dataset(self, filepath_1, filepath_2, filepath_3=None):
        if filepath_3 != None:
            # read first language data
            lang_1 = []
            with open(filepath_1, encoding='utf-8') as file:
                line = file.readline()
                cnt = 1
                while line:
                    lang_1.append(line.strip())
                    line = file.readline()
                    cnt += 1

            # read second language data
            lang_2 = []
            with open(filepath_2, encoding='utf-8') as file:

                # twi=file.read()
                line = file.readline()
                cnt = 1
                while line:
                    lang_2.append(line.strip())
                    line = file.readline()
                    cnt += 1
            # Read third Language data
            lang_3 = []
            with open(filepath_3, encoding='utf-8') as file:
                line = file.readline()
                cnt = 1
                while line:
                    lang_3.append(line.strip())
                    line = file.readline()
                    cnt += 1

            return lang_1, lang_2, lang_3
            
        else:
            # read first language data
            lang_1 = []
            with open(filepath_1, encoding='utf-8') as file:
                line = file.readline()
                cnt = 1
                while line:
                    lang_1.append(line.strip())
                    line = file.readline()
                    cnt += 1

            # read second language data
            lang_2 = []
            with open(filepath_2, encoding='utf-8') as file:

                # twi=file.read()
                line = file.readline()
                cnt = 1
                while line:
                    lang_2.append(line.strip())
                    line = file.readline()
                    cnt += 1

            return lang_1, lang_2

    # Define a helper function to remove string accents

    def removeStringAccent(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # normalize input twi sentence
    def normalize_twi(self, s):
        #s = self.removeStringAccent(s)
        s = s.lower()
        s = re.sub(r'([!.?])', r' \1', s)
        # s = re.sub(r'[^a-zA-Z.ƆɔɛƐ!?’]+', r' ', s)
        s = re.sub(r'\s+', r' ', s)
        return s

    # normalize input french sentence
    def normalize_FrEn(self, s):
        #s = self.removeStringAccent(s)
        s = s.lower()
        s = re.sub(r'([!.?])', r' \1', s)
        # s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
        s = re.sub(r'\s+', r' ', s)
        return s
    
    def writeTotxt(self,destination,data):
        with open(f'{destination}', 'w') as f:
            for line in data:
                 f.write(f"{line}\n")
