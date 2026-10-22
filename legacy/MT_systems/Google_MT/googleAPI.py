from googletrans import Translator, constants
# import nltk
# import numpy as np
# from nltk.translate.bleu_score import sentence_bleu
# import sacrebleu

# Use google API for bidirectional pivot translation of Twi and French
# pivot language = English
# instantiate a translator object
# initiate translator object
translator = Translator()
# Add Akan to the language supported by this package
# Note the googletrans package has not  been updated to capture the new additions by google since May 2022
# from https://translate.google.com/?sl=en&tl=ak&op=translate , the key and value for Twi is 'ak':'akan'
constants.LANGUAGES['ak'] = 'akan'


class GooglePivot:
    '''Perform Pivot Translation With Google Translate
    Args:
        param sentence: Sentence to be translated
        param src_key: The source language key
        param dest_key: The target language key
    '''
    def __init__(self):
        pass

    def evaluate(self, sentence, src_key, dest_key):
        eng_text = translator.translate(sentence, src=src_key, dest='en').text
        output = translator.translate(eng_text, dest=dest_key).text

        return output


class GoogleDirect:
    '''Perform Direct Translation With Google Translate
    Args:
        param sentence: Sentence to be translated
        param src_key: The source language key
        param dest_key: The target language key
    '''
    def __init__(self):
        pass

    def evaluate(self, sentences, src_key, dest_key):

        return translator.translate(sentences, src=src_key, dest=dest_key).text
