# Twi-French Machine Translation
French is a strategically and economically important language in the regions where the African language Twi is spoken. However, only a very small proportion of Twi speakers in Ghana speak French. The development of a Twi–French parallel corpus and corresponding machine translation applications would provide various advantages, including stimulating trade and job creation, supporting the Ghanaian diaspora in French-speaking nations, assisting French-speaking tourists and immigrants seeking medical care in Ghana, and facilitating numerous downstream natural language processing tasks. Since there are hardly any machine translation systems or parallel corpora between Twi and French that cover a modern and versatile vocabulary, our goal was to extend a modern Twi–English corpus with French and develop machine translation systems between Twi and French: Consequently, in this paper, we present our Twi–French corpus of 10,708 parallel sentences. Furthermore, we describe our machine translation experiments with this corpus. We investigated direct machine translation and cascading systems that use English as a pivot language. Our best Twi–French system is a direct state-of-the-art transformer-based machine translation system that achieves a BLEU score of 0.76. Our best French–Twi system, which is a cascading system that uses English as a pivot language, results in a BLEU score of 0.81. Both systems are fine tuned with our corpus, and our French–Twi system even slightly outperforms Google Translate on our test set by 7% relative.


This repository includes:
1. A script for fine-tuning a pre-trained OPUS-MT model ([fine_tune_opus.py](https://github.com/gyasifred/TW-FR-MT/blob/main/MT_systems/opus/fine_tune_opus.py)).
2. Scripts for translations using fine-tuned OPUS-MT models ([opus_direct_translate.py](https://github.com/gyasifred/TW-FR-MT/blob/main/MT_systems/opus/opus_direct_translate.py), [opus_pivot_translate.py](https://github.com/gyasifred/TW-FR-MT/blob/main/MT_systems/opus/opus_pivot_translate.py)).
3. Scripts to call the Google Translate API for translation ([googleAPIdirect_translate.py](https://github.com/gyasifred/TW-FR-MT/blob/main/MT_systems/Google_MT/googleAPIdirect_translate.py), [googleAPIpivot_translate.py](https://github.com/gyasifred/TW-FR-MT/blob/main/MT_systems/Google_MT/googleAPIpivot_translate.py)).
4. Scripts for evaluating translation quality ([get_bleu.py](https://github.com/gyasifred/TW-FR-MT/blob/main/MT_systems/evalution_scripts/get_bleu.py), [get_sacrebleu.py](https://github.com/gyasifred/TW-FR-MT/blob/main/MT_systems/evalution_scripts/get_sacrebleu.py), and [ter.py](https://github.com/gyasifred/TW-FR-MT/blob/main/MT_systems/evalution_scripts/ter.py)).
5. [TW-FR-EN Corpus](https://github.com/gyasifred/TW-FR-MT/tree/main/TW_FR_EN_corpus).


# Quickstart
We provide a [tutorials](https://github.com/gyasifred/TW-FR-MT/tree/main/tutorials) which contains the complete pipeline from fine-tuning to evaluation.
# Fine-tune OPUS-MT models

# Citation
BibTex

@Article{bdcc7020114,
AUTHOR = {Gyasi, Frederick and Schlippe, Tim},
TITLE = {Twi Machine Translation},
JOURNAL = {Big Data and Cognitive Computing},
VOLUME = {7},
YEAR = {2023},
NUMBER = {2},
ARTICLE-NUMBER = {114},
URL = {https://www.mdpi.com/2504-2289/7/2/114},
ISSN = {2504-2289},
DOI = {10.3390/bdcc7020114}
}
