# Twi-French Machine Translation
French is a strategically and economically important language in the regions where the African language Twi is spoken. But only a very small proportion of Twi speakers in Ghana speak French. Since there are hardly any machine translation systems or parallel corpora between Twi and French that cover a modern and versatile vocabulary, our goal was to extend the modern [English-Akuapem Twi](https://zenodo.org/record/4432117#.Y_hwwdLP1Nj) corpus of [Azunre et al., 2021](https://arxiv.org/abs/2103.15625) with French and develop machine translation systems between Twi and French.


In this repository, we show how to fine-tune pre-trained [OPUS-MT](https://github.com/Helsinki-NLP/Opus-MT) models for Twi-to-French machine translation.


This repository includes:
1. A script for fine-tuning a pre-trained OPUS-MT model ([fine_tune_opus.py](https://github.com/gyasifred/TW-FR-MT/blob/main/MT_systems/opus/fine_tune_opus.py)).
2. Scripts for translations using fine-tuned OPUS-MT models ([opus_direct_translate.py](https://github.com/gyasifred/TW-FR-MT/blob/main/MT_systems/opus/opus_direct_translate.py), [opus_pivot_translate.py](https://github.com/gyasifred/TW-FR-MT/blob/main/MT_systems/opus/opus_pivot_translate.py)).
3. Scripts to call the Google Translate API for translation ([googleAPIdirect_translate.py](https://github.com/gyasifred/TW-FR-MT/blob/main/MT_systems/Google_MT/googleAPIdirect_translate.py), [googleAPIpivot_translate.py](https://github.com/gyasifred/TW-FR-MT/blob/main/MT_systems/Google_MT/googleAPIpivot_translate.py)).
4. Scripts for evaluating translation quality ([get_bleu.py](https://github.com/gyasifred/TW-FR-MT/blob/main/MT_systems/evalution_scripts/get_bleu.py), [get_sacrebleu.py](https://github.com/gyasifred/TW-FR-MT/blob/main/MT_systems/evalution_scripts/get_sacrebleu.py)).
5. [TW-FR-EN Corpus](https://github.com/gyasifred/TW-FR-MT/tree/main/TW_FR_EN_corpus).


# Quickstart
We provide a [tutorials](https://github.com/gyasifred/TW-FR-MT/tree/main/tutorials) which contains the complete pipeline from fine-tuning to evaluation.

# Contributing
Please first clone this repo to your local machine, using a command line tool such as Cygwin or Anaconda Prompt:

```
git clone https://github.com/gyasifred/TW-FR-MT
```

Create a branch for your contributions, and check it out:

```
git branch <your-branch-name>
```

```
git checkout <your-branch-name>
```

Try to pick a branch name that describes your contribution.

Write your code, test it and then push to your branch:

```
git push origin <your-branch-name>
```

Create a pull request using the online GitHub repo page.

