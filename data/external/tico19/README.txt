TICO-19 data directory
=======================

The pipeline downloads the English-Kiswahili TMX automatically on first use
(SRC/corpora/tico19.py -> download_tmx). If the automatic download fails
(e.g. restricted network), install the data manually:

1. Download the translation memory:
     https://tico-19.github.io/data/TM/all.en-sw.tmx.zip
2. Extract the zip INTO THIS DIRECTORY (data/external/tico19/).
   The file all.en-sw.tmx should be visible here afterwards.
   (The importer skips the download if a .tmx is already present.)

Optional — official benchmark dev/test text files:

3. Download https://tico-19.github.io/data/tico19-testset.zip
4. Extract it so the files land in a "bench" subdirectory, i.e.
     data/external/tico19/bench/TICO19.dev.en
     data/external/tico19/bench/TICO19.dev.sw
     data/external/tico19/bench/TICO19.test.en
     data/external/tico19/bench/TICO19.test.sw
   Then import_tico19(split="dev") / split="test" reads the line pairs.
   NOTE: which split is reserved for evaluation is a Week 2 decision.
   Never train on the evaluation split.

Attribution (required, license CC BY 4.0):
  TICO-19: Translation Initiative for COVID-19, https://tico-19.github.io/
  Anastasopoulos et al., "TICO-19: the Translation Initiative for COvid-19",
  Proceedings of the 1st Workshop on NLP for COVID-19 (Part 2) at EMNLP 2020.
