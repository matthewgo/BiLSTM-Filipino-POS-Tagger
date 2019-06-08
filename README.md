Based on BiLSTM Sequence Tagging Tensorflow implementation of Guillaume Genthial:
https://github.com/guillaumegenthial/sequence_tagging


##BiLSTM + Conditional Random Fields + Char Embeddings + FastText Word Vectors 
####For NER, POS Tagging or any Sequence Tagging Tasks.
Some data paths in model/config.py and oov_generator.sh is currently hardcoded to work with Filipino data tested in my local machine.

**Requirements:**
- Clone fastText and update FASTTEXT file path in oov_generator.sh with the installation folder
- Download the binary file of the word vectors of the language you intend to use from https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md

Has three runnable Python scripts that you should run in order:
- *build_data.py* - loads the vocabulary (list of word in dataset), generate word vectors of out-of-vocab words, loads word vectors of words in fastText word vectors, and keeps a trimmed copy of all these word vectors for faster processing.
- *train.py* - feeds the train and dev set to our BiLSTM + CRF network
- *evaluate.py* - reads the test set and output model's accuracy. Also has an interactive interface to test out sentences to be tagged.

**_Sample data files to be uploaded soon._**
