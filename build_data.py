from model.config import Config
from model import data_utils
from model.data_utils import FilipinoPOSDataset, get_fasttext_vocab, get_vocabs, UNK, NUM, write_vocab, load_vocab,\
                            export_trimmed_fasttext_vectors, get_char_vocab

def main():
    config = Config(load=False)
    processing_word = data_utils.get_processing_word(lowercase=True)

    #Datasets
    test = FilipinoPOSDataset(config.filename_test, processing_word=processing_word)
    dev = FilipinoPOSDataset(config.filename_dev, processing_word=processing_word)
    train = FilipinoPOSDataset(config.filename_train, processing_word=processing_word)

    # Vocab Generators
    vocab_words, vocab_tags = get_vocabs([train, dev, test])
    vocab_fasttext = get_fasttext_vocab(config.filename_fasttext)

    #Build Word and Tag Vocab
    if config.use_fasttext_oov_vector_gen:
        vocab = vocab_words
    else:
        vocab = vocab_words & vocab_fasttext
    vocab.add(UNK)
    vocab.add(NUM)

    # Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    #Trim and (insert new) Fasttext vectors
    word_to_idx, idx_to_word = load_vocab(config.filename_words)
    export_trimmed_fasttext_vectors(word_to_idx, idx_to_word, config.filename_fasttext,
                                config.filename_fasttext_trimmed, config.dim_word, config.use_fasttext_oov_vector_gen)

    # Build and save char vocab
    train = FilipinoPOSDataset(config.filename_train)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.filename_chars)

if __name__ == "__main__":
    main()