import os
from root import ROOT_DIR


from .general_utils import get_logger
from model import data_utils


class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words, _ = data_utils.load_vocab(self.filename_words)
        self.vocab_tags, _ = data_utils.load_vocab(self.filename_tags)
        self.vocab_chars, _ = data_utils.load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = data_utils.get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag  = data_utils.get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (data_utils.get_trimmed_fasttext_vectors(self.filename_fasttext_trimmed)
                if self.use_pretrained else None)



    # general config
    dir_output = os.path.join(ROOT_DIR, "results/test-filipino-pos/")
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_word = 300
    dim_char = 100


    #fastText files
    fasttext_folder = os.path.join(ROOT_DIR, 'data/fasttext-filipino/')
    filename_fasttext= fasttext_folder + "cc.tl.300.vec"
    filename_fasttext_trimmed = fasttext_folder + "cc.tl.300.vec.trimmed.npz"
    use_pretrained = True

    # dataset
    folder = os.path.join(ROOT_DIR, "data/filipino-pos/")
    filename_dev = folder + 'dev.txt'
    filename_test = folder + 'test.txt'
    filename_train = folder + 'train.txt'


    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = folder + "words.txt"
    filename_oov_words = folder + "oov_words.txt"
    filename_tags = folder + "tags.txt"
    filename_chars = folder + "chars.txt"

    # training
    train_embeddings = False
    nepochs          = 15
    dropout          = 0.5
    batch_size       = 20
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 3

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True # if crf, training is 1.7x slower on CPU
    use_chars = True # if char embedding, training is 3.5x slower on CPU

    use_fasttext_oov_vector_gen = True

    filename_oov_result_vectors = folder + 'oov_vectors.vec'
