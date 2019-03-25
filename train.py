from model.data_utils import FilipinoPOSDataset
from model.tagger_model import TaggerModel
from model.config import Config

def main():
    config = Config()

    #build model
    model = TaggerModel(config)
    print(model.idx_to_tag)
    model.build()
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    dev = FilipinoPOSDataset(config.filename_dev, config.processing_word,
                       config.processing_tag, config.max_iter)
    train = FilipinoPOSDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    model.train(train, dev)

if __name__ == "__main__":
    main()
