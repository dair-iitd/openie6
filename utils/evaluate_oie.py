import argparse
from pytorch_lightning import Trainer
from model import set_seed
import params
from torch.utils.data import DataLoader
import utils
from run import predict, test
import pickle
from metric import Carb
import os

CARB_TEST = 'carb/data/test.txt'
TEMP_FILE = 'carb/data/test_conj_split.txt'
CARB_DATA_DIR = 'carb/data'
CARB_CONJ_TEST = 'carb/data/split_test.txt'
MAPPING_FILE = 'mapping.p'

def create_test_file(file):
    with open(file, 'r') as f:
        lines = f.read()
        lines = lines.replace("\\", "")

    mapping = {}
    sentences = []
    for line in lines.split('\n\n')[:-1]:
        list_sentences = line.strip().split('\n')
        if len(list_sentences) == 1:
            mapping[list_sentences[0]] = list_sentences[0]
            sentences.append(list_sentences[0])
        elif len(list_sentences) > 1:
            for sent in list_sentences[1:]:
                mapping[sent] = list_sentences[0]
                sentences.append(sent)
        else:
            assert False

    with open(CARB_CONJ_TEST, 'w') as f:
        for sent in sentences:
            f.write(sent + ' [unused1] [unused2] [unused3]')
            f.write('\n')
        f.write('\n')
    return mapping


def main(hparams):

    train_dataset, val_dataset, test_dataset, meta_data_vocab, pos_vocab, verb_vocab = utils.process_data(hparams)
    test_dataloader = DataLoader(test_dataset, batch_size=hparams.batch_size, collate_fn=utils.pad_data, num_workers=1)

    predict(hparams, None, meta_data_vocab, None, None, None, test_dataloader, pos_vocab, verb_vocab)

    temp_files = [f for f in os.listdir(CARB_DATA_DIR) if f.startswith('predictions_')]
    assert len(temp_files) == 1


    carb_test_conj_file = CARB_DATA_DIR + '/' + temp_files[0]

    mapping = create_test_file(carb_test_conj_file)

    pickle.dump(mapping, open(MAPPING_FILE, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = params.add_args(parser)
    hyperparams = parser.parse_args()
    set_seed(hyperparams.seed)

    main(hyperparams)
