import os

BASE_PATH = os.path.dirname(__file__)

# TRAIN_SAMPLE_PATH = os.path.join(BASE_PATH, '')
# TEST_SAMPLE_PATH = os.path.join(BASE_PATH, '.')
# LABEL_PATH = os.path.join(BASE_PATH, '')

TRAIN_SAMPLE_PATH = os.path.join(BASE_PATH, '')
TEST_SAMPLE_PATH = os.path.join(BASE_PATH, '')

LABEL_PATH = os.path.join(BASE_PATH, '')



BERT_PAD_ID = 0
TEXT_LEN = 50

BATCH_SIZE = 20

BERT_MODEL = os.path.join(BASE_PATH, '')
MODEL_DIR = os.path.join(BASE_PATH, '')

EMBEDDING_DIM = 512
NUM_FILTERS = 256
NUM_CLASSES = 
FILTER_SIZES = [2, 3, 4]

EPOCH = 100
LR = 1e-5

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    print(torch.tensor([1,2,3]).to(DEVICE))