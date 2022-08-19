import os

DATA_DIR = r"input\archive_1\cityscapes_data"
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'val')

TRAIN_FILES = os.listdir(TRAIN_DIR)
TEST_FILES = os.listdir(TEST_DIR)

CROP_SIZE = (128, 128)
N_EPOCHS = 1
BATCH_SIZE = 8
WEIGHTS_PATH = r"models\best.pth"
