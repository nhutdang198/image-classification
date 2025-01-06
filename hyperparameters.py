from datetime import datetime
import pathlib

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
NUMBER_OF_CLASSES = 2
EARLY_STOPPING_PATIENCE = 10

now = datetime.now().strftime('%y-%m-%d:%H:%M')
day = datetime.now().strftime('%y-%m-%d')

CHECKPOINTS = 'checkpoints'
BEST_ACCURACY_MODEL = CHECKPOINTS + '\\best_accuracy_model.keras'
VALIDATION_MODEL = CHECKPOINTS + '\\best_accuracy_model.keras'
DATASET_PATH  = str(pathlib.Path.cwd()) + '\\dataset'
TRAIN_FOLDER = 'train'