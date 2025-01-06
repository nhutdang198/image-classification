import json
import pandas
import pathlib
from bidict import bidict
from tensorflow import keras
from preprocessing import preprocess_image_dataset
from colorama import Fore
from pprint import pprint

from hyperparameters import IMAGE_SIZE, BATCH_SIZE, VALIDATION_MODEL, DATASET_PATH, TRAIN_FOLDER

def load_class_names(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def modify_image_path(filename):
    return DATASET_PATH + '/' + TRAIN_FOLDER + '/' + filename

if __name__ == '__main__':
    
    ### read cvs file which contains the image file names and their corresponding labels
    dataframe = pandas.read_csv(f'{DATASET_PATH}\\Training_set.csv')
    dataframe = pandas.DataFrame(dataframe)

    class_names = load_class_names('bimap.json')
    bimap = bidict(class_names)
    
    ### get data from the pandas dataframe
    dataframe['number_label'] = dataframe['label'].map(bimap)
    dataframe['filename'] = dataframe['filename'].apply(modify_image_path)

    features = dataframe['filename'].values 
    labels = dataframe['number_label'].values

    # print out the number of image paths
    print(Fore.GREEN + '### number of image paths: ' + Fore.RESET, end="")
    pprint(len(features))

    # print out the number of image paths
    print(Fore.GREEN + '### number of labels: ' + Fore.RESET, end="")
    pprint(len(labels))
    
    ### step: load the trained model
    model = keras.models.load_model(VALIDATION_MODEL)
    
    print(model.summary())
    
    ### preprocess data
    train_dataset, train_dataset_size = preprocess_image_dataset(features, labels, IMAGE_SIZE, BATCH_SIZE)
    
    ### steps: evaluate model
    test_loss, test_accuracy = model.evaluate(train_dataset)

    print(Fore.GREEN + "Test Accuracy: " + Fore.RESET + "{}".format(test_accuracy * 100))
    print(Fore.GREEN + "Test Accuracy: " + Fore.RESET + "{}".format(test_accuracy * 100))