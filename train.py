#!/usr/bin/env python
# coding: utf-8

# In[1]:


### import neccessary libraries
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as pyplot
import numpy
from PIL import Image
import os
import json
import pathlib
from colorama import Fore
from datetime import datetime
import pandas
from pprint import pprint
from PIL import Image
from bidict import bidict


# In[2]:


### import local libraries
from model import create_model
from preprocessing import preprocess_image_dataset
from  predict import predict 


# In[3]:


print(Fore.RED + '#############################################################################################' + Fore.RESET, end="\n")
print(Fore.GREEN + '############################### Image Classification Application ############################' + Fore.RESET, end="\n")
print(Fore.RED + '#############################################################################################' + Fore.RESET, end="\n")


# In[4]:


# get root directory
root = pathlib.Path.cwd()


# In[5]:


root


# In[6]:


### define tensorboard callbacks
log_dir = os.path.join("logs", "fit", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# In[7]:


### import hyperparameters
from hyperparameters import (
    IMAGE_SIZE, BATCH_SIZE, EPOCHS, BEST_ACCURACY_MODEL, DATASET_PATH, TRAIN_FOLDER
)


# In[8]:


### read cvs file which contains the image file names and their corresponding labels
dataframe = pandas.read_csv(f'{DATASET_PATH}\\Training_set.csv')
dataframe = pandas.DataFrame(dataframe)


# In[9]:


### get the unique classes
classes = dataframe["label"].unique()
# print out the number of image paths
print(Fore.GREEN + '### Label names: ' + Fore.RESET, end="")
pprint(classes)


# In[10]:


### change the image path to the absolute path
def modify_image_path(filename):
    return DATASET_PATH + '/' + TRAIN_FOLDER + '/' + filename

dataframe['filename'] = dataframe['filename'].apply(modify_image_path)


# In[11]:


### print out the first 10 rows of the dataframe
print(Fore.GREEN + '### Display 10 sample row' + Fore.RESET)
dataframe.head(10)


# In[12]:


### Load your images for testing
image1 = Image.open(dataframe.iloc[0].filename)
image2 = Image.open(dataframe.iloc[1].filename)
image3 = Image.open(dataframe.iloc[2].filename)
image4 = Image.open(dataframe.iloc[3].filename)
image5 = Image.open(dataframe.iloc[4].filename)

# Get the dimensions of the images
width, height = image1.size

# Create a new blank image with a width that can contain all 5 images
total_width = width * 5
combined_image = Image.new('RGB', (total_width, height))

# Paste the images into the new blank image
combined_image.paste(image1, (0, 0))
combined_image.paste(image2, (width, 0))
combined_image.paste(image3, (width * 2, 0))
combined_image.paste(image4, (width * 3, 0))
combined_image.paste(image5, (width * 4, 0))

# Show the combined image
print(Fore.GREEN + '### Display 5 images for testing' + Fore.RESET)
combined_image


# In[13]:


### get the shape of the image
image2 = Image.open(dataframe.iloc[1].filename)
math_image = numpy.array(image2)
image_shape = math_image.shape
print(Fore.GREEN + '### Shape of an image: ' + Fore.RESET, end='')
print(image_shape)


# In[14]:


### create bimap for classes
bimap = bidict()

# iterate through classes and add them to the bidict
for i, c in enumerate(classes):
    bimap[c] = i

with open('bimap.json', 'w') as json_file:
    json.dump(dict(bimap), json_file, indent=4)  # `indent=4` for pretty printing

print(Fore.GREEN + '### Class name map: ' + Fore.RESET)
pprint(bimap)


# In[15]:


### get data from the pandas dataframe
dataframe['number_label'] = dataframe['label'].map(bimap)

features = dataframe['filename'].values 
labels = dataframe['number_label'].values

# print out the number of image paths
print(Fore.GREEN + '### number of image paths: ' + Fore.RESET, end="")
pprint(len(features))

# print out the number of image paths
print(Fore.GREEN + '### number of labels: ' + Fore.RESET, end="")
pprint(len(labels))


# In[16]:


# # Define an ImageDataGenerator with various augmentations
# datagen = keras.preprocessing.image.ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )


# In[17]:


### preprocess data
train_dataset, train_dataset_size = preprocess_image_dataset(features, labels, IMAGE_SIZE, BATCH_SIZE)


# In[18]:


# Get the number of batches
num_batches = tensorflow.data.experimental.cardinality(train_dataset).numpy()

# Calculate total number of elements in the train dataset
total_elements_in_train = sum(1 for _ in train_dataset.unbatch())

# # Calculate total number of elements in the train dataset
# total_elements_in_validation = sum(1 for _ in validation_dataset.unbatch())

# print out the number of batches
print(Fore.GREEN + '### number of batches: ' + Fore.RESET, end="")
pprint(num_batches)

# print out the number of items in the dataset
print(Fore.GREEN + '### number of items in train dataset: ' + Fore.RESET, end="")
pprint(total_elements_in_train)

# # print out the number of items in the dataset
# print(Fore.GREEN + '### number of items in validation dataset: ' + Fore.RESET, end="")
# pprint(total_elements_in_validation)


# In[19]:


### create the model
model = create_model(image_shape, len(classes))
print(model.summary())


# In[20]:


### steps: choose optimizer and loss function
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)


# In[ ]:


### prepare for saving point
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=BEST_ACCURACY_MODEL,
    save_weights_only=False,  # Set to False to save the entire model
    verbose=1,
    save_best_only=True, # Save only when the metric improves 
    monitor='val_loss', # Monitor the validation loss 
    mode='min' # Save model with the minimum validation loss
)


# In[22]:


### train the model
history = model.fit(
    train_dataset,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    # validation_data=validation_dataset,
    # validation_batch_size=BATCH_SIZE,
    callbacks=[tensorboard_callback, checkpoint_callback],
    verbose=1
)


# In[23]:


### train accuracy
train_accuracy = history.history['accuracy']  # Training accuracy across epochs
print(f"Training accuracy: {train_accuracy}")

pyplot.plot(history.history['accuracy'], label='Training Accuracy')
pyplot.title('Model Accuracy')
pyplot.ylabel('Accuracy')
pyplot.xlabel('Epoch')
pyplot.legend(loc='upper left')
pyplot.show()


# In[24]:


### steps: evaluate model
test_loss, test_accuracy = model.evaluate(train_dataset)

print(Fore.GREEN + "Test Accuracy: " + Fore.RESET + "{}".format(test_accuracy * 100))
print(Fore.GREEN + "Test Accuracy: " + Fore.RESET + "{}".format(test_accuracy * 100))


# In[25]:


### step: load the trained model
best_accuracy_model = keras.models.load_model(BEST_ACCURACY_MODEL)
print(best_accuracy_model.summary())


# In[26]:


### test for predicting image label
image_path = dataframe.iloc[0].filename
label = dataframe.iloc[0].label


# In[27]:


### display image to be tested
print(Fore.GREEN + "### Display image to be tested " + Fore.RESET)
print(Fore.RED + "### Label: " + Fore.RESET, end="")
print(dataframe.iloc[0].label)
Image.open(image_path)


# In[28]:


### predict image label
probs, classes = predict(image_path=image_path, model=best_accuracy_model, top_k=5)


# In[29]:


# Print results
predictions = []
print(Fore.GREEN + "Image:" + Fore.RESET + " {}".format(image_path))
print(Fore.RED + "Predictions:" + Fore.RESET)
print("************************************************")
for prob, class_name in zip(probs, classes):
    print(Fore.GREEN + "Class: " + Fore.RESET + f"{class_name}".format(class_name), end="\n")
    print(Fore.GREEN + "Class name: " + Fore.RESET + f"{bimap.inverse[class_name]}".format(class_name), end="\n")
    print(Fore.RED + "Score: " + Fore.RESET + f"{prob:.4f}".format(prob), end="\n")
    print("************************************************")
    predictions.append({
        'class_index': class_name,
        'class_name': bimap.inverse[class_name],
        'score': prob
    })


# In[30]:


assert label == predictions[0]['class_name']

