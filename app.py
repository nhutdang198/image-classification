import gradio as gr
from tensorflow.keras.models import load_model
import tensorflow
import json
import pandas
import pathlib
from bidict import bidict
from tensorflow import keras
from datetime import datetime
from preprocessing import preprocess_image_dataset
from colorama import Fore
from pprint import pprint

### import hyperparameters
from hyperparameters import (
    BEST_ACCURACY_MODEL
)

# Load your trained model
model = load_model(BEST_ACCURACY_MODEL)

def load_class_names(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def classify_image(inp):
    inp = inp.reshape((1, *model.input_shape[1:]))  # Reshape input to match model input shape
    prediction = model.predict(inp)
    return prediction

bimap = bidict(load_class_names('bimap.json'))

# Example prediction function
def predict(inp):
    predictions = classify_image(inp)
    
    ### Get one predicted class
    # predicted_class = tensorflow.math.argmax(predictions, axis=-1)
    # number_class = predicted_class.numpy()[0]
    # class_name = bimap.inverse[int(number_class)]

    ### Get top 5 predicted classes
    probs, classes = tensorflow.math.top_k(predictions[0], 5)
    result = []
    print("************************************************")
    for prob, class_name in zip(probs, classes):
        index_class = int(class_name.numpy())
        print(Fore.GREEN + "Class: " + Fore.RESET + f"{class_name}".format(class_name), end="\n")
        print(Fore.GREEN + "Class name: " + Fore.RESET + f"{bimap.inverse[index_class]}", end="\n")
        print(Fore.RED + "Score: " + Fore.RESET + f"{prob:.4f}".format(prob), end="\n")
        result.append({"class": bimap.inverse[index_class], "score": f"{prob:.4f}".format(prob)})
        print("************************************************")
    return {"predicted_class": result}


# Create a Gradio interface
iface = gr.Interface(fn=predict, inputs="image", outputs="text", title="Butterfly Image Classifier")

# Launch the interface
iface.launch()
