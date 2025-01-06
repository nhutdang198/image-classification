import argparse
import tensorflow as tf
from tensorflow import keras
import numpy
import json
from colorama import Fore
from helper.decode_image import decode_image
from bidict import bidict

def load_class_names(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def predict(image_path, model, top_k=5):
  '''Predict the class (or classes) of an image using a trained deep learning model.

  Args:
    image_path (str): Path to the input image
    model (keras.Model): Trained model
    top_k (int, optional): Number of top classes to return. Defaults to 5.

  Returns:
    probs (numpy.array): Probabilities of the top k classes
    classes (numpy.array): Class indices of the top k classes
  '''
  image = decode_image(image_path)
  image = numpy.expand_dims(image, axis=0)           # Add batch dimension
  
  predictions = model.predict(image)
  top_k_values, top_k_indices = tf.math.top_k(predictions[0], top_k)

  # Convert tensors to numpy arrays
  probs = top_k_values.numpy()
  classes = top_k_indices.numpy()
  return probs, classes

def test_predict():
    STORE_PATH = './checkpoints/'
    model_name = 'best_accuracy_model.keras'

    # Load model from checkpoint
    model = keras.models.load_model(STORE_PATH + model_name)
    print(model.summary())

    # image to predict
    image_path = './butterfly-image-classification/Image_1.jpg'

    # total number of classes
    top_k = 5

    # Predict the class of the input image
    probs, classes = predict(image_path, model, top_k)
    class_names = load_class_names('bimap.json')
    top_classes = [class_names[str(idx)] for idx in classes]
    
    for prob, class_name in zip(probs, top_classes):
      print(f"Class: {class_name}, Probability: {prob:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict image class using a trained model.')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Model checkpoint to load')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top K classes to display')
    parser.add_argument('--category_names', type=str, default=None, help='Path to category names JSON file')

    args = parser.parse_args()

    # STORE_PATH = './trained_models/'
    
    # get model name    
    model_name = args.checkpoint
    
    # Load model from checkpoint
    model = keras.models.load_model(model_name)
    
    # image to predict
    image_path = args.image_path
    top_k = args.top_k
    
    # Predict the class of the input image
    probs, classes = predict(image_path, model, top_k)

    # Load class names if provided
    if args.category_names is not None:
        class_names = load_class_names(args.category_names)
        bimap = bidict(class_names)
    else:
        class_names = load_class_names('bimap.json')
        bimap = bidict(class_names)


    # Print results
    print(Fore.GREEN + "Image:" + Fore.RESET + " {}".format(image_path))
    print(Fore.RED + "Predictions:" + Fore.RESET)
    print("************************************************")
    for prob, class_name in zip(probs, classes):
        print(Fore.GREEN + "Class: " + Fore.RESET + f"{class_name}".format(class_name), end="\n")
        print(Fore.GREEN + "Class name: " + Fore.RESET + f"{bimap.inverse[class_name]}".format(class_name), end="\n")
        print(Fore.RED + "Score: " + Fore.RESET + f"{prob:.4f}".format(prob), end="\n")
        print("************************************************")