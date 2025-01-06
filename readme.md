# Image Classification Project

This project contains scripts and resources for training, predicting, and deploying an image classification model using TensorFlow/Keras.

## Project Structure


## Requirements

- Python
- TensorFlow/Keras
- Gradio
- Any other dependencies required by your scripts (please list them here)

## Scripts

### `app.py`

This script launches a Gradio interface for image classification.

#### Usage

```zsh
python app.py
```

### `predict.py`

This script is used to predict the class of an image using a trained model.

#### Usage

To predict the class of an image, use the predict.py script:

```zsh
python predict.py <image_path> <model_path>
```

Example:

```zsh
python predict.py ./classifier/butterfly-image-classification/train/Image_1.jpg ./classifier/checkpoints/best_accuracy_model.keras
```

### `train.py`

This script is used to train the image classification model.

#### Usage

first transform ipynb file to py file
```zsh
jupyter nbconvert --to script train.ipynb
```

Example:

```zsh
python train.py
```

## Installation

To install the required dependencies, run:

```zsh
pip install -r requirements.txt
```