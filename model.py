from tensorflow import keras

# input_shape = (224, 224, 3) # Define your input shape here
# Define a function to create and build the model
def create_model(input_shape, number_of_classes):
    model = keras.Sequential()    
    # Add layers one by one
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(number_of_classes, activation='softmax')) # Updated output layer with 75 classes
    return model