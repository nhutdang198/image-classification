import tensorflow as tf
from tensorflow import io

def decode_image(image_path: str, image_size: int = 224) -> any:
  image = io.read_file(image_path)
  image = io.decode_image(image, channels=3, expand_animations=False)
  image = tf.image.resize(image, [image_size, image_size])
  image = image / 255.0
  image = tf.cast(image, tf.float32)  # Normalize to [0, 1]
  return image
