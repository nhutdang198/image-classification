import tensorflow as tf
from tensorflow import io

def can_decode_image(image_path: str, image_size: int = 224) -> any:
  """
  Check if an image can be decoded. If it can, return the image path;
  otherwise, return None.

  Args:
    image_path (str): Path to the image to be checked
    image_size (int, optional): Size of the output image. Defaults to 224.

  Returns:
    any: The image path if the image can be decoded; otherwise, None
  """
  try:
    image = io.read_file(image_path)
    image = io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size])
    image = image / 255.0
    image = tf.cast(image, tf.float32)  # Normalize to [0, 1]
    return image_path
  except:
    return None
