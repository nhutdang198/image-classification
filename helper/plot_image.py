from tensorflow import io
import matplotlib.pyplot as pyplot
import numpy

def plot_image(image_path: str) -> any:
  """
  Plot a given image.

  Args:
    image_path (str): Path to the image to be plotted

  Returns:
    None
  """
  image = io.read_file(image_path)
  image = io.decode_image(image, channels=3, expand_animations=False)
  image = numpy.asarray(image)
  image = image / 255.0
  pyplot.figure()
  pyplot.axis('off')
  pyplot.imshow(image)
