import silence_tensorflow.auto
import PIL
import tensorflow as tf
import cv2
import remove_labels
import numpy as np
from PIL import Image

process_image = remove_labels.process_image


def histogram_equalization(image):
    """Applies histogram equalization on an image.

    Args:
        image (numpy.ndarray): An image as a NumPy array.

    Returns:
        numpy.ndarray: The histogram equalized image as a NumPy array.
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype("uint8")
    img_equalized = cv2.equalizeHist(img_gray)
    return cv2.cvtColor(img_equalized, cv2.COLOR_GRAY2RGB)


def contrast_limited_histogram_equalization(image):
    """Applies contrast limited adaptive histogram equalization (CLAHE) on an image.

    Args:
        image (numpy.ndarray): An image as a NumPy array.

    Returns:
        numpy.ndarray: The CLAHE image as a NumPy array.
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype("uint8")
    clahe = cv2.createCLAHE(clipLimit=0, tileGridSize=(14, 14))
    img_ahe = clahe.apply(img_gray)
    return cv2.cvtColor(img_ahe, cv2.COLOR_GRAY2RGB)


def gamma_correction(image, gamma=1.0):
    """Applies gamma correction on an image.

    Args:
        image (numpy.ndarray): An image as a NumPy array.
        gamma (float): The gamma value to use for gamma correction.

    Returns:
        numpy.ndarray: The gamma corrected image as a NumPy array.
    """
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(np.asarray(image * 255).astype("uint8"), table)


def median_filtering(image, ksize=3):
    """Applies median filtering on an image.

    Args:
        image (numpy.ndarray): An image as a NumPy array.
        ksize (int): The size of the kernel for the median filter.

    Returns:
        numpy.ndarray: The median filtered image as a NumPy array.
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_median = cv2.medianBlur(img_gray, ksize)
    return cv2.cvtColor(img_median, cv2.COLOR_GRAY2RGB)


def gaussian_filtering(image, ksize=(1, 1), sigmaX=3):
    """Applies Gaussian filtering on an image.

    Args:
        image (numpy.ndarray): An image as a NumPy array.
        ksize (tuple): The size of the kernel for the Gaussian filter.
        sigmaX (float): The standard deviation of the Gaussian kernel in the X direction.

    Returns:
        numpy.ndarray: The Gaussian filtered image as a NumPy array.
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_gaussian = cv2.GaussianBlur(img_gray, ksize, sigmaX)
    return cv2.cvtColor(img_gaussian, cv2.COLOR_GRAY2RGB)


def resize(image, image_size=384):
    """
    Resizes a given RGB image.

    Parameters:
    image (numpy.ndarray): The input RGB image.
    image_size (int): The target size for resizing.

    Returns:
    numpy.ndarray: The resized RGB image.
    """
    return cv2.resize(np.asarray(image).astype("uint8"), (image_size, image_size))


def preprocess(img):
    """Applies preprocessing steps to the input image.

    Args:
        img (PIL.Image): The input image.

    Returns:
        np.ndarray: The preprocessed image as a NumPy array.

    """
    x = process_image(np.asarray(img).astype("uint8"), True)
    # Apply Gaussian filtering
    x = gaussian_filtering(x)
    # Apply preprocessing for InceptionV3 model
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    return x


def transform_all(image, ahe):
    """Applies a series of image transformations to the input image.

    Args:
        image (PIL.Image): The input image.
        ahe (bool): If True, apply adaptive histogram equalization. If False, apply contrast-limited AHE.

    Returns:
        np.ndarray: The transformed image as a NumPy array.

    """
    if ahe:
        # Apply gamma correction
        image = gamma_correction(image, gamma=1.0)
        # Apply Gaussian filtering
        image = gaussian_filtering(image)
        # Apply histogram equalization
        image = histogram_equalization(image)
    else:
        # Apply gamma correction
        image = gamma_correction(image, gamma=0.4)
        # Apply contrast-limited AHE
        image = contrast_limited_histogram_equalization(image)

    return np.asarray(image).astype("uint8")


def apply_transformations(image, headless=True):
    """Applies image transformations and preprocessing steps to the input image.

    Args:
        image (PIL.Image): The input image.
        headless (bool): If True, apply preprocessing for headless models. If False, apply preprocessing for models with top layers.

    Returns:
        np.ndarray: The transformed and preprocessed image as a NumPy array.

    """
    image = np.asarray(image).astype("uint8")
    # Apply image transformations
    morph = process_image(image, True)
    # Apply preprocessing
    if headless:
        x = tf.keras.applications.inception_v3.preprocess_input(morph)
    else:
        x = preprocess(morph)

    return np.asarray(x).astype("float32")
