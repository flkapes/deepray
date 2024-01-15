import cv2
import numpy as np
import logging
import logging.config
import json
import skimage.exposure

with open("logging_config.json", "r") as config_file:
    config_dict = json.load(config_file)

logging.config.dictConfig(config_dict)

# Create a logger
logger = logging.getLogger(__name__)


def load_image(file_path: str) -> np.ndarray:
    """Load an image from the specified file path.

    Args:
        file_path (str): The file path to load the image from.

    Returns:
        numpy.ndarray: The loaded image as a NumPy array.
    """
    try:
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"File not found: {file_path}")
        return image
    except Exception as e:
        raise IOError(f"Error reading file {file_path}: {e}")


def apply_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale.

    Args:
        image (numpy.ndarray): The image to convert.

    Returns:
        numpy.ndarray: The grayscale image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(image: np.ndarray, sigma: float = 6) -> np.ndarray:
    """Apply a Gaussian blur filter to an image.

    Args:
        image (numpy.ndarray): The image to apply the filter to.
        sigma (float, optional): The standard deviation of the Gaussian filter. Defaults to 6.

    Returns:
        numpy.ndarray: The filtered image.
    """
    if sigma <= 0:
        raise ValueError("Sigma must be a positive number.")
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)


def apply_morphology(image: np.ndarray, kernel_size: int = 18) -> np.ndarray:
    """Apply morphological operations to an image.

    Args:
        image (numpy.ndarray): The image to apply the operations to.
        kernel_size (int, optional): The size of the kernel to use for morphological operations. Defaults to 18.

    Returns:
        numpy.ndarray: The processed image.
    """
    if kernel_size <= 0:
        raise ValueError("Kernel size must be a positive integer.")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def apply_threshold(image: np.ndarray) -> np.ndarray:
    """Apply Otsu's thresholding method to an image.

    Args:
        image (numpy.ndarray): The image to apply the thresholding to.

    Returns:
        numpy.ndarray: The thresholded image.
    """
    if not isinstance(image, np.ndarray):
        logger.error("Input for thresholding must be a NumPy array.")
        raise TypeError("Input must be a NumPy array.")

    logger.info("Applying thresholding to image.")
    return cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)[1]


def apply_contour_filtering(
    image: np.ndarray,
    thresh_image: np.ndarray,
    min_area: int = 265,
    max_area: int = 3600,
) -> np.ndarray:
    """Filter contours from an image based on area.

    Args:
        image (numpy.ndarray): The original image.
        thresh_image (numpy.ndarray): The thresholded image to use for finding contours.
        min_area (int, optional): The minimum area of contours to keep. Defaults to 265.
        max_area (int, optional): The maximum area of contours to keep. Defaults to 3600.

    Returns:
        numpy.ndarray: The image with filtered contours.
    """
    if not all(isinstance(i, np.ndarray) for i in [image, thresh_image]):
        logger.error("Inputs for contour filtering must be NumPy arrays.")
        raise TypeError("Inputs must be NumPy arrays.")
    if min_area <= 0 or max_area <= 0:
        logger.error("Area bounds for contour filtering must be positive.")
        raise ValueError("Area bounds must be positive.")

    logger.info("Applying contour filtering to image.")
    masked_image = image.copy()
    mean_val = int(np.mean(masked_image))
    contours, _ = cv2.findContours(
        thresh_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1
    )

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            cv2.drawContours(masked_image, [contour], 0, (mean_val), -1)

    return masked_image


def apply_stretch_intensity(image: np.ndarray):
    """Stretch the intensity of an image to enhance contrast.

    Args:
        img (numpy.ndarray): The image to stretch.

    Returns:
        numpy.ndarray: The stretched image.
    """
    if not isinstance(image, np.ndarray):
        logger.error("Input for intensity stretching must be a NumPy array.")
        raise TypeError("Input must be a NumPy array.")

    logger.info("Applying intensity stretch to image.")
    min_val, max_val = np.amin(image), np.amax(image)
    stretched_image = skimage.exposure.rescale_intensity(
        image, in_range=(min_val, max_val), out_range=(0, 255)
    )
    return stretched_image.astype(np.uint8)


def process_image(file_path, is_arr=True) -> np.ndarray:
    logger.info("Processing image.")
    try:
        if not is_arr:
            img = load_image(file_path)
        else:
            img = np.asarray(file_path)
            file_path = img.astype("uint8")
        gray = apply_grayscale(img)
        blur = apply_gaussian_blur(gray, 5)
        morph = apply_morphology(blur)
        thresh = apply_threshold(morph)
        masked = apply_contour_filtering(gray, thresh)
        result = apply_stretch_intensity(masked)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        logger.info("Image processing completed successfully.")
        return result
    except Exception as e:
        logger.error(f"Error occurred during image processing: {e}")
        raise
