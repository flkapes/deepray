import cv2
import numpy as np
import skimage.exposure


def load_image(file_path: str) -> np.ndarray:
    """Load an image from the specified file path.

    Args:
        file_path (str): The file path to load the image from.

    Returns:
        numpy.ndarray: The loaded image as a NumPy array.
    """
    image = cv2.imread(file_path)
    return image


def apply_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale.

    Args:
        image (numpy.ndarray): The image to convert.

    Returns:
        numpy.ndarray: The grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(image: np.ndarray, sigma: float = 6) -> np.ndarray:
    """Apply a Gaussian blur filter to an image.

    Args:
        image (numpy.ndarray): The image to apply the filter to.
        sigma (float, optional): The standard deviation of the Gaussian filter. Defaults to 6.

    Returns:
        numpy.ndarray: The filtered image.
    """
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)


def apply_morphology(image: np.ndarray, kernel_size: int = 18) -> np.ndarray:
    """Apply morphological operations to an image.

    Args:
        image (numpy.ndarray): The image to apply the operations to.
        kernel_size (int, optional): The size of the kernel to use for morphological operations. Defaults to 18.

    Returns:
        numpy.ndarray: The processed image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def apply_threshold(image: np.ndarray) -> np.ndarray:
    """Apply Otsu's thresholding method to an image.

    Args:
        image (numpy.ndarray): The image to apply the thresholding to.

    Returns:
        numpy.ndarray: The thresholded image.
    """
    return cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)[1]


def apply_contour_filtering(image: np.ndarray, thresh_image: np.ndarray, min_area: int = 265, max_area: int = 3600) -> np.ndarray:
    """Filter contours from an image based on area.

    Args:
        image (numpy.ndarray): The original image.
        thresh_image (numpy.ndarray): The thresholded image to use for finding contours.
        min_area (int, optional): The minimum area of contours to keep. Defaults to 265.
        max_area (int, optional): The maximum area of contours to keep. Defaults to 3600.

    Returns:
        numpy.ndarray: The image with filtered contours.
    """
    masked_image = image.copy()
    mean_val = int(np.mean(masked_image))
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

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
    min_val = int(np.amin(image))
    max_val = int(np.amax(image))
    stretched_image = exposure.rescale_intensity(image, in_range=(min_val, max_val), out_range=(0, 255))
    return stretched_image.astype(np.uint8)


def process_image(file_path, is_arr=True) -> np.ndarray:
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
    return result
