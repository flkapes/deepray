import cv2
import numpy as np
import skimage.exposure


def load_image(file_path):
    """Load an image from the specified file path.

    Args:
        file_path (str): The file path to load the image from.

    Returns:
        numpy.ndarray: The loaded image as a NumPy array.
    """
    return cv2.imread(file_path)


def to_grayscale(img):
    """Convert an image to grayscale.

    Args:
        img (numpy.ndarray): The image to convert.

    Returns:
        numpy.ndarray: The grayscale image.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(img, sigma=6):
    """Apply a Gaussian blur filter to an image.

    Args:
        img (numpy.ndarray): The image to apply the filter to.
        sigma (float, optional): The standard deviation of the Gaussian filter. Defaults to 6.

    Returns:
        numpy.ndarray: The filtered image.
    """
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


def apply_morphology(img, kernel_size=18):
    """Apply morphological operations to an image.

    Args:
        img (numpy.ndarray): The image to apply the operations to.
        kernel_size (int, optional): The size of the kernel to use for morphological operations. Defaults to 18.

    Returns:
        numpy.ndarray: The processed image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def apply_threshold(img):
    """Apply Otsu's thresholding method to an image.

    Args:
        img (numpy.ndarray): The image to apply the thresholding to.

    Returns:
        numpy.ndarray: The thresholded image.
    """
    return cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]


def filter_contours(img, thresh_img, min_area=265, max_area=3600):
    """Filter contours from an image based on area.

    Args:
        img (numpy.ndarray): The original image.
        thresh_img (numpy.ndarray): The thresholded image to use for finding contours.
        min_area (int, optional): The minimum area of contours to keep. Defaults to 265.
        max_area (int, optional): The maximum area of contours to keep. Defaults to 3600.

    Returns:
        numpy.ndarray: The image with filtered contours.
    """
    masked = img.copy()
    meanval = int(np.mean(masked))
    contours = cv2.findContours(thresh_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for cntr in contours:
        area = cv2.contourArea(cntr)
        if min_area < area < max_area:
            cv2.drawContours(masked, [cntr], 0, (meanval), -1)

    return masked


def stretch_intensity(img):
    """Stretch the intensity of an image to enhance contrast.

    Args:
        img (numpy.ndarray): The image to stretch.

    Returns:
        numpy.ndarray: The stretched image.
    """
    minval, maxval = int(np.amin(img)), int(np.amax(img))
    return skimage.exposure.rescale_intensity(
        img, in_range=(minval, maxval), out_range=(0, 255)
    ).astype(np.uint8)


def process_image(file_path, is_arr=True):
    if not is_arr:
        img = load_image(file_path)
    else:
        img = np.asarray(file_path)
        file_path = img.astype("uint8")
    gray = to_grayscale(img)
    blur = apply_gaussian_blur(gray, 5)
    morph = apply_morphology(blur)
    thresh = apply_threshold(morph)
    masked = filter_contours(gray, thresh)
    result = stretch_intensity(masked)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result
