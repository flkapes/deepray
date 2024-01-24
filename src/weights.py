from typing import List, Union, Dict
import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer
import logging
import logging.config
import json
import os

try:
    with open("logging_config.json", "r") as config_file:
        config_dict = json.load(config_file)
except:
    with open(os.environ["logging"], "r") as config_file:
        config_dict = json.load(config_file)
        
logging.config.dictConfig(config_dict)
logger = logging.getLogger(__name__)


def generate_class_weights(
    class_series: Union[List[int], np.ndarray],
    multi_class: bool = True,
    one_hot_encoded: bool = False,
) -> Dict[int, float]:
    """
    Calculates class weights for imbalanced datasets.

    Args:
        class_series: A list or array containing the class labels or one-hot encoded class values.
        multi_class: A boolean indicating whether the problem is multi-class or multi-label. Default is True.
        one_hot_encoded: A boolean indicating whether the class values are one-hot encoded. Default is False.

    Returns:
        class_weights: A dictionary mapping class labels to their corresponding class weights.
    """
    if multi_class:
        if one_hot_encoded:
            class_series = np.argmax(class_series, axis=1)

        class_labels = np.unique(class_series)
        class_weights = compute_class_weight(
            class_weight="balanced", classes=class_labels, y=class_series
        )
        return dict(zip(class_labels, class_weights))
    else:
        if not one_hot_encoded:
            mlb = MultiLabelBinarizer()
            class_series = mlb.fit_transform(class_series)

        n_samples, n_classes = class_series.shape

        class_count = np.sum(class_series, axis=0)
        class_weights = np.where(
            class_count > 0, n_samples / (n_classes * class_count), 1
        )
        class_labels = (
            range(len(class_weights)) if not one_hot_encoded else mlb.classes_
        )
        return dict(zip(class_labels, class_weights))


def get_weights(flow_from_dir: str) -> Dict[int, float]:
    """
    Generate class weights for imbalanced datasets.

    Args:
    - flow_from_dir: A string representing the directory path from which the images are loaded.

    Returns:
    - class_weights: A dictionary mapping class labels to their corresponding class weights.
    """

    # Create an image data generator
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator()

    # Set the target size, batch size, color mode, and class mode for the
    # generator
    train_gen = data_gen.flow_from_directory(
        flow_from_dir,
        target_size=(3, 3),
        batch_size=1,
        color_mode="rgb",
        class_mode="binary",
    )

    valid_gen = train_gen

    # Retrieve the labels from the generator and store them in a list
    labels = [valid_gen.labels[i] for i in range(len(valid_gen))]

    # Call the generate_class_weights function with the labels as input
    class_weights = generate_class_weights(labels)

    # Print the class indices obtained from the generator
    labels = list(train_gen.class_indices.keys())
    log_info = (
        f"Found {len(labels)} class indicies: \n\tLabel: {labels[0]} –– Index: 0 ––"
        f" Weight: {class_weights[0]}" +
        f"\n\tLabel: {labels[1]} –– Index: 1 –– Weight: {class_weights[1]}")
    logger.info(log_info)

    return class_weights
