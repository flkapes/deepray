import numpy as np
import tensorflow as tf
import logging
import logging.config
import os
import json
from utils import (
    check_image_size,
    check_batch_size,
    check_seed,
    check_data_dir,
)


with open("logging_config.json", "r") as config_file:
    config_dict = json.load(config_file)

logging.config.dictConfig(config_dict)

# Create a logger
logger = logging.getLogger(__name__)

transformations = {
    "resnet152v2": tf.keras.applications.resnet_v2.preprocess_input,
    "resnet101v2": tf.keras.applications.resnet_v2.preprocess_input,
    "resnet101": tf.keras.applications.resnet.preprocess_input,
    "resnet152": tf.keras.applications.resnet.preprocess_input,
    "densenet121": tf.keras.applications.densenet.preprocess_input,
    "densenet169": tf.keras.applications.densenet.preprocess_input,
    "densenet201": tf.keras.applications.densenet.preprocess_input,
    "inceptionv3": tf.keras.applications.inception_v3.preprocess_input,
    "inception_resnetv2": tf.keras.applications.inception_v3.preprocess_input,
    "xception": tf.keras.applications.inception_v3.preprocess_input,
    "vgg16": tf.keras.applications.vgg16.preprocess_input,
    "vgg19": tf.keras.applications.vgg19.preprocess_input,
}


def get_model_preproc(model_str: str):
    """Return the pre-processing function for a given model name.

    Args:
        model_str (str): The name of the image classification model to use.

    Returns:
        callable: The corresponding pre-processing function for the specified model.
    """
    model_str = model_str.strip().lower()
    if model_str in transformations:
        logger.info(f"Preprocessing function retrieved for model: {model_str}")
        return transformations[model_str]
    else:
        logger.error(f"Model preprocessing function not found for: {model_str}")
        raise ValueError(f"No preprocessing function found for model: {model_str}")


def load_images_with_augmentation_and_eval(
    dataset_directory,
    image_size,
    batch_size,
    dataset_type,
    model_name,
    validation_split=0,
    seed=44,
):
    """
    Load and preprocess images using tf.data.Dataset with data augmentation for training dataset and specific handling for evaluation dataset.

    Args:
        data_directory (str): Directory containing the dataset images.
        image_size (int): Size to which images should be resized.
        batch_size (int): Batch size for the dataset.
        dataset_type (str): Type of dataset ('train', 'valid', 'eval').
        model_name (callable): Model name.
        validation_split (float, optional): Proportion of data used for validation. Defaults to 0.
        seed (int, optional): Seed for data shuffling and transformations. Defaults to 44.

    Returns:
        tf.data.Dataset: The prepared dataset.
    """
    data_directory = check_data_dir(dataset_directory)
    image_size = check_image_size(image_size)
    batch_size = check_batch_size(batch_size)

    shuffle = dataset_type in ["training", "validation"]

    # Set seed for reproducibility
    tf.keras.utils.set_random_seed(seed)

    def process_path(file_path):
        logger.info(f"Processing file: {file_path}")

        parts = tf.strings.split(file_path, os.sep)
        label = parts[-2]
        # Log label extraction
        logger.info(f"Extracted label: {label}")

        image = tf.io.read_file(file_path)
        image = tf.io.decode_png(image, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.grayscale_to_rgb(image)
        image = tf.image.resize(image, [image_size, image_size])

        if dataset_type == "train":
            image = augment(image)

        # Log image properties after decoding and resizing
        logger.info(f"Image shape after resize: {image.shape}, dtype: {image.dtype}")

        # Log preprocessed image properties
        return image, tf.where(tf.equal(label, "positive"), 1.0, 0.0)

    list_ds = tf.data.Dataset.list_files(str(data_directory / "*/*"), seed=seed)
    total_images = tf.data.experimental.cardinality(list_ds).numpy()

    def augment(image):
        image = tf.image.random_flip_left_right(image, seed=seed)
        return image

    def configure_for_performance(ds):
        ds = ds.cache().batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    list_ds = tf.data.Dataset.list_files(
        str(data_directory / "*/*"), shuffle=shuffle, seed=seed
    )
    total_images = tf.data.experimental.cardinality(list_ds).numpy()

    # Splitting dataset for training and validation
    if dataset_type == "training":
        skip_count = int(total_images * validation_split)
        dataset = list_ds.skip(skip_count).map(
            lambda file_path: process_path(file_path),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    elif dataset_type == "validation":
        take_count = int(total_images * validation_split)
        dataset = list_ds.take(take_count).map(
            lambda file_path: process_path(file_path),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:  # For 'eval' dataset
        dataset = list_ds.map(
            lambda file_path: process_path(file_path),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = configure_for_performance(dataset)

    logger.info(
        "Dataset with augmentation and evaluation handling loaded and configured for"
        f" {dataset_type} type with {len(dataset)} batches."
    )
    return dataset
