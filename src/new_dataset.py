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
        logger.error(
            f"Model preprocessing function not found for: {model_str}")
        raise ValueError(
            f"No preprocessing function found for model: {model_str}")

import numpy as np
def load_images_with_augmentation_and_eval(dataset_directory, image_size, batch_size, dataset_type, model_name, validation_split=0, seed=44):
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
    shuffle = dataset_type in ['training', 'validation']
    tf.keras.utils.set_random_seed(seed)
    arg = {
        "directory": str(dataset_directory),
        "labels": 'inferred',
        "label_mode": 'binary',
        "class_names": None,
        "color_mode": 'rgb',
        "batch_size": batch_size,
        "image_size": (image_size, image_size),
        "shuffle": shuffle,
        "seed": seed,
        "validation_split": validation_split,
        "subset": dataset_type if shuffle else None,
        "interpolation": 'bilinear',
        "follow_links": False,
        "crop_to_aspect_ratio": False
    }
    data_directory = check_data_dir(dataset_directory)
    image_size = check_image_size(image_size)
    batch_size = check_batch_size(batch_size)

    dataset = tf.keras.utils.image_dataset_from_directory(
        **arg
    )
    class_names = dataset.class_names
    dataset = dataset.map(lambda x, y: (get_model_preproc(model_name)(x), y))
    image_batch, labels_batch = next(iter(dataset))
    first_image = image_batch[0]
    print(np.min(first_image), np.max(first_image), labels_batch[0])
    #def augment(image):
    #    image = tf.image.random_flip_left_right(image, seed=seed)
    #    image = tf.image.random_flip_up_down(image, seed=seed)
    #    return image

    #def preprocess_image(image):
        #if dataset_type == 'train':
        #     image = augment(image)
    #    return get_model_preproc(model_name)(image)

    def configure_for_performance(ds):
        ds = ds.cache()
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    dataset = configure_for_performance(dataset)

    logger.info(f"Dataset with augmentation and evaluation handling loaded and configured for {dataset_type} type with {len(dataset)} batches.")
    return dataset
