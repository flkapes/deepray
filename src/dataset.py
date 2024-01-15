import silence_tensorflow.auto
import tensorflow as tf
import remove_labels
import logging
import logging.config
import json
from utils import (
    check_image_size,
    check_batch_size,
    check_seed,
    check_data_dir,
    check_model_name,
    check_dataset_type,
    check_validation_split_value,
)
from models import list_models

with open("logging_config.json", "r") as config_file:
    config_dict = json.load(config_file)

logging.config.dictConfig(config_dict)

# Create a logger
logger = logging.getLogger(__name__)
# Import the process_image function from the remove_labels module and
# assign it to apply_transformations variable
apply_transformations = remove_labels.process_image

# Dictionary that maps image classification model names to their
# corresponding pre-processing functions
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


def create_data_generator(
    generator_type: str, validation_split: float, model_type: str
):
    """Create an instance of ImageDataGenerator class from the tf.keras.preprocessing.image module with the specified configuration.

    Args:
        generator_type (str): The type of generator to create (either "train", "valid", or "eval").
        validation_split (float): The proportion of the dataset to use for validation.
        model_type (str): The name of the image classification model to use.

    Returns:
        tf.keras.preprocessing.image.ImageDataGenerator: The created ImageDataGenerator object.
    """
    generator_type = check_dataset_type(generator_type)
    validation_split = check_validation_split_value(validation_split)

    preproc_func = get_model_preproc(model_type)

    logger.info(
        f"Initializing data generator. Type: {generator_type}, Validation Split:"
        f" {validation_split}")

    generator_mapping = {
        "train": lambda: tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            horizontal_flip=True,
            validation_split=validation_split,
            preprocessing_function=preproc_func,
            cval=0.0,
            fill_mode="constant",
        ),
        "valid": lambda: tf.keras.preprocessing.image.ImageDataGenerator(
            validation_split=validation_split,
            preprocessing_function=preproc_func,
        ),
        "eval": lambda: tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preproc_func,
        ),
    }

    logger.info(f"Data generator created successfully for {generator_type}")
    return generator_mapping[generator_type]()


def create_dataset(
    dataset_type: str,
    data_generator: tf.keras.preprocessing.image.ImageDataGenerator,
    validation_split: float,
    data_directory: str,
    batch_size: int,
    image_size: int,
    seed: int = 44,
) -> tf.keras.preprocessing.image.DirectoryIterator:
    """
    Create a batched generator object from the specified dataset type using the specified ImageDataGenerator object.

    Args:
        dataset_type (str): The type of dataset to create (either "train", "valid", or "eval").
        data_generator (tf.keras.preprocessing.image.ImageDataGenerator): The ImageDataGenerator object created using create_data_generator().
        validation_split (float): The proportion of the dataset to use for validation.
        data_directory (str): The path to the directory containing the dataset images.
        batch_size (int): The batch size to use for the dataset.
        image_size (int): The size to which images should be resized.
        seed (int, optional): The random seed to use for data augmentation. Defaults to 44.

    Returns:
        tf.keras.preprocessing.image.DirectoryIterator: The directory iterator for the dataset.
    """
    logger.info(
        f"Creating dataset of type {dataset_type} with image size {image_size}, batch"
        f" size {batch_size}")
    COLOR_MODE = "rgb"
    CLASS_MODE = "binary"

    dataset_type = check_dataset_type(dataset_type)
    target_size = (check_image_size(image_size), check_image_size(image_size))
    seed = check_seed(seed)
    batch_size = check_batch_size(batch_size)
    data_directory = check_data_dir(data_directory)

    color_mode = COLOR_MODE
    class_mode = CLASS_MODE

    arguments = {
        "directory": data_directory,
        "target_size": target_size,
        "batch_size": batch_size,
        "seed": seed,
        "color_mode": color_mode,
        "class_mode": class_mode,
    }

    if dataset_type == "train":
        arguments["shuffle"] = True
        arguments["subset"] = "training" if validation_split > 0 else None

    elif dataset_type == "valid":
        arguments["shuffle"] = False
        arguments["subset"] = "validation" if validation_split > 0 else None

    elif dataset_type == "eval":
        arguments["shuffle"] = False

    logger.info(f"Dataset created successfully for {dataset_type}")
    return data_generator.flow_from_directory(**arguments)
