import silence_tensorflow.auto
import tensorflow as tf
import remove_labels
from utils import check_image_size, check_batch_size, check_seed, check_data_dir, check_dataset_type, check_validation_split_value
from models import list_models

# Import the process_image function from the remove_labels module and assign it to apply_transformations variable
apply_transformations = remove_labels.process_image

# Dictionary that maps image classification model names to their corresponding pre-processing functions
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
    return transformations[model_str.strip().lower()]


def create_data_generator(
    generator_type: str, validation_split: float, model_type: str
):
    """Create an instance of ImageDataGenerator class from the tf.keras.preprocessing.image module with the specified configuration.

    Args:
        generator_type (str): The type of generator to create (either "train", "valid", or "eval").
        valid_split_value (float): The proportion of the dataset to use for validation.
        model_type (str): The name of the image classification model to use.

    Returns:
        tf.keras.preprocessing.image.ImageDataGenerator: The created ImageDataGenerator object.
    """
    generator_type = check_dataset_type(generator_type)
    validation_split = check_validation_split_value(validation_split)
    model_type = check_model_name(model, list_models)

    preproc_func = get_model_preproc(model_type)

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
        "data_directory": data_directory,
        "target_size": target_size,
        "batch_size": batch_size,
        "seed": seed,
        "color_mode": color_mode,
        "class_mode": class_mode
    }

    if dataset_type == "train":
        arguments["shuffle"] = True
        arguments["subset"] = 'training' if validation_split > 0 else None

    elif dataset_type == "valid":
       arguments["shuffle"] = False
       arguments["subset"] = 'validation' if validation_split > 0 else None

    elif dataset_type == "eval":
        arguments["shuffle"] = False

    return data_generator.flow_from_directory(**arguments)