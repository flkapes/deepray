import silence_tensorflow.auto
import tensorflow as tf
import os
import re
from pathlib import Path
import warnings
from exceptions import *
import logging
import logging.config
import json

try:
    with open("logging_config.json", "r") as config_file:
        config_dict = json.load(config_file)
except:
    with open(os.environ["logging"], "r") as config_file:
        config_dict = json.load(config_file)
logging.config.dictConfig(config_dict)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def get_next_folder_name(
        folder_root_path: str,
        model_name: str,
        bone_type: str):
    """
    Returns the highest number found in the filenames of a specific path.

    Args:
        folder_root_path (str): The root path of the folder.
        model_name (str): The name of the model.
        bone_type (str): The type of bone.

    Returns:
        int: The highest number found in the filenames of the specified path.
    """
    path = os.path.join(folder_root_path, bone_type, model_name)
    maxn = 0
    try:
        for file in os.listdir(path):
            num = int(re.search("run(\\d*)", file).group(1))
            maxn = num if num > maxn else maxn
    except BaseException:
        maxn = 0
    return maxn


def get_transform_from_name(transform_name, **kwargs):
    import transformations as T

    transform_name = transform_name.lower().strip()
    if transform_name == "ahe":
        return T.histogram_equalization
    elif transform_name == "clahe":
        return T.contrast_limited_histogram_equalization
    elif transform_name == "gaussian":
        return T.gaussian_filtering
    elif transform_name == "gamma":
        return T.gamma_correction
    elif transform_name == "all_transforms":
        return T.transform_all
    elif transform_name == "best_config":
        return T.apply_transformations
    elif transform_name == "label_remove":
        import remove_labels

        return remove_labels.preprocess_image


def get_optimizer(optimizer_name: str):
    optimizer_name = optimizer_name.lower().rstrip()
    if optimizer_name == "lookahead":
        return optim.Lookahead
    elif optimizer_name == "radam":
        return optim.RectifiedAdam
    elif optimizer_name == "lazyadam":
        return optim.LazyAdam
    elif optimizer_name == "adam":
        return tf.keras.optimizers.Adam
    elif optimizer_name == "adamw":
        return optim.AdamW
    elif optimizer_name == "sgd":
        return tf.keras.optimizer.SGD
    elif optimizer_name == "yogi":
        return optim.Yogi
    elif optimizer_name == "rmsprop":
        return tf.keras.optimizers.experimental.RMSprop
    else:
        return optim


def set_device():
    """
    Sets the device to be used for training. If a GPU is available, it sets the
    visible devices to GPU and sets memory growth to True. If a TPU is
    available, it connects to the TPU and initializes the TPU system. If neither
    GPU nor TPU is available, it sets the visible devices to CPU. Returns the
    dtype for mixed precision training.
    """
    if tf.config.list_physical_devices("GPU"):
        print("GPU is available")
        for i, gpu in enumerate(tf.config.list_physical_devices("GPU")):
            try:
                tf.config.experimental.set_visible_devices(
                    [], f"/device:GPU:{i}")
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        return "mixed_float16"
    elif "COLAB_TPU_ADDR" in os.environ:
        print("TPU is available")
        tpu_address = "grpc://" + os.environ["COLAB_TPU_ADDR"]
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_address)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        tf.config.experimental.set_default_device(resolver.master())
        return "mixed_float16"
    else:
        print("GPU and TPU are not available, using CPU")
        tf.config.set_visible_devices([], "GPU")
        tf.config.set_visible_devices([], "TPU")
        tf.config.set_visible_devices(
            tf.config.list_physical_devices("CPU"), "CPU")
        return "float32"


def check_dropout_range(dropout):
    """
    Check if the dropout rate is within the valid range of 0 and 1.

    Args:
        dropout (int): The dropout rate to be checked.

    Returns:
        int: The input dropout rate if it is valid.

    Raises:
        InvalidDropoutRateException: If the dropout rate is invalid.
    """
    if not 0 <= dropout <= 1:
        logger.error(
            f"Invalid Dropout Rate: {dropout}. Must be between 0 and 1.")
        raise InvalidDropoutRateException(
            "Dropout rate must be between 0 and 1.")
    return dropout


def check_regularizer(regularizer):
    """
    Check if the regularizer is valid and return it.

    Args:
        regularizer (str): The type of regularizer to be checked.

    Returns:
        str: The valid regularizer.

    Raises:
        InvalidRegularizerException: If the regularizer is not valid.
    """
    valid_regularizers = ["l1", "l2", "l1_l2"]
    if regularizer not in valid_regularizers:
        logger.error(
            f"Invalid Regularizer: {regularizer}. Valid options:"
            f" {', '.join(valid_regularizers)}"
        )
        raise InvalidRegularizerException(
            "Invalid regularization technique. Please choose from: "
            + ", ".join(valid_regularizers)
        )
    return regularizer


def check_seed(seed):
    """
    Check if the input seed is an integer and return it.

    Args:
        seed (int): the seed value to be checked

    Returns:
        int: the input seed if it is an integer

    Raises:
        InvalidSeedValueException: if the input seed is not an integer
    """
    if not isinstance(seed, int):
        logger.error(f"Invalid Seed Value: {seed}. Must be an integer.")
        raise InvalidSeedValueException("Seed value must be an integer.")
    return seed


def check_trainable_layers(trainable_layers):
    """
    Check if the input trainable_layers is an integer and return it.

    Args:
        trainable_layers (int): the number of trainable layers
    Returns:
        int: the number of trainable layers
    Raises:
        InvalidTrainableLayersException: if trainable_layers is not an integer
    """
    if not isinstance(trainable_layers, int):
        logger.error(
            f"Invalid Trainable Layers: {trainable_layers}. Must be an integer."
        )
        raise InvalidTrainableLayersException(
            "Trainable layers must be an integer.")
    return trainable_layers


def check_image_size(image_size):
    """
    Validates the given image size.

    Args:
        image_size (int): The size of the image to be validated.

    Returns:
        int: The validated image size.

    Raises:
        InvalidImageSizeException: If the image size is not an integer or not within the valid range.
    """
    if not isinstance(image_size, int):
        logger.error(f"Invalid Image Size: {image_size}. Must be an integer.")
        raise InvalidImageSizeException("Image size must be an integer.")
    if not 0 <= image_size <= 1024:
        logger.error(
            f"Invalid Image Size: {image_size}. Must be between 0 and 1024.")
        raise InvalidImageSizeException(
            "Image size must be between 0 and 1024.")
    return image_size


def check_model_name(model_name, model_dict):
    """
    Check if the model name exists in the model dictionary and raise an exception if not found.

    Args:
        model_name (str): The name of the model to check.
        model_dict (dict): A dictionary containing available model names.

    Returns:
        str: The model name if it exists in the dictionary.

    Raises:
        ModelNotFoundException: If the model name is not found in the dictionary.
    """
    if model_name not in model_dict():
        logger.error(
            f"Model Not Found: {model_name}. Available models:"
            f" {', '.join(model_dict())}"
        )
        raise ModelNotFoundException(
            "Model not found. Please choose from: " + ", ".join(model_dict())
        )
    return model_name


def check_dataset_type(dataset_type):
    """
    Check the dataset type and raise an exception if it is not 'train', 'valid', or 'eval'.

    Args:
        dataset_type (str): The type of dataset to check.

    Returns:
        str: The dataset type if it is valid.

    Raises:
        InvalidDatasetTypeException: If the dataset type is not 'train', 'valid', or 'eval'.
    """
    if dataset_type not in ["train", "valid", "eval"]:
        logger.error(
            f"Invalid Dataset Type: {dataset_type}. Valid types: 'train', 'valid',"
            " 'eval'")
        raise InvalidDatasetTypeException(
            "Invalid dataset type. Please choose from: 'train', 'valid', or 'eval'"
        )
    return dataset_type


def check_batch_size(batch_size):
    """
    Check the batch size for validity and return it if valid.

    Args:
        batch_size (int): The batch size to be checked.

    Returns:
        int: The valid batch size.

    Raises:
        InvalidBatchSizeException: If the batch size is not an integer.
    """
    if not isinstance(batch_size, int):
        logger.error(f"Invalid Batch Size: {batch_size}. Must be an integer.")
        raise InvalidBatchSizeException("Batch size must be an integer.")
    return batch_size


def check_data_dir(data_dir):
    """
    Check if the data directory exists and is a directory.

    Args:
        data_dir (str): The path to the data directory.

    Returns:
        str: The validated data directory path.

    Raises:
        InvalidDataDirException: If the data directory does not exist or is not a directory.
    """
    if not os.path.exists(data_dir):
        logger.error(f"Data Directory Does Not Exist: {data_dir}")
        raise InvalidDataDirException("Data directory does not exist.")
    if not os.path.isdir(data_dir):
        logger.error(f"Data Directory is Not a Directory: {data_dir}")
        raise InvalidDataDirException("Data directory is not a directory.")
    return data_dir


def check_validation_split_value(valid_split_value):
    """
    Check the validation split value.

    Args:
        valid_split_value (float) The value to be validated
        
    Returns:
        float: The validated split value, or None if the input is not valid
    """
    if not 0 <= valid_split_value <= 1:
        logger.error(
            f"Invalid Validation Split Value: {valid_split_value}. Must be between 0"
            " and 1.")
        return None
    return valid_split_value
