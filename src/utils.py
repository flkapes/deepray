import silence_tensorflow.auto
import tensorflow_addons as tfa
import tensorflow as tf
import os
import re
from pathlib import Path
import warnings
from exceptions import *
import logging
import logging.config
import json

with open("logging_config.json", "r") as config_file:
    config_dict = json.load(config_file)
logging.config.dictConfig(config_dict)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def get_next_folder_name(folder_root_path: str, model_name: str, bone_type: str):
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
    if tf.config.list_physical_devices("GPU"):
        print("GPU is available")
        for i, gpu in enumerate(tf.config.list_physical_devices("GPU")):
            try:
                tf.config.experimental.set_visible_devices([], f"/device:GPU:{i}")
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        return "mixed_float16"
    elif "COLAB_TPU_ADDR" in os.environ:
        print("TPU is available")
        tpu_address = "grpc://" + os.environ["COLAB_TPU_ADDR"]
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        tf.config.experimental.set_default_device(resolver.master())
        return "mixed_float16"
    else:
        print("GPU and TPU are not available, using CPU")
        tf.config.set_visible_devices([], "GPU")
        tf.config.set_visible_devices([], "TPU")
        tf.config.set_visible_devices(tf.config.list_physical_devices("CPU"), "CPU")
        return "float32"


def check_dropout_range(dropout):
    if not 0 <= dropout <= 1:
        logger.error(f"Invalid Dropout Rate: {dropout}. Must be between 0 and 1.")
        raise InvalidDropoutRateException("Dropout rate must be between 0 and 1.")
    return dropout


def check_regularizer(regularizer):
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
    if not isinstance(seed, int):
        logger.error(f"Invalid Seed Value: {seed}. Must be an integer.")
        raise InvalidSeedValueException("Seed value must be an integer.")
    return seed


def check_trainable_layers(trainable_layers):
    if not isinstance(trainable_layers, int):
        logger.error(
            f"Invalid Trainable Layers: {trainable_layers}. Must be an integer."
        )
        raise InvalidTrainableLayersException("Trainable layers must be an integer.")
    return trainable_layers


def check_image_size(image_size):
    if not isinstance(image_size, int):
        logger.error(f"Invalid Image Size: {image_size}. Must be an integer.")
        raise InvalidImageSizeException("Image size must be an integer.")
    if not 0 <= image_size <= 1024:
        logger.error(f"Invalid Image Size: {image_size}. Must be between 0 and 1024.")
        raise InvalidImageSizeException("Image size must be between 0 and 1024.")
    return image_size


def check_model_name(model_name, model_dict):
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
    if dataset_type not in ["train", "valid", "eval"]:
        logger.error(
            f"Invalid Dataset Type: {dataset_type}. Valid types: 'train', 'valid',"
            " 'eval'"
        )
        raise InvalidDatasetTypeException(
            "Invalid dataset type. Please choose from: 'train', 'valid', or 'eval'"
        )
    return dataset_type


def check_batch_size(batch_size):
    if not isinstance(batch_size, int):
        logger.error(f"Invalid Batch Size: {batch_size}. Must be an integer.")
        raise InvalidBatchSizeException("Batch size must be an integer.")
    return batch_size


def check_data_dir(data_dir):
    if not os.path.exists(data_dir):
        logger.error(f"Data Directory Does Not Exist: {data_dir}")
        raise InvalidDataDirException("Data directory does not exist.")
    if not os.path.isdir(data_dir):
        logger.error(f"Data Directory is Not a Directory: {data_dir}")
        raise InvalidDataDirException("Data directory is not a directory.")
    return data_dir


def check_validation_split_value(valid_split_value):
    if not 0 <= valid_split_value <= 1:
        logger.error(
            f"Invalid Validation Split Value: {valid_split_value}. Must be between 0"
            " and 1."
        )
        return None
    return valid_split_value
