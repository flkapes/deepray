import logging
import logging.config
import json
from typing import Optional
import time

import silence_tensorflow.auto
import tensorflow as tf
import tensorflow.keras.applications as models
from tensorflow.keras.layers import Dense, CenterCrop, Dropout, LeakyReLU
from tensorflow.keras.initializers import GlorotUniform
import os
from tensorflow.keras import Sequential

from utils import (
    check_model_name,
    check_dropout_range,
    check_image_size,
    check_regularizer,
    check_seed,
    check_trainable_layers,
)

try:
    with open("logging_config.json", "r") as config_file:
        config_dict = json.load(config_file)
except:
    with open(os.environ["logging"], "r") as config_file:
        config_dict = json.load(config_file)

logging.config.dictConfig(config_dict)
logger = logging.getLogger(__name__)

# Dictionary that maps model names to their corresponding Keras model classes
model_classes = {
    "resnet101": models.resnet.ResNet101,
    "resnet152": models.resnet.ResNet152,
    "densenet121": models.densenet.DenseNet121,
    "densenet169": models.densenet.DenseNet169,
    "densenet201": models.densenet.DenseNet201,
    "resnet101v2": models.resnet_v2.ResNet101V2,
    "resnet152v2": models.resnet_v2.ResNet152V2,
    "vgg16": models.vgg16.VGG16,
    "vgg19": models.vgg19.VGG19,
    "inceptionv3": models.InceptionV3,
    "inception_resnetv2": models.inception_resnet_v2.InceptionResNetV2,
    "xception": models.xception.Xception,
}


def get_model(model_name: str) -> tf.keras.Model:
    """Return the Keras model for a given model name.

    Args:
        model_name (str): The name of the model to retrieve.

    Returns:
        tf.keras.Model: The Keras model object for the specified model name.
    """
    model = model_classes[check_model_name(
        model_name, list_models).strip().lower()]
    logger.info(f"Model '{model}' successfully retrieved")
    return model


def list_models() -> list:
    """Return a list of available model names.

    Returns:
        list: A list of available model names.
    """
    logger.info(f"List of available models queried successfully.")
    return list(model_classes.keys())


def get_configured_model(
    model_name: str,
    image_size: int = 324,
    dropout: float = 0.3,
    seed: Optional[int] = None,
    trainable_layers: int = 0,
    regularizer: str = "l2",
) -> tf.keras.Model:
    """Return a compiled and configured instance of a Keras model.

    Args:
        model_name (str): The name of the Keras model to use.
        uncropped_image (int, optional): The size of the input image. Defaults to 324.
        crop_layer (int, optional): The size of the crop layer to use for model training. Defaults to 324.
        dropout (float, optional): The proportion of units to drop during training. Defaults to 0.3.
        seed (int, optional): The random seed to use for model initialization. Defaults to None.
        trainable_layers (int, optional): The number of fine-tuneable layers to use during training.
        regularizer (str, optional): The regularization technique to use. Defaults to "l2".

    Returns:
        tf.keras.Model: The compiled and configured instance of the specified Keras model.
    """
    retrieved_class = get_model(model_name)
    logger.info(
        f"Configuring model: {model_name} with image size {image_size}, dropout"
        f" {dropout}, seed {seed}")
    model_base = retrieved_class(
        weights="imagenet",
        input_shape=(
            check_image_size(image_size),
            check_image_size(image_size),
            3),
        include_top=False,
        pooling="avg",
    )

    model_base.trainable = False
    initializer = GlorotUniform(
        seed=int(time.time()) if seed is None else check_seed_value(seed)
    )

    trainable_layers = abs(check_trainable_layers(trainable_layers))
    logger.info("Found trainable layers %s", trainable_layers)
    for layer in model_base.layers[-trainable_layers:]:
        layer.trainable = True

    x = model_base.output
    x = Dense(
        128,
        kernel_regularizer=check_regularizer(regularizer),
        kernel_initializer=initializer,
        activation="relu",
    )(x)
    x = Dense(32, activation="relu", kernel_initializer=initializer)(x)

    x = Dropout(check_dropout_range(dropout))(x)

    output = Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=model_base.input, outputs=output)
    logger.info(f"Model '{model_name}' successfully configured")
    return model
