import silence_tensorflow.auto
import tensorflow as tf
import time
import tensorflow.keras.applications as models
from tensorflow.keras.layers import Dense, CenterCrop, Dropout, LeakyReLU
from tensorflow.keras import Sequential

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
    "resnet101v2": models.resnet_v2.ResNet101V2,
    "resnet152v2": models.resnet_v2.ResNet152V2,
    "xception": models.xception.Xception,
}


def get_model(model_name: str):
    """Return the Keras model for a given model name.

    Args:
        model_name (str): The name of the model to retrieve.

    Returns:
        tf.keras.Model: The Keras model object for the specified model name.
    """
    return model_classes[model_name.strip().lower()]


def list_models() -> list:
    """Return a list of available model names.

    Returns:
        list: A list of available model names.
    """
    return list(model_classes.keys())


def get_configured_model(
    model_name,
    uncropped_image=324,
    crop_layer=324,
    dropout=0.4,
    seed=None,
    trainable_layers=0,
    regularizer="l2",
):
    """Return a compiled and configured instance of a Keras model.

    Args:
        model_name (str): The name of the Keras model to use.
        uncropped_image (int, optional): The size of the input image. Defaults to 384.
        crop_layer (int, optional): The size of the crop layer to use for model training. Defaults to 384.
        dropout (float, optional): The proportion of units to drop during training. Defaults to 0.4.
        seed (int, optional): The random seed to use for model initialization. Defaults to None.
        trainable_layers (int, optional): The number of fine-tuneable layers to use during training.
        regularizer (str, optional): The regularization technique to use. Defaults to "l2".

    Returns:
        tf.keras.Model: The compiled and configured instance of the specified Keras model.
    """
    resized_image = uncropped_image
    retrieved_class = get_model(model_name)
    model_base = retrieved_class(
        weights="imagenet",
        input_shape=(crop_layer, crop_layer, 3),
        include_top=False,
        pooling="avg",
    )

    model_base.trainable = False
    initializer = tf.keras.initializers.GlorotUniform(
        seed=int(time.time()) if seed is None else seed
    )
    if trainable_layers != 0:
        if trainable_layers < 0:
            trainable_layers *= -1
        print("Found trainable layers " + str(trainable_layers))
        for layer in model_base.layers[-(trainable_layers):]:
            layer.trainable = True

    x = model_base.output

    dense1_added = Dense(
        128,
        kernel_regularizer=regularizer,
        kernel_initializer=initializer,
        activation="relu",
    )(x)
    dense2_added = Dense(32, activation="relu", kernel_initializer=initializer)(
        dense1_added
    )
    dropout1 = tf.keras.layers.Dropout(0.3)(dense2_added)
    sig = Dense(1, activation="sigmoid")(dropout1)
    model = tf.keras.Model(inputs=model_base.input, outputs=sig)
    return model
