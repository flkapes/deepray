import silence_tensorflow.auto
import tensorflow as tf
import remove_labels

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
    generator_type: str, valid_split_value: float, model_type: str
):
    """Create an instance of ImageDataGenerator class from the tf.keras.preprocessing.image module with the specified configuration.

    Args:
        generator_type (str): The type of generator to create (either "train", "valid", or "eval").
        valid_split_value (float): The proportion of the dataset to use for validation.
        model_type (str): The name of the image classification model to use.

    Returns:
        tf.keras.preprocessing.image.ImageDataGenerator: The created ImageDataGenerator object.
    """
    if generator_type == "train":
        if valid_split_value > 0:
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
                horizontal_flip=True,
                validation_split=valid_split_value,
                preprocessing_function=get_model_preproc(model_type),
                cval=0.0,
                fill_mode="constant",
            )
        else:
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
                horizontal_flip=True,
                preprocessing_function=get_model_preproc(model_type),
                cval=0.0,
                fill_mode="constant",
            )
        return data_gen
    elif generator_type == "valid":
        if valid_split_value > 0:
            valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                validation_split=valid_split_value,
                preprocessing_function=get_model_preproc(model_type),
            )
        else:
            valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=get_model_preproc(model_type),
            )
        return valid_datagen
    elif generator_type == "eval":
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=get_model_preproc(model_type),
        )
        return test_datagen


def create_dataset(
    dataset_type: str,
    data_generator,
    valid_split_value: float,
    data_directory: str,
    batch_size: int,
    image_size: int,
    model_crop_layer_size=None,
    seed=44,
):
    """Create a batched generator object from the specified dataset type using the specified ImageDataGenerator object.
     Args:
         dataset_type (str): The type of dataset to create (either "train", "valid", or "eval").
         data_generator (tf.keras.preprocessing.image.ImageDataGenerator): The ImageDataGenerator object created using create_data_generator().
         valid_split_value (float): The proportion of the dataset to use for validation.
         data_directory (str): The path to the directory containing the dataset images.
         batch_size (int): The batch size to use for the dataset.
         image_size (int): The size to which images should be resized.
         model_crop_layer_size (int, optional): The size of the crop layer to use for model training. Defaults to None.
         seed (int, optional): The random seed to use for data augmentation. Defaults to 44.
    Returns:
         tf.keras.preprocessing.image.DirectoryIterator: The directory iterator for the dataset.
    """
    if dataset_type == "train":
        dtype = "training"
        if valid_split_value > 0:
            train_set = data_generator.flow_from_directory(
                data_directory,
                target_size=(image_size, image_size),
                batch_size=batch_size,
                seed=seed,
                color_mode="rgb",
                class_mode="binary",
                subset=dtype,
                shuffle=True,
            )
        else:
            train_set = data_generator.flow_from_directory(
                data_directory,
                target_size=(image_size, image_size),
                batch_size=batch_size,
                seed=seed,
                color_mode="rgb",
                class_mode="binary",
                shuffle=True,
            )
        return train_set
    elif dataset_type == "valid":
        dataset_type = "validation"
        if valid_split_value > 0:
            valid_set = data_generator.flow_from_directory(
                data_directory,
                target_size=(image_size, image_size),
                batch_size=batch_size,
                seed=seed,
                color_mode="rgb",
                class_mode="binary",
                subset=dataset_type,
                shuffle=False,
            )
        else:
            valid_set = data_generator.flow_from_directory(
                data_directory,
                target_size=(image_size, image_size),
                batch_size=batch_size,
                seed=seed,
                color_mode="rgb",
                class_mode="binary",
                shuffle=False,
            )
        return valid_set
    elif dataset_type == "eval":
        test_set = data_generator.flow_from_directory(
            data_directory,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            seed=seed,
            color_mode="rgb",
            class_mode="binary",
            shuffle=False,
        )
        return test_set
