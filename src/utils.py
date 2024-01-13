import silence_tensorflow.auto
import tensorflow_addons as tfa
import tensorflow as tf
import os
import re
from pathlib import Path
import warnings

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
