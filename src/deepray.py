import os
import time
import warnings

import silence_tensorflow.auto
import tensorflow.keras.metrics as metrics
import tensorflow as tf
import bentoml
import focal_loss

from get_class_weights2 import do as get_weights
from transformations import apply_transformations
from dataset import create_dataset, create_data_generator
from models import get_configured_model
from utils import set_device, get_next_folder_name

# Suppress warnings for better readability
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set global policy for mixed precision training on GPU
tf.keras.mixed_precision.set_global_policy(set_device())


def train(PARAMS, train_dir=None, eval_dir=None):
    """
    Train a deep learning model using the specified parameters.

    Args:
        PARAMS (dict): A dictionary of model parameters.
        train_dir (str, optional): Path to the training directory. Defaults to None.
        eval_dir (str, optional): Path to the evaluation directory. Defaults to None.

    Returns:
        None

    """
    # Extract relevant parameters from PARAMS dictionary
    dataset_path = PARAMS["dataset_path"]
    body_part = PARAMS["body_part"]
    model_str = PARAMS["model"]
    checkpoint_save_path = PARAMS["checkpoint_save_path"]

    # Define paths for training, evaluation, and checkpoint directories
    if train_dir is not None and eval_dir is not None:
        eval_dir = eval_dir
        checkpoint_path = os.path.join(checkpoint_save_path, body_part, model_str)
    else:
        try:
            train_dir = os.path.join(dataset_path, "train", body_part)
            eval_dir = os.path.join(dataset_path, "valid", body_part)
        except:
            train_dir = os.path.join(dataset_path, body_part)
            eval_dir = os.path.join(dataset_path, body_part)
        checkpoint_path = os.path.join(checkpoint_save_path, body_part, model_str)
        new_folder = os.path.join(
            checkpoint_path,
            "run"
            + str(get_next_folder_name(checkpoint_save_path, body_part, model_str)),
        )
        os.makedirs(
            new_folder,
            exist_ok=True,
        )

    # Get the configured model
    model = get_configured_model(
        PARAMS["model"],
        image_size=PARAMS["image_size"],
        trainable_layers=PARAMS["fine_tune"],
        regularizer="l1_l2",
    )

    # Compile the model with an optimizer, loss function, and metrics
    model.compile(
        loss=focal_loss.BinaryFocalLoss(gamma=1.8),
        optimizer="adam",
        metrics=[
            metrics.BinaryAccuracy(name="bacc"),
            metrics.Precision(),
            metrics.Recall(),
            metrics.TruePositives(),
            metrics.FalsePositives(),
            metrics.TrueNegatives(),
            metrics.FalseNegatives(),
            metrics.AUC(),
            "mse",
        ],
    )

    # Build the model
    model.build(input_shape=(None, PARAMS["pre_size"], PARAMS["pre_size"], 3))

    # Define callbacks for the training process
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.12,
        patience=2,
        min_delta=0.01,
        verbose=1,
        min_lr=0.000000000001,
    )

    weights = get_weights(train_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.environ.get("LOG_DIR", f"tensorboard/{time.time()}"),
        histogram_freq=1,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PARAMS["patience"], restore_best_weights=True
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=2,
        filepath=os.path.join(
            checkpoint_path,
            "{epoch:04d}--VLoss{val_loss:04f}-Recall{val_recall:04f}-Precision{val_precision:0.4f}.h5",
        ),
    )

    callbacks = [early_stopping, model_checkpoint, reduce_lr, tensorboard_callback]

    # Print a summary of the model architecture
    print(
        f"{PARAMS['model']}        --        {PARAMS['body_part']}        --       "
        + f" ({PARAMS['pre_size']}px ,{PARAMS['pre_size']}px)"
    )
    if PARAMS["weights"] != "none":
        model.load_weights(PARAMS["weights"])
    print(model.summary())

    # Train the model using the created datasets, callbacks, and class weights
    if PARAMS["weights"] == "none":
        train_dataset = create_dataset(
            "train",
            create_data_generator(
                "train", PARAMS["train_val_split"], model_type=PARAMS["model"]
            ),
            PARAMS["train_val_split"],
            train_dir,
            PARAMS["train_batch_size"],
            PARAMS["pre_size"],
            PARAMS["image_size"],
        )
        valid_dataset = create_dataset(
            "valid",
            create_data_generator(
                "valid", PARAMS["train_val_split"], model_type=PARAMS["model"]
            ),
            PARAMS["train_val_split"],
            train_dir,
            PARAMS["valid_batch_size"],
            PARAMS["pre_size"],
            PARAMS["image_size"],
        )

        history = model.fit(
            train_dataset,
            steps_per_epoch=train_dataset.n // PARAMS["train_batch_size"],
            validation_data=valid_dataset,
            use_multiprocessing=False,
            validation_steps=valid_dataset.n // PARAMS["valid_batch_size"],
            callbacks=callbacks,
            epochs=PARAMS["max_epochs"],
            class_weight=weights,
        )

        bentoml.tensorflow.save_model(
            str("last_15_trainable" + PARAMS["model"])
            + "_"
            + PARAMS["body_part"],  # +"_"+str(time.time()),
            model,
            signatures={"__call__": {"batchable": True, "batch_dim": 0}},
            metadata={
                "ValidLoss": history.history["val_loss"][-1],
                "Recall": history.history["val_recall"][-1],
                "Precision": history.history["val_precision"][-1],
                "BinaryAccuracy": history.history["val_bacc"][-1],
                "Epochs": str(len(history.history["val_bacc"])),
                "AUC": history.history["val_auc"][-1],
                "TPR": history.history["val_true_positives"][-1],
                "FPR": history.history["val_false_positives"][-1],
                "TNR": history.history["val_true_negatives"][-1],
                "FNR": history.history["val_false_negatives"][-1],
                "MSE": history.history["val_mse"][-1],
            },
            custom_objects=None,
        )
    else:
        eval_dataset = create_dataset(
            "eval",
            create_data_generator("eval", 0, model_type=PARAMS["model"]),
            0,
            eval_dir,
            PARAMS["eval_batch_size"],
            PARAMS["pre_size"],
            PARAMS["image_size"],
        )
        eval = model.evaluate(eval_dataset, verbose=2)
        print(eval)


if __name__ == "__main__":
    import argparse

    training = argparse.ArgumentParser(
        "deepray.py",
        description=(
            "Run training or inference on 10 different models using the MURA dataset!"
        ),
        epilog="Thanks for using DeepRay! :)",
    )
    training.add_argument(
        "-d",
        "--dataset-path",
        action="store",
        type=str,
        required=True,
        help=(
            "The path to the MURASepatated dataset directory. I.E. This the subfolders"
            " of whatever you set here must be XR_*, and you must set the correct bone"
            " below, using option -p. Former Default Value: /workspace/MURASeparated"
        ),
    )
    training.add_argument(
        "-l",
        "--learning-rate",
        action="store",
        type=float,
        default=7e-3,
        help="The learning rate to be applied to the model being trained.",
    )
    training.add_argument(
        "-w",
        "--weight-decay",
        action="store",
        type=float,
        default=3e-5,
        help=(
            "The weight decay to be applied to the model being trained, if optimizer"
            " supports it. Can be 0 or float value."
        ),
    )
    training.add_argument(
        "-b",
        "--batch-size",
        action="store",
        type=int,
        default=[48, 24, 6],
        nargs="*",
        help=(
            "The batch size being used by the model either the training set, validation"
            " set, testing set or a combination of them. If you want the training set"
            " to have a batch size of 48, and the valid set to have a size of 24, and"
            " you are not providing a test set, use: --batch_size 48 24 0. You must set"
            " all three to a value, and if 0, the argument will be ignored."
        ),
    )
    training.add_argument(
        "-s",
        "--train-valid-split",
        action="store",
        type=float,
        default=0.2,
        help=(
            "The split of training and validation data. 0.2 would mean 80%% of training"
            " data will be used for training and 20%% will be used for validation."
        ),
    )
    training.add_argument(
        "-e",
        "--max-epochs",
        action="store",
        type=int,
        default=48,
        help="The maximum number of epochs training should go on for.",
    )
    training.add_argument(
        "--patience",
        action="store",
        type=int,
        default=6,
        help="The value set for early stopping patience..",
    )
    training.add_argument(
        "-p",
        "--body-part",
        action="store",
        type=str,
        default="XR_HUMERUS",
        help=(
            "The body part being trained on. This value will be appended to the dataset"
            " path, so make sure the name of the folder containing the positive and"
            " negative classes is set to this value."
        ),
    )
    training.add_argument(
        "-H",
        "--image-size",
        action="store",
        type=int,
        default=324,
        help="The image input size for the model. ",
    )
    training.add_argument(
        "-m",
        "--model",
        action="store",
        type=str,
        default="densenet201",
        help="The name of the model being trained. Include the subtype.",
    )
    training.add_argument(
        "-D",
        "--seed",
        action="store",
        type=int,
        default=None,
        help="The seed used to ensure training runs are reproducible.",
    )
    training.add_argument(
        "-x",
        "--docker",
        action="store",
        type=str,
        default="False",
        help=(
            "Argument that toggles the default dataset filepath for easier docker"
            " deployment."
        ),
    )
    training.add_argument(
        "-T",
        "--weights",
        type=str,
        default="none",
    )
    training.add_argument(
        "-U",
        "--trainable-layers",
        type=int,
        default=-8,
        help=(
            "Argument that toggles the default dataset filepath for easier docker"
            " deployment."
        ),
    )
    parsed = training.parse_args()

    if parsed:
        training_params = {
            "dataset_path": parsed.dataset_path.strip(),
            "weight_decay": parsed.weight_decay,
            "lr": parsed.learning_rate,
            "train_batch_size": parsed.batch_size[0],
            "valid_batch_size": parsed.batch_size[1],
            "eval_batch_size": parsed.batch_size[2],
            "train_val_split": parsed.train_valid_split,
            "max_epochs": parsed.max_epochs,
            "body_part": parsed.body_part.strip(),
            "checkpoint_save_path": "checkpoints/",
            "patience": parsed.patience,
            "model": parsed.model.strip(),
            "seed": parsed.seed,
            "image_size": parsed.image_size,
            "docker": parsed.docker,
            "weights": parsed.weights.strip(),
            "fine_tune": parsed.trainable_layers,
        }
        if training_params["docker"] == "meow":
            train_dir = f'/tmp/dset/MURASeparated/{training_params["body_part"]}'
            eval_dir = f'/tmp/dset/MURASeparated/valid/{training_params["body_part"]}'
            training_params["dataset_path"] = "/tmp/dset/MURASeparated"
        else:
            train_dir, eval_dir = None, None
        train(training_params, train_dir, eval_dir)
