import comet_ml
from comet_ml import Optimizer
import pathlib
import os
import time
import warnings
import json
import logging
import logging.config
import argparse

import silence_tensorflow.auto
import tensorflow.keras.metrics as metrics
import tensorflow as tf
import bentoml
import focal_loss

from weights import get_weights
from transformations import apply_transformations
from dataset import create_dataset, create_data_generator
from models import get_configured_model
from utils import set_device, get_next_folder_name

# Set up logging configuration
with open("logging_config.json", "r") as config_file:
    config_dict = json.load(config_file)

logging.config.dictConfig(config_dict)
pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.WARNING)

# Create a logger
logger = logging.getLogger(__name__)

# Suppress warnings for better readability
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set global policy for mixed precision training on GPU
tf.keras.mixed_precision.set_global_policy(set_device())

def optimize(model="resnet152v2", image_size=384, fine_tune=8, regularizer="l1_l2", weight_decay=0, learning_rate=7e-3, patience=3, checkpoint_path="./checkpoints", train_val_split=0.2, train_dir="../../MURASeparated", train_batch_size=48, valid_batch_size=48, max_epochs=2):

    model = get_configured_model(
        model,
        image_size=image_size,
        trainable_layers=fine_tune,
        regularizer=regularizer,
    )

    # Compile the model with an optimizer, loss function, and metrics
    model.compile(
        loss=focal_loss.BinaryFocalLoss(gamma=1.8),
        optimizer=tf.keras.optimizer.Adam(learning_rate=learning_rate, weight_decay=weight_decay),
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
    model.build(input_shape=(None, image_size, image_size, 3))

    # Define callbacks for the training process
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.12,
        patience=patience,
        min_delta=0.01,
        verbose=1,
        min_lr=0.000000000001,
    )

    tensorboard_folder_time = str(time.time())
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.environ.get("LOG_DIR", f"tensorboard/{tensorboard_folder_time}"),
        histogram_freq=1,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=2,
        filepath=str(
            checkpoint_path
            / "{epoch:04d}--VLoss{val_loss:04f}-Recall{val_recall:04f}-Precision{val_precision:0.4f}.h5"
        ),
    )

    callbacks = [early_stopping, model_checkpoint, reduce_lr, tensorboard_callback]

    # Print a summary of the model architecture
    logger.info(
        f"{PARAMS['model']}        --        {PARAMS['body_part']}        --       "
        + f" ({PARAMS['image_size']}px ,{PARAMS['image_size']}px)"
    )

    # Train the model using the created datasets, callbacks, and class weights
    weights = get_weights(train_dir)
    print("model weights not loaded")
    train_dataset = create_dataset(
        "train",
        create_data_generator(
            "train", train_val_split, model_type=model,
        ),
        train_val_split,
        train_dir,
        train_batch_size,
        image_size,
    )
    valid_dataset = create_dataset(
        "valid",
        create_data_generator(
            "valid", train_val_split, model_type=model,
        ),
        train_val_split,
        train_dir,
        valid_batch_size,
        image_size
    )
    history = model.fit(
        train_dataset,
        steps_per_epoch=train_dataset.n // train_batch_size,
        validation_data=valid_dataset,
        use_multiprocessing=False,
        validation_steps=valid_dataset.n // valid_batch_size,
        callbacks=callbacks,
        epochs=max_epochs,
        class_weight=weights,
    )

config = {
    "algorithm": "bayes",
    "spec": {
        "objective": "minimize",
        "metric": "loss",
    },
    "parameters": {
        "model_trainable_layers": {"type": "integer", "min": 2, "max": 16},
        "regularizer": {"type": "categorical", "values": ["l1", "l2", "l1_l2"]},
        "image_size": {"type": "integer", "min": 224, "max": 512},
        "learning_rate": {"type": "double", "min": 7e-6, "max": 7e-2},

    }
}

opt = Optimizer(config)

for experiment in opt.get_experiments(project_name="optimizer-search-02"):
    loss = optimize(fine_tune=experiment.get_parameter("model_trainable_layers"), regularizer=experiment.get_parameter("regularizer"), image_size=experiment.get_parameter("image_size"), learning_rate=experiment.get_parameter("learning_rate"))
    experiment.log_metric("loss", loss)