import os
import tensorflow as tf
import keras
import bentoml
import tensorflow.keras.metrics as metrics
import focal_loss

# Directory containing the model files
model_dir = "saved_checkpoints"


def is_keras_model_file(filename: str) -> bool:
    #Checks if the given filename is a Keras model file.
    return (
        filename.endswith(".h5")
        or filename.endswith(".hdf5")
        or os.path.isdir(os.path.join(model_dir, filename))
    )


def convert_models_to_bentos():
    for model_file in os.listdir(model_dir):
        if not is_keras_model_file(model_file):
            # Skip if not a Keras model file
            continue

        model_path = os.path.join(model_dir, model_file)

        try:
            model = keras.models.load_model(model_path)
        except Exception as e:
            print(f"Failed to load model {model_file}: {e}")
            continue

        # These are the same parameters that Keras will use during training.
        model.compile(
            loss=focal_loss.BinaryFocalLoss(gamma=1.8), # BinaryFocalLoss performs great!
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

        # Save as a BentoML service
        bentoml.tensorflow.save_model(
            "keras_" + model_file,
            model,
            signatures={"__call__": {"batchable": True, "batch_dim": 0}}, # Make sure batching is enabled for efficiency during inference.
        )

if __name__ == '__main__':
    convert_models_to_bentos()