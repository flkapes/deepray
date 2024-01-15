# import silence_tensorflow.auto
from PIL.Image import Image as PILImage
import numpy as np
import bentoml
from bentoml.io import JSON
import asyncio
import PIL.Image
from bentoml.io import NumpyNdarray
from bentoml.io import Image
import remove_labels
from dataset import get_model_preproc

# Load the TensorFlow models for different bones.
# At least the top model + one extra picked arbitrarily are served for
# each bone.
densenet201_xr_elbow = bentoml.tensorflow.get(
    "keras_densenet201_xr_elbow.h5:tf6u3pfragmk2edp"
).to_runner(max_latency_ms=180000)
densenet121_xr_elbow = bentoml.tensorflow.get(
    "keras_densenet121_xr_elbow.h5:g2p2chfrakmk2edp"
).to_runner(max_latency_ms=180000)
densenet201_xr_shoulder = bentoml.tensorflow.get(
    "keras_densenet201_xr_shoulder.h5:iagfy5fracmk2edp"
).to_runner(max_latency_ms=180000)
resnet101_xr_shoulder = bentoml.tensorflow.get(
    "keras_resnet101_xr_shoulder.h5:7diloafracmk2edp"
).to_runner(max_latency_ms=180000)
densenet169_xr_finger = bentoml.tensorflow.get(
    "keras_densenet169_xr_finger.h5:avst6qfragmk2edp"
).to_runner(max_latency_ms=180000)
resnet152_xr_wrist = bentoml.tensorflow.get(
    "keras_resnet152_xr_wrist.h5:y4lgiwfracmk2edp"
).to_runner(max_latency_ms=180000)
resnet152v2_xr_humerus = bentoml.tensorflow.get(
    "keras_resnet152v2_xr_humerus.h5:enzjxofragmk2edp"
).to_runner(max_latency_ms=180000)
resnet101v2_xr_forearm = bentoml.tensorflow.get(
    "keras_resnet101v2_xr_forearm.h5:ts2nf7vq76mk2edp"
).to_runner(max_latency_ms=180000)
resnet101_xr_hand = bentoml.tensorflow.get(
    "keras_resnet101_xr_hand.h5:flbojkfrakmk2edp"
).to_runner(max_latency_ms=180000)
xception_xr_hand = bentoml.tensorflow.get(
    "keras_xception_xr_hand.h5:fgiwuovq76mk2edp"
).to_runner(max_latency_ms=180000)
densenet169_xr_wrist = bentoml.tensorflow.get(
    "keras_densenet169_xr_wrist.h5:mn34thfq76mk2edp"
).to_runner(max_latency_ms=180000)
resnet101v2_xr_humerus = bentoml.tensorflow.get(
    "keras_resnet101v2_xr_humerus.h5:qt5irrfracmk2edp"
).to_runner(max_latency_ms=180000)
densenet201_xr_forearm = bentoml.tensorflow.get(
    "keras_densenet201_xr_forearm.h5:3ta5xafq76mk2edp"
).to_runner(max_latency_ms=180000)
resnet152_xr_finger = bentoml.tensorflow.get(
    "keras_resnet152_xr_finger.h5:rcangwfragmk2edp"
).to_runner(max_latency_ms=180000)
resnet101v2_xr_elbow = bentoml.tensorflow.get(
    "keras_resnet101v2_xr_elbow.h5:nhy4pcvragmk2edp"
).to_runner(max_latency_ms=180000)
resnet152_xr_hand = bentoml.tensorflow.get(
    "keras_resnet152_xr_hand.h5:x7bokgfragmk2edp"
).to_runner(max_latency_ms=180000)

# Define the BentoML service with all the model runners.
svc = bentoml.Service(
    name="deepray_tensorflow",
    runners=[
        densenet201_xr_shoulder,
        resnet101v2_xr_humerus,
        densenet201_xr_forearm,
        resnet152_xr_finger,
        resnet101_xr_shoulder,
        resnet101v2_xr_elbow,
        resnet152_xr_hand,
        resnet152_xr_wrist,
        densenet201_xr_elbow,
        densenet169_xr_finger,
        densenet169_xr_wrist,
        resnet152v2_xr_humerus,
        resnet101v2_xr_forearm,
        xception_xr_hand,
    ],
)


@svc.api(input=Image(), output=NumpyNdarray(dtype="float32"))
async def predict_image(
    f: PIL.Image.Image, ctx: bentoml.Context
) -> "np.ndarray[np.int64]":
    """
    Asynchronous API function to predict the class of the input image.
    You must include the request headers: 'model_type' and 'bone_type'.
    Args:
        f: PIL.Image.Image object, the input image.
        ctx: bentoml.Context object, provides access to the request headers.

    Returns:
        np.ndarray[np.int64]: The output tensor containing the prediction result.
    """
    assert isinstance(f, PILImage)
    request_headers = ctx.request.headers
    f = f.resize((324, 324)).convert("RGB")
    f = remove_labels.process_image(f, True)
    arr = np.array(f)  # / 255.0

    model_type = request_headers["model_type"]
    bone_type = request_headers["bone_type"]
    # remove_labels = request_headers['labels']
    arr = np.expand_dims(arr, 0).astype("float32")
    func = get_model_preproc(model_type)
    arr = func(arr)
    # preprocessing_func = get_model_preproc(model_type)
    # arr = preprocessing_func(arr)
    if bone_type == "XR_HUMERUS":
        if model_type == "resnet152v2":
            output_tensor = await resnet152v2_xr_humerus.async_run(arr)
        else:
            output_tensor = await resnet101v2_xr_humerus.async_run(arr)
    elif bone_type == "XR_ELBOW":
        if model_type == "densenet121":
            output_tensor = await densenet121_xr_elbow.async_run(arr)
        elif model_type == "resnet101v2":
            output_tensor = await resnet101v2_xr_elbow.async_run(arr)
        else:
            output_tensor = await densenet201_xr_elbow.async_run(arr)
    elif bone_type == "XR_HAND":
        if model_type == "xception":
            output_tensor = await xception_xr_hand.async_run(arr)
        elif model_type == "resnet152":
            output_tensor = await resnet152_xr_hand.async_run(arr)
        else:
            output_tensor = await resnet101_xr_hand.async_run(arr)
    elif bone_type == "XR_FINGER":
        if model_type == "densenet169":
            output_tensor = await densenet169_xr_finger.async_run(arr)
        elif model_type == "resnet152":
            output_tensor = await resnet152_xr_finger.async_run(arr)
    elif bone_type == "XR_FOREARM":
        if model_type == "densenet201":
            output_tensor = await densenet201_xr_forearm.async_run(arr)
        else:
            output_tensor = await resnet101v2_xr_forearm.async_run(arr)
    elif bone_type == "XR_WRIST":
        if model_type == "densenet169":
            output_tensor = await densenet169_xr_wrist.async_run(arr)
        elif model_type == "resnet152":
            output_tensor = await resnet152_xr_wrist.async_run(arr)

    elif bone_type == "XR_SHOULDER":
        if model_type == "resnet101":
            output_tensor = await resnet101_xr_shoulder.async_run(arr)
        else:
            output_tensor = await densenet201_xr_shoulder.async_run(arr)

    return output_tensor
