
# DeepRay

DeepRay is a collection of models and training code, built to classify X-Ray images of bones as normal or abnormal, such as with images in the MURA Dataset. This repository will contain the training code, and the code to convert trained models into an inference-compatible format for the framework BentoML. I have yet to publish the front end of this project, but I will soon. This was my dissertation project in my final year of University, and the goal here is to polish it up. Would love your thoughts on this, otherwise, enjoy!

## Available Model Types

- DenseNet121
- DenseNet169
- DenseNet201
- ResNet101
- ResNet152
- ResNet101V2
- ResNet152V2
- InceptionV3
- InceptionResNetV2
- Xception


## Dataset downloads & Pretrained Weights

### Autodownload script, make sure that you have installed the requirements already:
```bash
  pip install -r requirements.txt
  python download.py --dataset
  python download.py --saved-checkpoints
  python download.py --mura-dataset
```

### Download MURA directly from my Silicon Valley based Object Store

```bash
  wget -O MURA-v1.1.zip http://sjc1.vultrobjects.com/mura-dataset/MURA-v1.1.zip
  unzip MURA-v1.1.zip
  python extractimages.py
  rm MURA-v1.1.zip
```

### Download the pre-prepared MURA Dataset Folder from my Silicon Valley based Object Store:

```bash
wget -O MURASeparated.zip http://sjc1.vultrobjects.com/mura-dataset/MURASeparated.zip
unzip MURASeparated.zip
rm MURASeparated.zip
```

### Download Pretrained Weights Files from my Silicon Valley based Object Store

```bash
wget -O SavedCheckpoints.zip http://sjc1.vultrobjects.com/mura-dataset/SavedCheckpoints.zip
unzip SavedCheckpoints.zip
rm SavedCheckpoints.zip
```

## Argument Descriptions
```bash
  -d <training_and_validation_path>                   -> /root/MURASeparated
  -l <learning_rate>                                  -> 7e-3
  -w <weight_decay>                                   -> 0
  -m <model_name>                                     -> densenet201
  -b <train_batch_size> <val_batch_size> <eval_size>  -> 48 24 0
  -s <training_validation_split>                      -> 0.2
  -e <max_epochs>                                     -> 48
  -p <body_part>                                      -> XR_FOREARM
  -H <image_input_size>                               -> 324
  -L <crop_image_size>                                -> 324
  -D <seed>                                           -> 44
  -T <path_to_weights_file>                           -> path/to/saved_checkpoints/
  -U <no_of_layers_to_finetune>                       -> 8
  --patience <early_stopping_patience>                -> 4
```
## Run Locally

Go to the project directory, i.e. $HOME/Project/:

```bash
  cd Project/
```

Install dependencies:

```bash
  pip install -r requirements.txt
```

Enter src folder:
```bash
  cd src/
```

Run Training:
```bash
  python deepray.py -d <training_and_validation_path> -l 7e-3 -m resnet152 -b 48 24 0 -s 0.2 -e 48 -p XR_ELBOW -H 324 -L 324 -U 8 --patience 4
```

Run Evaluation / Load Weights:
```bash
  python deepray.py -d <evaluation_path> -l 7e-3 -m resnet152 -b 48 24 0 -s 0.2 -e 48 -p XR_ELBOW -H 324 -L 324 -U 8 --patience 4 -T /path/to/saved/weights/model.h5
```
