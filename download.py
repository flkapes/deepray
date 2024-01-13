import gdown
import sys
import argparse

datasetURL = "https://drive.google.com/file/d/12f2Z6TWkh5yl82DyI-Egg_Gi0qZSJovB/view?usp=share_link"
weights_folder = "https://drive.google.com/drive/folders/16hB14UCJErVv-hu69WShcgg0GBVYh8_O?usp=drive_link"
weights_zip_file = (
    "https://drive.google.com/file/d/1s220_qbdfvmjUPr9Jp-9jDlpAtjenI1Q/view?usp=sharing"
)


def download_dataset():
    output = "MURASeparated.zip"
    gdown.download(url=datasetURL, output=output, quiet=False, fuzzy=True)


def download_weights_folder():
    output = "saved_checkpoints.zip"
    gdown.download_folder(weights_zip_file, quiet=True, use_cookies=False)


def download_specific_weights(weight_file_name: str):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="DeepRay Data Downloader",
        usage="""A script used to download relevant data and weights for the DeepRay Final Project of Faris Kapes.""",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="download_dataset",
        action="store_true",
        help="""Include this argument to download the pre-transformed dataset. Size 3.3GB.""",
    )
    parser.add_argument(
        "-a",
        "--all-weights",
        dest="download_all_weights",
        action="store_true",
        help="""Include this argument to download the all of the pre-trained model weights. Size 9.05GB.""",
    )
    parser.add_argument(
        "-w",
        "--weight",
        nargs=1,
        default=None,
        dest="download_weight_file",
        help="""Include this argument to download the a single weight file from the pre-trained weights. Use the format: -w densenet121_xr_elbow !!! NOT IMPLEMENTED""",
    )

    args = parser.parse_args()

    if args.download_dataset == True:
        download_dataset()
    if args.download_all_weights == True and args.download_weight_file == None:
        download_weights_folder()
        quit(0)
    elif args.download_all_weights == True and args.download_weight_file != None:
        print(
            "You cannot specify both --all-weights/-a at the same time as --weight"
            " <weight>/-w <weight>"
        )
        quit(0)
    if args.download_all_weights == False and args.download_weight_file != None:
        download_specific_weights(args.download_weight_file)
