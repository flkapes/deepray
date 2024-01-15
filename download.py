import sys
import argparse
import hashlib
import requests
import zipfile
import tqdl
import os

saved_checkpoints_url = "http://sjc1.vultrobjects.com/mura-dataset/SavedCheckpoints.zip"
mura_dataset_url = "http://sjc1.vultrobjects.com/mura-dataset/MURA-v1.1.zip"
mura_separated_url = "http://sjc1.vultrobjects.com/mura-dataset/MURASeparated.zip"


def unzip_file(zip_path, extract_to=None):
    """
    Unzip a zip file.

    Args:
        zip_path (str): Path to the zip file.
        extract_to (str, optional): Directory to extract the files. Defaults to the same directory as the zip file.
    """
    if extract_to is None:
        extract_to = os.path.dirname(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")


def download_file(url, output):
    tqdl.download(url, output)
    print(f"Downloaded {output}")


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_and_check(url, output, expected_md5):
    download_file(url, output)
    downloaded_md5 = md5(output)
    if downloaded_md5 != expected_md5:
        print(
            f"Error: MD5 mismatch for {output}. Expected {expected_md5}, got"
            f" {downloaded_md5}"
        )
    else:
        print(f"MD5 check passed for {output}")
        unzip_file(output)  # Unzip after successful download and MD5 check


def download_prepped_dataset():
    expected_md5 = "2a480a89a815e41121468d982e1d525b"  # MD5 for MURASeparated.zip
    output = "MURASeparated.zip"
    download_and_check(mura_separated_url, output, expected_md5)


def download_saved_checkpoints():
    expected_md5 = "55c6ba7c3da437b6b74fd86b47d775f0"  # MD5 for SavedCheckpoints.zip
    output = "SavedCheckpoints.zip"
    download_and_check(saved_checkpoints_url, output, expected_md5)


def download_unprocessed_dataset():
    expected_md5 = "2b653718a9c55fcb6691d36b36f235de"  # MD5 for MURA-v1.1.zip
    output = "MURA-v1.1.zip"
    download_and_check(mura_dataset_url, output, expected_md5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="MURA Dataset Downloader",
        usage="""A script used to download datasets and saved checkpoints for MURA project.""",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="download_dataset",
        action="store_true",
        help=(
            "Include this argument to download the preprocessed MURA dataset. Size:"
            " approx 3.3GB."
        ),
    )
    parser.add_argument(
        "-s",
        "--saved-checkpoints",
        dest="download_saved_checkpoints",
        action="store_true",
        help="Include this argument to download saved checkpoints. Size: approx 9GB.",
    )
    parser.add_argument(
        "-m",
        "--mura-dataset",
        dest="download_mura_dataset",
        action="store_true",
        help=(
            "Include this argument to download the unprocessed MURA-v1.1 dataset. Size:"
            " approx 3.3GB."
        ),
    )
    args = parser.parse_args()

    if args.download_dataset:
        download_prepped_dataset()
    if args.download_saved_checkpoints:
        download_saved_checkpoints()
    if args.download_mura_dataset:
        download_unprocessed_dataset()
