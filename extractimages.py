import os
import sys
import shutil
import timeit

from numba import njit
import glob
import pandas as pd

CLASSES = {"0": "negative", "1": "positive"}


def moveImages(folderPathsCSV: str, newFolder: str) -> None:
    """This function simply converts the MURA dataset file structure to one that is compatible with the tensorflow/keras
    ImagaDataGenerator

    Args:
      folderPathsCSV (str): the path of the CSV file containing the
        paths for the images
      newFolder (str): the path of the folder where the images will
        be stored
    """
    df = pd.read_csv(folderPathsCSV)
    pathComponents = "/".join(folderPathsCSV.split("/")[:-2])
    try:
        trainOrValid = folderPathsCSV.split("/")[-1].split("_")[0]
    except ValueError:
        trainOrValid = folderPathsCSV.split("_")[0]
    for path, positiveOrNegative in df.itertuples(False):
        path2 = path.split("/")
        typeOfBone = path2[2]
        patientNum = path2[3]
        temp = path2[4].split("_")
        studyNum = temp[0]
        folderPath = os.path.join(
            newFolder, trainOrValid, typeOfBone, CLASSES[str(positiveOrNegative)]
        )
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        for imageFile in glob.glob(os.getcwd() + "/" + path + "/*.png"):
            imageNumber = imageFile.split("/")[-1]
            if os.path.exists(
                os.path.join(
                    folderPath, patientNum + "_" + studyNum + "_" + imageNumber
                )
            ):
                quit()
            shutil.copy2(
                imageFile,
                os.path.join(
                    folderPath, patientNum + "_" + studyNum + "_" + imageNumber
                ),
            )


if __name__ == "__main__":
    moveImages("MURA-v1.1/train_labeled_studies.csv", "MURASeparated/")
    moveImages("MURA-v1.1/valid_labeled_studies.csv", "MURASeparated/")
