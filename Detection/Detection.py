import os

import cv2

from cv2 import imread
from cv2 import IMREAD_COLOR

def hello():
    print("hello")


def module_wd(working_directory : str) -> None:
    project_folder = "LPDR-system"
    index_project_folder = str(os.getcwd()).find(
        project_folder) + len(project_folder)
    project_directory = str(os.getcwd())[0:index_project_folder]
    os.chdir(project_directory)
    os.chdir(working_directory)


