import os
import cv2
import numpy as np


def module_wd(working_directory: str):
    """module_wd This functions is used for development purposes.This would make whatever code directory you are working in the current working directory during runtime.

    Args:
        working_directory (str): This is the path to the directory you want to set as the current working directory. The path is relative to the parent directory which has the main.py file.
    """

    project_folder = "LPDR-system"
    index_project_folder = str(os.getcwd()).find(
        project_folder) + len(project_folder)
    project_directory = str(os.getcwd())[0:index_project_folder]
    ret = os.chdir(project_directory)
    ret = os.chdir(working_directory)
    return ret


def sharpen_image(file_path: str, output_directory: str) -> np.ndarray:
    """sharpen_image This function is used to apply sharpening processing to an image.

    Args:
        file (str): image file. This can be a jpg or png file.
        output_directory (str): The directory where the sharpened image will be saved.

    Returns:
        _type_: None
    """
    filename = file_path[-5:]
    image = cv2.imread(file_path, flags=cv2.IMREAD_COLOR)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpen_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    save_image = input("Do you want to save the sharpened image? (y/n): ")
    if save_image == "y":
        write_file = f"Sharpen-{filename}"
        cv2.imwrite(f"{output_directory}\\{write_file}", sharpen_image)
    else:
        print("Image not saved.")
    return sharpen_image
