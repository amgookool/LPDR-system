import logging as log
import threading
import time
import cv2
import os
import re

import warnings
warnings.filterwarnings("ignore")

work_dir = os.path.abspath(os.getcwd() + r"\Raw-Dataset")

RawImages_directory: str = os.path.join(work_dir + r"\Raw-images")
Batch1_directory: str = os.path.join(RawImages_directory + r"\Batch1")

detect_count = 1
segment_count = 1

_exec = Batch1_directory

log.basicConfig(filename=work_dir+r"\sorted-dataset.log",
                level=log.INFO, format='%(asctime)s : %(levelname)s : %(message)s')


def rename_file(file: str, store_directory: str,  filename_prefix: str, count: int):
    """rename_file
    This function will rename a file in a directory

    The rename file will have a prefix filename and a number as suffix

    Args:
        file (str): This will be the file we want to rename

        filename_prefix (str): This is the prefix filename

        count (int): This represents the suffix of the filename
    """
    file_pattern = r'\.JPG|\.png|\.PNG|\.jpg'
    try:
        if re.search(pattern=file_pattern, string=file):
            file_extension = file[-4:]
            if file.startswith(filename_prefix):
                return
            else:
                new_filename = f"{filename_prefix}{count}" + file_extension
                new_file = store_directory + "\\" + f"{new_filename}"
                log.info(
                    f"""
File Directory: {os.path.abspath(file)}
Renaming {file} --> {new_filename}
Saving to Directory:
{store_directory}\n
""")

                print(f"Renamed {file} to {new_filename}")
                os.rename(os.path.abspath(file), new_file)
        else:
            raise FileExistsError
    except FileExistsError as err:
        print(f"The file already exists\n {err}")
        return


def move_file_to_folder(file: str):
    """move_file_to_folder This function prompts the user for the destination folder of shown image

    Args:
        file (str): The file being sorted
    """
    DetectionImages_directory: str = os.path.join(work_dir + r"\Detection")

    SegmentationImages_directory: str = os.path.join(
        work_dir + r"\Segmentation")

    directory_dict = {1: (DetectionImages_directory, "Vehicle-"),
                      2: (SegmentationImages_directory, "LP-")}

    try:
        location_input = int(input(f"""\nPlease choose where to move {file}
    1 : {DetectionImages_directory}
    2 : {SegmentationImages_directory}
Selection: """))

        if location_input == 1 or location_input == 2:
            return directory_dict[location_input]
        else:
            raise ValueError
    except ValueError:
        print(f"\nInvalid integer input: {ValueError}\n")
        return move_file_to_folder(file)


def sort_dataset(raw_directory: str):
    """sort_dataset This function will sort the raw images into a Detection or Segmentation folder.

    The images in these folders will be used for annotating bounding boxes or segments for training the machine learning algorithms
    The function will also rename the files in the following manner:
        - Detection Folder -> Vehicle-{index}
        - Segmentation Folder -> LP-{index}
    Args:
        raw_directory (str): The directory that has the raw images
    """
    global detect_count, segment_count
    for file in os.listdir(os.chdir(raw_directory)):
        filepath = os.path.abspath(file)

        def show_image():
            image = cv2.imread(filepath, cv2.IMREAD_COLOR)
            cv2.namedWindow(f"{file}", cv2.WINDOW_NORMAL)
            cv2.imshow(f"{file}", image)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyWindow(f"{file}")

        thread_image = threading.Thread(target=show_image)
        thread_image.start()

        store_directory, prefix_filename = move_file_to_folder(file)

        if ("Vehicle") in prefix_filename:
            rename_file(file, store_directory, prefix_filename, detect_count)
            detect_count += 1
        elif ("LP") in prefix_filename:
            rename_file(file, store_directory, prefix_filename, segment_count)
            segment_count += 1
    print("Finished sorting the Dataset 🤮")
    return


if __name__ == "__main__":
    start_time = time.time()
    # sort_dataset(_exec)
    time.sleep(5)
    end_time = time.time()
    def get_finalTime (start_time:float,end_time:float):
        return end_time - start_time

    print(f"Execution time: {get_finalTime(start_time,end_time)} seconds")
