from keras_preprocessing.image import load_img, img_to_array
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import glob
import cv2
import os

import warnings
warnings.filterwarnings("ignore")

work_dir = os.path.abspath(os.getcwd() + r"\Datasets\LP-Detection")


def xml_to_csv_format(path: str) -> pd.DataFrame:
    """xml_to_csv_format 
    This function will convert the xml file to a pandas dataframe.

    Args:
        path (str): The path of the directory containing xml files

    Returns:
        pd.DataFrame: Pandas dataframe containing data on all xml files. Can be converted into a CSV file.
    """
    xml_files = list()
    for xfile in glob.glob(path + "/*.xml"):
        tree = ET.parse(xfile)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (str(os.path.abspath(path + "\\" + root.find('filename').text)),  # file path
                     root.find('filename').text,  # filename of image
                     int(root.find('size')[0].text),  # width
                     int(root.find('size')[1].text),  # height
                     member[0].text,  # object class name
                     int(member[4][0].text),  # bbox - xmin
                     int(member[4][1].text),  # bbox - ymin
                     int(member[4][2].text),  # bbox - xmax
                     int(member[4][3].text)  # bbox - ymax
                     )
            xml_files.append(value)
    column_name = ['filepath', 'filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_files, columns=column_name)
    xml_df.to_csv(path + ".csv", index=0)
    return xml_df


def xml_to_txt_format(path : str):
    xml_files = list()
    for xml_file in glob.glob(path + "/*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            filename = root.find("filename").text
            # width, height, depth


def rename_file(directory: str):
    """rename_file 
    This function will rename all the files in a directory

    The rename file will have "Car-" as prefix and the number of the file as suffix

    Args:
        directory (str): This will be the directory containing the files to rename
    """
    count = 0
    os.chdir(directory)
    for filename in os.listdir():
        if filename.endswith(".JPG" or ".jpg") or filename.endswith(".PNG" or ".png"):
            file_extension = filename[-4:]
            if filename.startswith("Car"):
                count += 1
                continue
            else:
                os.rename(filename, f"Car-{count}" + file_extension)
                count += 1
    return


def verify_annotations(data_df: pd.DataFrame):
    """verify_annotations This functions verifies that xml data generated on an image

    The function uses the xml data to draw the bounding boxes on the 

    Args:
        data_df (pd.DataFrame): _description_
    """
    image_path = data_df.get("filepath")
    # width = data_df.get("width")
    # height = data_df.get("height")
    xmin = data_df.get("xmin")
    ymin = data_df.get("ymin")
    xmax = data_df.get("xmax")
    ymax = data_df.get("ymax")
    for _img_path, _xmin, _ymin, _xmax, _ymax in zip(image_path, xmin, ymin, xmax, ymax):
        print(_img_path, _xmin, _ymin, _xmax, _ymax)
        # Image File
        image: np.ndarray = cv2.imread(_img_path, cv2.IMREAD_COLOR)

        # Bounding Box

        blue_rgb = (205, 0, 0)  # BGR
        cv2.rectangle(image, (_xmin, _ymin), (_xmax, _ymax), blue_rgb, 5)

        # Showing image
        cv2.namedWindow(f"{image_path}", cv2.WINDOW_NORMAL)
        cv2.imshow(f'{image_path}', image)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()


def image_bbox_resize(xml_df: pd.DataFrame, img_target_size : tuple = (224,224)) -> dict:
    image_data = label_data = list()
    bbox_df: pd.DataFrame = xml_df.get(
        ["filepath", "xmin", "ymin", "xmax", "ymax"])
    for index, data in bbox_df.iterrows():
        image_array : np.ndarray = cv2.imread(data['filepath'])
        height, width, depth = image_array.shape  # Height, Width, Depth

        # Image Preprocessing
        load_image = load_img(
            data['filepath'], target_size=img_target_size, color_mode="rgb")
        img_array = img_to_array(load_image)
        # Normalization of Image
        n_image = img_array/255

        image_data.append(n_image)

        # Normalization of Labels
        n_xmin, n_xmax = data['xmin']/width, data['xmax']/width
        n_ymin, n_ymax = data['ymin']/height, data['ymax']/height

        n_labels = (n_xmin, n_xmax, n_ymin, n_ymax)

        label_data.append(n_labels)
    return {
        "Image-Data": image_data,
        "Label-Data": label_data
    }


if __name__ == "__main__":
    testing_directory: str = os.path.join(work_dir + r"\Testing")
    training_directory: str = os.path.join(work_dir + r"\Training")
    xml_df: pd.DataFrame = xml_to_csv_format(training_directory)
    # image_paths: list = [x for x in xml_df.get("filepath")]
    # xml_paths = [a[:-3] + "xml" for a in xml_df.get("filepath")]
    # data = image_bbox_resize(xml_df=xml_df)
    # print(data['Label-Data'])
