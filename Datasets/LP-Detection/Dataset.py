from keras_preprocessing.image import load_img, img_to_array
import xml.etree.ElementTree as ET
from operator import itemgetter
import pandas as pd
import numpy as np
import glob
import cv2
import os

import warnings

warnings.filterwarnings("ignore")

work_dir = os.path.abspath(os.getcwd() + r"\Datasets\LP-Detection")
# condense_df = df[[x for x in df_columns if x != 'filepath']]
# if condense:
#   condense_df.to_csv(path_ + ".csv", index=0)
# else:
#   df.to_csv(path_+".csv", index=0)


def read_xml_format(path_: str) -> pd.DataFrame:
    """read_xml_format
    This function will read the xml file and convert it into a pandas dataframe.

    Args:
        path_ (str): The path of the directory containing xml files

    Returns:
        pd.DataFrame: Pandas dataframe containing data on all xml files. Can be converted into a CSV file.
    """
    xml_files = list()
    for xml_file in glob.glob(path_ + "/*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            filepath = os.path.abspath(
                path_ + "\\" + root.find("filename").text)
            filename = root.find("filename").text
            img_size = {
                "width": int(root.find("size")[0].text),
                "height": int(root.find("size")[1].text),
                "depth": int(root.find("size")[2].text),
            }
            classname = member[0].text
            bounding_box = {
                "xmin": int(member[4][0].text),
                "ymin": int(member[4][1].text),
                "xmax": int(member[4][2].text),
                "ymax": int(member[4][3].text),
            }
            xml_files.append(
                (
                    filepath,
                    filename,
                    img_size["width"],
                    img_size["height"],
                    img_size["depth"],
                    classname,
                    bounding_box["xmin"],
                    bounding_box["ymin"],
                    bounding_box["xmax"],
                    bounding_box["ymax"],
                )
            )

    df_columns = [
        "filepath",
        "filename",
        "width",
        "height",
        "depth",
        "classname",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]

    df = pd.DataFrame(xml_files, columns=df_columns)
    return df


def verify_vehicle_xml_annotations(path_: str):
    """verify_annotations This functions verifies that xml data generated on an image

    The function uses the xml data to draw the bounding boxes on the

    Args:
        path_ (str): The path of the directory containing xml files 
    """
    data_df = read_xml_format(path_)
    objects = list()
    indexed_imgs = list()

    image_path = data_df.get("filepath")
    filename = data_df.get("filename")
    xmin = data_df.get("xmin")
    ymin = data_df.get("ymin")
    xmax = data_df.get("xmax")
    ymax = data_df.get("ymax")

    for (_img_path, _fname, _xmin, _ymin, _xmax, _ymax) in zip(image_path, filename, xmin, ymin, xmax, ymax):
        f_index = int(_fname.split(".")[0].split("-")[2])

        if f_index not in indexed_imgs:
            indexed_imgs.append(f_index)
            object_ = {
                "index": f_index,
                "path": _img_path,
                "filename": _fname,
                "bboxs": [(_xmin, _ymin, _xmax, _ymax)]
            }
            objects.append(object_)

        if f_index in indexed_imgs and (_xmin, _ymin, _xmax, _ymax) not in object_['bboxs']:
            object_['bboxs'].append((_xmin, _ymin, _xmax, _ymax))

    objects.sort(key=itemgetter("index"))
    for o in objects:
        image: np.ndarray = cv2.imread(o['path'], cv2.IMREAD_COLOR)

        blue_bgr_scheme = (210, 0, 0)

        for box in o['bboxs']:
            cv2.rectangle(image, (box[0], box[1]),
                          (box[2], box[3]), blue_bgr_scheme, 8)
        
        cv2.namedWindow(f"{o['filename']}", cv2.WINDOW_NORMAL)
        cv2.imshow(f"{o['filename']}", image)
        
        if cv2.waitKey(0) & 0xFF == ord("q"):
            cv2.destroyAllWindows()


def YOLO_txt_format(path_: str) -> pd.DataFrame:
    """This function generates the txt files and returns a dataframe for the data needed to train the YOLO model.
    The data needed to train the algorithm are:

        class_id x_center y_center bbox_width bbox_height

    Args:
        path_ (str): The path of the directory containing images and xml files

    Returns:
        pd.DataFrame: A dataframe consisting of the required data for the YOLO algorithm
    """
    xml_df = read_xml_format(path_)

    # Preprocess Data for required ata for YOLO algorithm
    # Normalize the Data with respect to photo height and width
    xml_df["x_center"] = (xml_df["xmin"] + xml_df["xmax"]
                          ) / (2 * xml_df["width"])

    xml_df["y_center"] = (xml_df["ymin"] + xml_df["ymax"]
                          ) / (2 * xml_df["height"])

    xml_df["bbox_width"] = (xml_df["xmax"] - xml_df["xmin"]) / xml_df["width"]

    xml_df["bbox_height"] = (
        xml_df["ymax"] - xml_df["ymin"]) / xml_df["height"]

    # Slice Data for YOLO text file format
    data_values = xml_df[
        ["filename", "x_center", "y_center", "bbox_width", "bbox_height"]
    ].values

    data_paths = xml_df["filepath"].values

    for fpath, (fname, x, y, w, h) in zip(data_paths, data_values):
        fn, _ = fname.split(".")

        text_file_format = f"0 {x} {y} {w} {h}"
        file = fpath[:-4]

        with open(file + ".txt", mode="w") as f:
            f.write(text_file_format)
            f.close()

    return xml_df


def image_bbox_resize(
    xml_df: pd.DataFrame, img_target_size: tuple = (224, 224)
) -> dict:
    image_data = label_data = list()
    bbox_df: pd.DataFrame = xml_df.get(
        ["filepath", "xmin", "ymin", "xmax", "ymax"])
    for index, data in bbox_df.iterrows():
        image_array: np.ndarray = cv2.imread(data["filepath"])
        height, width, depth = image_array.shape  # Height, Width, Depth

        # Image Preprocessing
        load_image = load_img(
            data["filepath"], target_size=img_target_size, color_mode="rgb"
        )
        img_array = img_to_array(load_image)
        # Normalization of Image
        n_image = img_array / 255

        image_data.append(n_image)

        # Normalization of Labels
        n_xmin, n_xmax = data["xmin"] / width, data["xmax"] / width
        n_ymin, n_ymax = data["ymin"] / height, data["ymax"] / height

        n_labels = (n_xmin, n_xmax, n_ymin, n_ymax)

        label_data.append(n_labels)
    return {"Image-Data": image_data, "Label-Data": label_data}


if __name__ == "__main__":
    testing_directory: str = os.path.join(work_dir + r"\Testing")
    training_directory: str = os.path.join(work_dir + r"\Training")
    xml_df: pd.DataFrame = read_xml_format(training_directory)
    verify_vehicle_xml_annotations(training_directory)
    # xml_df.to_csv(training_directory + ".csv",index=0)
    # Datasets\LP-Detection\Training\1e88349d-Vehicle-61.JPG
    # image_paths: list = [x for x in xml_df.get("filepath")]
    # xml_paths = [a[:-3] + "xml" for a in xml_df.get("filepath")]
    # data = image_bbox_resize(xml_df=xml_df)
    # print(data['Label-Data'])
