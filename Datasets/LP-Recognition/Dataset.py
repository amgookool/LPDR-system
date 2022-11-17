from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from object_detection.utils import dataset_util
from collections import namedtuple
import xml.etree.ElementTree as ET

from operator import itemgetter
import tensorflow as tf
from PIL import Image

import pandas as pd
import numpy as np
import shutil

import glob
import sys
import cv2

import os
import io
import re

work_dir = os.path.join(os.getcwd(), "Datasets", "LP-Segmentation")
Data_directory: str = os.path.join(work_dir, "Data")
Train_directory: str = os.path.join(work_dir, "train")
Test_directory: str = os.path.join(work_dir, "test")


def read_xml_format(path_: str) -> pd.DataFrame:
    """read_xml_format
    This function will read the xml file and convert it into a pandas dataframe.

    Args:
        path_ (str): The path of the directory containing xml files

    Returns:
        pd.DataFrame: Pandas dataframe containing data on all xml files. Can be converted into a CSV file.
    """
    xml_files = list()
    xml_path = os.path.join(path_, "*.xml")
    for xml_file in glob.glob(xml_path):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            filepath = os.path.join(path_, root.find("filename").text)
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

    for (_img_path, _fname, _xmin, _ymin, _xmax, _ymax) in zip(
        image_path, filename, xmin, ymin, xmax, ymax
    ):
        f_index = int(_fname.split(".")[0].split("-")[2])

        if f_index not in indexed_imgs:
            indexed_imgs.append(f_index)
            object_ = {
                "index": f_index,
                "path": _img_path,
                "filename": _fname,
                "bboxs": [(_xmin, _ymin, _xmax, _ymax)],
            }
            objects.append(object_)

        if (
            f_index in indexed_imgs
            and (_xmin, _ymin, _xmax, _ymax) not in object_["bboxs"]
        ):
            object_["bboxs"].append((_xmin, _ymin, _xmax, _ymax))

    objects.sort(key=itemgetter("index"))
    for o in objects:
        image: np.ndarray = cv2.imread(o["path"], cv2.IMREAD_COLOR)

        blue_bgr_scheme = (210, 0, 0)

        for box in o["bboxs"]:
            cv2.rectangle(image, (box[0], box[1]),
                          (box[2], box[3]), blue_bgr_scheme, 8)

        cv2.namedWindow(f"{o['filename']}", cv2.WINDOW_NORMAL)
        cv2.imshow(f"{o['filename']}", image)

        if cv2.waitKey(0) & 0xFF == ord("q"):
            cv2.destroyAllWindows()


def generate_csv_file(xml_path: str, save_path: str, filename: str):
    """generate_csv_file This function generates a csv file containing the
    bounding boxes for each object

    Args:
        xml_df (pd.DataFrames): The dataframe containing the bounding boxes"""
    xml_df = read_xml_format(xml_path)

    new_df = xml_df[["filepath", "filename", "width", "height",
                     "classname", "xmin", "ymin", "xmax", "ymax"]]

    filename = filename + ".csv"
    save = os.path.join(save_path, filename)
    new_df.to_csv(save, index=0)


def generate_tf_record(imgs_path: str, csv_path: str, output_path: str, filename_: str):
    """generate_tf_record This function generates a tfrecord file containing the
    bounding boxes for each object
    """
    record_filename = os.path.join(output_path,  filename_ + ".record")

    writer = tf.io.TFRecordWriter(record_filename)

    def class_text_to_int(row_label):
        if row_label == "License-Plate" or row_label == "Character":
            return 1

    def split(df: pd.DataFrame, group):
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

    group = split(pd.read_csv(csv_path), "filename")

    for g in group:
        with tf.io.gfile.GFile(os.path.join(imgs_path, '{}'.format(g.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)

        width, height = image.size
        filename = g.filename.encode('utf8')

        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes = []
        classes_text = []

        for _, row in g.object.iterrows():
            xmins.append(row["xmin"] / width)
            xmaxs.append(row["xmax"]/width)
            ymins.append(row["ymin"]/height)
            ymaxs.append(row["ymax"]/height)
            classes.append(class_text_to_int(row["classname"]))
            encoded_class_text = row["classname"].encode("utf-8")
            classes_text.append(encoded_class_text)

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        writer.write(tf_example.SerializeToString())
    writer.close()


def split_data_images(training_percent: float = 0.8):
    xml_files = list()
    img_files = list()

    file_pattern = r"\.JPG|\.png|\.PNG|\.jpg|\.xml"

    for file in os.listdir(Data_directory):
        if re.search(pattern=file_pattern, string=file):
            if (file[-4:] == ".JPG" or file[-4:] == ".jpg" or file[-4:] == ".png" or file[-4:] == ".PNG"):
                file_path = os.path.join(Data_directory, file)
                img_files.append(file_path)

            if file[-4:] == ".xml":
                file_path = os.path.join(Data_directory, file)
                xml_files.append(file_path)

    data = {
        "image-file": img_files,
        "xml-file": xml_files,
    }
    df = pd.DataFrame(data=data)

    num_rows, _ = df.shape
    num_training = round(training_percent * num_rows)

    training_df: pd.DataFrame = df.iloc[:num_training, :]
    train_directory = os.path.join(work_dir, "train")
    for _, row in training_df.iterrows():
        shutil.move(row['image-file'], train_directory)
        shutil.move(row['xml-file'], train_directory)

    testing_df: pd.DataFrame = df.iloc[num_training:, :]
    test_directory = os.path.join(work_dir, "test")
    for _, row in testing_df.iterrows():
        shutil.move(row['image-file'], test_directory)
        shutil.move(row['xml-file'], test_directory)


if __name__ == "__main__":
    df: pd.DataFrame = read_xml_format(Data_directory)

    # split_data_images(training_percent=0.9)

    generate_csv_file(Train_directory, work_dir, "Train")
    generate_csv_file(Test_directory, work_dir, "Test")

    train_csv = os.path.join(work_dir, "Train.csv")
    test_csv = os.path.join(work_dir, "Test.csv")

    generate_tf_record(Train_directory, train_csv, work_dir, "Train")
    generate_tf_record(Test_directory, test_csv, work_dir, "Test")
