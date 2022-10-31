def create_csv(save_directory: str, xml_dataframe: pd.DataFrame):
    """create_csv 
    This function is used to generate a csv file containing xml data

    Args:
        save_directory (str): The directory where the csv file will be saved
        filename (str): The name of the CSV file
        xml_dataframe (pd.DataFrame): The pandas Dataframe being converted to CSV file
    """
    file = save_directory + ".csv"
    xml_dataframe.to_csv(file, index=0)


from keras_preprocessing.image import load_img, img_to_array
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import glob
import cv2
import os

import warnings
warnings.filterwarnings("ignore")

work_dir = os.path.abspath(os.getcwd() + r"\Detection\YoloV5")

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
    return xml_df
