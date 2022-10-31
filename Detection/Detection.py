import os
import cv2
import numpy as np
import pandas as pd
from keras_preprocessing.image import load_img, img_to_array

# import xml.etree.ElementTree as ET
# import pandas as pd
# import numpy as np
# import glob

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
    pass
    # image_paths: list = [x for x in xml_df.get("filepath")]
    # xml_paths = [a[:-3] + "xml" for a in xml_df.get("filepath")]
    # data = image_bbox_resize(xml_df=xml_df)
    # print(data['Label-Data'])