# from keras_preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np
import subprocess
import shutil
import cv2
import re
import os
import sys

# import xml.etree.ElementTree as ET
# import pandas as pd
# import numpy as np
# import glob


class YOLO_ALGO:
    algorithm_dir = os.getcwd() + r"\Detection\YoloV5"
    path_data_imgs = algorithm_dir + r"\data_images"
    data_imgs_train_dir = path_data_imgs + r"\train"
    data_imgs_test_dir = path_data_imgs + r"\test"

    def get_data_images(self, training_percent: float = 0.8):
        txt_files = list()
        img_files = list()

        file_pattern = r"\.JPG|\.png|\.PNG|\.jpg|\.txt"

        dataset_training_path = os.getcwd(
        ) + r"\Datasets\LP-Detection\Train-Test"

        for batch_folder in os.listdir(dataset_training_path):
            for file in os.listdir(
                    os.path.join(dataset_training_path + "\\" + batch_folder)):
                if re.search(pattern=file_pattern, string=file):
                    file_ext = file[-4:]

                    if file_ext == ".txt":
                        txt_files.append(
                            os.path.join(dataset_training_path + "\\" +
                                         batch_folder + "\\" + file))

                    elif (file_ext == ".JPG" or file_ext == ".jpg"
                          or file_ext == ".png" or file_ext == ".PNG"):
                        img_files.append(
                            os.path.join(dataset_training_path + "\\" +
                                         batch_folder + "\\" + file))
        data = {
            "txt-file": txt_files,
            "image-file": img_files,
        }
        df = pd.DataFrame(data=data)
        num_rows, _ = df.shape
        num_training = round(training_percent * num_rows)

        training_data: pd.DataFrame = df.iloc[:num_training, :]
        validation_data: pd.DataFrame = df.iloc[num_training:, :]

        for _, row in training_data.iterrows():
            shutil.copy2(row["txt-file"], self.data_imgs_train_dir)
            shutil.copy2(row["image-file"], self.data_imgs_train_dir)
        for _, row in validation_data.iterrows():
            shutil.copy2(row["txt-file"], self.data_imgs_test_dir)
            shutil.copy2(row["image-file"], self.data_imgs_test_dir)

    def train(
        self,
        pretrained_weights: bool = True,
        batch_size: int = 8,
        epochs: int = 50,
        model_name: str = "Model",
    ):
        yolo_models = {
            1: "yolov5n",
            2: "yolov5s",
            3: "yolov5m",
            4: "yolov5l",
            5: "yolov5x",
        }
        try:
            which_model: int = int(
                input(f"Which YOLO model should be used?:\n{yolo_models}: "))
            if which_model in yolo_models:
                selected_model = yolo_models.get(which_model)
            else:
                raise ValueError
        except ValueError as model_err:
            print(f"The value {model_err} is invalid\n")
            self.train()

        print(f"Activating Environment: {sys.prefix}")
        subprocess.run(".\yolo-env\Scripts\activate", shell=True)
        os.chdir(self.algorithm_dir)

        if pretrained_weights:
            subprocess.run(
                f"python train.py --data data.yaml --weights {selected_model+'.pt'} --batch-size {batch_size} --name {model_name} --epochs {epochs}",
                shell=True,
            )
        else:
            subprocess.run(
                f"python train.py --data data.yaml --weights '' --cfg {selected_model+'.yaml'} --batch-size {batch_size} --name {model_name} --epochs {epochs}",
                shell=True,
            )

        print(f"Deactivating Environment: {sys.prefix}")
        subprocess.run("deactivate", shell=True)

    def export_model(self, formats: list):
        weights_dir = self.algorithm_dir + r"\runs\train"
        weight_file = "best.pt"
        
        saved_models = {
            "Torchscript": "torchscript",
            "ONNX": "onnx",
            "OpenVINO": "openvino",
            "TensorRT": 'engine',
            "CoreML": 'coreml',
            "TF-SavedModel": "saved_model",
            "TF-GraphDef": "pb",
            "TF-Lite": "tflite",
            "PaddlePaddle": 'paddle'
        }

        os.chdir(self.algorithm_dir)
        path_ = os.path.abspath(weights_dir + r"\Model\weights" + "\\" + weight_file )
        
        export_formats = [saved_models[x] for x in formats]

        cmd = f"python export.py --weights {path_} --include {' '.join(map(str,export_formats))}"
        print(cmd)
        subprocess.run(cmd, shell=True)

    def evaluate_model(self):
        os.chdir(self.algorithm_dir)
        weight = os.path.join(self.algorithm_dir + "\\" + r"runs\train\Model2\weights\best.pt")
        code_func = f"python val.py --weights {weight} --data data.yaml --img 640"
        subprocess.run(code_func,shell=True)

    def predict(self):
        pass


if __name__ == "__main__":
    algo = YOLO_ALGO()
    # algo.get_data_images()
    # algo.train()
    # algo.evaluate_model()
    algo.export_model(["ONNX","TF-SavedModel"])

    # image_paths: list = [x for x in xml_df.get("filepath")]
    # xml_paths = [a[:-3] + "xml" for a in xml_df.get("filepath")]
    # data = image_bbox_resize(xml_df=xml_df)
    # print(data['Label-Data'])

# def image_bbox_resize(xml_df: pd.DataFrame, img_target_size : tuple = (224,224)) -> dict:
#     image_data = label_data = list()
#     bbox_df: pd.DataFrame = xml_df.get(
#         ["filepath", "xmin", "ymin", "xmax", "ymax"])
#     for index, data in bbox_df.iterrows():
#         image_array : np.ndarray = cv2.imread(data['filepath'])
#         height, width, depth = image_array.shape  # Height, Width, Depth

#         # Image Preprocessing
#         load_image = load_img(
#             data['filepath'], target_size=img_target_size, color_mode="rgb")
#         img_array = img_to_array(load_image)
#         # Normalization of Image
#         n_image = img_array/255

#         image_data.append(n_image)

#         # Normalization of Labels
#         n_xmin, n_xmax = data['xmin']/width, data['xmax']/width
#         n_ymin, n_ymax = data['ymin']/height, data['ymax']/height

#         n_labels = (n_xmin, n_xmax, n_ymin, n_ymax)

#         label_data.append(n_labels)
#     return {
#         "Image-Data": image_data,
#         "Label-Data": label_data
#     }
