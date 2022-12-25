import pandas as pd
import numpy as np
import subprocess
import shutil
import cv2
import re
import os
import sys
import tensorflow as tf


class YOLO_ALGO:
    algorithm_dir = os.path.join(os.getcwd(), "Detection", "YoloV5")
    path_data_imgs = os.path.join(algorithm_dir, "data_images")
    data_imgs_train_dir = os.path.join(path_data_imgs, "train")
    data_imgs_test_dir = os.path.join(path_data_imgs, "test")

    def get_data_images(self, training_percent: float = 0.8):
        txt_files = list()
        img_files = list()

        file_pattern = r"\.JPG|\.png|\.PNG|\.jpg|\.txt"

        dataset_training_path = os.path.join(
            os.getcwd(), "Datasets", "LP-Detection", "Train-Test"
        )

        for batch_folder in os.listdir(dataset_training_path):
            for file in os.listdir(os.path.join(dataset_training_path, batch_folder)):
                if re.search(pattern=file_pattern, string=file):
                    file_ext = file[-4:]

                    if file_ext == ".txt":
                        txt_files.append(
                            os.path.join(dataset_training_path,
                                         batch_folder, file)
                        )

                    elif (
                        file_ext == ".JPG"
                        or file_ext == ".jpg"
                        or file_ext == ".png"
                        or file_ext == ".PNG"
                    ):
                        img_files.append(
                            os.path.join(dataset_training_path,
                                         batch_folder, file)
                        )
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
                input(f"Which YOLO model should be used?:\n{yolo_models}: ")
            )
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

    def export_model(self, formats: list, model_folder: str = "Model"):
        weights_dir = os.path.join(self.algorithm_dir, "runs", "train")
        weight_file = "best.pt"

        saved_models = {
            "Torchscript": "torchscript",
            "ONNX": "onnx",
            "OpenVINO": "openvino",
            "TensorRT": "engine",
            "CoreML": "coreml",
            "TF-SavedModel": "saved_model",
            "TF-GraphDef": "pb",
            "TF-Lite": "tflite",
            "PaddlePaddle": "paddle",
        }

        os.chdir(self.algorithm_dir)
        path_ = os.path.join(weights_dir, model_folder, "weights", weight_file)

        export_formats = [saved_models[x] for x in formats]

        cmd = f"python export.py --weights {path_} --include {' '.join(map(str,export_formats))}"
        print(cmd)
        subprocess.run(cmd, shell=True)

    def evaluate_model(self, model_folder: str = "Model", img_size: int = 640):
        os.chdir(self.algorithm_dir)
        weight = os.path.join(
            self.algorithm_dir, "runs", "train", model_folder, "weights", "best.pt"
        )
        code_func = (
            f"python val.py --weights {weight} --data data.yaml --img {img_size}"
        )
        subprocess.run(code_func, shell=True)

    def predict(self, image, model_folder: str = "Model", img_size: int = 640, confidence_threshold: float = 0.5, probability_threshold: float = 0.5, show_predictions : bool = True) -> list:

        weights_folder = os.path.join(
            self.algorithm_dir, "runs", "train", model_folder, "weights")

        lp_detections: list = []

        # Image Preprocessing
        img = cv2.imread(image)

        img_ = img.copy()
        r, c, d = img_.shape
        max_rc = max(r, c)

        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:r, 0:c] = img_

        image_w, image_h = input_image.shape[:2]
        x_factor = image_w/img_size
        y_factor = image_h/img_size

        # Using OpenCV DNN module for inference of YOLO Model
        onnx_path = os.path.join(weights_folder, "best.onnx")
        network = cv2.dnn.readNetFromONNX(onnx_path)
        network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Using Model to make predictions
        blob = cv2.dnn.blobFromImage(
            input_image, 1/255, (img_size, img_size), swapRB=True, crop=False)
        network.setInput(blob)
        predictions = network.forward()

        # Predictions Outputs
        # CenterX, CenterY, Width, Height, Confidence, Probability Score
        detections = predictions[0]

        # Filtering Detections based on Confidence Threshold and Probability Score
        boxes = list()
        confs = list()

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.4:
                class_probability = row[5]
                if class_probability > 0.25:
                    cen_x, cen_y, w, h = row[0:4]
                    # Normalization of Predictions for Input
                    left = int((cen_x - 0.5 * w) * x_factor)
                    top = int((cen_y - 0.5 * h) * y_factor)
                    width = int(w*x_factor)
                    height = int(h*y_factor)
                    box = np.array([left, top, width, height])

                    confs.append(confidence)
                    boxes.append(box)

        # Performing Non Maximum Suppression on Detections
        boxes_np = np.array(boxes).tolist()
        confs_np = np.array(confs).tolist()
        indexs = cv2.dnn.NMSBoxes(
            boxes_np, confs_np, probability_threshold, confidence_threshold).flatten()

        print("Number of Detections:{0}".format(len(indexs)))
        # Drawing Bounding Boxes
        for i in indexs:
            x, y, w, h = boxes_np[i]
            bb_conf = confs_np[i]
            # Bounding Box for license plate
            # lp_image: np.ndarray = img_[y:y+h + 20, x:x+w + 10]
            lp_image: np.ndarray = img_[y:y+h + 5, x:x+w - 4]
            lp_detections.append(lp_image)
            
            # Drawing Predictions
            if show_predictions:
                # Bounding Box
                cv2.rectangle(img_, (x, y), (x+w, y+h), (255, 5, 255), 2)

                # Rectangle for confidence score
                conf_text = "License Plate:{:.0f}%".format(bb_conf * 100)
                cv2.rectangle(img_, (x, y-50), (x+w, y), (255, 5, 255), -1)
                cv2.putText(img_, conf_text,(x,y-10),cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)

                # Showing the Image
                cv2.namedWindow("Prediction", cv2.WINDOW_GUI_NORMAL)
                cv2.imshow("Prediction", img_)
                if cv2.waitKey(0) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
        return lp_detections


# if __name__ == "__main__":
#     algo = YOLO_ALGO()
    # algo.get_data_images()
    # algo.train()
    # algo.evaluate_model()
    # algo.export_model(["ONNX","TF-SavedModel"])
    # detections = algo.predict(
    #     image=,show_predictions=False)
    
    # for img in detections:
    #     cv2.namedWindow("Prediction", cv2.WINDOW_GUI_NORMAL)
    #     cv2.imshow("Prediction", img)
    #     if cv2.waitKey(0) & 0xFF == ord("q"):
    #         cv2.destroyAllWindows()
   

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
