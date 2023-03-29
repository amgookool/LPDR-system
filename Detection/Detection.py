# from Utils import show_image, draw_box, put_text
from .Utils import show_image, draw_box, put_text
import onnxruntime as ort
import pandas as pd
import numpy as np
import threading
import cv2
import os


class YOLO_ALGO:
    models_directory = os.path.join(os.getcwd(), "Detection", "Models")

    def __init__(self, model_file: str = "yolov5m.onnx", device: str = "GPU"):
        if "GPU" in device:
            self.provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif "CPU" in device:
            self.provider = ['CPUExecutionProvider', 'CUDAExecutionProvider']
        self.model_file = os.path.join(self.models_directory, model_file)
        self.session = self.load_model()
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def load_model(self):
        session = ort.InferenceSession(
            self.model_file, providers=self.provider)
        return session

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img.astype(np.float32)
        img = img / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def make_detection(self, img) -> np.ndarray:
        height, width = img.shape[:2]
        self.y_factor = height / 640
        self.x_factor = width / 640

        img_tensor = self.preprocess(img)
        # Run inference on the input image
        outputs = self.session.run(
            [self.output_name], {self.input_name: img_tensor})

        # Process the outputs
        output = outputs[0]
        output = output[0]

        return output

    def filter_detections(self, output, confidence_threshold: float = 0.5, probability_threshold: float = 0.4) -> list:
        predictions, boxes, confs = [], [], []
        boxes = output[:, :4]
        confs = output[:, 4]
        class_probs = output[:, 5:]

        # Apply the confidence and class probability thresholds to filter out low confidence predictions
        class_thresh = 0.4
        conf_thresh = 0.1
        mask = (confs > conf_thresh) & (np.max(class_probs, axis=1) > class_thresh)
        boxes = boxes[mask]
        confs = confs[mask]
        
        if len(boxes > 0) and len (confs > 0):
            # Apply non-maximum suppression to filter out overlapping bounding boxes
            indicies = cv2.dnn.NMSBoxes(boxes.tolist(), confs.tolist(), confidence_threshold, probability_threshold)
            if type(indicies) == tuple:
                pass
            else:
                indicies = indicies.reshape(-1)
                # Convert the boxes from YOLO format to pixel coordinates and return the filtered detections
                for index in indicies:
                    box = boxes[index]
                    x = (box[0] - 0.5 * (box[2]-4)) * self.x_factor
                    y = (box[1] - 0.45 * (box[3]+2)) * self.y_factor
                    width = box[2] * self.x_factor
                    height = box[3] * self.y_factor
                    confidence = confs[index]
                    predictions.append( ( (int(x), int(y), int(width), int(height)), confidence))
        return predictions


    def filter_detections_img(self, output) -> list:
        self.predictions = []
        probability_threshold = 0.5
        confidence_threshold = 0.5
        boxes, confs = [], []
        for i in range(len(output)):
            row = output[i]
            # print(f"Shape of Row: {row.shape}")
            confidence = row[4]
            if confidence > 0.3:
                class_prob = row[5]
                if class_prob > 0.5:
                    cenX, cenY, width, height = row[0:4]
                    # print(f"cenX:{cenX}, cenY:{cenY}, w:{width}, h:{height}")
                    # Normalization of Predictions for Input
                    left = (cenX - 0.5 * (width-4)) * self.x_factor
                    top = (cenY - 0.45 * (height+2)) * self.y_factor
                    width = width * self.x_factor
                    height = height * self.y_factor
                    box = np.array([left, top, width, height])

                    confs.append(confidence)
                    boxes.append(box)

        # Non-maximum suppression to filter overlapping bounding boxes
        boxes_lst = np.array(boxes).tolist()
        confs_lst = np.array(confs).tolist()

        indexs = cv2.dnn.NMSBoxes(
            boxes_lst,
            confs_lst,
            probability_threshold,
            confidence_threshold).flatten()

        for i in indexs:
            x, y, w, h = boxes_lst[i]
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            bb_conf = confs_lst[i]
            self.predictions.append(((x, y, w, h), bb_conf))
        return self.predictions

    def filter_detections_video(self, output):
        self.predictions = []
        probability_threshold = 0.2
        confidence_threshold = 0.3
        boxes, confs = [], []
        for i in range(len(output)):
            row = output[i]
            # print(f"Shape of Row: {row.shape}")
            confidence = row[4]
            if confidence > 0.1:
                class_prob = row[5]
                if class_prob > 0.1:
                    cenX, cenY, width, height = row[0:4]
                    # print(f"cenX:{cenX}, cenY:{cenY}, w:{width}, h:{height}")
                    # Normalization of Predictions for Input
                    left = (cenX - 0.5 * (width-4)) * self.x_factor
                    top = (cenY - 0.45 * (height+2)) * self.y_factor
                    width = width * self.x_factor
                    height = height * self.y_factor
                    box = np.array([left, top, width, height])

                    confs.append(confidence)
                    boxes.append(box)

        # Non-maximum suppression to filter overlapping bounding boxes
        boxes_lst = np.array(boxes).tolist()
        confs_lst = np.array(confs).tolist()

        if len(boxes_lst) > 0 and len(confs_lst) > 0:
            indexs = cv2.dnn.NMSBoxes(
                boxes_lst,
                confs_lst,
                probability_threshold,
                confidence_threshold)

            if type(indexs) == tuple:
                pass
            else:
                indexs = indexs.flatten()
                for i in indexs:
                    x, y, w, h = boxes_lst[i]
                    x = int(x)
                    y = int(y)
                    w = int(w)
                    h = int(h)
                    bb_conf = confs_lst[i]
                    self.predictions.append(((x, y, w, h), bb_conf))
        return self.predictions

    def get_lpObjects(self, img, predictions) -> list:
        lp_images = []
        if len(predictions) > 0:
            for i in predictions:
                bbox, conf = i
                x, y, w, h = bbox
                lp_image: np.ndarray = img[y: y+h + 1, x: x+w + 2]
                lp_images.append(lp_image)
        return lp_images

    def draw_prediction(self, img, x, y, w, h, conf, text: str = "YOLO-LP"):
        if "YOLO" in text:
            d_text = f"LP: {conf*100:.2f}%"
        else:
            d_text = f"{text}:{conf*100:.2f}%"
        img = draw_box(img, x, y, w, h, (255, 0, 0))
        img = put_text(img, d_text, x, y)
        return img


if __name__ == "__main__":
    # Load an image
    img = cv2.imread("/home/adrian/Desktop/LPDRV4/test1.jpg")
    video = cv2.VideoCapture("/home/adrian/Desktop/LPDRV4/Video-1.mp4")
    # Initialize the algorithm
    algo = YOLO_ALGO(device="GPU")
    
    # Inferene on Images
    def image_inference():
        detections = algo.make_detection(img)
        # Filter the detections to get the most confident predictions
        predictions = algo.filter_detections(detections,confidence_threshold=0.1)
        # Crop the LPs detected from the image
        lp_images = algo.get_lpObjects(img,predictions=predictions)
        # Show the LPs detected
        for n,i in enumerate(lp_images):
            show_image(i,f"LP{n+1}")
        # Show the predictions on the image
        for i in predictions:
            bbox, conf = i
            x,y,w,h = bbox
            algo.draw_prediction(img,x,y,w,h,conf)
        show_image(img,"YOLO-LP")

    def video_inference():
        numFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        test = []
        for _ in range(numFrames):
            success,frame = video.read()
            if not success:
                break
            
            detections = algo.make_detection(frame)
            predictions = algo.filter_detections(detections)
            lp_images = algo.get_lpObjects(frame,predictions=predictions)
            if len(lp_images) != 0:
                test.append(lp_images)
            # if len(lp_images) != 0:
            #     for n,i in enumerate(lp_images):
            #         show_image(i,f"LP{n+1}")
        print(len(test))
    video_inference()
