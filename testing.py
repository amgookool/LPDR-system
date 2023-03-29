from Utilities import show_image, show_video, calculate_fps
from Recognition import OCR_READER
from Detection import YOLO_ALGO
import pandas as pd
import numpy as np
import cv2
import os
from time import time


test_folder = os.path.join(os.getcwd(), "Testing")
videos_folder = os.path.join(test_folder, "Videos")
images_folder = os.path.join(test_folder, "Images")
daytime_folder = os.path.join(images_folder, "DayTime")
nightime_folder = os.path.join(images_folder, "NightTime")

def test_images(model_name:str, folder_path: str = daytime_folder,filename:str= "DayTime"):
    file_name = os.path.join(images_folder, f"{filename + '-' + model_name[0:-5]}.csv")
    data_list = []
    device = "GPU"
    detection_algo = YOLO_ALGO(device=device, model_file=model_name)
    recog_algo = OCR_READER(device=device)
    
    for height_folder in os.listdir(folder_path):
        height = float(height_folder[0:-6])
        for angle_folder in os.listdir(os.path.join(daytime_folder, height_folder)):
            angle = int(angle_folder[0:-3])
            for img in os.listdir(os.path.join(daytime_folder, height_folder, angle_folder)):
                image_path = os.path.join(
                    daytime_folder, height_folder, angle_folder, img)
                _, img_dist, img_angle, img_height = img[0:-4].split(
                    "-")
                image = cv2.imread(image_path)
                # Run Algorithms
                localization_detections = detection_algo.make_detection(image)
                localization_predictions = detection_algo.filter_detections(localization_detections)
                lp_images = detection_algo.get_lpObjects(image, localization_predictions)
                for lp_pred,lp_image in zip(localization_predictions,lp_images):
                    bbox, yolo_conf = lp_pred
                    x, y, w, h = bbox
                    recog_results = recog_algo.make_prediction(lp_image)
                    for r in recog_results:
                        lp_text, text_conf = r
                        data = {
                            "Name": img,
                            "Distance":img_dist,
                            "Angle":img_angle,
                            "Height":img_height,
                            "Detection-Confidence":yolo_conf,
                            "Text": lp_text,
                            "Text-Confidence": text_conf
                        }
                        data_list.append(data)
    df = pd.DataFrame(data_list)
    df.to_csv(file_name, index=False)


def test_image(model_name:str,filename:str):
    csv_file = os.path.join(os.getcwd(),"Testing","Images","result.csv")
    image_file_pre_path = os.path.join(os.getcwd(),"Testing","Images")
    img_file = os.path.join(image_file_pre_path,"NightTime","38.75height",filename)
    device = "GPU"
    text = str()
    detection_algo = YOLO_ALGO(device=device, model_file=model_name)
    recog_algo = OCR_READER(device=device)
    img = cv2.imread(img_file)
    localization_detections = detection_algo.make_detection(img)
    localization_predictions = detection_algo.filter_detections(localization_detections,confidence_threshold=0.05,probability_threshold=0.05)
    lp_images = detection_algo.get_lpObjects(img, localization_predictions)
    for lp_pred,lp_image in zip(localization_predictions,lp_images):
        bbox, yolo_conf = lp_pred
        x, y, w, h = bbox
        recog_results = recog_algo.make_prediction(lp_image)
        for r in recog_results:
            lp_text, text_conf = r
            text += lp_text
        print(f"Detection Confidence: {yolo_conf*100}")
        print(f"Text: {text}")
        print(f"Text Confidence: {text_conf*100}")


def test_video(filename:str,model_file:str, device:str="GPU"):
    print(model_file)
    duration_list,FPS_list = [],[]
    detection_algo = YOLO_ALGO(model_file= model_file,device=device)
    recog_algo = OCR_READER(device=device)
    video_file = os.path.join(os.getcwd(),"Testing","Videos",filename)
    cap = cv2.VideoCapture(video_file)
    numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prevFrame,newFrame = 0,0
    for idx in range(numFrames):
        success, frame = cap.read()
        startTime = time()
        detections = detection_algo.make_detection(frame)
        predictions = detection_algo.filter_detections(detections)
        lp_images = detection_algo.get_lpObjects(frame, predictions)
        if len(lp_images) != 0:
            for lp_pred, lp_image in zip(predictions,lp_images):
                bbox, yolo_conf = lp_pred
                x, y, w, h = bbox
                recog_results = recog_algo.make_prediction(lp_image)
                for r in recog_results:
                    lp_text, text_conf = r
        endTime = time()
        newFrame = time()
        fps,prevFrame = calculate_fps(prevFrame,newFrame)
        timeperFrame = endTime - startTime
        duration_list.append(timeperFrame)
        FPS_list.append(fps)
    duration_time = sum(duration_list)/len(duration_list)
    average_FPS = sum(FPS_list)/len(FPS_list)
    print(f"Duration Time: {duration_time*10e3}")
    print(f"Average FPS: {int(average_FPS)}")


if __name__ == "__main__":
    # print("5ft")
    # test_image(model_name="yolov5n.onnx",filename="50deg/NightTime-5ft-50deg-38.75height.JPG")
    # print("\n10ft")
    # test_image(model_name="yolov5n.onnx",filename="50deg/NightTime-10ft-50deg-38.75height.JPG")
    # print("\n15ft")
    # test_image(model_name="yolov5n.onnx",filename="50deg/NightTime-15ft-50deg-38.75height.JPG")
    # print("\n20ft")
    # test_image(model_name="yolov5n.onnx",filename="50deg/NightTime-20ft-50deg-38.75height.JPG")
    
    # test_images(model_name = "yolov5n.onnx",folder_path=daytime_folder,filename="DayTime")
    # test_images(model_name = "yolov5s.onnx",folder_path=daytime_folder,filename="DayTime")
    # test_images(model_name = "yolov5m.onnx",folder_path=daytime_folder,filename="DayTime")
    # test_images(model_name = "yolov5l.onnx",folder_path=daytime_folder,filename="DayTime")
    # test_images(model_name = "yolov5x.onnx",folder_path=daytime_folder,filename="DayTime")
    
    # test_images(model_name = "yolov5n.onnx",folder_path=nightime_folder,filename="NightTime")
    # test_images(model_name = "yolov5s.onnx",folder_path=nightime_folder,filename="NightTime")
    # test_images(model_name = "yolov5m.onnx",folder_path=nightime_folder,filename="NightTime")
    # test_images(model_name = "yolov5l.onnx",folder_path=nightime_folder,filename="NightTime")
    # test_images(model_name = "yolov5x.onnx",folder_path=nightime_folder,filename="NightTime")
    test_video(filename="Video-1.mp4",model_file = "yolov5x.onnx",device="CPU")
    pass
