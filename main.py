from Utilities import show_image,show_video,calculate_fps 
from Recognition import OCR_READER
from Detection import YOLO_ALGO
from time import time
import numpy as np
import argparse
import cv2

parser = argparse.ArgumentParser(description="Choosing the type of source (image/video) for system.")
parser.add_argument('-i',"--image", type=str,required=False,metavar="",help="The path to the image you want to run a prediction.")
parser.add_argument('-v',"--video", type=str,required=False,metavar="",help="The path to the video you want to run predictions on.")
parser.add_argument('-u',"--url", type=str, required=False,metavar="",help="The URL of your live video feed/stream.")
args = parser.parse_args()

device="GPU"
detection_algo = YOLO_ALGO(device=device)
recog_algo = OCR_READER(device=device)

def pipeline_img(img: np.ndarray):
    # Use algorithm to make detections on image
    detections = detection_algo.make_detection(img)
    # Filter the detections to get the most confident predictions
    predictions = detection_algo.filter_detections(detections)
    # Crop the LPs detected from the image 
    lp_images = detection_algo.get_lpObjects(img,predictions)
    # Iterate over the predictions and LPs and run the recognition algorithm on each LP
    for lp_pred,lp_img in zip(predictions,lp_images):
        bbox, _ = lp_pred
        x,y,w,h = bbox
        # Run recognition algorithm on LP
        recog_results = recog_algo.make_prediction(lp_img)
        # process the results
        for r in recog_results:
            lp_text,text_conf = r
            # Annotate prediction on image 
            img = detection_algo.draw_prediction(img,x,y,w,h,text_conf,lp_text)
            text_print = f"License Plate: {lp_text}\nConfidence: {text_conf*100:.2f}%"
            print(text_print)
    return img

def pipeline_video(frame:np.ndarray):
    detections = detection_algo.make_detection(frame)
    predictions = detection_algo.filter_detections(detections)
    lp_images = detection_algo.get_lpObjects(frame,predictions)
    if len(lp_images) != 0:
        for lp_pred, lp_image in zip(predictions,lp_images):
            bbox, _ = lp_pred
            x,y,w,h = bbox
            # Run recognition algorithm on LP
            recog_results = recog_algo.make_prediction(lp_image)
            for r in recog_results:
                lp_text, text_conf = r
                if text_conf > 0.8:
                    frame = detection_algo.draw_prediction(frame,x,y,w,h,text_conf,lp_text)
                    text_print = f"License Plate: {lp_text}\nConfidence: {text_conf*100:.2f}%"
                    print(text_print)
    return frame


if __name__ == "__main__":
    # Video Execution (URL)
    if args.url is None:
        pass
    else:
        vidCap = cv2.VideoCapture(args.video)
        winName = "LPDR-SYSTEM-URL"
        prevFrame, newFrame = 0,0
        while True:
            ret, frame = vidCap.read()
            frame = pipeline_video(frame)
            # Calculate the FPS
            newFrame = time()
            fps,prevFrame = calculate_fps(prevFrame,newFrame)
            
            # Show FPS on top right of frame
            fps_color = (255,0,0)
            cv2.putText(frame,f"FPS:{fps}",(frame.shape[1]-100,50),cv2.FONT_HERSHEY_SIMPLEX,1,fps_color,2)
            cv2.imshow(winName,frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        vidCap.release()
        cv2.destroyWindow(winName)
            
    
    # Image Execution
    if args.image is None:
        pass
    else:
        # Load an image
        winName = "LPDR-System"
        img = cv2.imread(args.image)
        img = pipeline_img(img)
        show_image(img,winName)

    # Video File Execution
    if args.video is None:
        pass
    else:
        vidCap = cv2.VideoCapture(args.video)
        winName = "LPDR-System"
        numFrames = int(vidCap.get(cv2.CAP_PROP_FRAME_COUNT))
        prevFrame, newFrame = 0,0
        for f_idx in range(numFrames):
            success, frame = vidCap.read()
            
            # Perform prediction
            frame = pipeline_video(frame)
            
            # Calculate the FPS
            newFrame = time()
            fps,prevFrame = calculate_fps(prevFrame,newFrame)
            
            # Show FPS on top right of frame
            fps_color = (255,0,0)
            cv2.putText(frame,f"FPS:{fps}",(frame.shape[1]-100,50),cv2.FONT_HERSHEY_SIMPLEX,1,fps_color,2)
            
            # Show the frame
            if True:
                show_video(frame,winName)
            else:
                break
        vidCap.release()
        cv2.destroyWindow(winName)