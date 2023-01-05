from Detection import YOLO_ALGO
from Segmentation import SEGMENT_ALGO
from Recognition import RECOG_ALGO
import cv2, sys
import argparse


parser = argparse.ArgumentParser(description="Choosing the type of source (image/video) for system.")
parser.add_argument('-i',"--image", type=str,required=False,metavar="",help="The path to the image you want to run a prediction.")
parser.add_argument('-v',"--video", type=str,required=False,metavar="",help="The path to the video you want to run predictions on.")
args = parser.parse_args()

if __name__ =="__main__":
    detection_model = YOLO_ALGO()
    segmentation_model = SEGMENT_ALGO()
    recognition_model = RECOG_ALGO()
    # if args.video is None and args.image is not None:
    if args.video is None:
        pass
    else:
        print("Video file found.")
    
    if args.image is None:
        pass
    else:
        lp_text = ""
        img = cv2.imread(args.image)
        lps : list = detection_model.predict(image_data=img, model_name="yolov5s.onnx", img_size=640)
        
        for plate in lps:
            chars_list = segmentation_model.process(plate)
            for char in chars_list:
                char = recognition_model.inference(image=char,model_name="alexnet--epoch34.h5")
        #         if char is None:
        #             continue
        #         else:
        #             lp_text.join(char)
        # print(lp_text)
            
            
    
    
    # localization = detection.YOLO_ALGO()
    # detections = localization.predict(
    #     image=r"Detection/IMG_4357.jpg", show_predictions=False)
    
    # segment = segmentation.SEGMENT_ALGO(detections)
    # segmentations = segment.process()
    
    # recog = recognition.RECOGNITION_ALGO(segmentations)
    # recog.recognize()

    # capture_object = cv2.VideoCapture(0)
    # while True:
    #     try:
    #         retval, frame = capture_object.read()
    #         if (retval == True):
    #             cv2.imshow("Video", frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             raise KeyboardInterrupt
    #     except KeyboardInterrupt:
    #         print("User exited the program.")
    #         capture_object.release()
    #         break
    # cv2.destroyAllWindows()
    # sys.exit(0)
