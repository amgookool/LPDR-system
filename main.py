from Detection import Detection as detection
from Segmentation import Segmentation as segmentation
from Recognition import Recognition as recognition
from Utilities import Utilities as utils
import cv2, sys
import tensorflow as tf


if __name__ =="__main__":
    localization = detection.YOLO_ALGO()
    detections = localization.predict(
        image=r"C:\Users\amgoo\Desktop\LPDR-system\Detection\IMG_4357.jpg", show_predictions=False)
    
    segment = segmentation.SEGMENT_ALGO(detections)
    segmentations = segment.process()
    
    recog = recognition.RECOGNITION_ALGO(segmentations)
    recog.recognize()

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
