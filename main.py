from Detection import Detection as detection
from Segmentation import Segmentation as segmentation
from Recognition import Recognition as recognition
from Utilities import Utilities as utils
import cv2, sys
import tensorflow as tf


if __name__ =="__main__":
    capture_object = cv2.VideoCapture(0)
    while True:
        try:
            retval, frame = capture_object.read()
            if (retval == True):
                cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            print("User exited the program.")
            capture_object.release()
            break
    cv2.destroyAllWindows()
    sys.exit(0)
