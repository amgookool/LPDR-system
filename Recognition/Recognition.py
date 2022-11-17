import pytesseract
import cv2
import os

tesseract_path = os.path.join("C:\\","Program Files", "Tesseract-OCR")
pytesseract.pytesseract.tesseract_cmd = os.path.join(tesseract_path,"tesseract.exe")

class RECOGNITION_ALGO:
    def __init__(self, characters_list: list):
        self.char_list = characters_list
        # "C:\Program Files\Tesseract-OCR\tesseract.exe"
        

    def recognize(self):
        for i in self.char_list:
            img = cv2.resize(i, (600, 360))

            prediction = pytesseract.image_to_string(img)
            print(prediction)
            print(type(prediction))

            cv2.namedWindow("Character", cv2.WINDOW_GUI_NORMAL)
            cv2.imshow("Character", img)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
