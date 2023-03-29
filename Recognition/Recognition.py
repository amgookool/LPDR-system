import numpy as np
import easyocr
import os

np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))


class OCR_READER:
    def __init__(self, device: str = "GPU") -> None:
        self.model_directory = os.path.join(os.getcwd(),"Recognition","Models")
        if "GPU" in device:
            isGPU = True
        elif "CPU" in device:
            isGPU = False
        self.reader = easyocr.Reader(
            lang_list=['en'], gpu=isGPU, model_storage_directory=self.model_directory,verbose=False)
        
    def make_prediction(self,image_data):
        chars = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
            'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
        predictions= self.reader.readtext(image_data, detail=1, paragraph=False, allowlist=chars)
        prediction_list = []
        for pred in predictions:
            pred_text = pred[1]
            pred_box = pred[0]
            pred_conf = pred[2]
            prediction_list.append((pred_text,pred_conf))
        
        return prediction_list
        
        
if __name__ == "__main__":
    ocr = OCR_READER()
    # image_data = cv2.imread("/home/adrian/Desktop/LPDRV4/Testing/Images/DayTime/38.75height/20deg/DayTime-5ft-20deg-38.75height.JPG")
    # predictions = ocr.make_prediction(image_data)
    # print(predictions)