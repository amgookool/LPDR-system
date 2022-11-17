import numpy as np
from functools import cmp_to_key
import cv2


class SEGMENT_ALGO:
    def __init__(self, lp_detections: list):
        self.lp_detections = lp_detections

    def process(self) -> list:
        for i in self.lp_detections:
            # Reshape image
            img = cv2.resize(i,(400,600),interpolation=cv2.INTER_NEAREST)
            
            #  Sharpen the image
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            img : np.ndarray = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

            # Apply Grayscale
            img : np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply GaussianBlur Filter
            img : np.ndarray = cv2.GaussianBlur(img, (5, 55), 0)

            # Apply Thresholding Filter
            # img : np.ndarray = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY+cv2.THRESH_OTSU,45,15)
            img : np.ndarray = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU,img)[1]

            # cv2.namedWindow("Processed-Image", cv2.WINDOW_GUI_NORMAL)
            # cv2.imshow("Processed-Image", img)

            # Connected Component Analysis on the image and init mask to store interested components 
            _,labels = cv2.connectedComponents(img)
            mask = np.zeros(img.shape, dtype=np.uint8)

            # Upper and Lower Bound Criteria (amount of connected pixels) that makes up a character
            # Need to find out how many white pixels (character blob) makes up a character for a 800X800 image
            # This will be used for masking the pixels for the characters to send to the OCR algorithm
            total_pixels = img.shape[0] * img.shape[1]
            pix_lboundary = total_pixels // 80
            pix_uboundary = total_pixels // 17

            #  Loop through all connected components
            for (ind,label) in enumerate(np.unique(labels)):
                # if label is background, ignore it and continue
                if label == 0:
                    continue
                else:
                    # Else, construct label mask to display only connected components
                    label_mask = np.zeros(img.shape, dtype=np.uint8)
                    label_mask[labels == label] = 255 # white blob at the pixel loaction 
                    num_pixels = cv2.countNonZero(label_mask) 
                
                # if number of pixels in component is between upper and lower boundaries, add it to mask
                if num_pixels >= pix_lboundary and num_pixels <= pix_uboundary:
                    mask = cv2.add(mask, label_mask)

            # cv2.namedWindow("Mask-Image", cv2.WINDOW_GUI_NORMAL)
            # cv2.imshow("Mask-Image", mask)

            # Find contourrs and get the bounding boxes for the contours
            contours, hier = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            # Soring the bounding boxes from left to right, top to bottom
            # contours = sorted(contours, key= lambda ctr:  cv2.boundingRect(ctr)[0])
            def contour_sort(a,b):
                bb_a = cv2.boundingRect(a)
                bb_b = cv2.boundingRect(b)
                if abs(bb_a[1] - bb_b[1]) <= 10:
                    return bb_a[0] - bb_b[0]
                else:
                    return bb_a[1] - bb_b[1]

            contours = sorted(contours, key= cmp_to_key(contour_sort))

            lp_characters = list()
            
            for ct in contours:
                x,y,w,h = cv2.boundingRect(ct)

                roi = mask[y:y + h, x:x + w]
                lp_characters.append(roi)

            # for ind,char in enumerate(lp_characters):
            #     cv2.namedWindow(f"Character {ind+1}", cv2.WINDOW_GUI_NORMAL)
            #     cv2.imshow(f"Character {ind+1}", char)
        
            # if cv2.waitKey(0) & 0xFF == ord("q"):
            #     cv2.destroyAllWindows()
        return lp_characters


