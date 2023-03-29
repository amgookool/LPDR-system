from functools import cmp_to_key
import numpy as np
import cv2


def show_image(img_data: np.ndarray, name: str):
    # cv2.WINDOW_NORMAL, cv2.WINDOW_AUTOSIZE, cv2.WINDOW_GUI_NORMAL, cv2.WINDOW_FULLSCREEN, cv2.WINDOW_KEEPRATIO, cv2.WINDOW_GUI_EXPANDED
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img_data)


def close_all_windows():
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()


# Storing the bounding boxes from left to right, top to bottom
def contour_sort(a,b):
    bb_a = cv2.boundingRect(a)
    bb_b = cv2.boundingRect(b)
    if abs(bb_a[1] - bb_b[1]) <= 10:
        return bb_a[0] - bb_b[0]
    else:
        return bb_a[1] - bb_b[1]


class SEGMENT_ALGO:

    def process(self, img_data: np.ndarray, show_process:bool = False) -> list:
        # Reshaping the image
        img_size=(600,300)
        resized_img : np.ndarray = cv2.resize(img_data,img_size,interpolation=cv2.INTER_NEAREST)
        
        # Sharpening the image
        kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
        sharpened_img : np.ndarray  = cv2.filter2D(src=resized_img,ddepth=-1,kernel=kernel)
        
        # Applying Grayscale
        gray_img :np.ndarray = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
        
        # Applying Gaussian Blurring Filter
        blurred_img :np.ndarray = cv2.GaussianBlur(gray_img, (5, 5), 0)
        
        # Applying Binary Threshold
        binary_img = cv2.adaptiveThreshold(blurred_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,45,15)
        # binary_img : np.ndarray = cv2.threshold(blurred_img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        if show_process:
            show_image(binary_img,"Binary Image")

        # Upper and Lower Bound Criteria (amount of connected pixels) that makes up a character
        # Need to find out how many white pixels (character blob) makes up a character
        # This will be used for masking the pixels for the characters to send to the OCR algorithm
        total_pixels = binary_img.shape[0] * binary_img.shape[1]
        
        lboundary_pixels = total_pixels // 70
        uboundary_pixels = total_pixels // 20

        # Connected Component Analysis on the image and init mask to store interested components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img)

        # Creating the mask
        mask = np.zeros(binary_img.shape,dtype=np.uint8)
        
        #  Loop through all connected components
        for (ind,label) in enumerate(np.unique(labels)):
            # if label is background, ignore it and continue
            if label == 0:
                continue
            else:
            # Else, construct label mask to display only connected components
                label_mask = np.zeros(binary_img.shape,dtype=np.uint8)
                label_mask[labels == label] = 255
                num_pixels = cv2.countNonZero(label_mask)
            
            # if number of pixels in component is between upper and lower boundaries, add it to mask
            if num_pixels >= lboundary_pixels and num_pixels <= uboundary_pixels:
                mask = cv2.add(mask,label_mask)
        
        if show_process:
            show_image(mask,"Mask")

        # Find contours and get the bounding boxes for the contours
        contours,hiers = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours,key=cmp_to_key(contour_sort))
        
        # Storing the bounding boxes from left to right, top to bottom
        lp_chars = list()
        target_height, target_width = 128,128

        MASK_COPY = mask.copy()
        MASK_COPY = cv2.cvtColor(MASK_COPY,cv2.COLOR_GRAY2RGB)
        
        for idx,ct in enumerate(contours):
            x,y,w,h = cv2.boundingRect(ct)
            cv2.rectangle(MASK_COPY,(x,y),(x+w,y+h),(0,255,0),2)
            roi :np.ndarray = mask[y:y+h, x:x+w]
            roi :np.ndarray = cv2.bitwise_not(roi)
            rows,cols = roi.shape
            padY = (target_height - rows) // 2 if rows < target_height else int(0.17 * rows)
            padX = (target_width - cols) // 2 if cols < target_width else int(0.45 * cols)
            boarder = cv2.copyMakeBorder(roi,padY,padY,padX,padX,cv2.BORDER_CONSTANT,None,255)
            char = cv2.cvtColor(boarder,cv2.COLOR_GRAY2RGB)
            lp_chars.append(char)
            if show_process:
                show_image(char,f"Char{idx+1}")
        if show_process:
            show_image(MASK_COPY,"Mask with Bounding Boxes")
        close_all_windows()
        return lp_chars


if __name__ == "__main__":
    algo = SEGMENT_ALGO()
    img = cv2.imread("Segmentation/LP-7.JPG",cv2.IMREAD_COLOR)
    algo.process(img_data=img,show_process=True)
