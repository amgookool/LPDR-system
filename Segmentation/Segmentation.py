from functools import cmp_to_key
from time import time
from PIL import Image
import numpy as np
import cv2

def display_execution_time(start,end):
    print("Execution Time: {:.3f} seconds".format(end-start))

def show_image(img_data:np.ndarray,name:str):
    # cv2.WINDOW_NORMAL, cv2.WINDOW_AUTOSIZE, cv2.WINDOW_GUI_NORMAL, cv2.WINDOW_FULLSCREEN, cv2.WINDOW_KEEPRATIO, cv2.WINDOW_GUI_EXPANDED
    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    cv2.imshow(name, img_data)

def close_all_windows():
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
    



class SEGMENT_ALGO:
    def process(self,img_data:np.ndarray) -> list:
        print("Segmentation: Starting Segmentation Stage")
        start = time()
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img_data)
        
        # Reshape image
        print("Segmentation: Resizing Image")
        # (width,height) : (300,225) *(600,200), (600,400), (400,200) 
        img_size = (600,300) 
        gpu_img = cv2.cuda.resize(gpu_img,img_size,interpolation=cv2.INTER_NEAREST)
        resized_img = gpu_img.download()
        
        # #  Sharpen the image
        # print("Segmentation: Sharpening Image")
        # kernel = np.array([[0, -1, 0],
        #                     [-1, 5, -1],
        #                     [0, -1, 0]])
        
        # sharpen = cv2.filter2D(src=resized_img, ddepth=-1, kernel=kernel)
        # gpu_img.upload(sharpen)
        # sharpened_img = gpu_img.download()
        
        # Apply Grayscale
        print("Segmentation: Applying GrayScale to Image")
        gpu_img.upload(resized_img)
        gpu_img = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
        grayscale_img = gpu_img.download()
        
        # Apply GaussianBlur Filter
        print("Segmentation: Blurring Image")
        blur = cv2.GaussianBlur(grayscale_img, (5, 5), 0)
        gpu_img.upload(blur)
        blurred_img = gpu_img.download()
        
        # Apply Thresholding Filter
        print("Segmentation: Binarization of image ")
        threshold = cv2.adaptiveThreshold(blurred_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,45,15)
        gpu_img.upload(threshold)
        binary_img : np.ndarray = gpu_img.download()
        show_image(binary_img,"Binary-IMage")
        
        # Upper and Lower Bound Criteria (amount of connected pixels) that makes up a character
        # Need to find out how many white pixels (character blob) makes up a character
        # This will be used for masking the pixels for the characters to send to the OCR algorithm
        total_pixels = binary_img.shape[0] * binary_img.shape[1]
        # lboundary,uboundary = [(80,17), (70,20), (75,15), (80,15)]
        pix_lboundary = total_pixels // 75
        pix_uboundary = total_pixels // 15

        # Connected Component Analysis on the image and init mask to store interested components 
        print("Segmentation: Conducting Connected Component Analysis")
        num_labels,labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img)
    
        # Creating the mask
        gpu_mask = cv2.cuda_GpuMat()
        mask = np.zeros(binary_img.shape,dtype=np.uint8)
        gpu_mask.upload(mask)
        
        #  Loop through all connected components
        for (ind,label) in enumerate(np.unique(labels)):
            # if label is background, ignore it and continue
            if label == 0:
                continue
            else:
                # Else, construct label mask to display only connected components
                gpu_label_mask = cv2.cuda_GpuMat()
                label_mask = np.zeros(binary_img.shape, dtype=np.uint8)
                label_mask[labels == label] = 255
                gpu_label_mask.upload(label_mask)
                
                num_pixels = cv2.cuda.countNonZero(gpu_label_mask)
                
            # if number of pixels in component is between upper and lower boundaries, add it to mask
            if num_pixels >= pix_lboundary and num_pixels <= pix_uboundary:
                gpu_mask = cv2.cuda.add(gpu_mask,gpu_label_mask)
        
        mask = gpu_mask.download()
        show_image(mask,"Mask-Image")
        
        # Find contours and get the bounding boxes for the contours
        print("Segmentation: Finding Contours on Masked Image")
        contours,hiers = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Storing the bounding boxes from left to right, top to bottom
        def contour_sort(a,b):
            bb_a = cv2.boundingRect(a)
            bb_b = cv2.boundingRect(b)
            if abs(bb_a[1] - bb_b[1]) <= 10:
                return bb_a[0] - bb_b[0]
            else:
                return bb_a[1] - bb_b[1]

        contours = sorted(contours, key= cmp_to_key(contour_sort))
        print("Segmentation: Number of contours = {0}".format(len(contours)))
        lp_characters = list()
        for idx,ct in enumerate(contours):
            x,y,w,h = cv2.boundingRect(ct)
            roi :np.ndarray = mask[y:y + h, x:x + w]
            # roi0 = roi
            # roi1 = roi
            # roi2 = roi
            # char_3d = np.stack((roi0,roi1,roi2),axis=2)
            # print(char_3d.shape)
            # show_image(img_data=char_3d,name=f"Character{idx}")
            roi_list = [ roi for i in range(3)]
            print(roi_list)
            break
            lp_characters.append(roi)

        print("Segmentation: Completed Segmentation")
        end = time()
        display_execution_time(start,end)
        
        # for i, char in enumerate(lp_characters):
        #     show_image(img_data=char,name=f"Character {i+1}")
        
        close_all_windows()
        return lp_characters


if __name__ == "__main__":
    algo = SEGMENT_ALGO()
    img = cv2.imread("Segmentation/LP-1.JPG",cv2.IMREAD_COLOR)
    chars = algo.process(img_data=img)
    # show_image(img,"Original Image")
    