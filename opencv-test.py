from cv2 import VideoCapture, imshow, imwrite, waitKey, destroyWindow
  
# initialize the camera

cam_port = 0
cam = VideoCapture()
  
# reading the input using the camera
result, image = cam.read()
  
# If image will detected without any error, 
if result:
    # showing result, it take frame name and image 
    # output
    imshow("CameraFeed", image)
  
    # saving image in local storage
    imwrite("Test.png", image)
  
    # If keyboard interrupt occurs, destroy image 
    # window
    waitKey(0)
    destroyWindow("CameraFeed")
  
# If captured image is corrupted, moving to else part
else:
    print("No image detected. Please! try again")