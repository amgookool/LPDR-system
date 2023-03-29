import numpy as np
import cv2

def show_image(img_data: np.ndarray, name: str):
    # cv2.WINDOW_NORMAL, cv2.WINDOW_AUTOSIZE, cv2.WINDOW_GUI_NORMAL, cv2.WINDOW_FULLSCREEN, cv2.WINDOW_KEEPRATIO, cv2.WINDOW_GUI_EXPANDED
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img_data)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyWindow(name)

def draw_box(img_data: np.ndarray, x: int, y: int, w: int, h: int, color:tuple=(0,0,0)) -> np.ndarray:
    cv2.rectangle(img_data, (x, y), (x+w, y+h), color, 2)
    return img_data

def put_text(img_data: np.ndarray, text: str, x: int, y: int, font=cv2.FONT_HERSHEY_DUPLEX, color: tuple = (255, 255, 255),bg_color:tuple=(0,0,0), size: float = 0.55, thickness: int = 1) -> np.ndarray:
    text_size,_ = cv2.getTextSize(text, font, size, thickness)
    texW, texH = text_size
    y -= 5 
    cv2.rectangle(img_data, (x, y), (x + texW, y - texH), bg_color, -1)
    cv2.putText(img_data, text, org=(x, y), fontFace=font,
                fontScale=size, color=color, thickness=thickness)
    return img_data