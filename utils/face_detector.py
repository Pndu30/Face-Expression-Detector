import cv2
import numpy as np

def cam_init(cam_idx: int = 0):
    try:
        cam = cv2.VideoCapture(cam_idx)
        cam.set(3, 640) # Width
        cam.set(4, 480) # Height

        cascade  = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
    except Exception as e:
        raise("Error initialization {e}")
