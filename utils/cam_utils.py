import cv2
import numpy as np
import time

def cam_init(cam_idx: int = 0):
    try:
        cam = cv2.VideoCapture(cam_idx)
        cam.set(3, 640) # Width
        cam.set(4, 480) # Height
        fps = cam.get(cv2.CAP_PROP_FPS)
        cascade  = cv2.CascadeClassifier('./utils/haarcascade_frontalface_default.xml')
        return cam, cascade, fps
    except Exception as e:
        raise("Error initialization {e}")
    
def cam_loop(cam, cascade, kill_key=13):
    ret, frame = cam.read()
    
    if not ret:
        print("Error: Failed to grab frame.")
        cam.release()
        cv2.destroyAllWindows()
        return 0
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cor = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(face_cor) == 0:
        pass
    else:
        x1, y1, w, h = face_cor[0]
        face_image = frame[y1:y1+h, x1:x1+w]
        cur_time = time.time()
        last_time = get_face_img(cur_time, last_time, face_image)
        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), [0, 255, 0], 3)
    
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(10) == kill_key:
        cam.release()
        cv2.destroyAllWindows()
        return 0
    
    return face_image
    
def get_face_img(cur_time, last_time, img):
    if cur_time -  last_time >= 1:
        last_time = cur_time
        filename = fr"img.jpg"
        cv2.imwrite(filename, img)
    return last_time

if __name__ == '__main__':
    try:
        cam, cascade, fps = cam_init()
        print(fps)
        cam_loop(cam, cascade)
    except KeyboardInterrupt:
        cam.release()
        cv2.destroyAllWindows()


