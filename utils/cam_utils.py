import cv2

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
        
    
def get_face_img(cur_time, last_time, img):
    if cur_time -  last_time >= 1:
        last_time = cur_time
        filename = fr"img.jpg"
        cv2.imwrite(filename, img)
    return last_time

if __name__ == '__main__':
    pass


