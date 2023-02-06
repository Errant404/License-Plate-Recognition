import cv2
import numpy as np
import os

if __name__ == "__main__":
    img_path = r"D:\yolov5-5.0\MyData\images\test"
    for filename in os.listdir(img_path):

        try:
            img = cv2.imread(os.path.join(img_path, filename), 0)
            cv2.imencode('.jpg', img)[1].tofile(r"D:\yolov5-5.0\MyData_Gray\images\test\{}".format(filename))
        except:
            pass

