import cv2
import os

# 参考 https://blog.csdn.net/qq_36516958/article/details/114274778
# https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#2-create-labels

# path = "/public/ccpd_green/test/"
path = r"D:\yolov5-5.0\MyData\images\train"

for filename in os.listdir(path):


    try:
        list1 = filename.split("-", 3)  # 第一次分割，以减号'-'做分割
        subname = list1[2]
        list2 = filename.split("-")[-3]
        class_len = len(list2.split('_'))
        if class_len == 8:
            cls = 0
        else:
            cls = 1
        lt, rb = subname.split("_", 1)  # 第二次分割，以下划线'_'做分割
        lx, ly = lt.split("&", 1)
        rx, ry = rb.split("&", 1)
        width = int(rx) - int(lx)
        height = int(ry) - int(ly)  # bounding box的宽和高
        cx = float(lx) + width / 2
        cy = float(ly) + height / 2  # bounding box中心点

        img_path = os.path.join(path, filename)
        img = cv2.imread(img_path)
        width = width / img.shape[1]
        height = height / img.shape[0]
        cx = cx / img.shape[1]
        cy = cy / img.shape[0]

        txtname = filename.split(".", 1)
        txtfile = path + txtname[0] + ".txt"
        # 绿牌是第0类，蓝牌是第1类
        with open(r"D:\yolov5-5.0\MyData\labels\train\{}.txt".format(txtname[0]), "w") as f:
            f.write(str(cls) + " " + str(cx) + " " + str(cy) + " " + str(width) + " " + str(height))

        print(filename)
    except:
        pass
