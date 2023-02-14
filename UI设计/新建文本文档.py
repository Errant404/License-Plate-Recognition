from PySide2.QtCore import QObject, Signal
from PySide2.QtWidgets import QApplication, QMessageBox, QPlainTextEdit, QTextBrowser, QLabel
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QImage, QPixmap
import time
import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import os
import argparse
from rich.console import Console
from tqdm.tk import tqdm
from rich import print
from threading import Thread


class MySignals(QObject):
    text_print = Signal(QTextBrowser, str)
    photo_print = Signal(QLabel, np.ndarray)


class Stats:

    def __init__(self, path):
        # 从文件中加载UI定义

        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit

        self.ui = QUiLoader().load('u.ui')
        self.path = path
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True)

        self.ms = MySignals()
        self.ui.Pbutton1.clicked.connect(self.task2)
        self.ms.text_print.connect(self.printTextToGui)
        self.ms.photo_print.connect(self.printPhotoToGui)

    def printTextToGui(self, widget, text):
        widget.append(str(text))
        widget.ensureCursorVisible()

    def printPhotoToGui(self, widget, img):
        image = QImage(img, img.shape[1], img.shape[0], QImage.Format_BGR888)
        image = QPixmap(image).scaled(400, 300)
        widget.setPixmap(image)

    def task2(self):
        self.ui.Pbutton1.setEnabled(False)

        def get_plate_imgs_crop(path, ocr):
            labels_path = os.path.join(path, 'labels')
            label_lists = os.listdir(labels_path)
            label_lists.sort(key=lambda x: os.path.getmtime((labels_path + "\\" + x)))

            list(map(lambda x: os.remove(os.path.join(labels_path, x)), label_lists[0: -1]))
            txtName = label_lists[-1]

            with open(os.path.join(labels_path, txtName)) as f:
                texts = f.readlines()
                num = len(texts)

            name = txtName.split('.')[0].replace('_', '')
            len_name = len(name)

            plate_path = os.path.join(path, 'crops\\Plate')

            img_lists = os.listdir(plate_path)
            img_lists.sort(key=lambda x: os.path.getmtime((plate_path + "\\" + x)))
            list(map(lambda x: os.remove(os.path.join(plate_path, x)), img_lists[0: -num]))
            img_names = img_lists[-num:]

            for imgName in img_names:
                img_path = os.path.join(plate_path, imgName)
                img = cv2.imread(img_path)
                result = ocr.ocr(img, cls=True)
                try:
                    print(f'[green]{result[0][0][1][0]}[/green]')
                    self.ms.text_print.emit(self.ui.textBrowser, result[0][0][1][0])

                except Exception:
                    print('[red]识别失败[/red]')
                    self.ms.text_print.emit(self.ui.textBrowser, '识别失败')

        thread = Thread(target=get_plate_imgs_crop, args=(self.path, self.ocr))
        thread.start()
        self.ui.Pbutton1.setEnabled(True)

    def task1(self):
        self.ui.Pbutton1.setEnabled(False)

        def txt_recognize(path, ocr):
            imgs_info = self.get_plate_imgs(path)
            rec_success, rec_failed, plates_total = 0, 0, 0
            dir_list = os.listdir(path)[2:]
            dir_list.sort(key=lambda x: os.path.getmtime((path + "\\" + x)))
            img_total = len(dir_list)
            for imageName in tqdm(dir_list, desc='正在识别车牌', leave=False, mininterval=0.0001):

                plate_num = next(imgs_info)
                plates_total += plate_num
                for i in range(plate_num):
                    img = next(imgs_info)

                    self.ms.photo_print.emit(self.ui.label, img)

                    """
                    cv2.namedWindow(imageName, cv2.WINDOW_AUTOSIZE)
                    cv2.imshow(imageName, img)
                    cv2.waitKey(0)
                    cv2.destroyWindow(imageName)
                    """

                    result = ocr.ocr(img, cls=True)

                    try:
                        print(f'[green]{result[0][0][1][0]}[/green]')
                        self.ms.text_print.emit(self.ui.textBrowser, result[0][0][1][0])
                        rec_success += 1
                    except Exception:
                        print('[red]识别失败[/red]')
                        self.ms.text_print.emit(self.ui.textBrowser, '识别失败')
                        rec_failed += 1
            return rec_success, rec_failed, img_total, plates_total

        thread = Thread(target=txt_recognize, args=(self.path, self.ocr))
        thread.start()
        self.ui.Pbutton1.setEnabled(True)

    def get_plate_imgs(self, path):
        labels_path = os.path.join(path, 'labels')
        label_lists = os.listdir(labels_path)
        label_lists.sort(key=lambda x: os.path.getmtime((labels_path + "\\" + x)))
        for txtName in label_lists:
            with open(os.path.join(labels_path, txtName)) as f:
                texts = f.readlines()
                yield len(texts)
                imgName = txtName.split('.')[0] + '.jpg'
                img_path = os.path.join(path, imgName)
                assert os.path.exists(img_path), "image file {} dose not exist.".format(img_path)
                Img = cv2.imread(img_path)

                for text in texts:
                    text = text.replace('\n', ' ').split(' ')
                    x_c, y_c, w, h = float(text[1]), float(text[2]), float(text[3]), float(text[4])
                    w, h, x_c, y_c = w * Img.shape[1], h * Img.shape[0], x_c * Img.shape[1], y_c * Img.shape[0]
                    xmin, xmax, ymin, ymax = x_c - w / 2, x_c + w / 2, y_c - h / 2, y_c + h / 2

                    img = Image.fromarray(Img)
                    img = img.crop((xmin, ymin, xmax, ymax))
                    img = img.resize((100, 30), Image.LANCZOS)
                    img = np.asarray(img)

                    yield img


if __name__ == "__main__":
    fileNames = os.listdir("D:/yolov5-7.0/runs/detect")
    fileNames.sort(key=lambda x: os.path.getmtime(("D:/yolov5-7.0/runs/detect" + "\\" + x)))
    fileName = fileNames[-1]

    path = os.path.join("D:/yolov5-7.0/runs/detect", fileName)

    app = QApplication([])
    stats = Stats(path)
    stats.ui.show()
    app.exec_()
