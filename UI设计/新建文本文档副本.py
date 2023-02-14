from PySide2 import QtCore
from PySide2.QtCore import QObject, Signal
from PySide2.QtWidgets import QApplication, QMessageBox, QPlainTextEdit, QTextBrowser, QLabel, QFileDialog, QMainWindow
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QImage, QPixmap, QIcon
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
from LIB.share import Shared_Info
from detect副本 import main, parse_opt


class MySignals(QObject):
    text_print = Signal(QTextBrowser, str)
    photo_print = Signal(QLabel, np.ndarray)


class Win_Login:

    def __init__(self):
        _, self.parser = parse_opt()
        self.ui = QUiLoader().load('Login.ui')
        self.ui.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.ui.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.choice = None
        self.ui.pushButton_2.setEnabled(False)
        self.ui.pushButton.setEnabled(False)
        self.ui.pushButton_2.clicked.connect(self.main_start)
        self.ui.pushButton.clicked.connect(self.detect)
        self.ui.pushButton_4.clicked.connect(self.help_message)
        self.ui.pushButton_5.clicked.connect(self.info_message)
        self.ui.buttonGroup.buttonClicked.connect(self.handleButtonClicked)

    @staticmethod
    def info_message():
        with open('info.txt', mode='r') as f:
            text = f.read()
        QMessageBox.about(QMainWindow(), '出品信息', text)

    @staticmethod
    def help_message():
        with open('help.txt', mode='r') as f:
            text = f.read()
        QMessageBox.about(QMainWindow(), '使用说明', text)

    def handleButtonClicked(self):
        chosen_button = self.ui.buttonGroup.checkedButton()
        choice = chosen_button.text()
        if choice == '摄像头识别':
            self.choice = '摄像头识别'
            self.ui.pushButton.setEnabled(True)
        elif choice == '图片识别':
            self.choice = '图片识别'
            self.ui.pushButton.setEnabled(True)

    def detect(self):
        if self.choice == '图片识别':
            self.parser.set_defaults(save_txt=True)
            self.parser.set_defaults(source='D:/YOLOv5_7.0/data/images')

            choice = QMessageBox.question(
                self.ui,
                '确认',
                '是否展示检测图片?')

            if choice == QMessageBox.Yes:
                self.parser.set_defaults(view_img=True)

            self.ui.pushButton.setEnabled(False)
            self.ui.radioButton.setEnabled(False)
            self.ui.radioButton_2.setEnabled(False)

            opt = self.parser.parse_args()

            main(opt)
            self.ui.pushButton_2.setEnabled(True)
            QMessageBox.about(QMainWindow(), '提示', '目标检测完毕，可以开始识别了！')

        elif self.choice == '摄像头识别':
            self.parser.set_defaults(save_txt=True)
            self.parser.set_defaults(save_crop=True)
            self.parser.set_defaults(source='0')

            QMessageBox.about(QMainWindow(), '提示', '实时目标检测成功调用，等待识别按钮亮起后开始识别。按q退出实时目标检测。')
            self.ui.pushButton.setEnabled(False)
            self.ui.radioButton.setEnabled(False)
            self.ui.radioButton_2.setEnabled(False)

            opt = self.parser.parse_args()
            self.ui.pushButton_2.setEnabled(True)
            main(opt)

    def main_start(self):
        Shared_Info.MainWindow = Win_Main(self.choice)
        Shared_Info.MainWindow.ui.show()
        self.ui.close()


class Win_Main:

    def __init__(self, choice):
        self.ui = QUiLoader().load('MainWindow.ui')
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True)

        self.choice = choice

        self.ms = MySignals()
        self.path = self.get_path("D:/YOLOv5_7.0/runs/detect")

        if self.choice == '摄像头识别':
            self.task = self.task2
        elif self.choice == '图片识别':
            self.task = self.task1

        self.ui.pushButton.clicked.connect(self.task)
        self.ui.pushButton_2.clicked.connect(self.backToLogin)

        self.ms.text_print.connect(self.printTextToGui)
        self.ms.photo_print.connect(self.printPhotoToGui)

    def backToLogin(self):

        Shared_Info.Login = Win_Login()
        Shared_Info.Login.ui.show()

        self.ui.close()

    def printTextToGui(self, widget, text):
        widget.append(str(text))
        widget.ensureCursorVisible()

    def printPhotoToGui(self, widget, img):
        image = QImage(img, img.shape[1], img.shape[0], QImage.Format_BGR888)
        image = QPixmap(image).scaled(400, 300)
        widget.setPixmap(image)
        widget.setScaledContents(True)

    def task2(self):

        def get_plate_imgs_crop(path, ocr):
            try:
                labels_path, label_lists = self.sortList_byTime(path, 'labels')
                plate_path, img_lists = self.sortList_byTime(path, 'crops\\Plate')
                txtName = label_lists[-1]
                with open(os.path.join(labels_path, txtName)) as f:
                    texts = f.readlines()
                    num = len(texts)
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
                list(map(lambda x: os.remove(os.path.join(plate_path, x)), img_lists[0: -1]))
                list(map(lambda x: os.remove(os.path.join(labels_path, x)), label_lists[0: -1]))
            except:
                self.ms.text_print.emit(self.ui.textBrowser, '尚未识别到车牌')

        thread = Thread(target=get_plate_imgs_crop, args=(self.path, self.ocr))
        thread.start()

    def task1(self):

        def txt_recognize(path, ocr):
            imgs_info = self.get_plate_imgs(path)
            rec_success, rec_failed, plates_total = 0, 0, 0
            dir_list = os.listdir(path)[1:]
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
                        self.ms.text_print.emit(self.ui.textBrowser, result[0][0][1][0].replace('.', '·'))
                        rec_success += 1
                    except Exception:
                        print('[red]识别失败[/red]')
                        self.ms.text_print.emit(self.ui.textBrowser, '识别失败')
                        rec_failed += 1
            print(f'[blink2 yellow]识别图片总数: {img_total}; [blink2 orange]识别车牌总数: {plates_total}; [/blink2 orange]'
                  f'[blink2 green]车牌识别成功数: {rec_success}; [/blink2 green][blink2 red]车牌识别失败数{rec_failed}; [/blink2 red]'
                  f'[blink2 purple]车牌识别率{round(rec_success * 100 / plates_total, 2)}%[/blink2 purple]')
            return rec_success, rec_failed, img_total, plates_total

        thread = Thread(target=txt_recognize, args=(self.path, self.ocr))
        thread.start()

    @staticmethod
    def get_path(dir: str):
        fileNames = os.listdir(dir)
        fileNames.sort(key=lambda x: os.path.getmtime((dir + "\\" + x)))
        fileName = fileNames[-1]
        path = os.path.join(dir, fileName)
        return path

    @staticmethod
    def sortList_byTime(path, dirName: str):
        new_path = os.path.join(path, dirName)
        lists = os.listdir(new_path)
        lists.sort(key=lambda x: os.path.getmtime((new_path + "\\" + x)))
        return new_path, lists

    def get_plate_imgs(self, path):
        labels_path, label_lists = self.sortList_byTime(path, 'labels')
        for txtName in label_lists:
            with open(os.path.join(labels_path, txtName)) as f:
                texts = f.readlines()
                yield len(texts)
                imgName = txtName.split('.')[0] + '.jpg'
                img_path = os.path.join(path, imgName)
                assert os.path.exists(img_path), "image file {} dose not exist.".format(img_path)
                Img = cv2.imread(img_path)

                self.ms.photo_print.emit(self.ui.label_2,
                                         np.asarray(Image.fromarray(Img).resize((460, 460), Image.CUBIC)))

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


def LOGIN_UI_SHOW():
    app = QApplication([])
    app.setWindowIcon(QIcon('images/logo.png'))
    Shared_Info.Login = Win_Login()
    Shared_Info.Login.ui.show()
    app.exec_()


if __name__ == "__main__":
    LOGIN_UI_SHOW()
