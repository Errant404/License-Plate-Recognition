import os

from PySide2 import QtCore, QtWidgets
from PySide2.QtCore import QPropertyAnimation, QRect, QEasingCurve, Qt
from PySide2.QtGui import QIcon, QColor
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QApplication, QFileDialog, QMessageBox, QMainWindow, QDesktopWidget
from LIB.share import Shared_Info
from detect副本 import main, parse_opt


class Win_Login:

    def __init__(self):

        self.ui = QUiLoader().load('Login.ui')
        self.ui.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.ui.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.pushButton_2.clicked.connect(self.main_start)
        self.choice = None

    def main_start(self):
        Shared_Info.MainWindow = Win_Main(self.choice)
        Shared_Info.MainWindow.ui.show()
        self.ui.close()


class Win_Main:

    def __init__(self, choice):
        self.ui = QUiLoader().load('MainWindow.ui')
        self.ui.pushButton_2.clicked.connect(self.backToLogin)
        self.choice = choice

    def backToLogin(self):
        Shared_Info.Login = Win_Login()
        Shared_Info.Login.ui.show()

        self.ui.close()


app = QApplication([])
app.setWindowIcon(QIcon('images/logo.png'))
Shared_Info.Login = Win_Login()
Shared_Info.Login.ui.show()
app.exec_()
