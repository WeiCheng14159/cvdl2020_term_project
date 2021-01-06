# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'demoUI_n.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIntValidator


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1579, 720)
        Form.setMinimumSize(QtCore.QSize(1579, 720))
        Form.setMaximumSize(QtCore.QSize(1579, 720))
        font = QtGui.QFont()
        font.setPointSize(10)
        Form.setFont(font)
        self.imageIndex = QtWidgets.QComboBox(Form)
        self.imageIndex.setGeometry(QtCore.QRect(1130, 660, 151, 31))
        font = QtGui.QFont()
        font.setFamily("PMingLiU")
        font.setPointSize(10)
        self.imageIndex.setFont(font)
        self.imageIndex.setTabletTracking(True)
        self.imageIndex.setObjectName("imageIndex")
        self.detectButton = QtWidgets.QPushButton(Form)
        self.detectButton.setGeometry(QtCore.QRect(1090, 580, 161, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.detectButton.setFont(font)
        self.detectButton.setObjectName("detectButton")
        self.originalImage = QtWidgets.QLabel(Form)
        self.originalImage.setGeometry(QtCore.QRect(20, 140, 500, 400))
        self.originalImage.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.originalImage.setFrameShape(QtWidgets.QFrame.Box)
        self.originalImage.setLineWidth(1)
        self.originalImage.setText("")
        self.originalImage.setScaledContents(True)
        self.originalImage.setWordWrap(False)
        self.originalImage.setObjectName("originalImage")
        self.preprocessingImage = QtWidgets.QLabel(Form)
        self.preprocessingImage.setGeometry(QtCore.QRect(540, 140, 500, 400))
        self.preprocessingImage.setFrameShape(QtWidgets.QFrame.Box)
        self.preprocessingImage.setLineWidth(1)
        self.preprocessingImage.setText("")
        self.preprocessingImage.setScaledContents(True)
        self.preprocessingImage.setWordWrap(False)
        self.preprocessingImage.setObjectName("preprocessingImage")
        self.detectionResultImage = QtWidgets.QLabel(Form)
        self.detectionResultImage.setGeometry(
            QtCore.QRect(1060, 140, 500, 400))
        self.detectionResultImage.setFrameShape(QtWidgets.QFrame.Box)
        self.detectionResultImage.setLineWidth(1)
        self.detectionResultImage.setText("")
        self.detectionResultImage.setScaledContents(True)
        self.detectionResultImage.setWordWrap(False)
        self.detectionResultImage.setObjectName("detectionResultImage")
        self.directoryBrowser = QtWidgets.QTextBrowser(Form)
        self.directoryBrowser.setGeometry(QtCore.QRect(250, 660, 731, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.directoryBrowser.setFont(font)
        self.directoryBrowser.setObjectName("directoryBrowser")
        self.browseButton = QtWidgets.QPushButton(Form)
        self.browseButton.setGeometry(QtCore.QRect(80, 660, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.browseButton.setFont(font)
        self.browseButton.setObjectName("browseButton")
        self.titleLabel = QtWidgets.QLabel(Form)
        self.titleLabel.setGeometry(QtCore.QRect(340, 10, 961, 101))
        font = QtGui.QFont()
        font.setFamily("PMingLiU")
        font.setPointSize(27)
        font.setBold(True)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.titleLabel.setFont(font)
        self.titleLabel.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.titleLabel.setTextFormat(QtCore.Qt.AutoText)
        self.titleLabel.setScaledContents(False)
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.titleLabel.setObjectName("titleLabel")
        self.originalImageLabel = QtWidgets.QLabel(Form)
        self.originalImageLabel.setGeometry(QtCore.QRect(220, 100, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.originalImageLabel.setFont(font)
        self.originalImageLabel.setObjectName("originalImageLabel")
        self.preprocessingLabel = QtWidgets.QLabel(Form)
        self.preprocessingLabel.setGeometry(QtCore.QRect(720, 100, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.preprocessingLabel.setFont(font)
        self.preprocessingLabel.setObjectName("preprocessingLabel")
        self.detectImageLabel = QtWidgets.QLabel(Form)
        self.detectImageLabel.setGeometry(QtCore.QRect(1130, 550, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.detectImageLabel.setFont(font)
        self.detectImageLabel.setObjectName("detectImageLabel")
        self.originalSlider = QtWidgets.QSlider(Form)
        self.originalSlider.setGeometry(QtCore.QRect(130, 600, 731, 22))
        self.originalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.originalSlider.setObjectName("originalSlider")
        self.orignalSliderLabel = QtWidgets.QLabel(Form)
        self.orignalSliderLabel.setGeometry(QtCore.QRect(30, 590, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.orignalSliderLabel.setFont(font)
        self.orignalSliderLabel.setObjectName("orignalSliderLabel")
        self.chooseImageLabel = QtWidgets.QLabel(Form)
        self.chooseImageLabel.setGeometry(QtCore.QRect(1020, 660, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.chooseImageLabel.setFont(font)
        self.chooseImageLabel.setObjectName("chooseImageLabel")
        self.detectResultLabel = QtWidgets.QLabel(Form)
        self.detectResultLabel.setGeometry(QtCore.QRect(1260, 100, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.detectResultLabel.setFont(font)
        self.detectResultLabel.setObjectName("detectResultLabel")
        self.detectResultLabel_2 = QtWidgets.QLabel(Form)
        self.detectResultLabel_2.setGeometry(QtCore.QRect(1370, 550, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.detectResultLabel_2.setFont(font)
        self.detectResultLabel_2.setObjectName("detectResultLabel_2")
        self.originalIndexEdit = QtWidgets.QLineEdit(Form)
        self.originalIndexEdit.setGeometry(QtCore.QRect(870, 590, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.originalIndexEdit.setFont(font)
        self.originalIndexEdit.setObjectName("originalIndexEdit")
        self.originalIndexEdit.setValidator(QIntValidator(0, 999))
        self.originalIndexEdit.setMaxLength(3)
        self.originalIndexEdit.setAlignment(Qt.AlignRight)
        self.detectionResultEdit = QtWidgets.QLineEdit(Form)
        self.detectionResultEdit.setGeometry(QtCore.QRect(1270, 590, 291, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.detectionResultEdit.setFont(font)
        self.detectionResultEdit.setObjectName("detectionResultEdit")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Detection Nodules"))
        self.detectButton.setText(_translate("Form", "Detect"))
        self.browseButton.setText(_translate("Form", "Browse Dataset"))
        self.titleLabel.setText(_translate("Form", "Evaluate The Malignancy Of Pulmonary Nodules \n"
                                           "Using 3-D Deep Leaky Noisy-OR Network \n"
                                           " "))
        self.originalImageLabel.setText(_translate("Form", "Original Image"))
        self.preprocessingLabel.setText(
            _translate("Form", "Preprocessing Image"))
        self.detectImageLabel.setText(_translate("Form", "Detect Image"))
        self.orignalSliderLabel.setText(_translate("Form", "Original Slice"))
        self.chooseImageLabel.setText(_translate("Form", "Choose Image"))
        self.detectResultLabel.setText(_translate("Form", "Detect Result"))
        self.detectResultLabel_2.setText(_translate("Form", "Detect Result"))
