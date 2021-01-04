# from ../DSB2017.layers import nms,iou
import sys                     # nopep8
sys.path.append("../DSB2017")  # nopep8
from layers import nms, iou    # nopep8

import cv2
import os
import sys
import glob
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import measure, morphology, segmentation

from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMainWindow, QApplication, QFileDialog

from demoUI import Ui_Form

app = QtWidgets.QApplication(sys.argv)
qtWidget = QtWidgets.QWidget()


class appMainWindow(Ui_Form):

    def __init__(self):

        self.inputDir = ""
        self.slideBarVal = 0

        super(appMainWindow, self).__init__()
        self.setupUi(qtWidget)

        # Called when detection button is clicked
        self.detectButton.clicked.connect(self.__detectionTraining)

        # Called when browse directory button is clicked
        self.browseButton.clicked.connect(self.__browserDirectory)

        # Called when user chooses diff images in a data set
        self.imageIndex.currentIndexChanged.connect(self.__datasetIndexChanged)
        self.imageIndex.currentIndexChanged.connect(self.__updateView)

        # Called when slide bar moves
        self.originalSlider.valueChanged.connect(self.__slideBarValueChange)
        self.originalSlider.valueChanged.connect(self.__updateView)

        # Called when user types an index
        self.originalIndexEdit.textChanged.connect(self.__sliceChange)
        self.originalIndexEdit.textChanged.connect(self.__updateView)

    def __browserDirectory(self):

        print("------Open Browser-------")
        dialog_style = QFileDialog.DontUseNativeDialog
        dialog_style |= QFileDialog.DontUseCustomDirectoryIcons

        # Choose dataset directory
        self.inputDir = QFileDialog.getExistingDirectory(
            None, 'Select a folder:',  "../Image",)
        self.directoryBrowser.setText(self.inputDir)

        if self.inputDir == '':
            print("not select file yet")
        else:
            # Set dataset directory
            self.datasetPath = ("/").join(self.inputDir.split("/")[0:-1])
            self.listDatasetPath = [f for f in os.listdir(self.datasetPath)]

            # Get list of images (in *.npy form)
            self.listOfImages = [f for f in os.listdir(
                self.inputDir) if f.endswith('.npy')]
            self.listOfImages.sort(reverse=True)

            # Clear before adding items to drop list
            self.imageIndex.clear()
            self.imageIndex.addItems(self.listOfImages)

    def __update_path(self):

        path2Img = self.inputDir + "/" + str(self.imageIndex.currentText())
        path2preImg = path2Img.replace('original', 'preprocess_result').replace(
            '_origin', 'pp_clean')
        return (path2Img, path2preImg)

    def __sliceChange(self, text):

        try:
            self.slideBarVal = int(text)
        except:
            print("Invalid index: ", text)

    def __datasetIndexChanged(self, indeximg):

        # Update image path
        self.__update_path()

        # Reset to num 0 slice
        self.originalSlider.setValue(0)

    def __slideBarValueChange(self):

        # Update slide bar value
        self.slideBarVal = self.originalSlider.value()

        # Update the slice value in text field
        self.originalIndexEdit.setText(str(self.slideBarVal))

    def __detectionTraining(self):

        from shutil import copyfile
        if len(self.inputDir) == 0:
            print("Choose source folder first!")
        else:
            self.Path_preprocessing1 = self.inputDir.split("/")[0:-1]
            self.Path_preprocessing_img = (
                "/").join(self.Path_preprocessing1) + "/2Preprocessing_img/"
            self.Path_preprocessing_label = (
                "/").join(self.Path_preprocessing1) + "/2Preprocessing_label/"
            self.Path_preprocessing_Detect = (
                "/").join(self.Path_preprocessing1) + "/3Detection_img/"
            self.Path_preprocessing_image = (
                "/").join(self.Path_preprocessing1) + "/preprocess_result/"
            self.path2PreImg = self.path2Image.replace(
                'original', 'preprocess_result').replace('_origin', 'pp_clean')
            for i in os.listdir(self.Path_preprocessing_img):
                os.remove(self.Path_preprocessing_img + i)
            for i in os.listdir(self.Path_preprocessing_label):
                os.remove(self.Path_preprocessing_label + i)
            for i in os.listdir(self.Path_preprocessing_Detect):
                os.remove(self.Path_preprocessing_Detect + i)
            # print(self.path2PreImg)
            copyfile(self.path2PreImg, self.Path_preprocessing_img +
                     os.path.basename(self.path2PreImg).replace("pp_clean", "_clean"))
            copyfile(self.path2PreImg.replace("pp_clean", "_label"), self.Path_preprocessing_img.replace(
                '_img', '_label') + os.path.basename(self.path2PreImg.replace("pp_clean", "_label")))

            sys.path.append("../DSB2017")
            from main import main_init
            main_init()
            appMainWindow.detectionImage(self)

    def detectionImage(self):

        Path_Detection_img = self.Path_preprocessing_img.replace(
            "2Preprocessing_img", "3Detection_img") + "/" + str(self.imageIndex.currentText()).replace("_origin", "_pbb")
        Path_Preprocessing_img = self.Path_preprocessing_img + "/" + \
            str(self.imageIndex.currentText()).replace("_origin", "_clean")
        [a, img] = np.load(Path_Preprocessing_img, allow_pickle=True)

        pbb = np.load(Path_Detection_img)

        pbb = pbb[pbb[:, 0] > -1]
        pbb = nms(pbb, 0.05)
        box = pbb[0].astype('int')[1:]
        print(box)
        text = "Slice: %s  Height: %s  Width: %s  Chanels: %s" % (
            box[0], box[1], box[2], box[3])
        self.detectionResultEdit.setText(text)
        ax = plt.subplot(1, 1, 1)
        ax.axis('off')
        plt.imshow(img[0, box[0]], 'gray')
        plt.axis('off')
        rect = patches.Rectangle((box[2]-box[3], box[1]-box[3]), box[3]
                                 * 2, box[3]*2, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        # plt.show()
        ax.figure.savefig(Path_Detection_img.replace(".npy", ".jpg"))
        plt.cla()
        img = QPixmap(Path_Detection_img.replace(".npy", ".jpg"))
        self.detectionResultImage.setPixmap(img)
        # self.detectionResultImage.setGeometry(1060, 120, 500, 400)
        self.detectionResultImage.show()

    # Called when either user changes images in dataset OR slide bar moves OR user types
    # a new slice
    def __updateView(self):

        if self.inputDir == "":
            print("Select dataset first !!")
        else:
            # Compute path to original image and preprocessing image
            (self.path2Image, self.path2PreImg) = self.__update_path()

            # Load preprocessed image of size (1, 278, 236, 328)
            [preProcImgSize, self.preProcImg] = np.load(
                self.path2PreImg, allow_pickle=True)

            # The maximum value of slide bar is set to the max num of slices of preprocessed image
            self.originalSlider.setMaximum(self.preProcImg.shape[1] - 1)

            # Load original image of size (139, 512, 512)
            self.loadNumpyImg = np.load(self.path2Image, allow_pickle=True)

            # scaledIndex maps the index from preprocessing -> original
            scaledIndex = int(
                self.slideBarVal / self.preProcImg.shape[1] * self.loadNumpyImg.shape[0])

            # Original image
            jpgFileName = self.path2Image.replace(".npy", ".jpg")

            # Write to file & read from file
            cv2.imwrite(jpgFileName, self.loadNumpyImg[scaledIndex])
            qImg = QPixmap(jpgFileName)

            self.originalImage.setPixmap(qImg)
            self.originalImage.show()

            # Preprocessing image
            jpgFileName = self.path2PreImg.replace('npy', 'png')

            # Write to file & read from file
            cv2.imwrite(jpgFileName, self.preProcImg[0, self.slideBarVal])
            preImg = QPixmap(jpgFileName)

            self.preprocessingImage.setPixmap(preImg)
            self.preprocessingImage.show()


def Main_Function():
    ui = appMainWindow()
    qtWidget.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    Main_Function()
