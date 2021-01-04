# from ../DSB2017.layers import nms,iou
import sys
sys.path.append("../DSB2017")
from layers import nms, iou
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
        super(appMainWindow, self).__init__()
        self.setupUi(qtWidget)
        # self.preprocessButton.clicked.connect(QCoreApplication.instance().quit)
        self.detectButton.clicked.connect(self.Detection_Training)
        self.browseButton.clicked.connect(self.Browser_directory)
        self.imageIndex.currentIndexChanged.connect(self.onCurrentIndexChanged)
        self.originalSlider.valueChanged.connect(self.choose_slice)
        self.originalIndexEdit.textChanged.connect(self.Index_change)
        self.input_dir = ""

    def Index_change(self, text):
        try:
            value = int(text)
        except:
            print("invalid index", text)

    def Browser_directory(self):
        self.input_dir = ""
        print("------Open Browser-------")
        dialog_style = QFileDialog.DontUseNativeDialog
        dialog_style |= QFileDialog.DontUseCustomDirectoryIcons
        self.input_dir = QFileDialog.getExistingDirectory(
            None, 'Select a folder:',  "../Image",)
        self.directoryBrowser.setText(self.input_dir)
        if self.input_dir == '':
            print("not select file yet")
        else:
            self.Path_Image = ("/").join(self.input_dir.split("/")[0:-1])
            self.List_Path_Image = [f for f in os.listdir(self.Path_Image)]

            self.List_Image = [f for f in os.listdir(
                self.input_dir) if f.endswith('.npy')]
            self.List_Image.sort(reverse=True)

            self.imageIndex.addItems(self.List_Image)
            self.Path_Original = self.input_dir + \
                "/" + str(self.imageIndex.currentText())
            self.Path_Preprocessing = self.Path_Image + \
                "/" + self.List_Path_Image[1]
            self.Path_Detection = self.Path_Image + \
                "/" + self.List_Path_Image[0]
            print(self.List_Path_Image)

    def onCurrentIndexChanged(self, indeximg):
        self.Path_img = self.input_dir + "/" + \
            str(self.imageIndex.currentText())

        self.Load_numpy_img = np.load(self.Path_img, allow_pickle=True)

        cv2.imwrite(self.Path_img.replace(".npy", ".jpg"),
                    self.Load_numpy_img[0, 0])
        img = QPixmap(self.Path_img.replace(".npy", ".jpg"))
        self.originalSlider.setValue(0)
        self.originalImage.setPixmap(img)
        self.originalImage.show()


##############################################################################

    def Show_Preprocessing(self):
        """"""
        self.Path_preprocessing = self.input_dir.split("/")[0:-1]
        self.Path_preprocessing = (
            "/").join(self.Path_preprocessing) + "2_Preprocessing_img"
        # print(self.Path_preprocessing)
        """"""

        Path_Preprocessing_img = self.Path_Preprocessing + \
            "/" + str(self.imageIndex.currentText())
        # print(self.Path_Preprocessing)
        self.Load_numpy_img = np.load(Path_Preprocessing_img)

        cv2.imwrite(Path_Preprocessing_img.replace(
            ".npy", ".jpg"), self.Load_numpy_img[0, 0])
        img = QPixmap(Path_Preprocessing_img.replace(".npy", ".jpg"))

        self.preprocessingImage.setPixmap(img)
        # self.preprocessingImage.setGeometry(540, 120, 500, 400)
        self.preprocessingImage.show()

    def Detection_Training(self):
        from shutil import copyfile
        if len(self.input_dir) == 0:
            print("Choose source folder first!")
        else:
            self.Path_preprocessing1 = self.input_dir.split("/")[0:-1]
            self.Path_preprocessing_img = (
                "/").join(self.Path_preprocessing1) + "/2Preprocessing_img/"
            self.Path_preprocessing_label = (
                "/").join(self.Path_preprocessing1) + "/2Preprocessing_label/"
            self.Path_preprocessing_Detect = (
                "/").join(self.Path_preprocessing1) + "/3Detection_img/"
            self.Path_preprocessing_image = (
                "/").join(self.Path_preprocessing1) + "/preprocess_result/"
            self.Path_pp = self.Path_img.replace(
                'original', 'preprocess_result').replace('_origin', 'pp_clean')
            # Lisst_del = [f for in os.listdir(self.Path_Preprocessing)]
            for i in os.listdir(self.Path_preprocessing_img):
                os.remove(self.Path_preprocessing_img + i)
            for i in os.listdir(self.Path_preprocessing_label):
                os.remove(self.Path_preprocessing_label + i)
            for i in os.listdir(self.Path_preprocessing_Detect):
                os.remove(self.Path_preprocessing_Detect + i)
            # print(self.Path_pp)
            copyfile(self.Path_pp, self.Path_preprocessing_img +
                     os.path.basename(self.Path_pp).replace("pp_clean", "_clean"))
            copyfile(self.Path_pp.replace("pp_clean", "_label"), self.Path_preprocessing_img.replace(
                '_img', '_label') + os.path.basename(self.Path_pp.replace("pp_clean", "_label")))

            sys.path.append("../DSB2017")
            from main import main_init
            main_init()
            appMainWindow.Detection_Image(self)

    def Detection_Image(self):
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

    def choose_slice(self):
        # [Phu] valueChanged.connect(self.changeValue)
        if self.input_dir == "":
            print("select dataset first !!")
        else:
            self.originalSlider.valueChanged.connect(self.changeValue)
            self.Path_img = self.input_dir + "/" + \
                str(self.imageIndex.currentText())
            self.Path_pp = self.Path_img.replace(
                'original', 'preprocess_result').replace('_origin', 'pp_clean')
            [self.spacing_pre, self.img_pre] = np.load(
                self.Path_pp, allow_pickle=True)
            self.originalSlider.setMinimum(0)
            self.originalSlider.setMaximum(self.img_pre.shape[1] - 1)

            #self.input_dir = QFileDialog.getExistingDirectory(None, 'Select a folder:',  "../Image",)

            self.Load_numpy_img = np.load(self.Path_img)

            # self.setWindowTitle("SpinBox demo")

    def changeValue(self):

        pre_size = self.originalSlider.value()
        self.originalIndexEdit.setText(str(pre_size))
        # print(size)
        # new_size = img.shape[1] spacing[0]
        img_size = int(pre_size / self.spacing_pre[0])

        cv2.imwrite(self.Path_pp.replace('npy', 'png'),
                    self.img_pre[0, pre_size])
        self.im = QPixmap(self.Path_pp.replace('npy', 'png'))
        self.preprocessingImage.setPixmap(self.im)
        self.preprocessingImage.show()

        print(pre_size, self.spacing_pre, img_size)

        """Show Original Image from Pre Number"""

        print(self.Load_numpy_img.shape)
        self.Load_numpy_img_new = np.expand_dims(self.Load_numpy_img, axis=0)
        # cv2.imwrite(self.Path_img.replace(".npy",".jpg"),self.Load_numpy_img_new[0,img_size])

        # plt.imshow(self.Load_numpy_img_new[0,img_size],'gray')
        plt.imsave(self.Path_img.replace(".npy", ".jpg"),
                   self.Load_numpy_img_new[0, img_size], cmap='gray')
        img = QPixmap(self.Path_img.replace(".npy", ".jpg"))

        self.originalImage.setPixmap(img)
        self.originalImage.show()

    # """"""
#################################### processing ###############################
        # self.im1 = QPixmap(joinfolder + "/Image_Pr" +  "/" + self.Path_Original )

        # # self.Input_label = QLabel()
        """
        img = np.load('./Image/Image_Pr/001_clean.npy')
        
        import matplotlib.pyplot as plt
        cv2.imwrite("./Image/Image_Pr/001_clean.jpg",img[0,25])
        # cv2.imshow("aa",img[0,25])

        # cv2.waitKey(0)
        # print(img[0,25,12,23])
        # img = Image.fromarray(img[0,25,5], 'RGB')
        # # print(img.shape)
        self.im = QPixmap('./Image/Image_Pr/001_clean.jpg' ) 
        self.preprocessingImage.setPixmap(self.im)
        self.preprocessingImage.setGeometry(450, 110, 411, 331) 
        self.preprocessingImage.show()
        # pbb = np.load('./bbox_result/000_pbb.npy')

        # pbb = pbb[pbb[:,0]>-1]
        # pbb = nms(pbb,0.05)
        # box = pbb[0].astype('int')[1:]
        # print(box)

        
        # ax = plt.subplot(1,1,1)
        # plt.imshow(img[0,box[0]],'gray')
        # plt.axis('off')


        # ListImageOut = [h for h in os.listdir(joinfolder) if h.endswith(".jpg")]
        # self.im2 = QPixmap(joinfolder + "/Image_Out" + "/" + self.Path_Original ) 
        # # self.Input_label = QLabel()
        # self.detectionResultImage.setPixmap(self.im2) 
        # self.detectionResultImage.setGeometry(880, 110, 411, 331) 
        # self.detectionResultImage.show()
        """

    # def ImageOut(self):

    #     FolderImageOut = "./Image_Out/"
    #     ListImageOut = [h for h in os.listdir(FolderImageOut) if h.endswith(".jpg")]

    # def ShowImage(self):
    #     lb = QtGui.QLabel(self)


def Main_Function():
    ui = appMainWindow()
    qtWidget.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    Main_Function()
