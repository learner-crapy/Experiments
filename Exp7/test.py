import numpy as np
import os
from sklearn.decomposition import PCA
import cv2


# function: extract the img feature with a input of image directory
# input: the image path of all the image, string
# output: a list with the narray with shape of (long, width), list
class PCA_IMG:
    OriginalImg = []
    Reshaped = []
    ImgArray = []
    ImgName = []
    Label = []
    Name = {'Gilbert': 0, 'Janet': 1, 'Luke': 2}
    pca = PCA(n_components=50)

    # function: initial the attribute and obtain the path and name
    # input: imgpath
    # output: initial self.ImgName
    def __init__(self, imgpath):
        self.OriginalImg = []
        self.Reshaped = []
        self.ImgArray = []
        self.ImgName = []
        self.Label = []
        self.ImgName = os.listdir(imgpath)
        for i in range(0, len(self.ImgName)):
            self.ImgName[i] = imgpath + '/' + self.ImgName[i]
        self.ReadImg()
        self.GetLabel()
        self.Reshaped_f()
        self.Extract()

    # function: read the img with gray type
    # input: none, get the img path from self.ImgName
    # output: initial self.OriginalImg, list
    def ReadImg(self):
        for i in range(0, len(self.ImgName)):
            img = cv2.imread(self.ImgName[i], 0)
            self.OriginalImg.append(img)

    # function: reshaped the image
    # input: none
    # output: initial self.Reshaped
    def Reshaped_f(self):
        for i in range(0, len(self.OriginalImg)):
            self.Reshaped.append(cv2.resize(self.OriginalImg[i], (150, 150)))

    # function: return the class accroding to the name of the img
    # ninput: name, string
    # output: return the class, 0 or 1 or 2
    def Classes(self, name):
        return self.Name[name]

    # funcion: get the label from the train data
    # input: none
    # output: initial self.Label
    def GetLabel(self):
        for i in range(0, len(self.ImgName)):
            name = self.ImgName[i].split('/')[-1].split('_')[0]
            self.Label.append(self.Classes(name))

    # function: extract the feature of evey img
    # input: none, get form self.OriginalImg
    # output: initial self.ImgArray
    def Extract(self):
        for i in range(0, len(self.Reshaped)):
            self.ImgArray.append(self.pca.fit_transform(self.Reshaped[i]))
            # print(self.ImgArray[i].shape)

ImgDirPath = './AllPictures'
ImgDirPath_1 = './AllPictures_1'
pca_img = PCA_IMG(ImgDirPath)
pca_img_1 = PCA_IMG(ImgDirPath_1)
test_data = np.array(pca_img_1.ImgArray)
train_data = np.array(pca_img.ImgArray)
print(test_data.shape)
print(train_data.shape)
