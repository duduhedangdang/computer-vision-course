import cv2
import numpy as np
import os
import pandas as pd

import pickle as pkl
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#查看图像的辅助功能
def showImgAndWait(titulo, img):
    cv2.imshow(titulo, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#计算直方图的函数
def genHistogram(angles, magnitudes):
    histogram = np.zeros(9, dtype=np.float32)

    for i in range(angles.shape[0]):
        for j in range(angles.shape[1]):
            bin1 = int( angles[i, j] // 20 )
            bin2 = int(( (angles[i, j] // 20) + 1) % 9)
            # 计算两个受影响的angles的vote值
            prop = (angles[i, j] - (bin1 * 20)) / 20

            # 计算两个受影响的bin的vote值
            vote1 = (1 - prop) * magnitudes[i, j]
            vote2 =  prop * magnitudes[i, j]

            histogram[bin1] += vote1
            histogram[bin2] += vote2

    return histogram

#function to generate the cells
def genCells(angles, magnitudes, cell_size):
    cells = []
    for i in range(0, np.shape(angles)[0], cell_size):
        row = []
        for j in range(0, np.shape(angles)[1], cell_size):
            #calculate the histogram
            histogram = genHistogram(angles[i:i + cell_size, j:j + cell_size],
                    magnitudes[i:i + cell_size, j:j + cell_size])
            row.append(histogram)
            
        cells.append(row)

    return np.array(cells, dtype=np.float32)


#函数生成块，带有高斯和归一化
# 重叠量2/3
def genBlocks(block_size, cells, stride=1, sigma=0.5, normalization_tipe=2, threshold=0):
    ini = block_size // 2
    fin = block_size // 2

    if block_size % 2 != 0:
        fin = fin + 1

    sigma = block_size *0.5
    first_stop = ini
    second_stop = ini

    if np.shape(cells)[0] % block_size == 0:
        first_stop = first_stop - 1

    if np.shape(cells)[1] % block_size == 0:
        second_stop = second_stop - 1

    blocks = []
    for i in range( 0, np.shape(cells)[0] - block_size, stride):
        for j in range( 0, np.shape(cells)[1] - block_size, stride):

            block = np.array(cells[i :i + block_size, j:j + block_size])
            block = block.flatten()
            #标准化
            if normalization_tipe == 0:
                block =  normalize_L1(block, threshold)
            elif normalization_tipe == 1:
                block = normalize_L1_sqrt(block, threshold)
            elif normalization_tipe == 2:
                block = normalize_L2(block, threshold)
            elif normalization_tipe == 3:
                block = normalize_L2_Hys(block, threshold)


            blocks.append(block)

    blocks = np.array(blocks, dtype=np.float32)
    return blocks



## 标准化函数
def normalize_L1(block, threshold):
    norm = np.sum(block) + threshold
    if norm != 0:
        return block / norm
    else:
        return block


def normalize_L1_sqrt(block, threshold):
    norm = np.sum(block) + threshold
    if norm != 0:
        return np.sqrt(block / norm)
    else:
        return block


def normalize_L2(block, threshold):
    norm = np.sqrt(np.sum(block * block) + threshold * threshold)
    if norm != 0:
        return block / norm
    else:
        return block


def normalize_L2_Hys(block, threshold):
    norm = np.sqrt(np.sum(block * block) + threshold * threshold)

    if norm != 0:
        block_aux = block / norm
        block_aux[block_aux > 0.2] = 0.2
        norm = np.sqrt(np.sum(block_aux * block_aux) + threshold * threshold)
        if norm != 0:
            return block_aux / norm
        else:
            return block_aux
    else:
        return block




def hog(img,gamma=0.3,cell_size=6,block_size =3,normalization_tipe=2):
#hog参数

    img = np.power(img,gamma, dtype=np.float32)
    # 梯度计算
    # 计算居中的水平和垂直渐变，不进行平滑处理
    kernel = np.asarray([-1, 0, 1])
    kernel_vacio = np.asarray([1])
    gradientsx = cv2.sepFilter2D(img, -1, kernel,kernel_vacio)
    gradientsy = cv2.sepFilter2D(img, -1,kernel_vacio, kernel)

    # 空间/方向 BINNING
    # 计算梯度的大小和角度
    magnitude, angle = cv2.cartToPolar(gradientsx, gradientsy, angleInDegrees=True)

    # 如果图像是彩色的，选择具有最大值的通道作为该像素的强度
    intensity = np.argmax(magnitude, axis=2)
    x, y = np.ogrid[:intensity.shape[0], :intensity.shape[1]]
    max_angle = angle[x, y, intensity]
    max_magnitude = magnitude[x, y, intensity]

    # 角度从0-360 到 0-180
    max_angle = (360 - max_angle) % 180
    #生成cells
    cells = genCells(max_angle,max_magnitude,cell_size)
    #使用归一化和高斯函数生成块
    blocks_normalized = genBlocks(block_size,cells,normalization_tipe=normalization_tipe)

    #return a vector 
    return blocks_normalized.flatten()


#加载所有图像
def loadImgs(fichero):
    images = []
    for dirName, subdirList, fileList in os.walk(fichero):
        for name in fileList:
            img = cv2.imread(fichero +"/"+name, 1)
            images.append(img)

    return np.array(images)

def generateFeatures(name,gamma=0.3,cell_size=6,block_size =3,normalization_tipe=2):
    images_pos = loadImgs("./data/images/pos_person")
    images_neg = loadImgs("./data/images/neg_person")

    #计算正描述数据并保存
    pos_data = np.array([hog(img,gamma,cell_size,block_size,normalization_tipe) for img in images_pos])
    pos_data = pd.DataFrame(pos_data)
    pos_data.to_csv("./data/descriptor/" +name +"_pos.dat", sep=" ",index=False,header=False)

    #计算负描述数据并保存
    neg_data = np.array([hog(img,gamma,cell_size,block_size,normalization_tipe) for img in images_neg])
    neg_data = pd.DataFrame(neg_data)
    neg_data.to_csv("./data/descriptor/"+name +"_neg.dat", sep=" ",index=False,header=False)



#训练SVM
def training(filename_svm,filename_dat,filename_roc,tittle_roc,kernel='rbf',test_size=0.2, cv=False):
    #下载两种数据
    pos_data = pd.read_table("./data/descriptor/"+filename_dat+"_pos.dat", sep=" ", header= None)
    neg_data = pd.read_table("./data/descriptor/"+filename_dat+"_neg.dat", sep=" ", header=None)

    #添加标签
    pos_data["etiqueta"] = np.ones(pos_data.shape[0],dtype=np.uint8)
    neg_data["etiqueta"] = np.zeros(neg_data.shape[0],dtype=np.uint8)

    #如果要做交叉验证
    if(cv):
        datos = pd.concat([pos_data,neg_data])
        x_datos = datos.drop('etiqueta', axis=1) 
        y_datos = datos['etiqueta']
        #执行交叉验证
        svclassifier2 = SVC(kernel=kernel,gamma= 3e-2)
        scores = cross_val_score(svclassifier2, x_datos, y_datos, cv=5)
        # 95%置信区间的平均精度
        print("Accuracy 5-cross validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    else:
        train_pos, test_pos = train_test_split(pos_data, test_size = test_size, random_state=50)
        train_neg, test_neg = train_test_split(neg_data, test_size = test_size, random_state=50)

        train = pd.concat([train_pos,train_neg])
        test = pd.concat([test_pos,test_neg])

        x_train = train.drop('etiqueta', axis=1) 
        y_train = train['etiqueta']

        x_test = test.drop('etiqueta', axis=1) 
        y_test = test['etiqueta']

        svclassifier = SVC(kernel=kernel, gamma= 3e-2)  
        svclassifier.fit(x_train, y_train)

        pkl.dump(svclassifier, open("./data/models/" + filename_svm +".p", "wb"))
        # 做出预测
        y_pred = svclassifier.predict(x_test)
        acc_test=accuracy_score(y_test, y_pred)
        print("Acc_test: (TP+TN)/(T+P)  %0.2f" % acc_test)

        #ROC图计算
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        patch = mpatches.Patch(color='blue', label='ROC curve. area = {}, error = {}'.format(np.round(roc_auc, 4),
                                                                                        np.round(1 - roc_auc, 4)))
        plt.legend(handles=[patch], loc='lower right')
        plt.plot(fpr, tpr, color='blue')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(tittle_roc)
        plt.savefig("./result/"+filename_roc, dpi=700)
        #plt.show()
        plt.clf()

#行人检测功能
def pedestrian_detection(img,file_classifier, hor_jump=4,vert_jump=4):
    
    result_img = np.copy(img)
    pyramid = [img]
    new_img = img
    #生成高斯金字塔（附加功能）
    while np.shape(new_img)[0] >= 128 and np.shape(new_img)[1] >= 64:
        new_img = cv2.pyrDown(new_img)
        pyramid.append(new_img)

    #使用SVM分类器
    svm = pkl.load(open(file_classifier,"rb"))
    for level, img_pyramid in zip(range(len(pyramid)), pyramid):
        for i in range(0, np.shape(img_pyramid)[0] - 128, vert_jump):
            for j in range(0, np.shape(img_pyramid)[1] - 64, hor_jump):

                sub_img = img_pyramid[i:i + 128, j:j + 64]
                dst = hog(sub_img, 0.3)
                prediction = svm.predict(dst.reshape(-1, dst.shape[0]))
                #行人检测
                if prediction[0] == 1.0:
                    upper_left =((j* 2**level,i * 2**level))
                    down_right = ( (j+64) * 2**level,(i+128) *2**level )
                    cv2.rectangle(result_img,upper_left,down_right,(0,255,0),thickness=1)
    return result_img


def main():
    print("GENERATING FEATURES")
    generateFeatures("Norm_L1",gamma=0.3,normalization_tipe=0)
    generateFeatures("Norm_L1_sqrt",gamma=0.3,normalization_tipe=1)
    generateFeatures("Norm_L2",gamma=0.3,normalization_tipe=2)
    generateFeatures("Norm_L2_hys",gamma=0.3,normalization_tipe=3)

    print("TRAINING LINEAR")
    training("svmL1-linear","Norm_L1","CurveROC-linear-L1","Norm L1 Linear",kernel='linear',cv=False)
    training("svmL1_sqrt-linear","Norm_L1_sqrt","CurveROC-linear-L1_sqrt","Norm L1_sqrt linear",kernel='linear',cv=False)
    training("svmL2-linear","Norm_L2","CurveROC-linear-L2","Norm L2 linear",kernel='linear',cv=False)
    training("svmL2_hys-linear","Norm_L2_hys","CurveROC-linear-L2_hys","Norm L2_hys linear",kernel='linear',cv=False)

    print("TRAINING RBF")
    training("svmL1-rbf","Norm_L1","CurveROC-rbf-L1","Norm L1 rbf",kernel='rbf',cv=False)
    training("svmL1_sqrt-rbf","Norm_L1_sqrt","CurveROC-rbf-L1_sqrt","Norm L1_sqrt rbf",kernel='rbf',cv=False)
    training("svmL2-rbf","Norm_L2","CurveROC-rbf-L2","Norm L2 rbf",kernel='rbf',cv=False)
    training("svmL2_hys-rbf","Norm_L2_hys","CurveROC-rbf-L2_hys","Norm L2_hys rbf",kernel='rbf',cv=False)


    print("TRAINING RBF CV")
    training("svmL2-rbf","Norm_L2","CurveROC-rbf-L2","Norm L2 rbf",kernel='rbf',cv=True)

    img = cv2.imread("./test/prueba.png", 1)
    #img = cv2.pyrDown(img)
    res = pedestrian_detection(img,"./data/models/svmL2-rbf.p",4, 8)
    cv2.imwrite("./result/result_prueba.png",res)

    img = cv2.imread("./test/prueba2.jpg", 1)
    img = cv2.pyrDown(img)
    res = pedestrian_detection(img,"./data/models/svmL2-rbf.p",4, 8)
    cv2.imwrite("./result/result_prueba2.png",res)

    img = cv2.imread("./test/prueba3.jpg", 1)
    #img = cv2.pyrDown(img)
    res = pedestrian_detection(img,"./data/models/svmL2-rbf.p",4, 8)
    cv2.imwrite("./result/result_prueba3.png",res)


    img = cv2.imread("./test/granda.jpg", 1)
    #img = cv2.pyrDown(img)
    res = pedestrian_detection(img,"./data/models/svmL2-rbf.p",4, 8)
    cv2.imwrite("./result/result_granada1.png",res)

    img = cv2.imread("./test/granda.jpg", 1)
    #img = cv2.pyrDown(img)
    res = pedestrian_detection(img,"./data/models/svmL2-rbf.p",4, 4)
    cv2.imwrite("./result/result_granada1.png",res)

    img = cv2.imread("./test/beatles.jpg", 1)
    img = cv2.pyrDown(img)
    res = pedestrian_detection(img,"./data/models/svmL2-rbf.p",4, 4)
    cv2.imwrite("./result/result_beatles.png",res)

    pass



if __name__ == "__main__":
    main()

#lineal L1
#Acc_test: (TP+TN)/(T+P)  0.95
#lineal l1_sqrt
#Acc_test: (TP+TN)/(T+P)  0.94
#lineal L2
#Acc_test: (TP+TN)/(T+P)  0.94
#lineal L2_lys
#Acc_test: (TP+TN)/(T+P)  0.93

#rbf L1
# Acc_test: (TP+TN)/(T+P)  0.94
#rbf L1_sqrt
# Acc_test: (TP+TN)/(T+P)  0.97
#rbf L2
# Acc_test: (TP+TN)/(T+P)  0.95
# rbf L2_hys
# Acc_test: (TP+TN)/(T+P)  0.96

#Accuracy 5-cross validation: 0.95 (+/- 0.00)

#Acc_test: (TP+TN)/(T+P)  0.94 gauss mask