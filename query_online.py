# -*- coding: utf-8 -*-
# Author: AI算法与图像处理
from extract_cnn_vgg16_keras import VGGNet

import numpy as np
import h5py
from matplotlib import pyplot as plt
#import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse


ap = argparse.ArgumentParser()
#ap.add_argument("-query", required = False, default='TestImages/0.png',
#	help = "Path to query which contains image to be queried")
# ap.add_argument("-index", required = False,default='LACEfeatureCNN.h5',
# 	help = "Path to index")
# ap.add_argument("-result", required = False,default='lace',
# 	help = "Path for output retrieved images")
# 总数据
ap.add_argument("-index", required = False,default='lol.h5',
	help = "Path to index")
ap.add_argument("-result", required = False,default='img',
	help = "Path for output retrieved images")

args = vars(ap.parse_args())


# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File(args["index"],'r')
# feats = h5f['dataset_1'][:]
feats = h5f['dataset_1'][:]
print(feats)
imgNames = h5f['dataset_2'][:]
print(imgNames)
h5f.close()
        
print("--------------------------------------------------")
print("               searching starts")
print("--------------------------------------------------")

def url_is_correct(index_t):
    
    if index_t >5:
        print('超出请求次数！！！')
        exit()

    try:
        url = input('请输入正确的图片路径：')
        
        queryDir = url
        
        src = mpimg.imread(queryDir)
        return queryDir
            
    except:
        print('有误的图片路径，请重新输入：')
    return url_is_correct(index_t+1)


while True:    
    print('----------**********-------------')
    op = input("退出请输 exit，查询请输 enter : ")
    if  op == 'exit':
        break
    else:
        # read and show query image
        # 设置多张图片共同显示
        figure,ax=plt.subplots(4,4)
        global index_t
        index_t = 1
        queryDir = url_is_correct(index_t)
        queryImg = mpimg.imread(queryDir)
        x=0
        ax[x][x].set_title('Test-Image',fontsize=10)
        ax[x][x].imshow(queryImg,cmap=plt.cm.gray)
        ax[x][x].axis('off') # 显示第一张测试图片
        #plt.title("Query Image")
        #plt.imshow(queryImg)
        #plt.show()

        # init VGGNet16 model
        model = VGGNet()

        # extract query image's feature, compute simlarity score and sort
        queryVec = model.extract_feat(queryDir)
        scores = np.dot(queryVec, feats.T)
        rank_ID = np.argsort(scores)[::-1]
        rank_score = scores[rank_ID]
        #print rank_ID
        #print(rank_score)


        # number of top retrieved images to show
        maxres = 15
        imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
        print("top %d images in order are: " %maxres, imlist)

        # # show top #maxres retrieved result one by one
        # for i,im in enumerate(imlist):
        #     image = mpimg.imread(args["result"]+"/"+str(im, 'utf-8'))
        #     plt.title("search output %d" %(i+1))
        #     plt.imshow(image)
        #     plt.show()

        # 显示多张图片

        for i,im in enumerate(imlist):
            image = mpimg.imread(args["result"]+"/"+str(im, 'utf-8'))
            im_name = str(im).split('\'')[1]
            ax[int((i+1)/4)][(i+1)%4].set_title('%d Image %s -- %.2f' % (i+1,im_name,rank_score[i]),fontsize=10)
            #ax[int(i/maxres)][i%maxres].set_title('Image_name is %s' % im,fontsize=2)
            ax[int((i+1)/4)][(i+1)%4].imshow(image,cmap=plt.cm.gray)
            ax[int((i+1)/4)][(i+1)%4].axis('off')
        plt.show()
