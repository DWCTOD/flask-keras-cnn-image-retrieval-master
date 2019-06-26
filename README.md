# Image Retrieval Engine Based on Keras

基于内容的图像检索（ Content-Based Image Retrieval ，CBIR）
## 演示

详细讲解、操作截图和视频演示

[以图搜图展示](https://mp.weixin.qq.com/s?__biz=MzU4NTY4Mzg1Mw==&mid=2247484752&idx=1&sn=063168237aab0d4fba2c0ae426be2e4a&chksm=fd8783b2caf00aa4bffcfc9f4730bdf2b3c5a4f1d3fdee91115cdd3fc23709a3ac70f1d9c876&token=403649858&lang=zh_CN#rd)

首发于公众号：AI算法与图像处理

![公众号](https://github.com/DWCTOD/flask-keras-cnn-image-retrieval-master/blob/master/img/qrcode_for_gh_cf77d20d7eb8_430.jpg)

## 环境

```python
import keras
Using Theano backend
```

本人采用的是keras 2.24版本，使用python3.6

原文作者备注：https://github.com/willard-yuan/flask-keras-cnn-image-retrieval

keras 2.0.1 及 2.0.5 版本均经过测试可用。推荐Python 2.7，支持Python 3.6.

此外需要numpy, matplotlib, os, h5py, argparse. 推荐使用anaconda安装

## 文件说明

img 文件夹：图片库——用于搭建索引库和后续查询结果的显示

 1.jpg 2.jpg ：用于测试

extract_cnn_vgg16_keras.py:提取图片特征

index.py：建立索引库并保存模型

lol.h5：本人事先建好的模型，可以用于直接测试

pachong.py：爬取百度的图片

query_online.py：运行代码实现以图搜图在线测试

## 使用

```python
#操作汇总
# （1）打开终端执行index.py代码
python3 index.py -database img -index lol.h5
# 此时已经将图片库转成索引库并保存和输出lol.h5模型
# (2)继续运行代码 query_online.py
python3 query_online.py
# 按照提示，如果要退出输入 exit 查询直接enter
# 输入测试图片 名字即可（如果测试图片额代码不在同一路径下需要增加路径——这里 设置相对路径）
```



备注：本人对源代码进行了一些修改

1）增加了一个异常处理操作，主要是为了方便，即使手误输错也能继续运行，这个在之前的文章中有讲解过；

[学会这招再也不怕手误让代码崩掉](http://mp.weixin.qq.com/s?__biz=MzU4NTY4Mzg1Mw==&mid=2247484695&idx=1&sn=530d383d799e1aaa4554747098c53e01&chksm=fd8783f5caf00ae38c93613aab97df7feb9d6b13c7018e1fa2f378ec86bb97d164e93e7a1a2b&scene=21#wechat_redirect)

（2）作者最终显示的结果只能一张一张的展示，没有对比图，因此我稍微修改了一下，让可视化的效果更加的美观一些，有兴趣的可以参考我的代码；

（3）我对参数输入也进行了修改，将模型名字和图片库的路径都固定了，这样子测试的时候比较方便，大家在使用的时候请注意下，如果修改了名字要对应起来。

![代码](https://github.com/DWCTOD/flask-keras-cnn-image-retrieval-master/blob/master/img/%E6%B7%B1%E5%BA%A6%E6%88%AA%E5%9B%BE_%E9%80%89%E6%8B%A9%E5%8C%BA%E5%9F%9F_20190508214737.png)

备注：对于第一次使用的小伙伴，请保持心态良好，一开始要下载VGG16的预训练模型，可能等待时间比较久，此时可以考虑泡一杯coffee，如果下载速度慢是由于国内镜像源的问题，可以自行百度如何切换“源”，当然也可以找其他人上传的模型链接例如： https://github.com/fchollet/deep-learning-models/releases