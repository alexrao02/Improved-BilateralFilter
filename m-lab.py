# coding: utf-8
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
from PIL import Image


def euclidean_distance(x1,y1,x2,y2):
    '''
    计算两个像素点之间的欧拉距离
    :param (x1,y1):像素点1的坐标, type=int
    :param (x2,y2):像素点2的坐标, type=int
    :return:两个像素点之间的欧拉距离, type=float
    '''
    return np.sqrt(np.square(x1-x2)+np.square(y1-y2))


def pixel_difference(p1,p2):
    '''
    计算两个像素点之间的灰度值距离
    :param p1: 像素点1的灰度值, type=int
    :param p2: 像素点2的灰度值, type=int
    :return: 两个像素点之间的灰度值距离, type=int
    '''
    return abs(int(p1)-int(p2))


def closeness_func(distance,sigma_d):
    '''
    接近程度函数, 对两个像素点的距离根据高斯函数求得度量双边滤波算法权重的第一项值
    :param distance: 两个像素点之间的距离, type=float
    :param sigma_d: 双边滤波算法的sigma元组的第一项sigma值, type=float
    :return: 像素点之间接近程度, 即双边滤波算法权重的第一项值, type=float
    '''
    c=np.exp(-0.5*np.square(distance/sigma_d))
    return c


def similarity_func(difference,sigma_r):
    '''
    灰度相似程度函数, 对两个像素点的灰度差根据高数函数形式变换求得度量双边滤波算法权重的第二项值
    :param difference: 两个像素点之间的灰度差, type=float
    :param sigma_r: 双边滤波算法的sigma元组的第二项sigma值, type=float
    :return: 灰度接近程度, 即双边滤波算法权重的第二项值, type=float
    '''
    s=np.exp(-0.5*np.square(difference/sigma_r))
    return s


def fai(t):
    if t <= 0.012546391983472:
        res = 5.851875995763328 * t + 0.125874125874126
    else:
        res = t ** 0.3684
    return float(res)


def MLAB(img):
    init_image_mat = np.array(img)  # 将原始图像转化成numpy数组
    filtered_image_mat=np.zeros_like(init_image_mat)
    #对图像的每一个像素值进行双边滤波操作, 一共有RGB三个通道分别操作
    for channel in range(init_image_mat.shape[2]):
        for x in range(init_image_mat.shape[0]):
            for y in range(init_image_mat.shape[1]):
                mlab_x = fai(init_image_mat[x][y][0]/95.047)
                mlab_y = fai(init_image_mat[x][y][1]/100)
                mlab_z = fai(init_image_mat[x][y][2]/108.883)
                if channel == 0:
                    filtered_image_mat[x][y][channel] = 114.4 * mlab_y - 14.4
                if channel == 1:
                    filtered_image_mat[x][y][channel] = 311.5 * (mlab_x - mlab_y)
                if channel == 2:
                    filtered_image_mat[x][y][channel] = 111 * (mlab_y - mlab_z)
    filtered_image = Image.fromarray(cv.cvtColor(filtered_image_mat, cv.COLOR_XYZ2RGB))
    return filtered_image

#用于彩色图的双边滤波器
def bilateral_filtering_color(kernel_size,sigma):
    '''
    双边滤波操作(彩色图), RGB三通道
    :param init_image: 原始图像路径, type=string
    :param kernel_size: filter大小, type=list
    :param sigma: sigma_d和sigma_r的值, type=list
    :return: 滤波操作后的图像, type=Image
    '''
    #img_lenna = Image.open(init_image_path)  # 用PIL中的Image.open打开图像
    img_lenna = cv.imread('red.png')
    init_image= cv.cvtColor(img_lenna, cv.COLOR_BGR2XYZ)

    init_image_mat = np.array(init_image)  # 将原始图像转化成numpy数组
    filtered_image_mat = np.zeros_like(init_image_mat)
    # 对图像的每一个像素值进行双边滤波操作, 一共有RGB三个通道分别操作
    for channel in range(init_image_mat.shape[2]):
        for x in range(init_image_mat.shape[0]):
            for y in range(init_image_mat.shape[1]):
                mlab_x = fai(init_image_mat[x][y][0] / 95.047)
                mlab_y = fai(init_image_mat[x][y][1] / 100)
                mlab_z = fai(init_image_mat[x][y][2] / 108.883)
                if channel == 0:
                    filtered_image_mat[x][y][channel] = 114.4 * mlab_y - 14.4
                if channel == 1:
                    filtered_image_mat[x][y][channel] = 311.5 * (mlab_x - mlab_y)
                if channel == 2:
                    filtered_image_mat[x][y][channel] = 111 * (mlab_y - mlab_z)
    sigma_d=sigma[0]
    sigma_r=sigma[1]
    filtered_image_mat_new=np.zeros_like(filtered_image_mat)
    #对图像的每一个像素值进行双边滤波操作, 一共有RGB三个通道分别操作
    for channel in range(filtered_image_mat.shape[2]):
        for x in range(filtered_image_mat.shape[0]):
            for y in range(filtered_image_mat.shape[1]):
                print("finish x=%d,y=%d"%(x,y))
                #每个像素的操作范围为以其为中心的kernel
                filtered_pixel=0
                weight_sum=0
                for i in range(-(kernel_size[0]//2),(kernel_size[0]//2)+1):
                    for j in range(-(kernel_size[1]//2),(kernel_size[1]//2)+1):
                        # 遍历周边滤波模板内的中心点的周边像素点
                        now_x=x+i
                        now_y=y+j
                        # 特判超出边界的不合法坐标
                        if now_x<0 or now_x>=filtered_image_mat.shape[0] or now_y<0 or now_y>=filtered_image_mat.shape[1]:
                            continue
                        dis=euclidean_distance(x,y,now_x,now_y) #距离
                        diff=pixel_difference(filtered_image_mat[x][y][channel],filtered_image_mat[now_x][now_y][channel]) #像素差
                        closeness=closeness_func(dis,sigma_d) #双边滤波函数的第一项权重值
                        similarity=similarity_func(diff,sigma_r) #双边滤波函数的第二项权重值
                        weight=closeness*similarity #双边滤波函数的权重: 第一项*第二项
                        filtered_pixel+=weight*init_image_mat[now_x][now_y][channel] #按权重聚合周边像素点值
                        weight_sum+=weight #记录总权重,用于标准化
                #标准化及更新
                filtered_image_mat_new[x][y][channel]=int(round(filtered_pixel/weight_sum))

    filtered_image=Image.fromarray(filtered_image_mat_new)
    # filtered_image = Image.fromarray(cv.cvtColor(filtered_image_mat, cv.COLOR_Lab2RGB))
    return filtered_image


if __name__=='__main__':
    kernel_size=[3,3]
    sigma=[10,10]
    filtered_image=bilateral_filtering_color(kernel_size,sigma)
    filtered_image.show()
    filtered_image.save('MLAB.png')
    img_lenna = cv.imread('lena.png')
    # init_image = cv.cvtColor(img_lenna, cv.COLOR_BGR2RGB) #可调
    # img = cv.imread("red.png", 1)
    # filtered_image_OpenCV = cv.bilateralFilter(img, 3, 10.0, 10.0)
    # cv.imwrite("new.png", filtered_image_OpenCV)