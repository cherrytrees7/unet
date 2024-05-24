'''
Authon: wangzhenlin
data:   2021.5.19
'''
from ast import Import
import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random

import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np
from utils.aug import Process16
import pydicom


class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        # self.mask_path = mask_path
        # print(os.path.join(self.data_path, 'image/*.png'))
        self.imgs_paths = glob.glob(os.path.join(self.data_path, 'img/*.png'))
        # self.imgs_paths = glob.glob(os.path.join(self.mask_path, 'image/*.png'))

        # self.img_list=os.listdir(self.data_path)
        # self.mask_list=os.listdir(self.mask_path)

        # self.img_paths=os.listdir(self.img_list)
        # self.mask_paths=os.listdir(self.mask_path)


    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_paths[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('img', 'mask')
        # label_path=label_path.replace(".dcm","-lesion.png")
    
        # 读取训练图片和标签图片
        # ds=pydicom.read_file(image_path)           
        # image = ds.pixel_array
        image = cv2.imread(image_path,1)
        label = cv2.imread(label_path,0)
        if image is None:
            print(image_path)
        if label is None:
            print(label_path)
        label=np.where(label>1,1,0)

        # label=np.where(label==100,1,label)
        # label=np.where(label==200,2,label)
        
   
       
        # c = random.choice([ 0, 1, 2,3,4,5,6])
        extend=Process16()
        # if c >=4 :
        #     image,label=extend.rotate_bound(image,label,5)  
        #     image,label=extend.warp(image,label,3)

        c = random.choice([ 0, 1, 2,3,4,5,6])
        if c >=4 :
            # image,label=extend.hsv_transform(image,label,5)  
            image,label=extend.randomHorizontalFlip(image,label,5)   

        c = random.choice([ 0, 1, 2,3,4,5,6])
        extend=Process16()
        if c >=4 :
            image,label=extend.stretching_h(image,label,5)  
            image,label=extend.stretching_w(image,label,5)
        # label=np.where(label>2,0,label)
        image=image.astype(np.uint8)
        label=label.astype(np.uint8)
        image=cv2.resize(image,(256,448),1)
        label=cv2.resize(label,(256,448),0)

        # label = np.where(label>10,1,0)

        image=image.astype(np.float32)
        image = transforms.ToTensor()(image)
        
        label=label.astype(np.float32)
        label = transforms.ToTensor()(label)
        # label=torch.squeeze(label,0)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_paths)


if __name__ == "__main__":
    isbi_dataset = ISBI_Loader(r"/Users/cherrytrees7/Downloads/model/data/train")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=6, 
                                               shuffle=True)
    for image, label in train_loader:
        print("image.shape: ",image.shape[1])
        print("label.shape: ",label.shape)