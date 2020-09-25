#coding:utf8
import os
from PIL import  Image
from torch.utils import data
import numpy as np
import torch

class OCRDataset(data.Dataset):
    
    def __init__(self, image_path, labelfile, maxlabellength = 40, imagesize=(28, 320), train=True):

        self.labels = self.readfile(labelfile)
        self.imgs = [k for k in self.labels.keys()]
        self.img_path = image_path
        self.maxlabellength = maxlabellength
        self.imagesize = imagesize
        self.train = train
        #self.fakeimg_path = 'C:\\Users\\a811241\\Downloads\\FakeOCR_ordered_alltext\\default'

    def readfile(self, filename):
        res = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for i in lines:
                res.append(i.strip())
        dic = {}
        for i in res:
            p = i.split(' ')
            dic[p[0]] = p[1:]
        return dic
        
    def __getitem__(self, index):
        '''
        一次返回一张图片的数据
        '''
        h, w = self.imagesize
        maxlabellength = self.maxlabellength
        if self.imgs[index][-3:] == 'png':
            img_path = os.path.join(self.img_path, self.imgs[index])
        elif self.imgs[index][-3:] == 'jpg':
            pass
            #img_path = os.path.join(self.fakeimg_path, self.imgs[index])
        else:
            assert 1==0, 'unknown img format'

        img = Image.open(img_path).convert('L')
        width, height = img.size[0], img.size[1]
        scale = height * 1.0 / h
        if scale != 1:
            width = int(width / scale)
            img = img.resize([width, h], Image.ANTIALIAS)
        if self.train:
            assert width <= w, 'the image width must be smaller than the training width config'
        img = np.array(img, 'f') / 255.0 - 0.5

        if width != w and self.train:
            #bg = np.ones((h, w, 3))*0.5
            bg = np.ones((h, w)) * 0.5
            st = np.random.randint(0, w-width)
            #st = 0 #fix the img
            #bg[:, st:st+width, :] = img
            bg[:, st:st + width] = img
            img = bg.astype(np.float32)
        img = torch.from_numpy(img)
        assert len(img.shape) == 2
        imagesize = img.shape[:2]
        #img = img.permute(2, 0, 1)#.contiguous()  # h w c ---> c h w
        labelstr = self.labels[self.imgs[index]]
        labels = torch.ones(maxlabellength)*10000
        labels[:len(labelstr)] = torch.tensor([int(k)+1 for k in labelstr])
        input_length = torch.zeros(1)
        label_length = torch.zeros(1)
        label_length[0] = len(labelstr)

        if (len(labelstr) <= 0):
            print("length of label is 0")
        input_length[0] = imagesize[1] // 8

        return img, torch.squeeze(labels.int()), torch.squeeze(label_length.int()), torch.squeeze(input_length.int())
    
    def __len__(self):
        return len(self.imgs)
