#-*- coding:utf-8 -*-
#'''
# Created on 18-12-11 上午10:03
#
# @Author: Greg Gao(laygin)
#'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from ctpn_model import CTPN_Model
from ctpn_utils import gen_anchor, bbox_transfor_inv, clip_box, filter_bbox,nms, TextProposalConnectorOriented
from ctpn_utils import resize
import config

imgs_path = './datasetss/testing_data/images/'
outpath = './evaluation/submit'
if not os.path.exists(outpath):
    os.makedirs(outpath)

prob_thresh = 0.7
width = 1200
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

weights_total = os.listdir(config.checkpoints_dir)
if '.DS_Store' in weights_total:
    weights_total.remove('.DS_Store')
for weight_name in weights_total:
    weights = os.path.join(config.checkpoints_dir, weight_name)
    model = CTPN_Model()
    model.load_state_dict(torch.load(weights, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()

    for imgname in tqdm(os.listdir(imgs_path)[:]):
        #print(imgname)
        img_path = os.path.join(imgs_path, imgname)
        image = cv2.imread(img_path)
        orih, oriw = image.shape[:2]
        image = resize(image, width=width)
        image_c = image.copy()
        h, w = image.shape[:2]
        image = image.astype(np.float32) - config.IMAGE_MEAN
        image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()


        with torch.no_grad():
            image = image.to(device)
            cls, regr = model(image)
            cls_prob = F.softmax(cls, dim=-1).cpu().numpy()
            regr = regr.cpu().numpy()
            anchor = gen_anchor((int(h / 16), int(w / 16)), 16)
            bbox = bbox_transfor_inv(anchor, regr)
            bbox = clip_box(bbox, [h, w])

            fg = np.where(cls_prob[0, :, 1] > prob_thresh)[0]
            select_anchor = bbox[fg, :]
            select_score = cls_prob[0, fg, 1]
            select_anchor = select_anchor.astype(np.int32)

            keep_index = filter_bbox(select_anchor, 1)

            # nms
            select_anchor = select_anchor[keep_index]
            select_score = select_score[keep_index]
            select_score = np.reshape(select_score, (select_score.shape[0], 1))
            nmsbox = np.hstack((select_anchor, select_score))
            keep = nms(nmsbox, 0.3)
            select_anchor = select_anchor[keep]
            select_score = select_score[keep]

            # text line-
            textConn = TextProposalConnectorOriented()
            text = textConn.get_text_lines(select_anchor, select_score, [h, w])
            #print(text)
        with open(os.path.join(outpath, 'res_img_'+imgname.split('.')[0]+'.txt'), 'w') as f:
            for line in text:
                l = np.array(line[:8]).reshape(4, 2)
                xmin = max(0, np.min(l[:, 0]))*oriw/width
                xmax = min(w-1, np.max(l[:, 0]))*oriw/width
                ymin = max(0, np.min(l[:, 1]))*oriw/width
                ymax = min(h-1, np.max(l[:, 1]))*oriw/width
                if xmin<1e-6: xmin=0
                if ymin<1e-6: ymin=0
                f.write(','.join([str(a) for a in [xmin, ymin, xmax, ymax]]+['random'])+'\n')

    os.chdir('./evaluation')
    os.system('python zip.py')
    result = os.popen('python script.py -c DetEva')
    print(result.readlines(), weight_name)
    os.chdir('..')

