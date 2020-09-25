import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from ctpn.ctpn_model import CTPN_Model
from ctpn.ctpn_utils import gen_anchor, bbox_transfor_inv, clip_box, filter_bbox,nms, TextProposalConnectorOriented
from ctpn.ctpn_utils import resize
import ctpn.config as config
from crnn.densecrnn_torch import DenseCRNN
import io
import torch
import json
from PIL import Image
from flask import Flask, jsonify, request
import numpy as np

app = Flask(__name__)
'''load detection model'''
class DetectModel:
    def __init__(self, prob_thresh=0.7, minedge=1200):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = CTPN_Model()
        pretrained_path = './ctpn/checkpoints/ctpn_ep15.pth.tar'
        if pretrained_path is not None:
            print('loading trained model from {}'.format(pretrained_path))
            self.model.load_state_dict(torch.load(pretrained_path, map_location=self.device)['model_state_dict'])
        self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.prob_thresh = prob_thresh
        self.minedge = minedge

    def inference(self, image):
        orih, oriw = image.shape[:2]
        image = resize(image, width=self.minedge)
        #image_c = image.copy()
        h, w = image.shape[:2]
        image = image.astype(np.float32) - config.IMAGE_MEAN
        image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()

        with torch.no_grad():
            image = image.to(self.device)
            cls, regr = self.model(image)
            cls_prob = F.softmax(cls, dim=-1).cpu().numpy()
            regr = regr.cpu().numpy()
            anchor = gen_anchor((int(h / 16), int(w / 16)), 16)
            bbox = bbox_transfor_inv(anchor, regr)
            bbox = clip_box(bbox, [h, w])

            fg = np.where(cls_prob[0, :, 1] > self.prob_thresh)[0]
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
            # print(text)
        res = []
        r = oriw/self.minedge
        for line in text:
            l = np.array(line[:8]).reshape(4, 2)
            xmin = max(0, np.min(l[:, 0])) * r
            xmax = min(w - 1, np.max(l[:, 0])) * r
            ymin = max(0, np.min(l[:, 1])) * r
            ymax = min(h - 1, np.max(l[:, 1])) * r
            res.append([xmin, ymin, xmax, ymax])
        res_np = np.array(res)
        return res_np


'''load crnn model'''
class OCRModel:
    def __init__(self):
        char_set = open('./crnn/real.txt', 'r', encoding='utf-8').readlines()
        self.char_set = ''.join([ch.strip('\n') for ch in char_set])
        nclass = len(char_set)+1  #+1 is for blank
        self.imagesize = (32, 320)
        self.maxlabellength = 40
        densenet = DenseCRNN(nClasses=nclass)
        pretrained_path = './crnn/checkpoints/DenseCRNN_49_6.pth'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if pretrained_path is not None:
            print('loading trained model from {}'.format(pretrained_path))
            densenet.load_state_dict(torch.load(pretrained_path, map_location=self.device))
        self.densenet = densenet.to(self.device)
        for p in self.densenet.parameters():
            p.requires_grad = False
        self.densenet.eval()

    def inference(self, subimages):
        subimages = subimages.to(self.device)
        if len(subimages.shape) == 3:        ##single channel at first
            subimages = subimages[:, None, :, :]
        preds = self.densenet(subimages)
        preds = preds.permute(1, 0, 2)  # T N C
        _, preds = preds.max(2)    # T N num_words
        preds = preds.transpose(1, 0) #N T
        preds = preds.cpu().numpy()
        sim_preds = []
        for l in preds:
            char_int_list = []
            for i, c in enumerate(l):
                if c != 0 and (not (i > 0 and l[i - 1] == l[i])):
                    char_int_list.append(int(c))
            char_list = ''.join([self.char_set[k - 1] for k in char_int_list]).strip()
            sim_preds.append(char_list)
        return sim_preds

    def recognize(self, image, bboxes):
        img = Image.fromarray(image).convert('L')
        img = np.array(img)
        W, H = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
        maxratio = int(max(W / H) + 1)
        hh = 32
        ww = hh*maxratio
        batchsize = len(bboxes)
        subimages = np.ones((batchsize, hh, ww))
        for i, [x1, y1, x2, y2] in enumerate(bboxes):
            w, h = x2-x1+1, y2-y1+1
            subimg = img[y1:y2 + 1, x1:x2 + 1]
            resized_w = int(w*32./h)
            subimg = Image.fromarray(subimg).resize([resized_w, 32], Image.ANTIALIAS)
            subimg = np.array(subimg, 'f') / 255.0 - 0.5
            bg = np.ones((hh, ww)) * 0.5
            st = (ww-resized_w)//2
            bg[:, st:st+resized_w] = subimg
            subimages[i, :, :] = bg
        words = self.inference(torch.tensor(subimages).float())
        return {'bboxes': bboxes.tolist(), 'texts': words}

detmodel = DetectModel()
ocrmodel = OCRModel()

def get_prediction(image):
    # 1.  detection predictions
    image = np.asarray(image)
    boxes = detmodel.inference(image)
    boxes = boxes.astype(np.int64)  # box has been resized
    '''
    出于速度考虑，不保存中间的各个结果,做个crnn的并行化处理
    '''
    recognition = ocrmodel.recognize(image, boxes)
    return recognition

@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}
    # Ensure an image was properly uploaded to our endpoint.
    if request.method == 'POST':
        if request.files.get("image"):
            # Read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            data["predictions"] = get_prediction(image)
            # Indicate that the request was a success.
            data["success"] = True

    # Return the data dictionary as a JSON response.
    return jsonify(data)

if __name__ == '__main__':
    app.run()