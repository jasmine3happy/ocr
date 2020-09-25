#-*- coding:utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision

##两个loss只写了batchsize=1的情况
class RPN_REGR_Loss(nn.Module):
    def __init__(self, device, sigma=9.0):
        super(RPN_REGR_Loss, self).__init__()
        self.sigma = sigma
        self.device = device

    def forward(self, input, target):
        '''
        smooth L1 loss
        :param input:y_preds
        :param target: y_true
        :return:
        '''
        try:
            cls = target[0, :, 0]
            regr = target[0, :, 1:3]
            regr_keep = (cls == 1).nonzero()[:, 0]
            regr_true = regr[regr_keep]
            regr_pred = input[0][regr_keep]
            diff = torch.abs(regr_true - regr_pred)
            less_one = (diff<1.0/self.sigma).float()
            loss = less_one * 0.5 * diff ** 2 * self.sigma + torch.abs(1- less_one) * (diff - 0.5/self.sigma)
            loss = torch.sum(loss, 1)
            loss = torch.mean(loss) if loss.numel() > 0 else torch.tensor(0.0)
        except Exception as e:
            print('RPN_REGR_Loss Exception:', e)
            # print(input, target)
            loss = torch.tensor(0.0)

        return loss.to(self.device)


class RPN_CLS_Loss(nn.Module):
    def __init__(self,device):
        super(RPN_CLS_Loss, self).__init__()
        self.device = device

    def forward(self, input, target):
        y_true = target[0][0]
        cls_keep = (y_true != -1).nonzero()[:, 0]
        cls_true = y_true[cls_keep].long()
        cls_pred = input[0][cls_keep]
        loss = F.nll_loss(F.log_softmax(cls_pred, dim=-1), cls_true)  # original is sparse_softmax_cross_entropy_with_logits
        # loss = nn.BCEWithLogitsLoss()(cls_pred[:,0], cls_true.float())  # 18-12-8
        #loss = torch.clamp(torch.mean(loss), 0, 10) if loss.numel() > 0 else torch.tensor(0.0)
        loss = torch.mean(loss)
        return loss.to(self.device)


class OHEM_Loss(nn.Module):
    def __init__(self, device):
        super(OHEM_Loss, self).__init__()
        self.device = device

    def smooth_l1_loss(self, regr_pred, regr_target, sigma):
        diff = torch.abs(regr_target - regr_pred)
        less_one = (diff < 1.0 / sigma).float()
        loss = less_one * 0.5 * diff ** 2 * sigma + torch.abs(1 - less_one) * (diff - 0.5 / sigma)
        loss = torch.sum(loss, 1)
        return loss

    def forward(self, cls_pred, cls_target, loc_pred, loc_target, anchors, smooth_l1_sigma=10.0, batch_size=400):
        """
        Arguments:
            batch_size (int): number of sampled rois for bbox head training
            loc_pred (FloatTensor): [R, 2], location of rois
            loc_target (FloatTensor): [R, 2], location of rois
            cls_pred (FloatTensor): [R, C]
            cls_target (LongTensor): [R]
        Returns:
            cls_loss, loc_loss (FloatTensor)
        """
        cls_pred, cls_target = cls_pred[0], cls_target[0][0] #(R, 2) (R,)
        loc_pred, loc_target = loc_pred[0], loc_target[0] #(R, 2)  (R, 2)
        anchors = anchors[0]
        pos_inds = (cls_target==1).nonzero()[:, 0]
        ##for positive rois
        pos_cls_loss = F.cross_entropy(cls_pred[pos_inds], cls_target[pos_inds].long(), reduction='none', ignore_index=-1)
        pos_loc_loss = self.smooth_l1_loss(loc_pred[pos_inds], loc_target[pos_inds], sigma=smooth_l1_sigma)
        # 这里先暂存下正常的分类loss和回归loss
        loss = pos_cls_loss + pos_loc_loss
        # 然后对分类和回归loss求和
        sorted_pos_loss, idx = torch.sort(loss, descending=True)
        # import pdb
        # pdb.set_trace()
        keep = torchvision.ops.nms(anchors[pos_inds][idx], sorted_pos_loss, 0.7)
        sorted_pos_loss, idx = sorted_pos_loss[keep], idx[keep]
        # 再对loss进行降序排列
        keep_num = min(sorted_pos_loss.size()[0], batch_size//2)
        # 得到需要保留的loss数量
        if keep_num < sorted_pos_loss.size()[0]:
            # 这句的作用是如果保留数目小于现有loss总数，则进行筛选保留，否则全部保留
            keep_idx_cuda = idx[:keep_num]
            # 保留到需要keep的数目
            pos_cls_loss = pos_cls_loss[keep_idx_cuda]
            pos_loc_loss = pos_loc_loss[keep_idx_cuda]
            # 分类和回归保留相同的数目
        #cls_loss = pos_cls_loss.sum() / keep_num
        loc_loss = pos_loc_loss.sum() / keep_num

        ##for negative rois
        neg_inds = (cls_target==0).nonzero()[:, 0]
        neg_cls_loss = F.cross_entropy(cls_pred[neg_inds], cls_target[neg_inds].long(), reduction='none', ignore_index=-1)
        sorted_neg_loss, idx = torch.sort(neg_cls_loss, descending=True)
        keep = torchvision.ops.nms(anchors[neg_inds][idx], sorted_neg_loss, 0.7)
        sorted_pos_loss, idx = sorted_neg_loss[keep], idx[keep]
        # 再对loss进行降序排列
        keep_num_neg = min(sorted_neg_loss.size()[0], batch_size // 2)
        # 得到需要保留的loss数量
        if keep_num_neg < sorted_neg_loss.size()[0]:
            # 这句的作用是如果保留数目小于现有loss总数，则进行筛选保留，否则全部保留
            keep_idx_cuda = idx[:keep_num_neg]
            # 保留到需要keep的数目
            neg_cls_loss = neg_cls_loss[keep_idx_cuda]
            # 分类和回归保留相同的数目
        cls_loss = (neg_cls_loss.sum() + pos_cls_loss.sum())/ (keep_num+keep_num_neg)
        # 然后分别对分类和回归loss求均值
        return cls_loss, loc_loss

    def forward2(self, cls_pred, cls_target, loc_pred, loc_target, smooth_l1_sigma=10.0, batch_size=300):
        """
        Arguments:
            batch_size (int): number of sampled rois for bbox head training
            loc_pred (FloatTensor): [R, 2], location of rois
            loc_target (FloatTensor): [R, 2], location of rois
            cls_pred (FloatTensor): [R, C]
            cls_target (LongTensor): [R]
        Returns:
            cls_loss, loc_loss (FloatTensor)
        """
        cls_pred, cls_target = cls_pred[0], cls_target[0][0] #(R, 2) (R,)
        loc_pred, loc_target = loc_pred[0], loc_target[0] #(R, 2)  (R, 2)
        pos_inds = (cls_target==1).nonzero()[:, 0]
        if len(pos_inds)>batch_size//2:
            perm = torch.randperm(len(pos_inds))
            idx = perm[:batch_size//2]
            pos_inds = pos_inds[idx]
        keep_num = min(len(pos_inds), batch_size//2)
        ##for positive rois
        pos_cls_loss = F.cross_entropy(cls_pred[pos_inds], cls_target[pos_inds].long(), reduction='none', ignore_index=-1)
        pos_loc_loss = self.smooth_l1_loss(loc_pred[pos_inds], loc_target[pos_inds], sigma=smooth_l1_sigma)
        #cls_loss = pos_cls_loss.sum() / keep_num
        loc_loss = pos_loc_loss.sum() / keep_num
        ##for negative rois
        neg_inds = (cls_target==0).nonzero()[:, 0]
        if len(neg_inds)>batch_size//2:
            perm = torch.randperm(len(neg_inds))
            idx = perm[:batch_size//2]
            neg_inds = neg_inds[idx]
        keep_num_neg = min(len(neg_inds), batch_size // 2)
        neg_cls_loss = F.cross_entropy(cls_pred[neg_inds], cls_target[neg_inds].long(), reduction='none', ignore_index=-1)
        cls_loss = (neg_cls_loss.sum() + pos_cls_loss.sum())/ (keep_num+keep_num_neg)
        # 然后分别对分类和回归loss求均值
        return cls_loss, loc_loss

class BasicConv(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 relu=True,
                 bn=True,
                 bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CTPN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.vgg16(pretrained=False)
        layers = list(base_model.features)[:-1]
        self.base_layers = nn.Sequential(*layers)  # block5_conv3 output
        self.rpn = BasicConv(512, 512, 3,1,1,bn=False)
        self.brnn = nn.GRU(512,128, bidirectional=True, batch_first=True)
        self.lstm_fc = BasicConv(256, 512,1,1,relu=True, bn=False)
        self.rpn_class = BasicConv(512, 10*2, 1, 1, relu=False,bn=False)
        self.rpn_regress = BasicConv(512, 10 * 2, 1, 1, relu=False, bn=False)

    def forward(self, x):
        x = self.base_layers(x)
        # rpn
        x = self.rpn(x)  #bs c h w

        x1 = x.permute(0,2,3,1).contiguous()  # channels last #bs h w c
        b = x1.size()  # batch_size, h, w, c
        x1 = x1.view(b[0]*b[1], b[2], b[3])
        x2, _ = self.brnn(x1) #bs*h  w  c

        xsz = x.size()
        x3 = x2.view(xsz[0], xsz[2], xsz[3], 256)  # torch.Size([4, 20, 20, 256])
        x3 = x3.permute(0,3,1,2).contiguous()  # bs c h w
        x3 = self.lstm_fc(x3) #bs c h w
        x = x3

        cls = self.rpn_class(x)
        regr = self.rpn_regress(x)

        cls = cls.permute(0,2,3,1).contiguous()
        regr = regr.permute(0,2,3,1).contiguous()

        cls = cls.view(cls.size(0), cls.size(1)*cls.size(2)*10, 2)
        regr = regr.view(regr.size(0), regr.size(1)*regr.size(2)*10, 2)

        return cls, regr
