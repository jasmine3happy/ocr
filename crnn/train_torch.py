#-*- coding:utf-8 -*-
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils import data
from densenet_torch import DenseNet
from densecrnn_torch import DenseCRNN
from densenet_wrapper import DenseNet121
from dataset import OCRDataset
import utils


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_input = None
            return

    def next(self):
        data = self.next_data
        self.preload()
        return data

def val(crnn, val_loader, criterion, iteration, device):
    print('Start val')

    char_set = open('real.txt', 'r', encoding='utf-8').readlines()
    char_set = ''.join([ch.strip('\n') for ch in char_set])

    for p in crnn.parameters():
        p.requires_grad = False
    crnn.eval()
    n_correct = 0
    loss_avg = utils.averager()
    for i_batch, (image, labels, label_length, input_length) in enumerate(val_loader):
        image = image.to(device)
        labels = labels.to(device)
        label_length = label_length.to(device)
        input_length = input_length.to(device)
        if len(image.shape) == 3:  ##singe channel at first
            image = image[:, None, :, :]
        preds = crnn(image)
        preds = preds.permute(1, 0, 2)  # T N C
        batch_size = image.size(0)
        cost = criterion(preds, labels, input_length, label_length) / batch_size
        loss_avg.add(cost)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0)
        preds = preds.cpu().numpy()

        sim_preds = []
        for l in preds:
            char_int_list = []
            char_list = []
            for i, c in enumerate(l):
                if c != 0 and (not (i > 0 and l[i - 1] == l[i])):
                    char_int_list.append(int(c))
                if c == 0:
                    char_int_list.append(2)
            char_list = ''.join([char_set[k - 1] for k in char_int_list])#.strip()
            sim_preds.append(char_list)

        targets = []
        for l in labels.cpu().numpy():
            char_int_list = []
            char_list = []
            for i in l:
                if i == 10000:
                    break
                char_int_list.append(int(i))
            char_list = ''.join([char_set[k - 1] for k in char_int_list]).strip()
            targets.append(char_list)

        for pred, target in zip(sim_preds, targets):  # word correct
            if pred == target:
                n_correct += 1
        if (i_batch+1)%20 == 0:
            print('[%d][%d/%d]' %
                      (iteration, i_batch, len(val_loader)))

    print(n_correct)
    accuracy = n_correct / (128*len(val_loader))
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))

    return accuracy

def train(crnn, train_loader, criterion, iteration, device, optimizer, val_loader):

    for p in crnn.parameters():
        p.requires_grad = True
    crnn.train()
    loss_avg = utils.averager()
    for i_batch, (image, labels, label_length, input_length) in enumerate(train_loader):
        image = image.to(device)
        labels = labels.to(device)
        label_length = label_length.to(device)
        input_length = input_length.to(device)
        if len(image.shape) == 3:  ##singe channel at first
            image = image[:, None, :, :]
        preds = crnn(image)
        preds = preds.permute(1, 0, 2)  # T N C
        batch_size = image.size(0)

        cost = criterion(preds, labels, input_length, label_length) / batch_size
        crnn.zero_grad()
        cost.backward()
        optimizer.step()
        loss_avg.add(cost)
        if (i_batch+1) % 20 == 0:
            with open('train.log.txt', 'a+') as f:
                line = '[%d][%d/%d] Loss: %f' % (iteration, i_batch, len(train_loader), loss_avg.val())
                f.write(line + '\n')
            print('[%d][%d/%d] Loss: %f' %
                  (iteration, i_batch, len(train_loader), loss_avg.val()))
            loss_avg.reset()
        '''
        if (i_batch+1)%20000 == 0:
            torch.save(densenet.state_dict(),
                       '{0}/densenet121_Rec_done_{1}_{2}.pth'.format('./checkpoints', iteration, i_batch))
        '''

def main(densenet, train_loader, val_loader, criterion, optimizer):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    densenet = densenet.to(device)
    criterion = criterion.to(device)
    Iteration = 0
    save_dir = './checkpoints'
    niter =50
    bestaccuracy = 0
    while Iteration < niter:
        train(densenet, train_loader, criterion, Iteration, device, optimizer, val_loader)
        ## max_i: cut down the consuming time of testing, if you'd like to validate on the whole testset, please set it to len(val_loader)

        #accuracy = val(densenet, val_loader, criterion, Iteration, device)
        #for p in densenet.parameters():
          #  p.requires_grad = True

        accuracy = 6
        if accuracy >= bestaccuracy:
            torch.save(densenet.state_dict(),
                       '{0}/DenseCRNN_{1}_{2}.pth'.format(save_dir, Iteration, accuracy))
            torch.save(densenet.state_dict(), '{0}/densenet_best.pth'.format(save_dir))
            #bestaccuracy = accuracy
        print("is best accuracy: {0}".format(accuracy))
        if (Iteration+1)%10 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.4
        Iteration += 1


if __name__ == '__main__':
    char_set = open('real.txt', 'r', encoding='utf-8').readlines()
    char_set = ''.join([ch.strip('\n') for ch in char_set])
    nclass = len(char_set)+1  #+1 is for blank
    print('Num of classes is {}'.format(nclass))
    imagesize = (32, 320)
    batch_size = 128
    maxlabellength = 40
    image_path = 'D:\\workspace\\OCR_dataset\\annotations\\images'
    train_labelfile = './real_data/train_data_valid47.txt'
    val_labelfile = './train_data310.txt'
    train_dataset = OCRDataset(image_path, train_labelfile, maxlabellength = maxlabellength, imagesize=imagesize, train=True)
    val_dataset = OCRDataset(image_path, val_labelfile, maxlabellength = maxlabellength, imagesize=imagesize, train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # shuffle=True, just for time consuming.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = torch.nn.CTCLoss(reduction='sum')

    #densenet = DenseNet(nClasses=nclass)
    densenet = DenseCRNN(nClasses=nclass)
    pretrained_path = './checkpoints/DenseCRNN_pretrained.pth'
    if pretrained_path is not None:
        print('loading pretrained model from {}'.format(pretrained_path))
        densenet.load_state_dict(torch.load(pretrained_path), strict=False)

    # setup optimizer
    optimizer = optim.Adam(densenet.parameters(), lr=0.0005, betas=(0.5, 0.999))
    #optimizer = optim.Adadelta(densenet.parameters(), lr=0.001)
    main(densenet, train_loader, val_loader, criterion, optimizer)