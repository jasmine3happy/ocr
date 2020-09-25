#-*- coding:utf-8 -*-
from torch.utils.data import DataLoader
import torch
import torch.backends.cudnn as cudnn
from densenet_torch import DenseNet
from densecrnn_torch import DenseCRNN
from dataset import OCRDataset
from densenet_wrapper import DenseNet121

def test(crnn, test_loader, char_set):

    print('Start test')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    crnn = crnn.to(device)
    for p in crnn.parameters():
        p.requires_grad = False
    crnn.eval()
    n_correct = 0
    lines = []
    for i_batch, (image, labels, _, _) in enumerate(test_loader):
        #print(i_batch)
        if i_batch > 10000:
            break
        image = image.to(device)
        labels = labels.to(device)
        if len(image.shape) == 3:        ##singe channel at first
            image = image[:, None, :, :]
        preds = crnn(image)
        preds = preds.permute(1, 0, 2)  # T N C
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0)
        preds = preds.cpu().numpy()

        sim_preds = []
        for l in preds:
            char_int_list = []
            char_list = []
            for i,c in enumerate(l):
                if c != 0 and (not (i > 0 and l[i - 1] == l[i])):
                    char_int_list.append(int(c))
            char_list = ''.join([char_set[k - 1] for k in char_int_list ]).strip()
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

        for pred, target in zip(sim_preds, targets):  #word correct
            print(pred, "  ", target)
            if pred == target:
                n_correct += 1
            else:
                lines.append(pred+ "   "+ target+'\n')
    with open('wrong.txt', 'w', encoding='utf-8') as f:
        for l in lines:
            f.write(l)

    accuracy = n_correct / min(10000, len(test_dataset))
    print('test num is ', len(test_dataset))
    print('accuray: %f' % (accuracy))
    return accuracy

if __name__ == '__main__':
    char_set = open('real.txt', 'r', encoding='utf-8').readlines()
    char_set = ''.join([ch.strip('\n') for ch in char_set])
    nclass = len(char_set)+1  #+1 is for blank
    imagesize = (32, 320)
    batch_size = 1
    maxlabellength = 40
    image_path = 'D:\\workspace\\OCR_dataset\\annotations\\images'
    #image_path = 'C:\\Users\\a811241\\Desktop\\ppt'

    test_labelfile = 'C:\\Users\\a811241\\Downloads\\FakeOCR_ordered_alltext\\data_test.txt'
    test_labelfile = './test_data310.txt'
    test_dataset = OCRDataset(image_path, test_labelfile, maxlabellength = maxlabellength, train=False, imagesize=imagesize)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    densenet = DenseCRNN(nClasses=nclass)
    pretrained_path = './checkpoints/DenseCRNN_49_6.pth'
    if pretrained_path is not None:
        print('loading pretrained model from {}'.format(pretrained_path))
        densenet.load_state_dict(torch.load(pretrained_path))

    test(densenet, test_loader, char_set)