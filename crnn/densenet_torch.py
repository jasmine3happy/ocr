import torch
from torch import nn
import torch.nn.functional as F
import math

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels, eps=1.1e-5)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False, eps=1.1e-5)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels, eps=1.1e-5)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.dropout(out, p=0.2, training=self.training)
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet(nn.Module):
    def __init__(self, growthRate=8, depth=28, height = 32, nClasses=80, bottleneck=None):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 8*growthRate
        self.conv1 = nn.Conv2d(1, nChannels, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        #nOutChannels = int(math.floor(nChannels*reduction))
        nOutChannels = 128
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        #nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels, eps=1.1e-5)
        self.classifier = nn.Linear(768, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.shape)
        out = self.conv1(x)
        #print(out.shape)
        #import pdb
        #pdb.set_trace()
        out = self.dense1(out)
        #print(out.shape)
        out = self.trans1(out)
        #print(out.shape)
        out = self.dense2(out)
        #print(out.shape)
        out = self.trans2(out)
        #print(out.shape)
        out = self.dense3(out)
        #print(out.shape)

        out = F.relu(self.bn1(out)) #n c h w ---> n w  h c
        out = out.permute(0, 3, 2, 1)
        #print(out.shape)
        out = torch.flatten(out, start_dim=2, end_dim=3) #n w h*c
        #print(out.shape)

        out = F.log_softmax(self.classifier(out), dim=-1)
        #print(out.shape)
        return out

if __name__ == '__main__':
    #input = Input(shape=(32, 280, 1), name='the_input')
    #dense_cnn(input, 5000)
    input = torch.randn(128, 1, 32, 280)
    densenet = DenseNet()
    o = densenet(input)
