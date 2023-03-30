import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pylab as pl
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as Data
import pandas as pd
import numpy as np



seed = 666
batch_size = 16
lr = 0.0001
num_epochs = 240
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#Network Structure
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv1d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=10, kernel_size=32, stride=1, padding=1),
                           nn.BatchNorm1d(10),
                           nn.ReLU(),
                           nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                   )

b2 = nn.Sequential(*resnet_block(10, 10, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(10, 20, 2))
b4 = nn.Sequential(*resnet_block(20, 40, 2))
b5 = nn.Sequential(*resnet_block(40, 80, 2))


class RESNet(nn.Module):
    def __init__(self):
        super(RESNet, self).__init__()
        self.net = nn.Sequential(b1, b2, b3, b4, b5)
        self.AvgPool = nn.AdaptiveAvgPool1d(40)


    def forward(self, x):
        x = self.net(x)
        x = self.AvgPool(x)
        return x

class selfattention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cnn = RESNet()
        self.in_channels = in_channels
        self.query = nn.Conv1d(in_channels, in_channels*3, kernel_size=1, stride=1)
        self.key = nn.Conv1d(in_channels, in_channels*3, kernel_size=1, stride=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)


        self.fc = nn.Sequential(
            nn.Linear(80 * 40, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(100, 2)
        )

        self.ConvDrop = nn.Dropout(0.1)
    def forward(self, input):
        a = self.cnn(input)
        batch_size, channels, length = a.shape
        q = self.query(a).view(batch_size, -1, length).permute(0, 2, 1)
        k = self.key(a).view(batch_size, -1, length)
        v = self.value(a).view(batch_size, -1, length)

        attn_matrix = self.softmax(attn_matrix)
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))
        out = out.view(*a.shape)
        out = self.gamma * out + a
        out = out.view(-1, 80 * 40)
        out = self.ConvDrop(self.fc(out))
        return out





net = selfattention(80)
if torch.cuda.is_available():
    net = net.cuda()


weights = [0.2, 0.8]
class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2)
if torch.cuda.is_available():
    criterion = criterion.cuda()
optimizer = torch.optim.Adam(net.parameters(),lr=lr)

start = time.time()



total_train_step = 0
total_test_step = 0



#scheduler = StepLR(optimizer, step_size=10, gamma=0.95)

strat_time = time.time()

train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

for epoch in range(num_epochs):
    running_loss_train = 0
    running_loss_test = 0
    print("------Start of training epoch {}------".format(epoch + 1))


    # testing start
    net.train()
    total_train = 0
    count_train = 0
    correct_train = 0

    total_train_loss = 0
    total_train_accuracy = 0
    for i, data in enumerate(train_loader,0):
        csvs, labels = data
        if torch.cuda.is_available():
            csvs = csvs.cuda()
            labels = labels.cuda()
        csvs = csvs.float()
        csvs = csvs.permute(0, 2, 1)
        optimizer.zero_grad()
        outputs = net(csvs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss_train += loss.item()
        total_train_step = total_train_step + 1

        if total_train_step % 100 == 0:
            print("train_num: {}, train_Loss: {:.4f}".format(total_train_step, loss.item()))

        correct_train += (outputs.argmax(1) == labels).sum()
        total_train += labels.size(0)
        count_train += 1
    print("train_Acc：{:.6f}%".format(count_train, 100 * correct_train / total_train))
    train_acc_list.append(correct_train / total_train)
    end_time = time.time()
    print("train_time: {:.2f}".format(end_time - strat_time))
    train_loss_list.append(running_loss_train)
    #scheduler.step()

    net.eval()
    total_test = 0
    count_test = 0
    correct_test = 0

    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            csvs, labels = data
            if torch.cuda.is_available():
                csvs = csvs.cuda()
                labels = labels.cuda()
            csvs = csvs.float()
            csvs = csvs.permute(0, 2, 1)
            outputs = net(csvs)
            loss_test = criterion(outputs, labels)
            running_loss_test += loss_test.item()
            correct_test += (outputs.argmax(1) == labels).sum()
            total_test += labels.size(0)
            count_test += 1
        print("test_Acc：{:.6f}%".format(count_test, 100 * correct_test / total_test))
        print("test_Loss: {:.4f}".format(loss_test.item()))
        test_acc_list.append(correct_test / total_test)
        test_loss_list.append(running_loss_test)


#acc
plt.figure(figsize=(12, 12), dpi=100)
plt.subplot(2, 2, 1)
test_acc_lines = plt.plot(test_acc_list, 'r', lw=1, label='acc')
plt.title("test_acc")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend(["The Accuracy of Test"])

plt.subplot(2, 2, 2)
test_acc_lines = plt.plot(train_acc_list, 'g', lw=1, label='acc')
plt.title("train_acc")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend(["The Accuracy of Train"])


#loss
plt.subplot(2, 2, 3)
test_loss_lines = plt.plot(train_loss_list, 'b', lw=1, label='loss')
plt.title("train_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["The Loss of Train"])

plt.subplot(2, 2, 4)
test_loss_lines = plt.plot(test_loss_list, 'y', lw=1, label='loss')
plt.title("test_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["The Loss of Test"])

pl.show()




