"""
# author:杨顺帆
# contact: tarsmail@163.com
# datetime:2021/6/18 10:32
# software: PyCharm
文件说明：
"""
from __future__ import print_function
import torch
import math
import torch.utils.data
import pandas as pd
import os
import datetime
import torch.utils.data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data.dataset import Dataset
import wfdb
import pywt
import seaborn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

torch.set_default_tensor_type(torch.DoubleTensor)

# 是否用
is_cuda = False
num_epochs = 100
batch_size = 10
torch.manual_seed(123)
in_channels_ = 1
num_classes = 12
allow_label_leakage = True

device = torch.device("cuda:2" if is_cuda else "cpu")

list_target = []

# 测试集在数据集中所占的比例
RATIO = 0.3

data_path = "data/mit-bih-arrhythmia-database-1.0.0"

ecg_all_class_set = []
count_number_of_class = {'+': 0, 'N': 0, 'A': 0, 'V': 0, '~': 0, '|': 0, 'Q': 0}


# 读取心电数据和对应标签,并对数据进行小波去噪
def getDataSet(number, X_data, Y_data):
    ecgClassSet = ['+', 'N', 'A', 'V', '~', '|', 'Q']

    # 读取心电数据记录
    print("正在读取 " + number + " 号心电数据...")
    record = wfdb.rdrecord('data/mit-bih-arrhythmia-database-1.0.0/' + number, channel_names=['MLII'])
    data = record.p_signal.flatten()
    rdata = denoise(data=data)

    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann('data/mit-bih-arrhythmia-database-1.0.0/' + number, 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol
    for c in Rclass:
        if c not in ecg_all_class_set:
            ecg_all_class_set.append(c)
    # 去掉前后的不稳定数据
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end

    # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
    # X_data在R波前后截取长度为300的数据点
    # Y_data将NAVLR按顺序转换为01234
    while i < j:
        try:
            lable = ecgClassSet.index(Rclass[i])
            if Rlocation[i] - 1199 < 0 or Rlocation[i] + 2401 > Rlocation[len(Rlocation) - 1] or Rclass[
                i] not in ecgClassSet:
                i += 1
                continue
            x_train = rdata[Rlocation[i] - 1199:Rlocation[i] + 2401]
            X_data.append(x_train)
            Y_data.append(lable)
            temp_class = Rclass[i]
            print(count_number_of_class[temp_class])
            count_number_of_class[temp_class] = count_number_of_class[temp_class] + 1
            i += 1
        except ValueError:
            i += 1
    return


# 小波去噪预处理
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


# 加载数据集并进行预处理
def loadData():
    numberSet = ['100', '101']
    dataSet = []
    lableSet = []
    for n in numberSet:
        getDataSet(n, dataSet, lableSet)

    # 转numpy数组,打乱顺序
    dataSet = np.array(dataSet).reshape(-1, 3600)
    lableSet = np.array(lableSet).reshape(-1, 1)
    print(dataSet.shape)
    print(lableSet.shape)
    train_ds = np.hstack((dataSet, lableSet))
    np.random.shuffle(train_ds)

    # 数据集及其标签集
    X = train_ds[:, :3600].reshape(-1, 1, 3600)
    Y = train_ds[:, 3600]

    # 测试集及其标签集
    shuffle_index = np.random.permutation(len(X))
    test_length = int(RATIO * len(shuffle_index))
    test_index = shuffle_index[:test_length]
    train_index = shuffle_index[test_length:]
    X_test, Y_test = X[test_index], Y[test_index]
    X_train, Y_train = X[train_index], Y[train_index]
    return X_train, Y_train, X_test, Y_test


def basic_layer(in_channels, out_channels, kernel_size, batch_norm=False, max_pool=True, conv_stride=1, padding=0
                , pool_stride=2, pool_size=2):
    layer = nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=conv_stride,
                  padding=padding),
        nn.ReLU()
    )
    if batch_norm:
        layer = nn.Sequential(
            layer,
            nn.BatchNorm1d(num_features=out_channels)
        )
    if max_pool:
        layer = nn.Sequential(
            layer,
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        )
    return layer


class mit_bih_classifier(nn.Module):
    def __init__(self, in_channels=in_channels_):
        super(mit_bih_classifier, self).__init__()
        self.cnn = nn.Sequential(
            basic_layer(in_channels=in_channels, out_channels=128, kernel_size=50, batch_norm=True,
                        max_pool=True, conv_stride=3, pool_stride=3, pool_size=2),
            basic_layer(in_channels=128, out_channels=32, kernel_size=7, batch_norm=True,
                        max_pool=True, conv_stride=1, pool_stride=2, pool_size=2),
            basic_layer(in_channels=32, out_channels=32, kernel_size=10, batch_norm=False,
                        max_pool=False, conv_stride=1),
            basic_layer(in_channels=32, out_channels=128, kernel_size=5, conv_stride=2, pool_stride=2,
                        max_pool=True, batch_norm=False),
            basic_layer(in_channels=128, out_channels=256, kernel_size=15, conv_stride=1, pool_size=2,
                        max_pool=True, batch_norm=False),
            basic_layer(in_channels=256, out_channels=512, kernel_size=5, conv_stride=1,
                        max_pool=False, batch_norm=False),
            basic_layer(in_channels=512, out_channels=128, kernel_size=3, conv_stride=1,
                        max_pool=False, batch_norm=False),
            nn.Flatten(),
            nn.Linear(in_features=128 * 3 * 3, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=.1),
            nn.Linear(512, out_features=num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.cnn(x)


class get_pytorch_dataset(Dataset):
    def __init__(self, data, target, transforms_=None, ):
        self.data = data
        self.target = target
        self.transforms = transforms_

    def __getitem__(self, item):
        return self.data[item], self.target[item]

    def __len__(self):
        return self.data.shape[0]


device = torch.device("cuda:2" if is_cuda else "cpu")
model = mit_bih_classifier().to(device).double()
lr = 0.0003

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
loss_fn = nn.CrossEntropyLoss()

X_train, Y_train, X_test, Y_test = loadData()
train_dataset = get_pytorch_dataset(X_train, Y_train)
test_dataset = get_pytorch_dataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
# test_dataset = CustomDatasetFromCSV('./data/Arrhythmia_dataset.pkl')
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10, shuffle=True)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_fn(output.double(), target.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if batch_idx % 10 == 0:
            print('训练回合: {} [{}/{} ({:.0f}%)]\t损失: {:.10f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100 * batch_idx / len(train_loader),
                loss.item()))

    print('====> 训练回合: {} 平均损失: {:.10f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    total_accuracy = 0.00000
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target.long())
            test_loss += loss.item()
            # 准确率
            prediction = torch.argmax(output, dim=1)
            accuracy = (prediction == target).sum().float()
            total_accuracy = total_accuracy + accuracy
            print("========>predicted:", prediction)
            print("========>target:", target)

    print('整体测试集上的损失: {:.10f}'.format(test_loss))
    test_loss /= len(test_loader.dataset)
    print('整体测试集上的平均损失: {:.10f}'.format(test_loss))
    print("整体测试集上的正确率: {:.10f}".format(total_accuracy / len(test_dataset)))


if __name__ == "__main__":
    print(count_number_of_class)
    print(ecg_all_class_set)
    for epoch in range(1, num_epochs + 1):
        train(epoch)
        test(epoch)
        print(len(list_target))
