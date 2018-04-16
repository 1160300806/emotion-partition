from torch import nn
import torch
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
from gensim import corpora, models, matutils
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
SEN_MAX = 30
TRAIN_SET_NUM = 2500
TEST_SET_NUM = 500

torch.manual_seed(1)


def readData():
    senPath = '/home/liberty/Sentiment/sentiment-data/After.txt'
    posPath = '/home/liberty/Sentiment/sentiment-data/Pos.txt'
    posList = []
    senList = []
    with open(senPath, "r") as senIn:
        with open(posPath, "r") as posIn:
            for (sen, pos) in zip(senIn.readlines(), posIn.readlines()):
                pos = eval(pos.replace("\n", ""))
                if pos == -1:
                    posList.append(0)
                elif pos == 0:
                    posList.append(1)
                else:
                    posList.append(2)
                senList.append(sen.lower().replace('\n', '').split(" "))
    return posList, senList


# 生成训练集合、测试集合
posList, senList = readData()
train_senData = senList[:TRAIN_SET_NUM]
train_posData = posList[:TRAIN_SET_NUM]
test_senData = senList[TRAIN_SET_NUM:]
test_posData = posList[TRAIN_SET_NUM:]
# 生成测试集合上的字典
dictionary = corpora.Dictionary(train_senData)

EPOCH = 30
BATCH_SIZE = 50
TIME_STEP = SEN_MAX
INPUT_STEP = len(dictionary)
LR = 0.01
LEN = len(dictionary)


def Train_Data(data):
    ans = np.zeros((TRAIN_SET_NUM, SEN_MAX, LEN), dtype='float32')
    for i in range(TRAIN_SET_NUM):
        sen = data[i]
        length = len(sen)
        for j in range(SEN_MAX):
            if j > length - 1:
                break
            else:
                ans[i, j, dictionary.token2id[sen[j]]] = 1.0
    return torch.from_numpy(ans).type(torch.FloatTensor)


def Test_Data(data):
    ans = np.zeros((TEST_SET_NUM, SEN_MAX, LEN), dtype='float32')
    for i in range(TEST_SET_NUM):
        sen = data[i]
        length = len(sen)
        for j in range(SEN_MAX):
            if j > length - 1:
                break
            elif sen[j] in dictionary.token2id.keys():
                ans[i, j, dictionary.token2id[sen[j]]] = 1.0
            else:
                ans[i, j, dictionary.token2id['unk']] = 1.0
    return torch.from_numpy(ans).type(torch.FloatTensor)


def Train_Targets(data):
    return torch.LongTensor(data)


def Test_Targets(data):
    return torch.LongTensor(data)


train_data = Train_Data(train_senData)
train_targets = Train_Targets(train_posData)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=LEN,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(64, 3)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
torch_dataset = Data.TensorDataset(
    data_tensor=train_data, target_tensor=train_targets)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(loader):
        batch_x = Variable(x.view(-1, SEN_MAX, LEN))
        batch_y = Variable(y)
        output = rnn(batch_x)
        loss = loss_func(output, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(
                Variable(train_data))  # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y == train_targets) / float(TRAIN_SET_NUM)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0],
                  '| test accuracy: %.2f' % accuracy)

test_data = Test_Data(test_senData)
test_targets = Test_Targets(test_posData)

test_output = rnn(Variable(test_data))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
accuracy = sum(pred_y == test_targets) / float(TEST_SET_NUM)
print('Dev accuracy: %2.f' % accuracy)
