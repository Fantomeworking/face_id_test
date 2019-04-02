import numpy as np
import numpy
import cap
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import json


EPOCH = 100           # 训练整批数据多少次
BATCH_SIZE = 20
LR = 0.001          # 学习率
error=1#初始错误值

#读取用户字典
with open('user.json','r') as f:
    user1=json.loads(f.readline())
e1=[e1 for e1,e2 in user1.items()]
print(e1)
e2=[e2 for e1,e2 in user1.items()]
Pass=0
while Pass==0:
    #输入用户名：
    ID=input('输入用户名：')
    #用户名序列号
    if str(ID) in e1:
        print('用户名重复，请重新输入')#检测用户名是否重复
    else:
        #读取摄像头数据
        Face_Train=cap.cap1(BATCH_SIZE)
        user1.update({str(ID):e2[-1]+1})
        Pass=1

Face_all_Train=numpy.load("all_Face.npy")
Face_all_Train=numpy.append(Face_all_Train,Face_Train,0)


#BATCH=np.random.choice(range(1,Face_all_Train.shape[0]+1),BATCH_SIZE,replace=False)

Face_Train_x = torch.from_numpy(Face_all_Train).type(torch.FloatTensor).cuda()
np_y=np.array([np.full(20,x1) for x1 in np.array(list(user1.values()))[1:]])
Face_Train_y=torch.from_numpy(np.array([[[np_y.reshape(np_y.shape[1]*np_y.shape[0])]]]).transpose(3,1,2,0)).type(torch.LongTensor).cuda()
print(Face_Train_x.size(),Face_Train_y.size(),np_y.reshape(np_y.shape[1]*np_y.shape[0]))

train_loader = Data.DataLoader(dataset=Data.TensorDataset(
    Face_Train_x,Face_Train_y), batch_size=BATCH_SIZE, shuffle=True)
#b_y转化为long
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (3, 200, 200)
            nn.Conv2d(3,9,5,1,2),      # output shape (9, 100, 100)
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(9, 18, 5, 1, 2),
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (18, 50, 50)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(18, 36, 5, 1, 2),
            nn.ReLU(),  # activationf
            nn.MaxPool2d(2),  # output shape (36, 25, 25)
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(36 * 25 * 25, 50)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output

cnn = CNN()
cnn.cuda()
# optimize all cnn parameters
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted

for x in range(EPOCH):

    for step, (b_x,b_y) in enumerate(train_loader):   # 分配 batch data, normalize x when iterate train_loader
        
        output = cnn(b_x.cuda())               # cnn output
        loss = loss_func(output, b_y[:,0,0,0].cuda())   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
    error=loss.item()

torch.save(cnn.state_dict(),'cnn1_c')#存储训练结果
numpy.save("all_Face.npy",Face_all_Train)#存储所有的脸部数据
numpy.save("/home/guimu/桌面/tset/人脸识别/face_data/"
            +str(ID)+"_Face.npy",Face_Train)#此处训练新加的脸部数据
with open ('user.json','w') as f:
    json.dump(user1,f)