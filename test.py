import json
import cap
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#读取json
with open('user.json','r') as f:
	user=json.loads(f.readline())

#人脸识别模块
now_data=cap.cap1(1)
now_data= torch.from_numpy(now_data).type(torch.FloatTensor).cuda()

#读取神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (3, 200, 200)
            nn.Conv2d(3,9,5,1,2),      # output shape (9, 100, 100)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(9, 18, 5, 1, 2),
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (36, 50, 50)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(18, 36, 5, 1, 2),
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (36, 25, 25)
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(36 * 25 * 25, 50)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 36*25*25)
        output = self.out(x)
        return output
cnn = CNN()
cnn.cuda()
print(cnn)
cnn.load_state_dict(torch.load('cnn1_c'))
#用神经网络模型验证人脸
out=cnn(now_data)

pred_y = torch.max(F.softmax(out, 1), 1)[1].cuda().data.squeeze()
print(torch.max(F.softmax(out, 1), 1))
#验证成功打出GJ
for name,number in user.items():
	if pred_y == number:
		print(pred_y,name,number,"good job")




