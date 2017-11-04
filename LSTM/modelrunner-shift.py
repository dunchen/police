import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import random

random.seed()
x_dim=150
h_dim=150
h2_dim=50
h3_dim=25



random.seed()
x_dim=150
h_dim=150
h2_dim=50
h3_dim=25
inputx=[[[0 for shift in range(150)] for col in range(40000)] for row in range(4)]

test_ix=[0 for row in range(45000)]

pred_loss =nn.MSELoss()
stop_loss=nn.CrossEntropyLoss()

fil=pd.read_csv('../all_type_150_count/all_type_150_count_d_reg_shift.csv')
n=107669
day=0
temp_time=20141001
for i in range(n):
        time=fil.ix[i,2].astype(int)
        if time!=temp_time:
                if time==20141001:
                        day=-1
                day=day+1
                temp_time=time
        reg=fil.ix[i,3]
        shift=(fil.ix[i,1]).astype(int)
        if not(pd.isnull(reg)):
                reg2=reg.astype(int)
                count=np.asscalar(fil.ix[i,4])
                #print(mon-10,reg2,count)
                if count<100:
                        inputx[shift][day][reg2]=count/100
                else:
                        inputx[shift][day][reg2]=1
inputx=Variable(torch.FloatTensor(inputx),requires_grad=False)
'''
inputx=[[[0 for shift in range(150)] for col in range(40000)] for row in range(4)]
inputx=[[0 for col in range(150)] for row in range(45000)]
inputt=[0 for row in range(45000)]

test_ix=[0 for row in range(45000)]

pred_loss =nn.MSELoss()
stop_loss=nn.CrossEntropyLoss()


fil=pd.read_csv('../all_type_150_count/all_type_150_count_d_reg_shift.csv')
n=41582-2
day=0
temp_time=20141001
for i in range(n):
        time=fil.ix[i+2,1].astype(int)
        if time!=temp_time:
                day=day+1
                temp_time=time
                inputt[day]=time
        reg=fil.ix[i+2,2]
        if not(pd.isnull(reg)):
                reg2=reg.astype(int)
                count=np.asscalar(fil.ix[i+2,3])
                #print(mon-10,reg2,count)
                if count<200:
                        inputx[day][reg2]=count/200
                else:
                        inputx[day][reg2]=1
inputx=Variable(torch.FloatTensor(inputx),requires_grad=False)
'''


current_shift=3
test_num=0
for i in range(day):
        if (random.random()<(1/13)):
                test_ix[i]=1
                test_num=test_num+1


class stopsign(nn.Module):
    def __init__(self):
        super(stopsign, self).__init__()
        self.fc1=nn.Linear(x_dim+h_dim,h2_dim)
        self.fc2=nn.Linear(h2_dim,h3_dim)
        self.fc3=nn.Linear(h3_dim,2)
        self.relu=nn.ReLU()
        self.softmax=nn.Softmax()

    def forward(self,x):
        ht=self.relu(self.fc1(x))
        ht2=self.relu(self.fc2(ht))
        ht3=self.softmax(self.relu(self.fc3(ht2)))
        return ht3

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm = nn.LSTMCell(x_dim,h_dim)

    def forward(self, x):
        outputs = []
        outputsy = []
        h_t = Variable(torch.zeros(1,h_dim), requires_grad=False)
        c_t = Variable(torch.zeros(1,h_dim), requires_grad=False)
        i=0
        y=torch.FloatTensor([[0,1]])
        while i<30:
            h_t, c_t = self.lstm(inputx[current_shift][x-i-1], (h_t, c_t))
            temp_input=Variable(torch.cat((h_t,inputx[current_shift][x-i-1].view(1,150)),1).data,requires_grad=False)
            y=ystop(temp_input)
            outputs = outputs+ [h_t]
            i=i+1
            outh=h_t
            if (random.random()>y[0][0].data).numpy():
                print(y[0][0].data[0])
                #print(i)
                break


        for j in range(min(10,x-i)):
            h_t, c_t = self.lstm(inputx[current_shift][x-i-1-j], (h_t, c_t))
            temp_input=Variable(torch.cat((h_t,inputx[current_shift][x-i-1-j].view(1,150)),1).data,requires_grad=False).cuda()
            outputs=outputs+ [h_t]

        return outh,outputs,i+9

model=Sequence()
model.load_state_dict(torch.load('./savednet-dynamic-model-1000002017-11-03 20:06:26'))
ystop=stopsign()
ystop.load_state_dict(torch.load('./savednet-dynamic-stopsign-1000002017-11-03 20:06:26'))




xh,_,_=model(40)
xt=pred_loss(xh,inputx[40]).data[0]
for i in range(day):
        if (test_ix[i]==1) and (i>40):
                xh,_,_=model(i)
                xt=xt+pred_loss(xh,inputx[current_shift][i]).data[0]
print(xt/test_num)

