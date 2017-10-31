import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

x_dim=150
h_dim=150
inputx=[[0 for col in range(150)] for row in range(45000)]
inputt=[0 for row in range(45000)]

fil=pd.read_csv('all_type_150_count_d_reg.csv')
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
        h_t = Variable(torch.zeros(1,h_dim), requires_grad=False).cuda()
        c_t = Variable(torch.zeros(1,h_dim), requires_grad=False).cuda()
        i=0
        y=torch.FloatTensor([[0,1]])
        while i<30:
            h_t, c_t = self.lstm(inputx[x-i-1].cuda(), (h_t, c_t))
            temp_input=Variable(torch.cat((h_t,inputx[x-i-1].view(1,150)),1).data,requires_grad=False)
            y=ystop(temp_input)
            outputs = outputs+ [h_t]
            i=i+1
            outh=h_t
            if (random.random()>y[0][0].data).cpu().numpy():
                #print(y[0][0].data[0])
                #print(i)
                break


        for j in range(min(10,x-i)):
            h_t, c_t = self.lstm(inputx[x-i-1-j].cuda(), (h_t, c_t))
            temp_input=Variable(torch.cat((h_t,inputx[x-i-1-j].view(1,150)),1).data,requires_grad=False).cuda()
            outputs=outputs+ [h_t]

        return outh,outputs,i+9

model=Sequence()
model.load_state_dict(torch.load('./savednet-dynamic-model-1000002017-10-31 02:03:03'))
ystop=stopsign()
ystop.load_state_dict(torch.load('./savednet-dynamic-stopsign-1000002017-10-31 02:02:46'))
