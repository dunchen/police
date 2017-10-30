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

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm = nn.LSTMCell(x_dim,h_dim)

    def forward(self, x):
        #outputs,outputsy = []
        h_t = Variable(torch.zeros(1,h_dim), requires_grad=False)
        c_t = Variable(torch.zeros(1,h_dim), requires_grad=False)


        for i in range(10):
            h_t, c_t = self.lstm(inputx[x-(10-i)], (h_t, c_t))
        return h_t

model=Sequence()
model.load_state_dict(torch.load('./savednet-500000'))

print(model(44))
