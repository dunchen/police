from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import svm
import random

random.seed()
num=3000
x_dim=100
h_dim=100
h2_dim=50
h3_dim=25

pred_loss =nn.MSELoss()
stop_loss=nn.CrossEntropy()

class stopsign(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.fc1=nn.Linear(x_dim+h_dim,h2_dim)
        self.fc2=nn.Linear(h2_dim,h3_dim)
        self.fc3=nn.Linear(h3_dim,2)
        self.relu=nn.ReLU()
        self.softmax()=nn.Softmax()     
    
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
        outputs,outputsy = []
        h_t = Variable(torch.zeros(h_dim).double(), requires_grad=False)
        c_t = Variable(torch.zeros(h_dim).double(), requires_grad=False)

        y=0
        i=0      
        while random.random()>y[0]:
            h_t, c_t = self.lstm(x[i], (h_t, c_t))
            y=ystop(torch.cat((h_t.data,x[i+1].data))
            outputs+= [h_t]
            outputsy+= [y]
            i=i+1
            outh=h_t

        for j in range(30):
            h_t, c_t = self.lstm(x[i+j], (h_t, c_t))
            y=ystop(torch.cat((h_t.data,x[i+1].data))
            outputs+= [h_t]
            outputsy+= [y]

        return outh,outputs,outputsy,i+29

def trainstopsign(outputs,outputsy,size,target):
    minx=9999
    minloss=9999
    for i in range(size):
        loss=pred_loss(outputs[i],target)
        if loss<minloss:
            minloss=loss
            minx=i
    p1=torch.longTensor(1)
    p1[0]=1
    p1[1]=0
    p2=torch.longTensor(1)
    p2[0]=0
    p2[1]=1

    for i in range(size):
        if minx<i:
            loss=stop_loss(outputsy[i],p1)
            loss.backward()
            stop_optimizer.step()
        else:
            loss=stop_loss(outputsy[i],p2)
            loss.backward()
            stop_optimizer.step()
    return

def trainpred(out,target):
    loss=pred_loss(out,target)
    loss.backward()
    pred_optimizer.step()
    return

def svm_label(outputs,size,target):
    label=[]
    loss=criterion(outputs[0],target)
    for i in range(size-1):
        losst=criterion(outputs[i+1],target)
        if loss>losst:
            label=+[1]
        else:
            label=+[0]
    return label
    

model=Sequence()
ystop=stopsign()
pred_optimizer=optim.LBFGS(model.parameters(),lr=0.8)
stop_optimizer=optim.Adam(ystop.parameters(),lr=1e-3)

for i in range(num):
	out,outs,outsy,size=model(inputx[i])
	trainstopsign(outs,outsy,size,inputy[i])
        trainpred(out,y[i])





