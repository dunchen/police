from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#from sklearn import svm

import random
import pandas as pd

random.seed()
x_dim=150
h_dim=150
h2_dim=50
h3_dim=25
inputx=[[0 for col in range(150)] for row in range(100)]

pred_loss =nn.MSELoss()
stop_loss=nn.CrossEntropyLoss()

fil=pd.read_csv('all_type_150_count.csv')
n=1727-2
num=46-9
for i in range(n):
	mon=fil.ix[i+2,1].astype(int)
	reg=fil.ix[i+2,2]
	if not(pd.isnull(reg)):
		reg2=reg.astype(int)
		count=np.asscalar(fil.ix[i+2,3])
		#print(mon-10,reg2,count)
		inputx[mon-10][reg2]=count/(count+1)
inputx=Variable(torch.FloatTensor(inputx),requires_grad=False)

#print(inputx[34])


def min(a,b):
	if a<b:
		return a
	else:
		return b

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
        while 1==1:
            h_t, c_t = self.lstm(inputx[x-i-1], (h_t, c_t))
            y=ystop(torch.cat((h_t,inputx[x-i-1].view(1,150)),1))
            outputs = outputs+ [h_t]
            outputsy =outputsy+ [y]
            i=i+1
            outh=h_t
            if (random.random()>y[0][0].data).numpy(): break

        for j in range(min(3,35-x)):
            h_t, c_t = self.lstm(inputx[x-i-1-j], (h_t, c_t))
            y=ystop(torch.cat((h_t,inputx[x-i-1-j].view(1,150)),1))
            outputs=outputs+ [h_t[0]]
            outputsy=outputsy+ [y]
        
        return outh,outputs,outputsy,i+min(3,35-x)

def trainstopsign(outputs,outputsy,size,target):
    minx=9999
    minloss=Variable(torch.FloatTensor([9999]))
    for i in range(size):
        loss=pred_loss(outputs[i],target)
        if (loss.data<minloss.data).numpy():
            minloss=loss
            minx=i
    p1=Variable(torch.LongTensor([0]))
    p2=Variable(torch.LongTensor([1]))

    for i in range(size):
        if minx<i:
            stop_optimizer.zero_grad()
            loss=stop_loss(outputsy[i],p1)
            loss.backward()
            stop_optimizer.step()
            stop_optimizer.zero_grad()
        else:
            stop_optimizer.zero_grad()
            loss=stop_loss(outputsy[i],p2)
            loss.backward()
            stop_optimizer.step()
    return


def trainpred(out,target):
    loss=pred_loss(out,target)
    loss.backward()
    pred_optimizer.step()
    return

'''
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
''' 

model=Sequence()
ystop=stopsign()
pred_optimizer=optim.SGD(model.parameters(),lr=0.5)
stop_optimizer=optim.Adam(ystop.parameters(),lr=1e-3)

#print(inputx[35])
#print(model(35))
#io=pred_loss(model((5)*7),inputx[(5)*7])
#print(io)

for i in range(1000):
	t=random.randint(5,46-9)
	if not(t % 7):
		pred_optimizer.zero_grad()
		stop_optimizer.zero_grad()
		out,outs,outsy,size=model(t)
		trainpred(out,inputx[t])
		trainstopsign(outs,outsy,size,inputx[t])

x=pred_loss(model(35),inputx[35])
for i in range(35): x=x+pred_loss(model(i),inputx[i])
print(x/36)




